"""Job registry, event bus and worker execution.

Analysis runs on a background thread so the API stays responsive. Progress is
published to an in-process event bus which the WebSocket endpoint forwards to
the browser. All progress numbers come from real counters - processed frame
index, measured throughput, live track count - never from a timer.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import settings

log = logging.getLogger("visiontrack.jobs")


class JobStatus:
    READY = "ready"
    UPLOADING = "uploading"
    QUEUED = "queued"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"

    TERMINAL = frozenset({COMPLETE, FAILED, CANCELLED, READY})
    RUNNING = frozenset({QUEUED, EXTRACTING, ANALYZING})


@dataclass
class JobHandle:
    """Live control surface for one running job."""

    job_id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
    started_at: float = field(default_factory=time.time)
    future: Any = None
    last_progress: dict[str, Any] = field(default_factory=dict)

    def cancel(self) -> None:
        self.cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()


class EventBus:
    """Fan-out of job events to any number of WebSocket subscribers.

    Publishers are worker threads; subscribers live on the asyncio loop, so
    delivery hops threads via `call_soon_threadsafe`. Each subscriber also gets
    a replay of recent events so a client that connects mid-job is immediately
    in sync instead of waiting for the next tick.
    """

    def __init__(self, replay: int = 40):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._history: dict[str, deque[dict[str, Any]]] = {}
        self._replay = replay
        self._lock = threading.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        with self._lock:
            self._subscribers.setdefault(job_id, set()).add(queue)
            history = list(self._history.get(job_id, ()))
        for event in history:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover
                break
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id)
            if subs:
                subs.discard(queue)
                if not subs:
                    self._subscribers.pop(job_id, None)

    def publish(self, job_id: str, event: dict[str, Any]) -> None:
        event = {**event, "jobId": job_id, "at": time.time()}
        with self._lock:
            hist = self._history.setdefault(job_id, deque(maxlen=self._replay))
            # Keep the replay buffer useful: collapse consecutive progress ticks.
            if event.get("type") == "progress" and hist and hist[-1].get("type") == "progress":
                hist[-1] = event
            else:
                hist.append(event)
            queues = list(self._subscribers.get(job_id, ()))

        if not queues:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def _dispatch() -> None:
            for q in queues:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    # Slow consumer: drop the oldest and keep the newest state.
                    try:
                        q.get_nowait()
                        q.put_nowait(event)
                    except Exception:  # pragma: no cover
                        pass

        try:
            loop.call_soon_threadsafe(_dispatch)
        except RuntimeError:  # pragma: no cover - loop shutting down
            pass

    def clear(self, job_id: str) -> None:
        with self._lock:
            self._history.pop(job_id, None)


class JobManager:
    """Owns the worker pool and the set of in-flight jobs."""

    def __init__(self, max_workers: int | None = None):
        workers = max_workers or max(1, settings.max_concurrent_jobs)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vt-job")
        # Exports get their own pool: analysis is GPU-bound and deliberately
        # serialised, but waiting for it before writing a CSV would be absurd.
        self._export_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vt-export")
        self._handles: dict[str, JobHandle] = {}
        self._lock = threading.Lock()
        self.bus = EventBus()

    # -- lifecycle -------------------------------------------------------------

    def _submit_to(
        self, pool: ThreadPoolExecutor, key: str, target: Callable[[JobHandle], None]
    ) -> JobHandle:
        with self._lock:
            existing = self._handles.get(key)
            if existing is not None and existing.future is not None and not existing.future.done():
                raise JobAlreadyRunning(key)
            handle = JobHandle(job_id=key)
            self._handles[key] = handle

        def _run() -> None:
            try:
                target(handle)
            except Exception:  # pragma: no cover - target handles its own errors
                log.exception("task %s crashed outside its error handler", key)
            finally:
                with self._lock:
                    if self._handles.get(key) is handle:
                        self._handles.pop(key, None)

        handle.future = pool.submit(_run)
        return handle

    def submit(self, job_id: str, target: Callable[[JobHandle], None]) -> JobHandle:
        return self._submit_to(self._pool, job_id, target)

    def submit_export(self, export_id: str, target: Callable[[JobHandle], None]) -> JobHandle:
        return self._submit_to(self._export_pool, export_id, target)

    def handle(self, job_id: str) -> JobHandle | None:
        with self._lock:
            return self._handles.get(job_id)

    def cancel(self, job_id: str) -> bool:
        handle = self.handle(job_id)
        if handle is None:
            return False
        handle.cancel()
        return True

    def is_running(self, job_id: str) -> bool:
        handle = self.handle(job_id)
        return handle is not None and handle.future is not None and not handle.future.done()

    def running_ids(self) -> list[str]:
        with self._lock:
            return [
                jid for jid, h in self._handles.items()
                if h.future is not None and not h.future.done()
            ]

    def active_count(self) -> int:
        return len(self.running_ids())

    def shutdown(self, wait: bool = False) -> None:
        for jid in self.running_ids():
            self.cancel(jid)
        self._pool.shutdown(wait=wait, cancel_futures=True)
        self._export_pool.shutdown(wait=wait, cancel_futures=True)

    # -- progress publishing ---------------------------------------------------

    def emit(self, job_id: str, event_type: str, **payload: Any) -> None:
        self.bus.publish(job_id, {"type": event_type, **payload})

    def emit_progress(self, handle: JobHandle, payload: dict[str, Any]) -> None:
        handle.last_progress = payload
        self.bus.publish(handle.job_id, {"type": "progress", **payload})


class JobAlreadyRunning(RuntimeError):
    def __init__(self, job_id: str):
        super().__init__(f"Job {job_id} is already running")
        self.job_id = job_id


class ProgressTracker:
    """Turns raw frame counters into an honest ETA.

    The rate is an exponentially-weighted average of *recently observed*
    frames-per-second, so the estimate reflects what this machine is actually
    doing right now - including a slow patch on a crowded scene - rather than a
    fixed guess.
    """

    def __init__(self, total_units: int, *, smoothing: float = 0.25):
        self.total = max(1, total_units)
        self.smoothing = smoothing
        self.start = time.perf_counter()
        self._last_time = self.start
        self._last_units = 0
        self.rate: float | None = None

    def update(self, units_done: int) -> dict[str, Any]:
        now = time.perf_counter()
        dt = now - self._last_time
        d_units = units_done - self._last_units

        if dt >= 0.20 and d_units > 0:
            instant = d_units / dt
            self.rate = instant if self.rate is None else (
                self.smoothing * instant + (1 - self.smoothing) * self.rate
            )
            self._last_time = now
            self._last_units = units_done

        elapsed = now - self.start
        # Fall back to the whole-run average until the first window closes.
        rate = self.rate if self.rate else (units_done / elapsed if elapsed > 0 else 0.0)
        remaining = max(0, self.total - units_done)
        eta = remaining / rate if rate and rate > 0 else None

        return {
            "processed": units_done,
            "total": self.total,
            "progress": min(1.0, units_done / self.total),
            "elapsed": round(elapsed, 2),
            "rate": round(rate, 2) if rate else 0.0,
            "etaSeconds": round(eta, 1) if eta is not None else None,
        }

    @property
    def average_rate(self) -> float:
        elapsed = time.perf_counter() - self.start
        return (self._last_units / elapsed) if elapsed > 0 else 0.0


def new_job_id() -> str:
    return uuid.uuid4().hex[:16]


manager = JobManager()
