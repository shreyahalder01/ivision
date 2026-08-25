"""The analysis pipeline.

    VIDEO -> FRAME EXTRACTION -> DETECTION -> CONFIDENCE FILTER -> TRACKER
          -> PERSISTENT IDs -> RESULT STORAGE -> ANNOTATION -> ANALYTICS/EXPORT

Design constraints that shape this module:

* Nothing proportional to video length is held in memory. Frames stream in over
  a pipe, detections flush to SQLite in blocks.
* Cancellation is checked every batch, and whatever has been flushed stays
  valid, so a cancelled job yields real partial results and can be resumed from
  the last persisted frame.
* Progress is measured, never simulated. Frame counters come from the decoder;
  the ETA comes from observed throughput.
* A GPU OOM halves the batch and retries rather than failing the job.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import COCO_CLASSES, RESULT_DIR, settings
from ..errors import error_payload
from . import analytics, detector, store
from .detector import DetectorUnavailable
from .ffmpeg import FFmpegError, FFmpegUnavailable, probe_video
from .frames import FrameReader, inference_dimensions
from .jobs import JobHandle, JobStatus, ProgressTracker, manager
from .tracker import PersistentTracker, TrackerConfig

log = logging.getLogger("visiontrack.pipeline")


class AnalysisCancelled(Exception):
    """Raised internally to unwind to the partial-save path."""


_error = error_payload


def class_ids_for(names: Sequence[str], available: dict[int, str]) -> list[int]:
    lookup = {v.lower(): k for k, v in available.items()}
    return sorted({lookup[n.lower()] for n in names if n.lower() in lookup})


def _results_path(job_id: str) -> Path:
    return RESULT_DIR / f"{job_id}.json"


def write_results_file(
    job_id: str,
    *,
    fps: float,
    frame_stride: int,
    width: int,
    height: int,
    frame_map: dict[int, list[list[float]]],
    tracks: list[dict[str, Any]],
    partial: bool,
) -> Path:
    """Compact playback payload.

    Detections are stored as positional arrays keyed by frame number and
    normalised to 0..1 so the overlay renders at any display size. This is ~5x
    smaller than an array of objects, which matters at 70k+ detections.
    """
    payload = {
        "jobId": job_id,
        "schema": "visiontrack.results/1",
        "fps": fps,
        "frameStride": frame_stride,
        "sourceWidth": width,
        "sourceHeight": height,
        "partial": partial,
        # [trackId, classId, x, y, w, h, confidence] - all box values normalised
        "fields": ["trackId", "classId", "x", "y", "w", "h", "confidence"],
        "frames": {str(k): v for k, v in sorted(frame_map.items())},
        "tracks": [
            {
                "trackId": t["track_id"],
                "classId": t["class_id"],
                "className": t["class_name"],
                "firstFrame": t["first_frame"],
                "lastFrame": t["last_frame"],
                "firstSeen": t["first_seen"],
                "lastSeen": t["last_seen"],
                "duration": t["duration"],
                "detectionCount": t["detection_count"],
                "avgConfidence": t["avg_confidence"],
                "maxSpeed": t["max_speed"],
                "path": t.get("path") or [],
            }
            for t in tracks
        ],
    }
    dest = _results_path(job_id)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp.replace(dest)  # atomic: a reader never sees a half-written file
    return dest


def load_results_file(job_id: str) -> dict[str, Any] | None:
    path = _results_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_analysis(job_id: str, handle: JobHandle, *, resume: bool = False) -> None:
    """Execute one analysis job. Owns all status transitions for that job."""
    job = store.get_job(job_id)
    if job is None:
        log.error("job %s vanished before analysis started", job_id)
        return

    t_start = time.time()
    store.update_job(
        job_id, status=JobStatus.EXTRACTING, stage="Preparing analysis",
        started_at=t_start, error=None, progress=0.0,
    )
    manager.emit(job_id, "status", status=JobStatus.EXTRACTING, stage="Preparing analysis")

    detections_buffer: list[store.DetRow] = []
    frame_map: dict[int, list[list[float]]] = {}
    tracker: PersistentTracker | None = None
    reader: FrameReader | None = None
    processed = 0
    start_frame = 0
    resolved_key = job.get("model", "auto")
    device = "cpu"

    try:
        # ---------------------------------------------------------- 1. validate
        video_path = Path(job["video_path"])
        if not video_path.exists():
            raise PipelineFailure(_error(
                "video_missing", "SOURCE VIDEO MISSING",
                "The video file for this job is no longer on disk.",
                "The upload was deleted or moved after the job was created.",
                "Re-upload the footage to run this analysis again.",
            ))

        selected = list(job.get("selectedClasses") or [])
        if not selected:
            raise PipelineFailure(_error(
                "no_classes", "NO OBJECT CLASSES SELECTED",
                "Analysis needs at least one object class to look for.",
                "The job was created with an empty class selection.",
                "Select one or more classes, then start the analysis again.",
            ))

        metadata = job.get("videoMetadata")
        if not metadata:
            manager.emit(job_id, "status", status=JobStatus.EXTRACTING, stage="Extracting metadata")
            metadata = probe_video(video_path).to_dict()
            store.update_job(job_id, metadata=metadata)

        fps = float(metadata.get("fps") or 30.0)
        src_w = int(metadata.get("width") or 1920)
        src_h = int(metadata.get("height") or 1080)
        total_frames = int(metadata.get("frame_count") or metadata.get("frameCount") or 0)
        frame_stride = max(1, int(job.get("frame_stride") or 1))

        # -------------------------------------------------- 2. runtime + model
        env = detector.require_runtime()
        device = env["device"]

        resolved_key, note = detector.resolve_model(job.get("model", "auto"), metadata=metadata)
        if note:
            manager.emit(job_id, "notice", message=note)
        store.update_job(job_id, resolved_model=resolved_key, device=device)

        def weight_progress(message: str, fraction: float) -> None:
            manager.emit(
                job_id, "stage", stage=message,
                stageProgress=fraction, status=JobStatus.EXTRACTING,
            )

        manager.emit(job_id, "status", status=JobStatus.EXTRACTING, stage="Initializing model")
        det = detector.Detector(resolved_key, device=device)
        det.load(weight_progress)
        det.warmup()

        names = det.class_names
        wanted_ids = class_ids_for(selected, names)
        if not wanted_ids:
            raise PipelineFailure(_error(
                "classes_unsupported", "SELECTED CLASSES NOT SUPPORTED",
                "None of the selected classes exist in this detection model.",
                f"{detector.MODEL_SPECS[resolved_key].label} does not expose: "
                f"{', '.join(selected)}.",
                "Select classes supported by this model, or choose a different model.",
            ))

        # ------------------------------------------------------- 3. resume point
        if resume:
            last = store.last_processed_frame(job_id)
            if last >= 0:
                # Restart on a stride boundary after the last persisted frame.
                start_frame = last + frame_stride
                store.clear_detections(job_id, from_frame=start_frame)
                manager.emit(
                    job_id, "notice",
                    message=f"Resuming from frame {start_frame:,} of {total_frames:,}",
                )
        else:
            store.clear_detections(job_id)
            _results_path(job_id).unlink(missing_ok=True)

        # ------------------------------------------------- 4. streaming decode
        inf_w, inf_h = inference_dimensions(src_w, src_h)
        # Speeds are reported in source pixels even though tracking runs on the
        # downscaled frames, so "px/frame" means what a user would expect.
        speed_scale = src_w / inf_w if inf_w else 1.0

        tracking_method = job.get("tracking_method", "persistent")
        tracker = PersistentTracker(
            TrackerConfig.for_method(tracking_method, fps),
            id_offset=store.max_track_id(job_id) if resume else 0,
            frame_width=inf_w, frame_height=inf_h,
        )

        confidence = float(job.get("confidence", 0.30))
        iou = float(job.get("iou", 0.50))
        use_byte = tracker.cfg.use_byte_association
        # BYTE needs sub-threshold candidates, so query the detector lower and
        # split the results ourselves. Without BYTE we let the model filter.
        query_conf = min(confidence, tracker.cfg.low_conf_floor) if use_byte else confidence

        planned = max(1, (total_frames - start_frame) // frame_stride) if total_frames else 0
        progress = ProgressTracker(planned or 1)

        store.update_job(
            job_id, status=JobStatus.ANALYZING, stage="Detecting objects",
            total_frames=total_frames,
        )
        manager.emit(
            job_id, "status", status=JobStatus.ANALYZING, stage="Detecting objects",
            model=resolved_key, device=device, deviceName=env.get("deviceName"),
            gpuAccelerated=env.get("gpuAccelerated", False),
            totalFrames=total_frames, frameStride=frame_stride,
            batchSize=detector.batch_size_for(env, resolved_key),
        )

        reader = FrameReader(
            video_path, width=inf_w, height=inf_h, fps=fps,
            start_frame=start_frame, stride=frame_stride,
        )

        batch_size = detector.batch_size_for(env, resolved_key)
        batch_frames: list[np.ndarray] = []
        batch_indices: list[int] = []
        class_totals: dict[str, int] = {}
        last_emit = 0.0
        detections_written = store.detection_count(job_id) if resume else 0

        def flush_detections(force: bool = False) -> None:
            nonlocal detections_buffer, detections_written
            if not detections_buffer:
                return
            if force or len(detections_buffer) >= 2000:
                store.insert_detections(detections_buffer)
                detections_written += len(detections_buffer)
                detections_buffer = []

        def run_batch() -> None:
            """Detect + track one batch, with OOM backoff."""
            nonlocal batch_size, processed, last_emit
            if not batch_frames:
                return

            attempt_frames = list(batch_frames)
            attempt_indices = list(batch_indices)
            results: list[dict[str, np.ndarray]] = []
            chunk = len(attempt_frames)

            while True:
                try:
                    results = []
                    for i in range(0, len(attempt_frames), chunk):
                        results.extend(det.detect_batch(
                            attempt_frames[i:i + chunk],
                            conf=query_conf, iou=iou, class_ids=wanted_ids,
                        ))
                    break
                except DetectorUnavailable as exc:
                    if exc.code == "gpu_oom" and chunk > 1:
                        chunk = max(1, chunk // 2)
                        batch_size = chunk
                        detector.release_models()
                        det.load()
                        manager.emit(
                            job_id, "notice",
                            message=f"GPU memory tight - reduced batch size to {chunk}",
                        )
                        continue
                    raise

            for local_i, (frame_index, result) in enumerate(zip(attempt_indices, results)):
                image = attempt_frames[local_i]
                timestamp = frame_index / fps if fps > 0 else 0.0

                xyxy, confs, clss = result["xyxy"], result["conf"], result["cls"]
                # -------------------------------- confidence filtering (real)
                high = confs >= confidence
                low = (~high) if use_byte else np.zeros_like(high)

                hi_boxes = xyxy[high]
                hi_conf = confs[high]
                hi_cls = clss[high]
                hi_names = [names.get(int(c), COCO_CLASSES.get(int(c), str(c))) for c in hi_cls]

                lo_boxes = xyxy[low] if use_byte else None
                lo_conf = confs[low] if use_byte else None
                lo_cls = clss[low] if use_byte else None
                lo_names = (
                    [names.get(int(c), COCO_CLASSES.get(int(c), str(c))) for c in lo_cls]
                    if use_byte and lo_cls is not None else None
                )

                live = tracker.update(
                    hi_boxes, [int(c) for c in hi_cls], hi_names, [float(c) for c in hi_conf],
                    frame_index, timestamp, image,
                    low_boxes=lo_boxes,
                    low_class_ids=[int(c) for c in lo_cls] if lo_cls is not None else None,
                    low_class_names=lo_names,
                    low_confidences=[float(c) for c in lo_conf] if lo_conf is not None else None,
                )

                rows: list[list[float]] = []
                for obj in live:
                    x1, y1, x2, y2 = obj["box"]
                    nx = max(0.0, min(1.0, x1 / inf_w))
                    ny = max(0.0, min(1.0, y1 / inf_h))
                    nw = max(0.0, min(1.0 - nx, (x2 - x1) / inf_w))
                    nh = max(0.0, min(1.0 - ny, (y2 - y1) / inf_h))
                    if nw <= 0 or nh <= 0:
                        continue
                    detections_buffer.append((
                        job_id, frame_index, round(timestamp, 4), obj["trackId"],
                        obj["classId"], obj["className"], round(obj["confidence"], 5),
                        round(nx, 5), round(ny, 5), round(nw, 5), round(nh, 5),
                    ))
                    rows.append([
                        obj["trackId"], obj["classId"],
                        round(nx, 4), round(ny, 4), round(nw, 4), round(nh, 4),
                        round(obj["confidence"], 3),
                    ])
                    class_totals[obj["className"]] = class_totals.get(obj["className"], 0) + 1
                if rows:
                    frame_map[frame_index] = rows

                processed += 1

            flush_detections()

            # ------------------------------------------ real progress reporting
            snapshot = progress.update(processed)
            now = time.perf_counter()
            if now - last_emit >= 0.20:
                last_emit = now
                last_index = attempt_indices[-1] if attempt_indices else start_frame
                manager.emit_progress(handle, {
                    "status": JobStatus.ANALYZING,
                    "stage": "Detecting objects",
                    "frame": last_index,
                    "totalFrames": total_frames,
                    "processedFrames": processed,
                    "plannedFrames": progress.total,
                    "progress": snapshot["progress"],
                    "fps": snapshot["rate"],
                    "etaSeconds": snapshot["etaSeconds"],
                    "elapsed": snapshot["elapsed"],
                    "activeTracks": tracker.active_count(),
                    "uniqueObjects": tracker._next_public - 1,
                    "detections": detections_written + len(detections_buffer),
                    "classTotals": dict(class_totals),
                    "batchSize": batch_size,
                    "timestamp": (attempt_indices[-1] / fps) if attempt_indices and fps else 0.0,
                })

            batch_frames.clear()
            batch_indices.clear()

        # ----------------------------------------------------- 5. the main loop
        for frame_index, frame in reader.frames():
            if handle.cancelled:
                raise AnalysisCancelled()
            # The pipe hands us a view into the read buffer; copy before it moves.
            batch_frames.append(frame.copy())
            batch_indices.append(frame_index)
            if len(batch_frames) >= batch_size:
                run_batch()
                if processed % settings.persist_every_frames == 0:
                    flush_detections(force=True)
                    store.update_job(
                        job_id, processed_frames=processed,
                        progress=min(1.0, processed / max(1, progress.total)),
                    )

        run_batch()  # trailing partial batch
        flush_detections(force=True)

        # ------------------------------------------------------- 6. finalize
        manager.emit(job_id, "status", status=JobStatus.ANALYZING, stage="Building tracks")
        _finalize(
            job_id, tracker, fps=fps, frame_stride=frame_stride,
            inf_w=inf_w, inf_h=inf_h, src_w=src_w, src_h=src_h,
            speed_scale=speed_scale, metadata=metadata, selected=selected,
            frame_map=frame_map, partial=False,
        )

        elapsed = time.time() - t_start
        eff_fps = processed / elapsed if elapsed > 0 else 0.0
        detector.record_throughput(resolved_key, device, eff_fps)

        summary = store.get_job(job_id)
        store.update_job(
            job_id, status=JobStatus.COMPLETE, stage=None, progress=1.0,
            processed_frames=processed, finished_at=time.time(),
            partial=0, processing_fps=round(eff_fps, 2), error=None,
        )
        manager.emit(
            job_id, "complete",
            status=JobStatus.COMPLETE,
            processedFrames=processed,
            elapsed=round(elapsed, 2),
            processingFps=round(eff_fps, 2),
            results=(summary or {}).get("results"),
        )
        log.info(
            "job %s complete: %d frames in %.1fs (%.1f fps) on %s",
            job_id, processed, elapsed, eff_fps, device,
        )

    except AnalysisCancelled:
        _save_partial(
            job_id, tracker, detections_buffer, frame_map, processed,
            job=job, reason="cancelled",
        )

    except DetectorUnavailable as exc:
        _fail(job_id, exc.to_error(), detections_buffer, tracker, frame_map, processed, job)

    except FFmpegUnavailable as exc:
        _fail(job_id, _error(
            "ffmpeg_unavailable", "VIDEO ENGINE UNAVAILABLE",
            str(exc),
            "No usable FFmpeg binary could be located on this machine.",
            "Install FFmpeg and restart the backend, or run "
            "`pip install imageio-ffmpeg` for a bundled build.",
            exc.detail(),
        ), detections_buffer, tracker, frame_map, processed, job)

    except FFmpegError as exc:
        _fail(job_id, _error(
            "decode_failed", "VIDEO COULD NOT BE DECODED",
            exc.message,
            "The file may use an unsupported codec, or be truncated or corrupted.",
            "Re-encode the footage to H.264 MP4 and try again.",
            exc.detail(),
        ), detections_buffer, tracker, frame_map, processed, job)

    except MemoryError:
        _fail(job_id, _error(
            "out_of_memory", "INSUFFICIENT MEMORY",
            "The system ran out of memory during analysis.",
            "The video resolution or batch size exceeded available RAM.",
            "Close other applications, or select a smaller model and retry.",
        ), detections_buffer, tracker, frame_map, processed, job)

    except PipelineFailure as exc:
        _fail(job_id, exc.payload, detections_buffer, tracker, frame_map, processed, job)

    except Exception as exc:  # unexpected - still explain, still keep partials
        log.exception("job %s failed unexpectedly", job_id)
        _fail(job_id, _error(
            "unexpected", "ANALYSIS FAILED",
            "Analysis stopped because of an unexpected error.",
            f"{exc.__class__.__name__}: {exc}",
            "Retry the analysis. If it keeps failing, try a different model or video.",
            repr(exc),
        ), detections_buffer, tracker, frame_map, processed, job)

    finally:
        if reader is not None:
            reader.close()


class PipelineFailure(Exception):
    def __init__(self, payload: dict[str, Any]):
        super().__init__(payload.get("message", "Analysis failed"))
        self.payload = payload


def _finalize(
    job_id: str,
    tracker: PersistentTracker | None,
    *,
    fps: float,
    frame_stride: int,
    inf_w: int,
    inf_h: int,
    src_w: int,
    src_h: int,
    speed_scale: float,
    metadata: dict[str, Any],
    selected: Sequence[str],
    frame_map: dict[int, list[list[float]]],
    partial: bool,
) -> dict[str, Any]:
    """Build the tracks table, summary and playback file from tracker state."""
    rows: list[tuple] = []
    if tracker is not None:
        rows = analytics.summarize_tracks_for_storage(
            tracker.finalize(), fps=fps, frame_stride=frame_stride,
            width=inf_w, height=inf_h, job_id=job_id,
        )
        # Rescale motion metrics from inference pixels to source pixels.
        rescaled: list[tuple] = []
        for r in rows:
            r = list(r)
            r[12] = round(float(r[12]) * speed_scale, 3)  # max_speed
            r[13] = round(float(r[13]) * speed_scale, 3)  # avg_speed
            r[14] = round(float(r[14]) * speed_scale, 2)  # distance
            rescaled.append(tuple(r))
        rows = rescaled
        store.replace_tracks(job_id, rows)

    tracks = store.get_tracks(job_id)
    total_detections = store.detection_count(job_id)
    summary = analytics.build_summary(
        tracks, total_detections=total_detections, metadata=metadata,
        selected_classes=list(selected), frame_stride=frame_stride,
    )
    summary["partial"] = partial
    store.update_job(job_id, summary=summary)

    write_results_file(
        job_id, fps=fps, frame_stride=frame_stride,
        width=src_w, height=src_h, frame_map=frame_map,
        tracks=tracks, partial=partial,
    )
    return summary


def _save_partial(
    job_id: str,
    tracker: PersistentTracker | None,
    buffer: list[store.DetRow],
    frame_map: dict[int, list[list[float]]],
    processed: int,
    *,
    job: dict[str, Any],
    reason: str,
) -> None:
    """Persist everything analysed so far so a cancel is never destructive."""
    try:
        if buffer:
            store.insert_detections(buffer)
            buffer.clear()
        metadata = job.get("videoMetadata") or {}
        fps = float(metadata.get("fps") or 30.0)
        src_w = int(metadata.get("width") or 1920)
        src_h = int(metadata.get("height") or 1080)
        inf_w, inf_h = inference_dimensions(src_w, src_h)
        summary = _finalize(
            job_id, tracker, fps=fps,
            frame_stride=max(1, int(job.get("frame_stride") or 1)),
            inf_w=inf_w, inf_h=inf_h, src_w=src_w, src_h=src_h,
            speed_scale=src_w / inf_w if inf_w else 1.0,
            metadata=metadata, selected=job.get("selectedClasses") or [],
            frame_map=frame_map, partial=True,
        )
    except Exception:
        log.exception("job %s: failed to save partial results", job_id)
        summary = None

    store.update_job(
        job_id, status=JobStatus.CANCELLED, stage=None, partial=1,
        processed_frames=processed, finished_at=time.time(),
        error=_error(
            "cancelled", "ANALYSIS CANCELLED",
            "Analysis was cancelled before it finished.",
            "The job was stopped from the interface.",
            "Resume to continue from the last saved frame, or start over.",
        ),
    )
    manager.emit(
        job_id, "cancelled",
        status=JobStatus.CANCELLED, processedFrames=processed,
        partial=True, results=summary,
    )
    log.info("job %s cancelled after %d frames (partials saved)", job_id, processed)


def _fail(
    job_id: str,
    error: dict[str, Any],
    buffer: list[store.DetRow],
    tracker: PersistentTracker | None,
    frame_map: dict[int, list[list[float]]],
    processed: int,
    job: dict[str, Any],
) -> None:
    """Record a failure, keeping any partial work that was already analysed."""
    partial_saved = False
    try:
        if buffer:
            store.insert_detections(buffer)
            buffer.clear()
        if processed > 0 and tracker is not None:
            metadata = job.get("videoMetadata") or {}
            fps = float(metadata.get("fps") or 30.0)
            src_w = int(metadata.get("width") or 1920)
            src_h = int(metadata.get("height") or 1080)
            inf_w, inf_h = inference_dimensions(src_w, src_h)
            _finalize(
                job_id, tracker, fps=fps,
                frame_stride=max(1, int(job.get("frame_stride") or 1)),
                inf_w=inf_w, inf_h=inf_h, src_w=src_w, src_h=src_h,
                speed_scale=src_w / inf_w if inf_w else 1.0,
                metadata=metadata, selected=job.get("selectedClasses") or [],
                frame_map=frame_map, partial=True,
            )
            partial_saved = True
    except Exception:
        log.exception("job %s: could not save partials after failure", job_id)

    error = {**error, "partialSaved": partial_saved, "processedFrames": processed}
    store.update_job(
        job_id, status=JobStatus.FAILED, stage=None, error=error,
        processed_frames=processed, finished_at=time.time(),
        partial=1 if partial_saved else 0,
    )
    manager.emit(job_id, "failed", status=JobStatus.FAILED, error=error)
    log.error("job %s failed: %s (%s)", job_id, error.get("code"), error.get("message"))
