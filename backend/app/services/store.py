"""SQLite persistence for jobs, detections and tracks.

Detections are the high-volume table (a 4-minute 30fps clip with 10 objects per
frame is ~72k rows), so they are written in batches inside a single transaction
and read back through indexed frame ranges. Every write path is safe to
interrupt: a cancelled job keeps whatever was flushed.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ..config import DB_PATH

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS jobs (
    id                  TEXT PRIMARY KEY,
    filename            TEXT NOT NULL,
    original_name       TEXT NOT NULL,
    status              TEXT NOT NULL,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    started_at          REAL,
    finished_at         REAL,
    video_path          TEXT NOT NULL,
    poster_path         TEXT,
    sprite_path         TEXT,
    sprite_meta         TEXT,
    metadata_json       TEXT,
    classes_json        TEXT NOT NULL,
    confidence          REAL NOT NULL,
    iou                 REAL NOT NULL DEFAULT 0.5,
    model               TEXT NOT NULL,
    resolved_model      TEXT,
    tracking_method     TEXT NOT NULL,
    annotation_style    TEXT NOT NULL DEFAULT 'box_label',
    frame_stride        INTEGER NOT NULL DEFAULT 1,
    progress            REAL NOT NULL DEFAULT 0,
    processed_frames    INTEGER NOT NULL DEFAULT 0,
    total_frames        INTEGER NOT NULL DEFAULT 0,
    stage               TEXT,
    device              TEXT,
    error_json          TEXT,
    summary_json        TEXT,
    partial             INTEGER NOT NULL DEFAULT 0,
    processing_fps      REAL,
    parent_job_id       TEXT
);

CREATE TABLE IF NOT EXISTS detections (
    job_id      TEXT NOT NULL,
    frame       INTEGER NOT NULL,
    ts          REAL NOT NULL,
    track_id    INTEGER NOT NULL,
    class_id    INTEGER NOT NULL,
    class_name  TEXT NOT NULL,
    confidence  REAL NOT NULL,
    x           REAL NOT NULL,
    y           REAL NOT NULL,
    w           REAL NOT NULL,
    h           REAL NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_det_job_frame ON detections(job_id, frame);
CREATE INDEX IF NOT EXISTS idx_det_job_track ON detections(job_id, track_id);

CREATE TABLE IF NOT EXISTS tracks (
    job_id          TEXT NOT NULL,
    track_id        INTEGER NOT NULL,
    class_id        INTEGER NOT NULL,
    class_name      TEXT NOT NULL,
    first_frame     INTEGER NOT NULL,
    last_frame      INTEGER NOT NULL,
    first_seen      REAL NOT NULL,
    last_seen       REAL NOT NULL,
    duration        REAL NOT NULL,
    detection_count INTEGER NOT NULL,
    avg_confidence  REAL NOT NULL,
    max_confidence  REAL NOT NULL,
    max_speed       REAL NOT NULL DEFAULT 0,
    avg_speed       REAL NOT NULL DEFAULT 0,
    distance        REAL NOT NULL DEFAULT 0,
    gap_count       INTEGER NOT NULL DEFAULT 0,
    path_json       TEXT,
    PRIMARY KEY(job_id, track_id),
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS exports (
    id          TEXT PRIMARY KEY,
    job_id      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    fmt         TEXT NOT NULL,
    status      TEXT NOT NULL,
    progress    REAL NOT NULL DEFAULT 0,
    path        TEXT,
    size_bytes  INTEGER,
    options_json TEXT,
    error_json  TEXT,
    created_at  REAL NOT NULL,
    finished_at REAL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_exports_job ON exports(job_id);
"""

_local = threading.local()
_init_lock = threading.Lock()
_initialised = False


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def get_conn() -> sqlite3.Connection:
    """One connection per thread; the worker pool and API threads stay isolated."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    return conn


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


@contextmanager
def read_connection() -> Iterator[sqlite3.Connection]:
    """A short-lived, read-only-by-convention connection for streaming scans.

    Exports walk every detection row while simultaneously committing progress
    updates. Doing both on one connection means COMMIT fires while a SELECT
    cursor is still active, which SQLite may refuse. WAL mode lets an
    independent reader hold a consistent snapshot for as long as it needs
    without blocking those writes.
    """
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    global _initialised
    with _init_lock:
        if _initialised:
            return
        conn = get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()
        # Any job left mid-flight by a hard shutdown is not silently "running".
        conn.execute(
            """UPDATE jobs SET status='cancelled', partial=1, stage='interrupted',
                   error_json=?, updated_at=?
               WHERE status IN ('queued','extracting','analyzing','exporting')""",
            (
                json.dumps({
                    "code": "interrupted",
                    "title": "ANALYSIS INTERRUPTED",
                    "message": "The application closed while this job was running.",
                    "cause": "The backend process stopped before analysis finished.",
                    "action": "Resume the job to continue from the last saved frame.",
                }),
                time.time(),
            ),
        )
        conn.commit()
        _initialised = True


# --------------------------------------------------------------------------- jobs

def _job_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for key, target in (
        ("metadata_json", "videoMetadata"),
        ("classes_json", "selectedClasses"),
        ("error_json", "error"),
        ("summary_json", "results"),
        ("sprite_meta", "spriteMeta"),
    ):
        raw = d.pop(key, None)
        d[target] = json.loads(raw) if raw else None
    return d


def create_job(job: dict[str, Any]) -> None:
    now = time.time()
    with transaction() as conn:
        conn.execute(
            """INSERT INTO jobs (
                id, filename, original_name, status, created_at, updated_at,
                video_path, poster_path, sprite_path, sprite_meta, metadata_json,
                classes_json, confidence, iou, model, tracking_method,
                annotation_style, frame_stride, total_frames, stage, parent_job_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                job["id"], job["filename"], job["original_name"], job["status"],
                now, now, job["video_path"], job.get("poster_path"),
                job.get("sprite_path"),
                json.dumps(job["sprite_meta"]) if job.get("sprite_meta") else None,
                json.dumps(job["metadata"]) if job.get("metadata") else None,
                json.dumps(job.get("classes", [])),
                job.get("confidence", 0.30), job.get("iou", 0.50),
                job.get("model", "auto"), job.get("tracking_method", "persistent"),
                job.get("annotation_style", "box_label"),
                job.get("frame_stride", 1),
                job.get("total_frames", 0), job.get("stage"),
                job.get("parent_job_id"),
            ),
        )


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    mapping = {
        "metadata": ("metadata_json", json.dumps),
        "classes": ("classes_json", json.dumps),
        "error": ("error_json", lambda v: json.dumps(v) if v is not None else None),
        "summary": ("summary_json", lambda v: json.dumps(v) if v is not None else None),
        "sprite_meta": ("sprite_meta", lambda v: json.dumps(v) if v is not None else None),
    }
    cols: list[str] = []
    vals: list[Any] = []
    for key, value in fields.items():
        col, coerce = mapping.get(key, (key, None))
        cols.append(f"{col}=?")
        vals.append(coerce(value) if coerce else value)
    cols.append("updated_at=?")
    vals.append(time.time())
    vals.append(job_id)
    with transaction() as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(cols)} WHERE id=?", vals)


def get_job(job_id: str) -> dict[str, Any] | None:
    row = get_conn().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _job_row_to_dict(row) if row else None


def list_jobs(limit: int = 100, offset: int = 0, status: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM jobs"
    args: list[Any] = []
    if status:
        sql += " WHERE status=?"
        args.append(status)
    sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    return [_job_row_to_dict(r) for r in get_conn().execute(sql, args).fetchall()]


def count_jobs(status: str | None = None) -> int:
    if status:
        row = get_conn().execute("SELECT COUNT(*) c FROM jobs WHERE status=?", (status,)).fetchone()
    else:
        row = get_conn().execute("SELECT COUNT(*) c FROM jobs").fetchone()
    return int(row["c"]) if row else 0


def delete_job(job_id: str) -> dict[str, Any] | None:
    job = get_job(job_id)
    if job is None:
        return None
    with transaction() as conn:
        conn.execute("DELETE FROM detections WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM tracks WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM exports WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    return job


def find_job_by_content(video_path: str) -> dict[str, Any] | None:
    """Used to de-duplicate re-uploads of an identical file."""
    row = get_conn().execute(
        "SELECT * FROM jobs WHERE video_path=? ORDER BY created_at DESC LIMIT 1",
        (video_path,),
    ).fetchone()
    return _job_row_to_dict(row) if row else None


# --------------------------------------------------------------- detections/tracks

DetRow = tuple[str, int, float, int, int, str, float, float, float, float, float]


def insert_detections(rows: Sequence[DetRow]) -> None:
    if not rows:
        return
    with transaction() as conn:
        conn.executemany(
            """INSERT INTO detections
               (job_id, frame, ts, track_id, class_id, class_name, confidence, x, y, w, h)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )


def clear_detections(job_id: str, *, from_frame: int | None = None) -> None:
    with transaction() as conn:
        if from_frame is None:
            conn.execute("DELETE FROM detections WHERE job_id=?", (job_id,))
            conn.execute("DELETE FROM tracks WHERE job_id=?", (job_id,))
        else:
            conn.execute("DELETE FROM detections WHERE job_id=? AND frame>=?", (job_id, from_frame))


def trim_tracks(job_id: str, from_frame: int, *, width: int, height: int, fps: float) -> int:
    """Re-derive the surviving track rows after detections were rolled back.

    Resuming discards every detection at or after `from_frame`, which leaves the
    track rows describing frames that no longer exist - an inspector claiming an
    object was visible for eight seconds when only three seconds of it survived.
    Tracks that began inside the discarded range are dropped entirely; the rest
    are recomputed from the detections that remain, at full precision rather than
    from the down-sampled stored path.

    Returns the number of tracks removed.
    """
    with transaction() as conn:
        cur = conn.execute(
            "DELETE FROM tracks WHERE job_id=? AND first_frame>=?", (job_id, from_frame)
        )
        dropped = cur.rowcount or 0

        survivors = [
            int(r["track_id"]) for r in
            conn.execute("SELECT track_id FROM tracks WHERE job_id=?", (job_id,))
        ]
        for tid in survivors:
            rows = conn.execute(
                "SELECT frame, ts, confidence, x, y, w, h FROM detections "
                "WHERE job_id=? AND track_id=? ORDER BY frame", (job_id, tid),
            ).fetchall()
            if not rows:
                # Every one of this track's detections was in the discarded
                # range even though it started before it - nothing left to describe.
                conn.execute(
                    "DELETE FROM tracks WHERE job_id=? AND track_id=?", (job_id, tid)
                )
                dropped += 1
                continue

            confs = [float(r["confidence"]) for r in rows]
            pts = [
                (int(r["frame"]),
                 (float(r["x"]) + float(r["w"]) / 2) * width,
                 (float(r["y"]) + float(r["h"]) / 2) * height)
                for r in rows
            ]
            distance = 0.0
            max_speed = 0.0
            speeds: list[float] = []
            for (f0, x0, y0), (f1, x1, y1) in zip(pts, pts[1:]):
                step = math.hypot(x1 - x0, y1 - y0)
                distance += step
                span = max(1, f1 - f0)
                speed = step / span
                speeds.append(speed)
                max_speed = max(max_speed, speed)

            first_frame, last_frame = pts[0][0], pts[-1][0]
            path = [
                [f, round(cx / width, 5), round(cy / height, 5)]
                for f, cx, cy in _downsample(pts, 160)
            ]
            conn.execute(
                """UPDATE tracks SET
                     first_frame=?, last_frame=?, first_seen=?, last_seen=?, duration=?,
                     detection_count=?, avg_confidence=?, max_confidence=?,
                     max_speed=?, avg_speed=?, distance=?, path_json=?
                   WHERE job_id=? AND track_id=?""",
                (
                    first_frame, last_frame,
                    round(float(rows[0]["ts"]), 4), round(float(rows[-1]["ts"]), 4),
                    round(max(0.0, (last_frame - first_frame) / fps) if fps > 0 else 0.0, 4),
                    len(rows),
                    round(sum(confs) / len(confs), 5), round(max(confs), 5),
                    round(max_speed, 3),
                    round(sum(speeds) / len(speeds), 3) if speeds else 0.0,
                    round(distance, 2), json.dumps(path),
                    job_id, tid,
                ),
            )
    return dropped


def _downsample(points: Sequence[Any], budget: int) -> Sequence[Any]:
    if len(points) <= budget:
        return points
    step = len(points) / budget
    return [points[min(len(points) - 1, int(i * step))] for i in range(budget)]


def last_processed_frame(job_id: str) -> int:
    row = get_conn().execute(
        "SELECT MAX(frame) m FROM detections WHERE job_id=?", (job_id,)
    ).fetchone()
    return int(row["m"]) if row and row["m"] is not None else -1


def max_track_id(job_id: str) -> int:
    row = get_conn().execute(
        "SELECT MAX(track_id) m FROM detections WHERE job_id=?", (job_id,)
    ).fetchone()
    return int(row["m"]) if row and row["m"] is not None else 0


def iter_detections(job_id: str, start: int = 0, end: int | None = None) -> Iterable[sqlite3.Row]:
    if end is None:
        return get_conn().execute(
            "SELECT * FROM detections WHERE job_id=? AND frame>=? ORDER BY frame, track_id",
            (job_id, start),
        )
    return get_conn().execute(
        "SELECT * FROM detections WHERE job_id=? AND frame>=? AND frame<=? ORDER BY frame, track_id",
        (job_id, start, end),
    )


def stream_detections(
    conn: sqlite3.Connection, job_id: str, *, by_track: bool = False
) -> Iterable[sqlite3.Row]:
    """Ordered detection scan on a caller-owned connection (see `read_connection`)."""
    order = "track_id, frame" if by_track else "frame, track_id"
    return conn.execute(
        f"SELECT * FROM detections WHERE job_id=? ORDER BY {order}", (job_id,)
    )


def detection_count(job_id: str) -> int:
    row = get_conn().execute(
        "SELECT COUNT(*) c FROM detections WHERE job_id=?", (job_id,)
    ).fetchone()
    return int(row["c"]) if row else 0


def replace_tracks(
    job_id: str, rows: Sequence[tuple], *, keep_existing: bool = False
) -> None:
    """Write the track table for a job.

    `keep_existing` is for resumed jobs: the resumed tracker only knows about
    objects seen after the resume point, so wiping the table would orphan every
    detection written before it. Track IDs are offset past the highest stored ID
    on resume, so an upsert cannot collide with an earlier segment's tracks.
    """
    with transaction() as conn:
        if not keep_existing:
            conn.execute("DELETE FROM tracks WHERE job_id=?", (job_id,))
        if rows:
            verb = "INSERT OR REPLACE INTO" if keep_existing else "INSERT INTO"
            conn.executemany(
                f"""{verb} tracks (
                    job_id, track_id, class_id, class_name, first_frame, last_frame,
                    first_seen, last_seen, duration, detection_count, avg_confidence,
                    max_confidence, max_speed, avg_speed, distance, gap_count, path_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )


def get_tracks(job_id: str) -> list[dict[str, Any]]:
    rows = get_conn().execute(
        "SELECT * FROM tracks WHERE job_id=? ORDER BY first_frame, track_id", (job_id,)
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        raw = d.pop("path_json", None)
        d["path"] = json.loads(raw) if raw else []
        out.append(d)
    return out


def get_track(job_id: str, track_id: int) -> dict[str, Any] | None:
    row = get_conn().execute(
        "SELECT * FROM tracks WHERE job_id=? AND track_id=?", (job_id, track_id)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    raw = d.pop("path_json", None)
    d["path"] = json.loads(raw) if raw else []
    return d


# ------------------------------------------------------------------------ exports

def create_export(export: dict[str, Any]) -> None:
    with transaction() as conn:
        conn.execute(
            """INSERT INTO exports (id, job_id, kind, fmt, status, options_json, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                export["id"], export["job_id"], export["kind"], export["fmt"],
                export.get("status", "queued"),
                json.dumps(export.get("options", {})), time.time(),
            ),
        )


def update_export(export_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols, vals = [], []
    for key, value in fields.items():
        if key == "error":
            key, value = "error_json", (json.dumps(value) if value is not None else None)
        cols.append(f"{key}=?")
        vals.append(value)
    vals.append(export_id)
    with transaction() as conn:
        conn.execute(f"UPDATE exports SET {', '.join(cols)} WHERE id=?", vals)


def get_export(export_id: str) -> dict[str, Any] | None:
    row = get_conn().execute("SELECT * FROM exports WHERE id=?", (export_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    for key, target in (("options_json", "options"), ("error_json", "error")):
        raw = d.pop(key, None)
        d[target] = json.loads(raw) if raw else None
    return d


def list_exports(job_id: str) -> list[dict[str, Any]]:
    rows = get_conn().execute(
        "SELECT * FROM exports WHERE job_id=? ORDER BY created_at DESC", (job_id,)
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for key, target in (("options_json", "options"), ("error_json", "error")):
            raw = d.pop(key, None)
            d[target] = json.loads(raw) if raw else None
        out.append(d)
    return out


def all_video_paths() -> set[str]:
    return {r["video_path"] for r in get_conn().execute("SELECT video_path FROM jobs")}


def all_artifact_paths() -> set[str]:
    paths: set[str] = set()
    for col in ("video_path", "poster_path", "sprite_path"):
        for r in get_conn().execute(f"SELECT {col} p FROM jobs WHERE {col} IS NOT NULL"):
            paths.add(r["p"])
    for r in get_conn().execute("SELECT path p FROM exports WHERE path IS NOT NULL"):
        paths.add(r["p"])
    return paths
