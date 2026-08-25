"""Export pipeline: annotated/clean MP4, JSON, CSV, COCO and YOLO annotations.

Two rules shape everything here:

* **Streaming.** Detection rows are scanned in frame order straight out of
  SQLite and written as they are read. A million-detection job exports with the
  same memory footprint as a hundred-detection one. Video exports likewise
  stream frames through an FFmpeg decode pipe into an encode pipe.
* **Real progress.** Every format reports progress from a counter of work
  actually done - rows written, frames encoded - so the UI's percentage is
  measured, never interpolated from a timer.

Exports run on the shared worker pool and are cancellable. A cancelled or failed
export deletes its partial file: a truncated CSV that looks complete is worse
than no file.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from ..config import CLASS_GROUPS, EXPORT_DIR, VERSION, settings
from ..errors import error_payload
from . import store
from .annotate import Annotator, OverlayOptions, format_timecode, resolve_style
from .ffmpeg import FFmpegError, FFmpegUnavailable, hardware_encoders, require_ffmpeg
from .frames import FrameReader, FrameWriter
from .jobs import JobHandle, ProgressTracker, manager

log = logging.getLogger("visiontrack.exporters")

VIDEO_FORMATS = frozenset({"mp4"})
DATA_FORMATS = frozenset({"json", "csv", "coco", "yolo"})
ALL_FORMATS = VIDEO_FORMATS | DATA_FORMATS

SCHEMA_VERSION = "visiontrack.export/1"


class ExportStatus:
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportCancelled(Exception):
    """Unwinds to the cleanup path so no partial artifact is left behind."""


def new_export_id() -> str:
    return uuid.uuid4().hex[:16]


# ------------------------------------------------------------------ small utils

def _safe_stem(name: str, fallback: str) -> str:
    """Filesystem-safe stem derived from the user's original filename."""
    stem = Path(name or "").stem
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "-" for c in stem).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return (cleaned[:60] or fallback)


def _round(value: Any, digits: int) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if f != f or f in (math.inf, -math.inf):
        return 0.0
    return round(f, digits)


def _px(det: Any, width: int, height: int) -> tuple[float, float, float, float]:
    """Normalised geometry -> pixel x, y, w, h at the given frame size."""
    x = float(det["x"]) * width
    y = float(det["y"]) * height
    w = float(det["w"]) * width
    h = float(det["h"]) * height
    return x, y, w, h


class _Progress:
    """Throttled progress publisher shared by every exporter.

    Writes to SQLite and the event bus at most ~8x/second: a per-row update
    would cost more than the export itself.
    """

    def __init__(self, export_id: str, job_id: str, total: int, *, label: str):
        self.export_id = export_id
        self.job_id = job_id
        self.label = label
        self.tracker = ProgressTracker(max(1, total))
        self._last_emit = 0.0
        self._last_pct = -1.0

    def __call__(self, done: int, *, force: bool = False) -> None:
        snap = self.tracker.update(done)
        now = time.monotonic()
        pct = round(snap["progress"] * 100, 1)
        if not force and (now - self._last_emit < 0.125 or pct == self._last_pct):
            return
        self._last_emit = now
        self._last_pct = pct
        store.update_export(self.export_id, progress=snap["progress"])
        manager.emit(
            self.job_id, "export_progress",
            exportId=self.export_id,
            stage=self.label,
            progress=snap["progress"],
            processed=snap["processed"],
            total=snap["total"],
            rate=snap["rate"],
            etaSeconds=snap["etaSeconds"],
        )


def _check(handle: JobHandle | None) -> None:
    if handle is not None and handle.cancelled:
        raise ExportCancelled()


# ---------------------------------------------------------------- export options

class ExportOptions:
    """Normalised export request.

    `contents` gates which fields appear in data formats and which overlays are
    burned into video, so one set of checkboxes drives both.
    """

    def __init__(self, payload: dict[str, Any] | None = None, *, fmt: str = "json"):
        d = payload or {}
        self.raw = d
        self.fmt = fmt
        self.annotated = bool(d.get("annotated", True))
        self.overlay = OverlayOptions.from_payload(
            d.get("contents") or d, stamps_default=fmt in DATA_FORMATS
        )

        # Video geometry. None means "keep the source value".
        self.width: int | None = self._opt_int(d.get("width"))
        self.height: int | None = self._opt_int(d.get("height"))
        self.fps: float | None = self._opt_float(d.get("fps") or d.get("frameRate"))
        self.crf = max(0, min(51, int(d.get("crf") or 20)))
        self.include_audio = bool(d.get("includeAudio", True))

        # Data formats.
        self.pretty = bool(d.get("pretty", False))
        self.include_tracks = bool(d.get("includeTracks", True))
        self.min_confidence = self._opt_float(d.get("minConfidence")) or 0.0
        classes = d.get("classes") or d.get("onlyClasses")
        self.classes: set[str] | None = (
            {str(c).lower() for c in classes} if classes else None
        )

    @staticmethod
    def _opt_int(value: Any) -> int | None:
        try:
            i = int(value)
        except (TypeError, ValueError):
            return None
        return i if i > 0 else None

    @staticmethod
    def _opt_float(value: Any) -> float | None:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return f if f > 0 and f == f else None

    def keeps(self, det: Any) -> bool:
        if float(det["confidence"]) < self.min_confidence:
            return False
        if self.classes is not None and str(det["class_name"]).lower() not in self.classes:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        o = self.overlay
        return {
            "annotated": self.annotated,
            "width": self.width, "height": self.height, "fps": self.fps,
            "crf": self.crf, "includeAudio": self.include_audio,
            "pretty": self.pretty, "includeTracks": self.include_tracks,
            "minConfidence": self.min_confidence,
            "classes": sorted(self.classes) if self.classes else None,
            "contents": {
                "boxes": o.boxes, "classes": o.class_labels,
                "confidence": o.confidence, "trackIds": o.track_ids,
                "frameNumbers": o.frame_numbers, "timestamps": o.timestamps,
                "motionPaths": o.motion_paths, "trails": o.trails,
                "style": o.style,
            },
        }


def output_dimensions(
    src_w: int, src_h: int, want_w: int | None, want_h: int | None
) -> tuple[int, int]:
    """Requested export size, aspect-preserved when only one edge is given.

    Both dimensions are forced even because yuv420p subsamples chroma by two.
    """
    if src_w <= 0 or src_h <= 0:
        return 640, 360
    if want_w and want_h:
        w, h = want_w, want_h
    elif want_w:
        w = want_w
        h = int(round(want_w * src_h / src_w))
    elif want_h:
        h = want_h
        w = int(round(want_h * src_w / src_h))
    else:
        w, h = src_w, src_h
    return max(2, w - (w % 2)), max(2, h - (h % 2))


# ------------------------------------------------------------------- CSV export

def _csv_columns(opt: ExportOptions) -> list[str]:
    o = opt.overlay
    cols: list[str] = []
    if o.frame_numbers:
        cols.append("frame")
    if o.timestamps:
        cols += ["timestamp_seconds", "timecode"]
    if o.track_ids:
        cols.append("track_id")
    if o.class_labels:
        cols += ["class_id", "class_name"]
    if o.confidence:
        cols.append("confidence")
    if o.boxes:
        cols += ["x", "y", "width", "height", "x2", "y2", "center_x", "center_y"]
    # A CSV with no columns is useless; frame + track is the minimum that still
    # identifies a row.
    return cols or ["frame", "track_id"]


def _export_csv(
    dest: Path, job: dict[str, Any], opt: ExportOptions,
    progress: _Progress, handle: JobHandle | None,
) -> dict[str, Any]:
    meta = job.get("videoMetadata") or {}
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    cols = _csv_columns(opt)
    o = opt.overlay
    written = 0

    with store.read_connection() as conn, dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for i, det in enumerate(store.stream_detections(conn, job["id"])):
            if i % 512 == 0:
                _check(handle)
                progress(i)
            if not opt.keeps(det):
                continue
            x, y, w, h = _px(det, width, height)
            row: list[Any] = []
            if o.frame_numbers:
                row.append(int(det["frame"]))
            if o.timestamps:
                row += [_round(det["ts"], 4), format_timecode(float(det["ts"]))]
            if o.track_ids:
                row.append(int(det["track_id"]))
            if o.class_labels:
                row += [int(det["class_id"]), det["class_name"]]
            if o.confidence:
                row.append(_round(det["confidence"], 4))
            if o.boxes:
                row += [
                    _round(x, 2), _round(y, 2), _round(w, 2), _round(h, 2),
                    _round(x + w, 2), _round(y + h, 2),
                    _round(x + w / 2, 2), _round(y + h / 2, 2),
                ]
            if not row:  # degenerate column set
                row = [int(det["frame"]), int(det["track_id"])]
            writer.writerow(row)
            written += 1

    progress(progress.tracker.total, force=True)
    return {"rows": written, "columns": cols, "coordinateSpace": f"{width}x{height} pixels"}


# ------------------------------------------------------------------ JSON export

def _detection_object(det: Any, opt: ExportOptions, width: int, height: int) -> dict[str, Any]:
    o = opt.overlay
    out: dict[str, Any] = {}
    if o.frame_numbers:
        out["frame"] = int(det["frame"])
    if o.timestamps:
        out["timestamp"] = _round(det["ts"], 4)
        out["timecode"] = format_timecode(float(det["ts"]))
    if o.track_ids:
        out["trackId"] = int(det["track_id"])
    if o.class_labels:
        out["classId"] = int(det["class_id"])
        out["className"] = det["class_name"]
    if o.confidence:
        out["confidence"] = _round(det["confidence"], 5)
    if o.boxes:
        x, y, w, h = _px(det, width, height)
        out["boundingBox"] = {
            "x": _round(x, 2), "y": _round(y, 2),
            "width": _round(w, 2), "height": _round(h, 2),
        }
        out["boundingBoxNormalized"] = {
            "x": _round(det["x"], 5), "y": _round(det["y"], 5),
            "width": _round(det["w"], 5), "height": _round(det["h"], 5),
        }
    if not out:
        out = {"frame": int(det["frame"]), "trackId": int(det["track_id"])}
    return out


def _track_object(track: dict[str, Any], opt: ExportOptions) -> dict[str, Any]:
    out = {
        "trackId": int(track["track_id"]),
        "className": track["class_name"],
        "classId": int(track["class_id"]),
        "firstSeen": _round(track["first_seen"], 3),
        "lastSeen": _round(track["last_seen"], 3),
        "firstFrame": int(track["first_frame"]),
        "lastFrame": int(track["last_frame"]),
        "duration": _round(track["duration"], 3),
        "detectionCount": int(track["detection_count"]),
        "averageConfidence": _round(track["avg_confidence"], 4),
        "maxConfidence": _round(track["max_confidence"], 4),
        "maxSpeedPxPerFrame": _round(track["max_speed"], 2),
        "avgSpeedPxPerFrame": _round(track["avg_speed"], 2),
        "distancePx": _round(track["distance"], 1),
        "gapCount": int(track.get("gap_count") or 0),
    }
    if opt.overlay.motion_paths:
        # [frame, normalised cx, normalised cy] - resolution independent.
        out["motionPath"] = track.get("path") or []
    return out


def _job_descriptor(job: dict[str, Any]) -> dict[str, Any]:
    """The reproducible settings block: enough to rerun this exact analysis."""
    return {
        "jobId": job["id"],
        "filename": job.get("original_name") or job.get("filename"),
        "status": job.get("status"),
        "createdAt": job.get("created_at"),
        "finishedAt": job.get("finished_at"),
        "partial": bool(job.get("partial")),
        "settings": {
            "selectedClasses": job.get("selectedClasses") or [],
            "confidenceThreshold": job.get("confidence"),
            "iouThreshold": job.get("iou"),
            "model": job.get("model"),
            "resolvedModel": job.get("resolved_model"),
            "trackingMethod": job.get("tracking_method"),
            "annotationStyle": job.get("annotation_style"),
            "frameStride": job.get("frame_stride"),
            "device": job.get("device"),
        },
        "processing": {
            "processedFrames": job.get("processed_frames"),
            "totalFrames": job.get("total_frames"),
            "processingFps": job.get("processing_fps"),
        },
    }


def _export_json(
    dest: Path, job: dict[str, Any], opt: ExportOptions,
    progress: _Progress, handle: JobHandle | None,
) -> dict[str, Any]:
    meta = job.get("videoMetadata") or {}
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    indent = "  " if opt.pretty else ""
    nl = "\n" if opt.pretty else ""
    written = 0

    def dump(value: Any) -> str:
        if opt.pretty:
            return json.dumps(value, indent=2, ensure_ascii=False)
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

    header = {
        "schema": SCHEMA_VERSION,
        "generator": f"VisionTrack AI {VERSION}",
        "exportedAt": time.time(),
        "job": _job_descriptor(job),
        "video": meta,
        "summary": job.get("results") or {},
        "exportOptions": opt.to_dict(),
        "coordinateSpace": {
            "boundingBox": f"pixels in a {width}x{height} frame",
            "boundingBoxNormalized": "fractions of frame width/height (0..1)",
        },
    }

    # The detections array is streamed, so the header is written as a partial
    # object rather than dumping one giant dict.
    with store.read_connection() as conn, dest.open("w", encoding="utf-8") as fh:
        fh.write("{" + nl)
        for key, value in header.items():
            fh.write(f'{indent}{json.dumps(key)}:{" " if opt.pretty else ""}{dump(value)},{nl}')

        if opt.include_tracks:
            tracks = store.get_tracks(job["id"])
            fh.write(f'{indent}"tracks":{" " if opt.pretty else ""}[{nl}')
            for i, t in enumerate(tracks):
                sep = "," if i < len(tracks) - 1 else ""
                fh.write(f"{indent * 2}{dump(_track_object(t, opt))}{sep}{nl}")
            fh.write(f"{indent}],{nl}")

        fh.write(f'{indent}"detections":{" " if opt.pretty else ""}[{nl}')
        first = True
        for i, det in enumerate(store.stream_detections(conn, job["id"])):
            if i % 512 == 0:
                _check(handle)
                progress(i)
            if not opt.keeps(det):
                continue
            if not first:
                fh.write("," + nl)
            fh.write(f"{indent * 2}{dump(_detection_object(det, opt, width, height))}")
            first = False
            written += 1
        fh.write(f"{nl}{indent}]{nl}}}{nl}")

    progress(progress.tracker.total, force=True)
    return {"detections": written}


# ------------------------------------------------------------------ COCO export

def _export_coco(
    dest: Path, job: dict[str, Any], opt: ExportOptions,
    progress: _Progress, handle: JobHandle | None,
) -> dict[str, Any]:
    """COCO detection JSON.

    One `image` entry per analysed frame, keyed by frame number so the ids line
    up with the source video. Track identity is not part of the COCO detection
    spec, so it is carried in the widely-used `attributes.track_id` extension
    plus a top-level `track_id` field - both places tools look.
    """
    meta = job.get("videoMetadata") or {}
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    stem = _safe_stem(job.get("original_name") or job.get("filename") or "", job["id"])
    o = opt.overlay

    frames_seen: set[int] = set()
    categories: dict[int, str] = {}
    written = 0

    with store.read_connection() as conn, dest.open("w", encoding="utf-8") as fh:
        fh.write("{")
        fh.write('"info":' + json.dumps({
            "description": f"VisionTrack AI detections for {job.get('original_name')}",
            "version": SCHEMA_VERSION,
            "generator": f"VisionTrack AI {VERSION}",
            "date_created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "video": {
                "filename": job.get("original_name"),
                "fps": meta.get("fps"), "duration": meta.get("duration"),
                "frameCount": meta.get("frame_count"),
            },
            "model": job.get("resolved_model") or job.get("model"),
            "confidenceThreshold": job.get("confidence"),
            "trackingMethod": job.get("tracking_method"),
        }, separators=(",", ":")))
        fh.write(',"licenses":[],"annotations":[')

        ann_id = 0
        first = True
        for i, det in enumerate(store.stream_detections(conn, job["id"])):
            if i % 512 == 0:
                _check(handle)
                progress(i)
            if not opt.keeps(det):
                continue
            frame = int(det["frame"])
            cid = int(det["class_id"])
            frames_seen.add(frame)
            categories.setdefault(cid, str(det["class_name"]))
            x, y, w, h = _px(det, width, height)
            ann_id += 1
            ann: dict[str, Any] = {
                "id": ann_id,
                "image_id": frame,
                "category_id": cid,
                "bbox": [_round(x, 2), _round(y, 2), _round(w, 2), _round(h, 2)],
                "area": _round(w * h, 2),
                "iscrowd": 0,
                "segmentation": [],
            }
            if o.confidence:
                ann["score"] = _round(det["confidence"], 5)
            if o.track_ids:
                tid = int(det["track_id"])
                ann["track_id"] = tid
                ann["attributes"] = {"track_id": tid}
            if not first:
                fh.write(",")
            fh.write(json.dumps(ann, separators=(",", ":")))
            first = False
            written += 1

        fh.write('],"images":[')
        fps = float(meta.get("fps") or 30.0)
        for i, frame in enumerate(sorted(frames_seen)):
            if i:
                fh.write(",")
            fh.write(json.dumps({
                "id": frame,
                "file_name": f"{stem}_{frame:06d}.jpg",
                "width": width,
                "height": height,
                "frame_number": frame,
                "timestamp": round(frame / fps, 4) if fps > 0 else 0.0,
            }, separators=(",", ":")))

        fh.write('],"categories":[')
        for i, (cid, name) in enumerate(sorted(categories.items())):
            if i:
                fh.write(",")
            fh.write(json.dumps({
                "id": cid, "name": name,
                "supercategory": _supercategory(name),
            }, separators=(",", ":")))
        fh.write("]}")

    progress(progress.tracker.total, force=True)
    return {
        "annotations": written,
        "images": len(frames_seen),
        "categories": len(categories),
    }


def _supercategory(name: str) -> str:
    for group, members in CLASS_GROUPS.items():
        if name in members:
            return group
    return "object"


# ------------------------------------------------------------------ YOLO export

_YOLO_README = """VisionTrack AI - YOLO annotation export
=======================================

labels/<name>_<frame>.txt holds one line per detection in that frame:

    <class_index> <center_x> <center_y> <width> <height>{extra}

All geometry is normalised to 0..1 against the source frame size ({width}x{height}),
which is the format YOLO training expects.

class_index refers to classes.txt in this archive - indices are contiguous and
cover only the classes actually detected in this video, so the file can be used
as a dataset label set directly. data.yaml carries the same mapping in the layout
Ultralytics reads.

Frames with no detections have no label file. Frame numbers are absolute source
frame indices{stride_note}.

Generated by VisionTrack AI {version} from {filename}.
Model: {model} | Confidence threshold: {conf} | Tracking: {tracking}
"""


def _export_yolo(
    dest: Path, job: dict[str, Any], opt: ExportOptions,
    progress: _Progress, handle: JobHandle | None,
) -> dict[str, Any]:
    """A zip of per-frame YOLO label files plus classes.txt and data.yaml."""
    import zipfile

    meta = job.get("videoMetadata") or {}
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    stem = _safe_stem(job.get("original_name") or job.get("filename") or "", job["id"])
    o = opt.overlay
    stride = int(job.get("frame_stride") or 1)

    # Contiguous class indices, which is what a YOLO dataset requires. The
    # mapping back to the model's own ids is recorded in classes.txt.
    with store.read_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT class_id, class_name FROM detections WHERE job_id=? "
            "ORDER BY class_id", (job["id"],)
        ).fetchall()
    present = [(int(r["class_id"]), str(r["class_name"])) for r in rows]
    if opt.classes is not None:
        present = [(c, n) for c, n in present if n.lower() in opt.classes]
    index_of = {cid: i for i, (cid, _n) in enumerate(present)}

    written = 0
    files = 0
    with store.read_connection() as conn, zipfile.ZipFile(
        dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        current_frame: int | None = None
        lines: list[str] = []

        def flush() -> None:
            nonlocal files
            if current_frame is None or not lines:
                return
            zf.writestr(f"labels/{stem}_{current_frame:06d}.txt", "\n".join(lines) + "\n")
            files += 1

        for i, det in enumerate(store.stream_detections(conn, job["id"])):
            if i % 512 == 0:
                _check(handle)
                progress(i)
            if not opt.keeps(det):
                continue
            cid = int(det["class_id"])
            if cid not in index_of:
                continue
            frame = int(det["frame"])
            if frame != current_frame:
                flush()
                current_frame, lines = frame, []

            nx, ny = float(det["x"]), float(det["y"])
            nw, nh = float(det["w"]), float(det["h"])
            parts = [
                str(index_of[cid]),
                f"{nx + nw / 2:.6f}", f"{ny + nh / 2:.6f}",
                f"{nw:.6f}", f"{nh:.6f}",
            ]
            if o.confidence:
                parts.append(f"{float(det['confidence']):.4f}")
            if o.track_ids:
                parts.append(str(int(det["track_id"])))
            lines.append(" ".join(parts))
            written += 1
        flush()

        extra = ""
        if o.confidence:
            extra += " <confidence>"
        if o.track_ids:
            extra += " <track_id>"
        zf.writestr("classes.txt", "\n".join(
            f"{i} {name}  # model class id {cid}" for i, (cid, name) in enumerate(present)
        ) + "\n")
        zf.writestr("data.yaml", (
            f"# VisionTrack AI export - {job.get('original_name')}\n"
            f"path: .\ntrain: labels\nval: labels\n"
            f"nc: {len(present)}\n"
            f"names:\n" + "".join(f"  {i}: {name}\n" for i, (_c, name) in enumerate(present))
        ))
        zf.writestr("README.txt", _YOLO_README.format(
            extra=extra, width=width, height=height,
            stride_note=f", sampled every {stride} frames" if stride > 1 else "",
            version=VERSION,
            filename=job.get("original_name") or job.get("filename"),
            model=job.get("resolved_model") or job.get("model"),
            conf=job.get("confidence"), tracking=job.get("tracking_method"),
        ))

    progress(progress.tracker.total, force=True)
    return {"labels": written, "labelFiles": files, "classes": len(present)}


# ----------------------------------------------------------------- video export

class _FrameRateMapper:
    """How many output frames each source frame becomes.

    Emitting `out_fps / src_fps` frames per source frame - accumulated so the
    fractional remainder is never lost - preserves the clip's duration exactly
    while changing its frame rate. Same semantics as FFmpeg's `fps` filter, but
    done here because the annotation has to happen per source frame.
    """

    def __init__(self, src_fps: float, out_fps: float):
        src = src_fps if src_fps > 0 else 30.0
        out = out_fps if out_fps > 0 else src
        self.ratio = out / src
        self._credit = 0.0

    def emits(self) -> int:
        self._credit += self.ratio
        n = int(self._credit)
        self._credit -= n
        return n


class _DetectionCursor:
    """Walks detections in frame order in lockstep with the frame iterator.

    Keeping both streams ordered means an export never materialises the
    detection set - memory stays flat whether the job found 500 objects or
    500,000.
    """

    def __init__(self, cursor: Iterable[Any]):
        self._it = iter(cursor)
        self._peek: Any = next(self._it, None)

    def at(self, frame_index: int) -> list[dict[str, Any]]:
        # Discard anything we somehow passed (defensive; both streams ascend).
        while self._peek is not None and int(self._peek["frame"]) < frame_index:
            self._peek = next(self._it, None)
        out: list[dict[str, Any]] = []
        while self._peek is not None and int(self._peek["frame"]) == frame_index:
            out.append(dict(self._peek))
            self._peek = next(self._it, None)
        return out


def _copy_with_progress(
    src: Path, dest: Path, progress: _Progress, handle: JobHandle | None
) -> int:
    """Byte-wise copy so an unannotated, untransformed export is lossless.

    Re-encoding the user's own footage to hand it back would throw away quality
    for nothing.
    """
    total = max(1, src.stat().st_size)
    done = 0
    chunk = 4 * 1024 * 1024
    progress.tracker.total = total
    with src.open("rb") as fin, dest.open("wb") as fout:
        while True:
            _check(handle)
            buf = fin.read(chunk)
            if not buf:
                break
            fout.write(buf)
            done += len(buf)
            progress(done)
    progress(total, force=True)
    return done


def _export_video(
    dest: Path, job: dict[str, Any], opt: ExportOptions,
    progress: _Progress, handle: JobHandle | None,
) -> dict[str, Any]:
    from .ffmpeg import transcode

    meta = job.get("videoMetadata") or {}
    src_path = Path(job["video_path"])
    if not src_path.exists():
        raise ExportFailure(error_payload(
            "source_missing", "SOURCE VIDEO MISSING",
            "The original video file for this job is no longer on disk.",
            "The uploaded file was moved or deleted after the analysis ran.",
            "Re-upload the footage and run the analysis again.",
            str(src_path),
        ))

    src_w = int(meta.get("width") or 0)
    src_h = int(meta.get("height") or 0)
    src_fps = float(meta.get("fps") or 30.0)
    total_frames = int(meta.get("frame_count") or job.get("total_frames") or 0)
    out_w, out_h = output_dimensions(src_w, src_h, opt.width, opt.height)
    out_fps = opt.fps or src_fps
    resized = (out_w, out_h) != (src_w, src_h)
    refps = abs(out_fps - src_fps) > 1e-6

    # -- clean copy / straight transcode ------------------------------------
    if not opt.annotated or not opt.overlay.draws_anything:
        if not resized and not refps:
            size = _copy_with_progress(src_path, dest, progress, handle)
            return {
                "mode": "copy", "width": src_w, "height": src_h, "fps": src_fps,
                "bytes": size, "encoder": "none (stream copy)",
                "note": "Original file copied unchanged - no re-encode, no quality loss.",
            }
        progress.tracker.total = max(1, total_frames)
        last = 0

        def on_progress(fraction: float) -> None:
            nonlocal last
            _check(handle)
            last = int(fraction * max(1, total_frames))
            progress(last)

        transcode(
            src_path, dest,
            width=out_w if resized else None,
            height=out_h if resized else None,
            fps=out_fps if refps else None,
            crf=opt.crf, on_progress=on_progress, total_frames=total_frames,
        )
        progress(progress.tracker.total, force=True)
        return {
            "mode": "transcode", "width": out_w, "height": out_h, "fps": out_fps,
            "encoder": (hardware_encoders() or ["libx264"])[0],
        }

    # -- annotated render ---------------------------------------------------
    require_ffmpeg()
    style, notice = resolve_style(
        opt.overlay.style, has_masks=False, has_keypoints=False
    )
    opt.overlay.style = style

    annotator = Annotator(
        out_w, out_h, options=opt.overlay, fps=out_fps,
    )
    paths = (
        annotator.static_paths(store.get_tracks(job["id"]))
        if opt.overlay.motion_paths else None
    )

    stride = max(1, int(job.get("frame_stride") or 1))
    progress.tracker.total = max(1, total_frames)

    audio = src_path if (opt.include_audio and meta.get("has_audio")) else None
    reader = FrameReader(
        src_path, width=out_w, height=out_h, fps=src_fps, hwaccel=True
    )
    writer = FrameWriter(
        dest, width=out_w, height=out_h, fps=out_fps,
        crf=opt.crf, audio_source=audio,
    )
    mapper = _FrameRateMapper(src_fps, out_fps)

    read_frames = 0
    written_frames = 0
    annotated_frames = 0
    held_frames = 0

    try:
        writer.open()
        with store.read_connection() as conn:
            cursor = _DetectionCursor(store.stream_detections(conn, job["id"]))
            reader.open()
            held: list[dict[str, Any]] = []
            held_age = 0

            for frame_index, raw in reader.frames():
                if read_frames % 16 == 0:
                    _check(handle)

                dets = [d for d in cursor.at(frame_index) if opt.keeps(d)]
                if dets:
                    held, held_age = dets, 0
                    annotated_frames += 1
                elif stride > 1 and held and held_age < stride - 1:
                    # Detection ran every `stride` frames. Holding the last
                    # result across the gap keeps the overlay steady instead of
                    # strobing; the export metadata records that this happened.
                    held_age += 1
                    dets = held
                    held_frames += 1
                else:
                    held, held_age = [], 0

                copies = mapper.emits()
                if not copies:
                    read_frames += 1
                    if read_frames % 8 == 0:
                        progress(read_frames)
                    continue

                # np.frombuffer gives a read-only view; annotation needs its
                # own writable buffer.
                frame = raw.copy()
                annotator.draw(
                    frame, dets,
                    frame_index=frame_index,
                    timestamp=frame_index / src_fps if src_fps > 0 else 0.0,
                    paths=paths,
                )
                for _ in range(copies):
                    writer.write(frame)
                    written_frames += 1

                read_frames += 1
                if read_frames % 8 == 0:
                    progress(read_frames)
        writer.close()
    except ExportCancelled:
        writer.abort()
        raise
    except FFmpegError:
        writer.abort()
        raise
    finally:
        reader.close()

    progress(progress.tracker.total, force=True)
    result = {
        "mode": "annotated",
        "width": out_w, "height": out_h, "fps": out_fps,
        "encoder": writer.encoder,
        "sourceFrames": read_frames,
        "outputFrames": written_frames,
        "annotatedFrames": annotated_frames,
        "audio": bool(audio),
        "style": style,
    }
    if held_frames:
        result["heldFrames"] = held_frames
        result["note"] = (
            f"Detection sampled every {stride} frames; overlays were held across "
            f"{held_frames} intermediate frames rather than interpolated."
        )
    if notice:
        result["notice"] = notice
    return result


# ---------------------------------------------------------------- orchestration

class ExportFailure(Exception):
    """Carries a fully-formed error payload up to the failure handler."""

    def __init__(self, payload: dict[str, Any]):
        super().__init__(payload.get("message", "Export failed"))
        self.payload = payload


_WRITERS: dict[str, Callable[..., dict[str, Any]]] = {
    "mp4": _export_video,
    "json": _export_json,
    "csv": _export_csv,
    "coco": _export_coco,
    "yolo": _export_yolo,
}

_SUFFIX = {"mp4": ".mp4", "json": ".json", "csv": ".csv", "coco": ".json", "yolo": ".zip"}

_LABEL = {
    "mp4": "Encoding video",
    "json": "Writing JSON",
    "csv": "Writing CSV",
    "coco": "Writing COCO JSON",
    "yolo": "Packaging YOLO labels",
}

_DESCRIPTOR = {
    "mp4": "annotated", "json": "detections", "csv": "detections",
    "coco": "coco", "yolo": "yolo-labels",
}


def _dest_path(job: dict[str, Any], export_id: str, fmt: str, opt: ExportOptions) -> Path:
    stem = _safe_stem(job.get("original_name") or job.get("filename") or "", job["id"])
    tag = _DESCRIPTOR[fmt]
    if fmt == "mp4" and not (opt.annotated and opt.overlay.draws_anything):
        tag = "clean"
    return EXPORT_DIR / f"{stem}_{tag}_{export_id[:8]}{_SUFFIX[fmt]}"


def validate_request(job: dict[str, Any] | None, fmt: str) -> dict[str, Any]:
    """Reject an impossible export up front, with a reason the UI can show."""
    if job is None:
        return error_payload(
            "job_not_found", "JOB NOT FOUND",
            "This analysis job no longer exists.",
            "It may have been deleted from Job History.",
            "Return to the dashboard and pick another job.",
        )
    if fmt not in ALL_FORMATS:
        return error_payload(
            "unsupported_format", "UNSUPPORTED EXPORT FORMAT",
            f"'{fmt}' is not an export format VisionTrack produces.",
            "The request named a format outside the supported set.",
            f"Choose one of: {', '.join(sorted(ALL_FORMATS))}.",
        )
    if fmt in VIDEO_FORMATS and not Path(job["video_path"]).exists():
        return error_payload(
            "source_missing", "SOURCE VIDEO MISSING",
            "The original video for this job is no longer on disk.",
            "The uploaded file was moved or deleted after analysis.",
            "Re-upload the footage to export video. Data exports still work.",
            job["video_path"],
        )
    if fmt in DATA_FORMATS and store.detection_count(job["id"]) == 0:
        return error_payload(
            "no_detections", "NOTHING TO EXPORT",
            "This analysis found no objects, so there are no annotations to write.",
            "No detection passed the confidence threshold for the selected classes.",
            "Re-run the analysis with a lower confidence threshold or more classes.",
        )
    return {}


def start_export(
    job_id: str, fmt: str, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Register an export and hand it to the worker pool. Returns the row."""
    fmt = (fmt or "").strip().lower()
    job = store.get_job(job_id)
    problem = validate_request(job, fmt)
    if problem:
        raise ExportFailure(problem)
    assert job is not None

    opt = ExportOptions(options, fmt=fmt)
    export_id = new_export_id()
    store.create_export({
        "id": export_id,
        "job_id": job_id,
        "kind": "video" if fmt in VIDEO_FORMATS else "data",
        "fmt": fmt,
        "status": ExportStatus.QUEUED,
        "options": opt.to_dict(),
    })
    manager.emit(
        job_id, "export_queued",
        exportId=export_id, format=fmt, status=ExportStatus.QUEUED,
    )
    manager.submit_export(export_id, lambda h: run_export(export_id, h))
    row = store.get_export(export_id)
    return row or {"id": export_id, "status": ExportStatus.QUEUED}


def cancel_export(export_id: str) -> bool:
    return manager.cancel(export_id)


def run_export(export_id: str, handle: JobHandle | None = None) -> None:
    """Produce one export artifact. Owns its own error and cleanup handling."""
    record = store.get_export(export_id)
    if record is None:
        log.warning("export %s vanished before it started", export_id)
        return
    job = store.get_job(record["job_id"])
    if job is None:
        _fail_export(export_id, record, None, error_payload(
            "job_not_found", "JOB NOT FOUND",
            "The analysis this export belongs to was deleted.",
            "The job was removed from Job History while the export was queued.",
            "Start a new analysis.",
        ))
        return

    fmt = record["fmt"]
    opt = ExportOptions(record.get("options"), fmt=fmt)
    dest = _dest_path(job, export_id, fmt, opt)

    total_units = (
        int((job.get("videoMetadata") or {}).get("frame_count") or 0)
        if fmt in VIDEO_FORMATS
        else store.detection_count(job["id"])
    )
    progress = _Progress(export_id, job["id"], total_units, label=_LABEL[fmt])

    store.update_export(
        export_id, status=ExportStatus.RUNNING, path=str(dest), progress=0.0, error=None
    )
    manager.emit(
        job["id"], "export_started",
        exportId=export_id, format=fmt, status=ExportStatus.RUNNING,
        stage=_LABEL[fmt], total=total_units,
    )

    started = time.perf_counter()
    try:
        detail = _WRITERS[fmt](dest, job, opt, progress, handle)
        if not dest.exists():
            raise ExportFailure(error_payload(
                "no_output", "EXPORT PRODUCED NO FILE",
                "The export finished but no file was written.",
                "The encoder or writer exited without creating output.",
                "Retry the export. If it fails again, try a different format.",
            ))
        size = dest.stat().st_size
        elapsed = time.perf_counter() - started
        store.update_export(
            export_id, status=ExportStatus.COMPLETE, progress=1.0,
            size_bytes=size, finished_at=time.time(), path=str(dest),
        )
        manager.emit(
            job["id"], "export_complete",
            exportId=export_id, format=fmt, status=ExportStatus.COMPLETE,
            filename=dest.name, sizeBytes=size,
            elapsed=round(elapsed, 2), detail=detail,
        )
        log.info(
            "export %s (%s) complete: %s, %.1f KiB in %.1fs",
            export_id, fmt, dest.name, size / 1024, elapsed,
        )

    except ExportCancelled:
        _discard(dest)
        store.update_export(
            export_id, status=ExportStatus.CANCELLED, finished_at=time.time(),
            path=None, error=error_payload(
                "cancelled", "EXPORT CANCELLED",
                "The export was cancelled before it finished.",
                "You cancelled it.",
                "Start the export again whenever you are ready.",
            ),
        )
        manager.emit(
            job["id"], "export_cancelled",
            exportId=export_id, format=fmt, status=ExportStatus.CANCELLED,
        )
        log.info("export %s cancelled", export_id)

    except ExportFailure as exc:
        _fail_export(export_id, record, dest, exc.payload)

    except FFmpegUnavailable as exc:
        _fail_export(export_id, record, dest, error_payload(
            "ffmpeg_unavailable", "VIDEO ENGINE UNAVAILABLE",
            str(exc),
            "No usable FFmpeg binary could be located on this machine.",
            "Install FFmpeg and restart the backend, or export a data format "
            "instead - those need no video engine.",
            exc.detail(),
        ))

    except FFmpegError as exc:
        _fail_export(export_id, record, dest, error_payload(
            "encode_failed", "EXPORT ENCODING FAILED",
            exc.message,
            "FFmpeg stopped while encoding the annotated video.",
            "Try a lower resolution, or export without annotations.",
            exc.detail(),
        ))

    except MemoryError:
        _fail_export(export_id, record, dest, error_payload(
            "out_of_memory", "INSUFFICIENT MEMORY",
            "The system ran out of memory during the export.",
            "The requested output resolution needs more memory than is free.",
            "Close other applications or export at a smaller resolution.",
        ))

    except OSError as exc:
        _fail_export(export_id, record, dest, error_payload(
            "write_failed", "COULD NOT WRITE THE EXPORT FILE",
            "The export file could not be written to disk.",
            f"The filesystem reported: {exc.strerror or exc}",
            "Free up disk space and try again.",
            f"{type(exc).__name__}: {exc}",
        ))

    except Exception as exc:  # still explain, still clean up
        log.exception("export %s failed unexpectedly", export_id)
        _fail_export(export_id, record, dest, error_payload(
            "unexpected", "EXPORT FAILED",
            "The export stopped because of an unexpected error.",
            f"{exc.__class__.__name__}: {exc}",
            "Retry the export. If it keeps failing, try a different format.",
            repr(exc),
        ))


def _discard(dest: Path | None) -> None:
    """Remove a partial artifact - a truncated file that looks whole is a trap."""
    if dest is None:
        return
    try:
        dest.unlink(missing_ok=True)
    except OSError:  # pragma: no cover
        log.warning("could not remove partial export %s", dest)


def _fail_export(
    export_id: str, record: dict[str, Any], dest: Path | None, payload: dict[str, Any]
) -> None:
    _discard(dest)
    store.update_export(
        export_id, status=ExportStatus.FAILED, error=payload,
        finished_at=time.time(), path=None,
    )
    manager.emit(
        record["job_id"], "export_failed",
        exportId=export_id, format=record.get("fmt"),
        status=ExportStatus.FAILED, error=payload,
    )
    log.warning("export %s failed: %s (%s)", export_id, payload.get("code"), payload.get("message"))


# ------------------------------------------------------------------- discovery

def format_catalog() -> list[dict[str, Any]]:
    """What the export panel offers, including honest capability caveats."""
    encoders = hardware_encoders()
    return [
        {
            "id": "mp4", "kind": "video", "label": "MP4 VIDEO",
            "extension": ".mp4",
            "description": "H.264 video, with or without burned-in annotations.",
            "supportsResolution": True, "supportsFrameRate": True,
            "encoder": (encoders or ["libx264"])[0],
            "hardwareAccelerated": bool(encoders),
        },
        {
            "id": "json", "kind": "data", "label": "JSON",
            "extension": ".json",
            "description": "Full detections, tracks and analysis settings.",
        },
        {
            "id": "csv", "kind": "data", "label": "CSV",
            "extension": ".csv",
            "description": "One row per detection - opens in any spreadsheet.",
        },
        {
            "id": "coco", "kind": "data", "label": "COCO JSON",
            "extension": ".json",
            "description": "COCO detection format, with track ids as attributes.",
        },
        {
            "id": "yolo", "kind": "data", "label": "YOLO ANNOTATIONS",
            "extension": ".zip",
            "description": "Per-frame YOLO label files, classes.txt and data.yaml.",
        },
    ]


CONTENT_FIELDS = (
    # `video` = burned into the frame, `data` = included as a field/column.
    # Frame numbers and timestamps are essential in a data file but an opt-in
    # on-screen stamp in a video, so the defaults differ.
    {"id": "boxes", "label": "BOUNDING BOXES", "video": True, "data": True},
    {"id": "classes", "label": "CLASS NAMES", "video": True, "data": True},
    {"id": "confidence", "label": "CONFIDENCE SCORES", "video": True, "data": True},
    {"id": "trackIds", "label": "TRACK IDS", "video": True, "data": True},
    {"id": "frameNumbers", "label": "FRAME NUMBERS", "video": False, "data": True},
    {"id": "timestamps", "label": "TIMESTAMPS", "video": False, "data": True},
    {"id": "motionPaths", "label": "MOTION PATHS", "video": False, "data": False},
    {"id": "trails", "label": "OBJECT TRAILS", "video": False, "data": False},
)

ANNOTATION_STYLES = (
    {"id": "box_label", "label": "BOX + LABEL", "supported": True,
     "description": "Bounding box with class, ID and confidence."},
    {"id": "box_only", "label": "BOX ONLY", "supported": True,
     "description": "Clean boxes with no text."},
    {"id": "label_only", "label": "LABEL ONLY", "supported": True,
     "description": "Labels without boxes - least occlusion of the footage."},
    {"id": "mask", "label": "SEGMENTATION MASK", "supported": False,
     "description": "Needs a segmentation checkpoint (yolo11*-seg)."},
    {"id": "skeleton", "label": "SKELETON", "supported": False,
     "description": "Needs a pose checkpoint (yolo11*-pose)."},
)
