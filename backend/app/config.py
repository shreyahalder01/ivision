"""Runtime configuration and filesystem layout for VisionTrack AI."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("VISIONTRACK_DATA", BACKEND_ROOT / "data"))

UPLOAD_DIR = DATA_ROOT / "uploads"
RESULT_DIR = DATA_ROOT / "results"
EXPORT_DIR = DATA_ROOT / "exports"
MODEL_DIR = DATA_ROOT / "models"
TMP_DIR = DATA_ROOT / "tmp"
THUMB_DIR = DATA_ROOT / "thumbs"
DB_PATH = DATA_ROOT / "visiontrack.db"

for _d in (UPLOAD_DIR, RESULT_DIR, EXPORT_DIR, MODEL_DIR, TMP_DIR, THUMB_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str = os.environ.get("VISIONTRACK_HOST", "127.0.0.1")
    port: int = _env_int("VISIONTRACK_PORT", _env_int("PORT", 8787))

    # Upload constraints. 2 GiB default ceiling; a clear error beats a silent stall.
    max_upload_bytes: int = _env_int("VISIONTRACK_MAX_UPLOAD_MB", 2048) * 1024 * 1024
    upload_chunk_bytes: int = 1024 * 1024

    # Container formats we accept. Codec support is probed per-file by ffprobe.
    allowed_extensions: frozenset[str] = frozenset(
        {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    )

    # Processing. Batch sizes are re-derived at runtime from the actual device.
    max_concurrent_jobs: int = _env_int("VISIONTRACK_MAX_JOBS", 1)
    detection_batch_cuda: int = _env_int("VISIONTRACK_BATCH_CUDA", 8)
    detection_batch_cpu: int = _env_int("VISIONTRACK_BATCH_CPU", 4)
    # Detections are flushed to SQLite in blocks so a cancel/crash keeps partials.
    persist_every_frames: int = _env_int("VISIONTRACK_FLUSH_FRAMES", 120)
    progress_interval_frames: int = _env_int("VISIONTRACK_PROGRESS_FRAMES", 3)

    # Inference letterbox size handed to YOLO. 640 is the YOLO11 training size.
    inference_size: int = _env_int("VISIONTRACK_IMGSZ", 640)

    # Temp file reaping.
    tmp_ttl_seconds: int = _env_int("VISIONTRACK_TMP_TTL", 6 * 3600)
    orphan_upload_ttl_seconds: int = _env_int("VISIONTRACK_UPLOAD_TTL", 14 * 24 * 3600)

    # Explicit binary overrides. Discovery is layered; see services/ffmpeg.py.
    ffmpeg_path: str | None = os.environ.get("VISIONTRACK_FFMPEG")
    ffprobe_path: str | None = os.environ.get("VISIONTRACK_FFPROBE")

    force_cpu: bool = _env_flag("VISIONTRACK_FORCE_CPU")
    allow_weight_download: bool = _env_flag("VISIONTRACK_ALLOW_MODEL_DOWNLOAD", True)

    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip()
            for o in os.environ.get(
                "VISIONTRACK_CORS",
                "*",
            ).split(",")
            if o.strip()
        )
    )


settings = Settings()

VERSION = "1.0.0"

# COCO class ids exposed by the stock YOLO11 checkpoints. The UI surfaces the
# eight headline classes first but every entry here is selectable via search,
# which is what makes the selector forward-compatible with richer models.
COCO_CLASSES: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
    47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot",
    52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
    57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
    66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
    70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}

# Headline classes pinned to the top of the selector.
FEATURED_CLASSES: tuple[str, ...] = (
    "person", "car", "truck", "bus", "dog", "cat", "bicycle", "motorcycle",
)

# Coarse grouping used by the analytics panel (vehicles vs people vs animals).
CLASS_GROUPS: dict[str, tuple[str, ...]] = {
    "people": ("person",),
    "vehicles": ("car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane"),
    "animals": ("dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"),
}
