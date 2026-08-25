"""YOLO11 detection engine.

Responsibilities:

* Report honestly whether real inference is possible. If torch/ultralytics are
  missing the API says so and the UI enters an explicitly-labelled Demo Mode -
  we never fabricate detections and present them as analysis.
* Resolve the `auto` model choice against the actual device and video, and cap
  the model size to what the GPU's VRAM can hold (a 4 GB card should not be
  handed yolo11l and then OOM mid-job).
* Cache loaded models across jobs, and surface weight downloads as progress
  rather than a silent stall.
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..config import COCO_CLASSES, MODEL_DIR, settings

log = logging.getLogger("visiontrack.detector")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    weights: str
    label: str
    speed: str
    accuracy: str
    params_m: float
    size_mb: int
    # Rough relative inference cost, used for the time estimate. Calibrated
    # against measured throughput once a job has run.
    cost: float
    min_vram_gb: float


MODEL_SPECS: dict[str, ModelSpec] = {
    "yolo11n": ModelSpec("yolo11n", "yolo11n.pt", "YOLO11 NANO", "Very Fast", "Good", 2.6, 6, 1.0, 1.0),
    "yolo11s": ModelSpec("yolo11s", "yolo11s.pt", "YOLO11 SMALL", "Fast", "Better", 9.4, 19, 1.5, 1.6),
    "yolo11m": ModelSpec("yolo11m", "yolo11m.pt", "YOLO11 MEDIUM", "Balanced", "High", 20.1, 39, 2.6, 2.4),
    "yolo11l": ModelSpec("yolo11l", "yolo11l.pt", "YOLO11 LARGE", "Slower", "Highest", 25.3, 50, 3.6, 3.4),
}

MODEL_ORDER = ("yolo11n", "yolo11s", "yolo11m", "yolo11l")


class DetectorUnavailable(RuntimeError):
    """Real inference is not possible; carries a UI-ready diagnosis."""

    def __init__(self, message: str, *, cause: str = "", action: str = "", code: str = "ai_unavailable"):
        super().__init__(message)
        self.code = code
        self.message = message
        self.cause = cause
        self.action = action

    def to_error(self) -> dict[str, str]:
        return {
            "code": self.code,
            "title": "ANALYSIS COULD NOT START",
            "message": self.message,
            "cause": self.cause,
            "action": self.action,
        }


# ------------------------------------------------------------------ environment

_env_lock = threading.Lock()
_env_cache: dict[str, Any] | None = None


def probe_environment(refresh: bool = False) -> dict[str, Any]:
    """Inspect torch/ultralytics/CUDA once and cache the verdict."""
    global _env_cache
    with _env_lock:
        if _env_cache is not None and not refresh:
            return _env_cache

        info: dict[str, Any] = {
            "torchAvailable": False, "torchVersion": None,
            "ultralyticsAvailable": False, "ultralyticsVersion": None,
            "cudaAvailable": False, "cudaVersion": None,
            "device": "cpu", "deviceName": "CPU",
            "vramTotalGb": None, "gpuAccelerated": False,
            "aiAvailable": False, "reason": None,
        }
        try:
            import torch  # type: ignore

            info["torchAvailable"] = True
            info["torchVersion"] = torch.__version__
            try:
                cuda_ok = bool(torch.cuda.is_available()) and not settings.force_cpu
            except Exception:
                cuda_ok = False
            if cuda_ok:
                props = torch.cuda.get_device_properties(0)
                info.update({
                    "cudaAvailable": True,
                    "cudaVersion": getattr(torch.version, "cuda", None),
                    "device": "cuda:0",
                    "deviceName": props.name,
                    "vramTotalGb": round(props.total_memory / 1024**3, 2),
                    "gpuAccelerated": True,
                })
            else:
                import platform

                info["deviceName"] = platform.processor() or "CPU"
                if settings.force_cpu:
                    info["reason"] = "CPU inference forced via VISIONTRACK_FORCE_CPU."
        except Exception as exc:
            info["reason"] = f"PyTorch is not installed ({exc.__class__.__name__})."

        try:
            import ultralytics  # type: ignore

            info["ultralyticsAvailable"] = True
            info["ultralyticsVersion"] = ultralytics.__version__
        except Exception as exc:
            info["reason"] = info["reason"] or f"Ultralytics is not installed ({exc.__class__.__name__})."

        info["aiAvailable"] = info["torchAvailable"] and info["ultralyticsAvailable"]
        if not info["aiAvailable"] and not info["reason"]:
            info["reason"] = "The detection runtime is incomplete."
        _env_cache = info
        return info


def require_runtime() -> dict[str, Any]:
    env = probe_environment()
    if not env["aiAvailable"]:
        missing = []
        if not env["torchAvailable"]:
            missing.append("torch")
        if not env["ultralyticsAvailable"]:
            missing.append("ultralytics")
        raise DetectorUnavailable(
            "The detection runtime is not installed.",
            cause=f"Missing Python package(s): {', '.join(missing)}.",
            action=(
                "Install the inference runtime, then restart the backend:\n"
                "pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cu126\npip install ultralytics"
            ),
            code="runtime_missing",
        )
    return env


# ---------------------------------------------------------------------- weights

def weight_path(spec: ModelSpec) -> Path:
    return MODEL_DIR / spec.weights


def is_cached(key: str) -> bool:
    spec = MODEL_SPECS.get(key)
    return bool(spec and weight_path(spec).exists() and weight_path(spec).stat().st_size > 0)


def ensure_weights(key: str, on_progress: Callable[[str, float], None] | None = None) -> Path:
    """Return a local weights file, downloading via ultralytics if needed."""
    spec = MODEL_SPECS.get(key)
    if spec is None:
        raise DetectorUnavailable(
            f"Unknown detection model '{key}'.",
            cause="The requested model is not in the registry.",
            action="Choose a different model.",
            code="unknown_model",
        )

    dest = weight_path(spec)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    if not settings.allow_weight_download:
        raise DetectorUnavailable(
            f"The {spec.label} weights are not available locally.",
            cause="Automatic model download is disabled.",
            action=f"Place {spec.weights} in {MODEL_DIR} or enable downloads.",
            code="model_missing",
        )

    if on_progress:
        on_progress(f"Downloading {spec.label} weights ({spec.size_mb} MB)", 0.0)

    # Ultralytics resolves and downloads the checkpoint into its own cwd; do it
    # in the model dir so the artefact lands where we expect it.
    try:
        from ultralytics import YOLO  # type: ignore

        import os

        prev = os.getcwd()
        os.chdir(MODEL_DIR)
        try:
            YOLO(spec.weights)  # triggers the download
        finally:
            os.chdir(prev)
    except Exception as exc:
        raise DetectorUnavailable(
            f"The {spec.label} model could not be downloaded.",
            cause=f"{exc.__class__.__name__}: {exc}",
            action="Check your network connection, or select a model that is already cached.",
            code="model_download_failed",
        ) from exc

    if not dest.exists():
        # Some versions drop it in cwd or the ultralytics assets dir; go find it.
        for candidate in (Path.cwd() / spec.weights, MODEL_DIR / spec.weights):
            if candidate.exists():
                if candidate != dest:
                    shutil.move(str(candidate), str(dest))
                break

    if not dest.exists():
        raise DetectorUnavailable(
            f"The {spec.label} weights are missing after download.",
            cause="The download reported success but no weights file was written.",
            action="Retry, or select a different model.",
            code="model_missing",
        )
    if on_progress:
        on_progress(f"{spec.label} weights ready", 1.0)
    return dest


def cached_models() -> dict[str, bool]:
    return {key: is_cached(key) for key in MODEL_ORDER}


# ------------------------------------------------------------------ model choice

def vram_ceiling(env: dict[str, Any]) -> str:
    """Largest model this device can host without thrashing."""
    if not env.get("cudaAvailable"):
        return "yolo11m"  # CPU: large is technically fine but painfully slow
    vram = env.get("vramTotalGb") or 0.0
    usable = vram * 0.75  # leave headroom for the OS/display and batch tensors
    allowed = [k for k in MODEL_ORDER if MODEL_SPECS[k].min_vram_gb <= usable]
    return allowed[-1] if allowed else "yolo11n"


def resolve_model(requested: str, *, metadata: dict[str, Any] | None = None) -> tuple[str, str | None]:
    """Map a request (possibly 'auto') to a concrete key.

    Returns (key, note) where note explains any automatic adjustment so the UI
    can show it rather than silently doing something unexpected.
    """
    env = probe_environment()
    ceiling = vram_ceiling(env)
    ceiling_idx = MODEL_ORDER.index(ceiling)

    if requested == "auto":
        meta = metadata or {}
        duration = float(meta.get("duration") or 0)
        pixels = int(meta.get("width") or 1920) * int(meta.get("height") or 1080)
        if env.get("cudaAvailable"):
            # GPU: favour accuracy, backing off for long or high-res footage.
            pick = "yolo11m"
            if duration > 600 or pixels > 1920 * 1080 * 2:
                pick = "yolo11s"
            if duration > 1800:
                pick = "yolo11n"
        else:
            pick = "yolo11n" if duration > 60 or pixels > 1280 * 720 else "yolo11s"
        idx = min(MODEL_ORDER.index(pick), ceiling_idx)
        key = MODEL_ORDER[idx]
        note = (
            f"AUTO selected {MODEL_SPECS[key].label} for "
            f"{int(duration)}s of {meta.get('width')}x{meta.get('height')} on "
            f"{env.get('deviceName')}"
        )
        return key, note

    if requested not in MODEL_SPECS:
        raise DetectorUnavailable(
            f"Unknown detection model '{requested}'.",
            cause="The selected model is not in the registry.",
            action="Choose AUTO or a listed YOLO11 model.",
            code="unknown_model",
        )

    idx = MODEL_ORDER.index(requested)
    if idx > ceiling_idx:
        key = MODEL_ORDER[ceiling_idx]
        return key, (
            f"{MODEL_SPECS[requested].label} needs more VRAM than this device has "
            f"({env.get('vramTotalGb')} GB); using {MODEL_SPECS[key].label} instead."
        )
    return requested, None


# Measured activation cost per 640px frame, in GB. Model weights are tiny by
# comparison (yolo11s is ~38 MB at fp32); the CUDA context plus cuDNN workspace
# is the real fixed cost, hence _CUDA_FIXED_OVERHEAD_GB.
_PER_FRAME_GB = {"yolo11n": 0.05, "yolo11s": 0.09, "yolo11m": 0.15, "yolo11l": 0.19}
_CUDA_FIXED_OVERHEAD_GB = 0.95


def batch_size_for(env: dict[str, Any], key: str) -> int:
    """Largest safe batch for this device. The pipeline halves it on OOM."""
    if not env.get("cudaAvailable"):
        return max(1, settings.detection_batch_cpu)
    vram = env.get("vramTotalGb") or 4.0
    per_frame = _PER_FRAME_GB.get(key, 0.15)
    budget = vram * 0.70 - _CUDA_FIXED_OVERHEAD_GB
    if budget <= per_frame:
        return 1
    return int(max(1, min(settings.detection_batch_cuda, budget // per_frame)))


# ------------------------------------------------------------------- throughput

# Measured frames/sec, keyed by (model, device). Seeded with conservative
# estimates and overwritten by real measurements as jobs complete, which is what
# makes the ETA converge on truth instead of staying hard-coded.
_throughput: dict[str, float] = {}
_throughput_lock = threading.Lock()

# Baselines are the yolo11n figure measured on reference hardware (RTX 3050
# laptop, 640px letterbox, batch 8 => 176 FPS synthetic). Held slightly under
# the synthetic number because real footage carries NMS and tracking cost.
_BASELINE_CUDA_FPS = 150.0
_BASELINE_CPU_FPS = 7.5      # yolo11n @640 on a mid-range CPU


def _throughput_key(key: str, device: str) -> str:
    return f"{key}@{'cuda' if device.startswith('cuda') else 'cpu'}"


def estimated_fps(key: str, device: str) -> float:
    with _throughput_lock:
        measured = _throughput.get(_throughput_key(key, device))
    if measured and measured > 0:
        return measured
    spec = MODEL_SPECS.get(key) or MODEL_SPECS["yolo11n"]
    base = _BASELINE_CUDA_FPS if device.startswith("cuda") else _BASELINE_CPU_FPS
    return max(0.5, base / spec.cost)


def record_throughput(key: str, device: str, fps: float) -> None:
    """Blend a fresh measurement into the running estimate."""
    if fps <= 0:
        return
    tk = _throughput_key(key, device)
    with _throughput_lock:
        prev = _throughput.get(tk)
        _throughput[tk] = fps if prev is None else (0.65 * prev + 0.35 * fps)


def estimate_seconds(key: str, metadata: dict[str, Any], *, frame_stride: int = 1) -> float:
    env = probe_environment()
    device = env.get("device", "cpu")
    frames = int(metadata.get("frameCount") or metadata.get("frame_count") or 0)
    if frames <= 0:
        duration = float(metadata.get("duration") or 0)
        fps = float(metadata.get("fps") or 30)
        frames = int(duration * fps)
    to_process = max(1, frames // max(1, frame_stride))

    # Resolution penalty: letterboxing 4K costs more on the decode/resize side.
    pixels = int(metadata.get("width") or 1920) * int(metadata.get("height") or 1080)
    scale = max(1.0, (pixels / (1920 * 1080)) ** 0.35)

    infer = to_process / estimated_fps(key, device) * scale
    overhead = 2.0 + to_process * 0.0009  # decode + tracking + persistence
    return round(infer + overhead, 1)


def model_catalog(metadata: dict[str, Any] | None = None, *, frame_stride: int = 1) -> list[dict[str, Any]]:
    """Registry entries with per-device estimates - drives the model selector."""
    env = probe_environment()
    ceiling_idx = MODEL_ORDER.index(vram_ceiling(env))
    out: list[dict[str, Any]] = []
    for idx, key in enumerate(MODEL_ORDER):
        spec = MODEL_SPECS[key]
        out.append({
            "key": key,
            "label": spec.label,
            "speed": spec.speed,
            "accuracy": spec.accuracy,
            "paramsM": spec.params_m,
            "sizeMb": spec.size_mb,
            "cached": is_cached(key),
            "supported": idx <= ceiling_idx,
            "minVramGb": spec.min_vram_gb,
            "estimatedFps": round(estimated_fps(key, env.get("device", "cpu")), 1),
            "estimatedSeconds": (
                estimate_seconds(key, metadata, frame_stride=frame_stride) if metadata else None
            ),
        })
    return out


# --------------------------------------------------------------------- inference

_model_cache: dict[str, Any] = {}
_model_lock = threading.Lock()


class Detector:
    """Thin wrapper over an ultralytics YOLO model with a stable output shape."""

    def __init__(self, key: str, *, device: str, imgsz: int | None = None):
        self.key = key
        self.spec = MODEL_SPECS[key]
        self.device = device
        self.imgsz = imgsz or settings.inference_size
        self._model = None

    def load(self, on_progress: Callable[[str, float], None] | None = None) -> None:
        require_runtime()
        path = ensure_weights(self.key, on_progress)
        cache_key = f"{self.key}:{self.device}"
        with _model_lock:
            model = _model_cache.get(cache_key)
            if model is None:
                if on_progress:
                    on_progress(f"Loading {self.spec.label}", 0.0)
                try:
                    from ultralytics import YOLO  # type: ignore

                    model = YOLO(str(path))
                    model.to(self.device)
                except Exception as exc:
                    raise DetectorUnavailable(
                        f"The {self.spec.label} model could not be loaded.",
                        cause=f"{exc.__class__.__name__}: {exc}",
                        action="Try a smaller model, or switch to CPU inference.",
                        code="model_load_failed",
                    ) from exc
                _model_cache[cache_key] = model
            self._model = model

    @property
    def class_names(self) -> dict[int, str]:
        names = getattr(self._model, "names", None) if self._model else None
        if isinstance(names, dict) and names:
            return {int(k): str(v) for k, v in names.items()}
        return dict(COCO_CLASSES)

    def warmup(self) -> None:
        if self._model is None:
            return
        try:
            blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self._model.predict(blank, imgsz=self.imgsz, verbose=False, device=self.device)
        except Exception:  # pragma: no cover - warmup is best-effort
            pass

    def detect_batch(
        self,
        frames: Sequence[np.ndarray],
        *,
        conf: float,
        iou: float,
        class_ids: Sequence[int] | None,
    ) -> list[dict[str, np.ndarray]]:
        """Run detection over a batch. Returns per-frame xyxy/conf/cls arrays."""
        if self._model is None:
            raise DetectorUnavailable(
                "The detection model is not loaded.",
                cause="Inference was requested before the model finished loading.",
                action="Restart the analysis.",
                code="model_not_loaded",
            )
        if not frames:
            return []

        kwargs: dict[str, Any] = {
            "imgsz": self.imgsz, "conf": float(conf), "iou": float(iou),
            "verbose": False, "device": self.device,
        }
        if class_ids:
            kwargs["classes"] = list(class_ids)

        try:
            results = self._model.predict(list(frames), **kwargs)
        except Exception as exc:
            message = str(exc).lower()
            if "out of memory" in message or "cuda" in message and "memory" in message:
                raise DetectorUnavailable(
                    "The GPU ran out of memory during detection.",
                    cause=(
                        f"{self.spec.label} at batch size {len(frames)} exceeded the "
                        "available VRAM."
                    ),
                    action="Select a smaller model, or force CPU inference and retry.",
                    code="gpu_oom",
                ) from exc
            raise DetectorUnavailable(
                "Detection failed while processing a frame batch.",
                cause=f"{exc.__class__.__name__}: {exc}",
                action="Retry the analysis, or select a different model.",
                code="inference_failed",
            ) from exc

        out: list[dict[str, np.ndarray]] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                out.append({
                    "xyxy": np.zeros((0, 4), dtype=np.float32),
                    "conf": np.zeros((0,), dtype=np.float32),
                    "cls": np.zeros((0,), dtype=np.int32),
                })
                continue
            out.append({
                "xyxy": boxes.xyxy.cpu().numpy().astype(np.float32),
                "conf": boxes.conf.cpu().numpy().astype(np.float32),
                "cls": boxes.cls.cpu().numpy().astype(np.int32),
            })
        return out


def release_models() -> None:
    """Drop cached models and free VRAM (used when a job hits OOM)."""
    with _model_lock:
        _model_cache.clear()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def gpu_memory() -> dict[str, Any] | None:
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {
            "freeGb": round(free / 1024**3, 2),
            "totalGb": round(total / 1024**3, 2),
            "usedGb": round((total - free) / 1024**3, 2),
        }
    except Exception:
        return None
