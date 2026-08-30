"""FFmpeg / FFprobe integration.

Binary discovery is deliberately layered and never assumes a global install:

  1. VISIONTRACK_FFMPEG / VISIONTRACK_FFPROBE environment overrides
  2. A bundled binary under backend/bin/
  3. imageio-ffmpeg's vendored wheel binary (pip-installable, no system dep)
  4. PATH lookup via shutil.which
  5. Well-known install locations per platform

The resolved paths are cached and surfaced through /api/system/capabilities so
the UI can explain precisely what is wrong instead of failing opaquely.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..config import BACKEND_ROOT, settings

log = logging.getLogger("visiontrack.ffmpeg")

_WINDOWS = platform.system() == "Windows"
_EXE = ".exe" if _WINDOWS else ""

# Hide the console window that subprocess would otherwise flash on Windows.
_CREATE_NO_WINDOW = 0x08000000 if _WINDOWS else 0


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg/ffprobe invocation fails, carrying diagnostics."""

    def __init__(self, message: str, *, stderr: str = "", command: Sequence[str] = ()):
        super().__init__(message)
        self.message = message
        self.stderr = stderr.strip()
        self.command = list(command)

    def detail(self) -> str:
        """Last few stderr lines - what the UI shows in 'technical details'."""
        lines = [ln for ln in self.stderr.splitlines() if ln.strip()]
        return "\n".join(lines[-12:])


class FFmpegUnavailable(FFmpegError):
    """No usable ffmpeg binary could be located."""


@dataclass(frozen=True)
class BinaryInfo:
    path: str | None
    version: str | None
    source: str  # how we found it - shown in the diagnostics panel
    available: bool


def _run(cmd: Sequence[str], *, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=_CREATE_NO_WINDOW,
    )


def _probe_version(path: str) -> str | None:
    try:
        proc = _run([path, "-version"], timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    first = (proc.stdout or "").splitlines()
    if not first:
        return None
    m = re.search(r"version\s+(\S+)", first[0])
    return m.group(1) if m else first[0].strip()


def _candidate_paths(name: str) -> Iterable[tuple[str, str]]:
    """Yield (path, source-label) candidates in priority order."""
    override = settings.ffmpeg_path if name == "ffmpeg" else settings.ffprobe_path
    if override:
        yield override, "environment override"

    bundled = BACKEND_ROOT / "bin" / f"{name}{_EXE}"
    if bundled.exists():
        yield str(bundled), "bundled binary"

    found = shutil.which(name)
    if found:
        yield found, "system PATH"

    if _WINDOWS:
        roots = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
            Path(r"C:\ffmpeg"),
        ]
        for root in roots:
            if not root or not str(root) or not root.exists():
                continue
            direct = root / "ffmpeg" / "bin" / f"{name}.exe"
            if direct.exists():
                yield str(direct), f"install dir ({root.name})"
            try:
                for hit in root.glob(f"**/bin/{name}.exe"):
                    yield str(hit), f"install dir ({root.name})"
                    break
            except OSError:
                continue
    else:
        for base in ("/usr/bin", "/usr/local/bin", "/opt/homebrew/bin", "/snap/bin"):
            p = Path(base) / name
            if p.exists():
                yield str(p), f"install dir ({base})"

    # Last resort. imageio-ffmpeg ships a real ffmpeg build inside the wheel,
    # which guarantees the app works with no system dependency - but it is an
    # "essentials" build with a narrower encoder set than a full system install,
    # so it is only used when nothing better exists. It has no ffprobe, so this
    # only ever satisfies the ffmpeg half.
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg  # type: ignore

            vendored = imageio_ffmpeg.get_ffmpeg_exe()
            if vendored and Path(vendored).exists():
                yield vendored, "imageio-ffmpeg (vendored fallback)"
        except Exception:  # pragma: no cover - optional dependency
            pass


@lru_cache(maxsize=4)
def resolve_binary(name: str) -> BinaryInfo:
    for path, source in _candidate_paths(name):
        version = _probe_version(path)
        if version:
            log.info("resolved %s -> %s (%s, %s)", name, path, source, version)
            return BinaryInfo(path=path, version=version, source=source, available=True)
    log.warning("could not resolve a working %s binary", name)
    return BinaryInfo(path=None, version=None, source="not found", available=False)


def ffmpeg_info() -> BinaryInfo:
    return resolve_binary("ffmpeg")


def ffprobe_info() -> BinaryInfo:
    info = resolve_binary("ffprobe")
    if info.available:
        return info
    # A vendored/bundled ffmpeg usually sits next to an ffprobe of the same build.
    fm = ffmpeg_info()
    if fm.available and fm.path:
        sibling = Path(fm.path).with_name(f"ffprobe{_EXE}")
        if sibling.exists():
            version = _probe_version(str(sibling))
            if version:
                return BinaryInfo(str(sibling), version, "sibling of ffmpeg", True)
    return info


def require_ffmpeg() -> str:
    info = ffmpeg_info()
    if not info.available or not info.path:
        raise FFmpegUnavailable(
            "FFmpeg is not available. Install FFmpeg and add it to PATH, "
            "set VISIONTRACK_FFMPEG to the binary, or run "
            "`pip install imageio-ffmpeg` for a vendored build."
        )
    return info.path


def require_ffprobe() -> str:
    info = ffprobe_info()
    if not info.available or not info.path:
        raise FFmpegUnavailable(
            "FFprobe is not available. It ships alongside FFmpeg - install the "
            "full FFmpeg package or set VISIONTRACK_FFPROBE."
        )
    return info.path


_HW_ENCODER_ORDER = (
    "h264_nvenc",
    "h264_qsv",
    "h264_amf",
    "h264_videotoolbox",
    "h264_vaapi",
)

# Extra flags each hardware encoder needs before it will even initialise.
_HW_ENCODER_PROBE_FLAGS: dict[str, list[str]] = {
    "h264_nvenc": ["-preset", "p1"],
    "h264_qsv": [],
    "h264_amf": ["-usage", "ultralowlatency"],
    "h264_videotoolbox": [],
    "h264_vaapi": ["-vf", "format=nv12,hwupload"],
}


@lru_cache(maxsize=1)
def compiled_hardware_encoders() -> list[str]:
    """H.264 hardware encoders this *build* advertises, best first.

    Presence here means nothing about the host: cloud FFmpeg builds routinely
    list h264_nvenc on machines with no NVIDIA device at all. Use
    :func:`hardware_encoders` for the answer that reflects real hardware.
    """
    info = ffmpeg_info()
    if not info.available or not info.path:
        return []
    try:
        proc = _run([info.path, "-hide_banner", "-encoders"], timeout=25)
    except (OSError, subprocess.SubprocessError):
        return []
    text = proc.stdout or ""
    return [enc for enc in _HW_ENCODER_ORDER if re.search(rf"\b{enc}\b", text)]


def _encoder_works(ffmpeg_path: str, encoder: str) -> bool:
    """Actually encode two synthetic frames with `encoder`.

    This is the only reliable test: it forces FFmpeg to open the device, load
    the driver and initialise the session, which is exactly what fails at run
    time on GPU-less hosts. Output goes to the null muxer so nothing is written.
    """
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:r=25",
        "-frames:v",
        "2",
    ]
    if encoder == "h264_vaapi":
        # VAAPI needs an explicit device before the input is parsed.
        cmd[3:3] = ["-vaapi_device", os.environ.get("VISIONTRACK_VAAPI_DEVICE", "/dev/dri/renderD128")]
    cmd += _HW_ENCODER_PROBE_FLAGS.get(encoder, [])
    cmd += ["-c:v", encoder, "-f", "null", "-"]
    try:
        proc = _run(cmd, timeout=25)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


@lru_cache(maxsize=1)
def hardware_encoders() -> list[str]:
    """H.264 hardware encoders that genuinely work on *this host*, best first.

    Every candidate the build advertises is functionally probed once, then the
    result is cached for the process lifetime. On a CPU-only box (Render, most
    containers) this returns `[]` and callers fall back to libx264.
    """
    if settings.force_cpu:
        log.info("VISIONTRACK_FORCE_CPU set - hardware video encoders disabled")
        return []
    info = ffmpeg_info()
    if not info.available or not info.path:
        return []
    working: list[str] = []
    for enc in compiled_hardware_encoders():
        if _encoder_works(info.path, enc):
            log.info("hardware encoder %s verified on this host", enc)
            working.append(enc)
        else:
            log.info("hardware encoder %s advertised by build but unusable here", enc)
    if not working:
        log.info("no usable hardware H.264 encoder - using libx264 (CPU)")
    return working


def preferred_encoder() -> str:
    """The encoder every writer should default to: real hardware, else libx264."""
    encoders = hardware_encoders()
    return encoders[0] if encoders else "libx264"


@lru_cache(maxsize=1)
def hwaccel_decode_available() -> bool:
    """Whether `-hwaccel auto` buys anything on this host.

    FFmpeg silently degrades `-hwaccel auto` to software decode, so passing it
    is not dangerous - but on GPU-less hosts it is pure noise in the logs, and
    on partially-broken driver setups it can abort the decode outright. Probing
    once and omitting the flag when it does nothing keeps the CPU path clean.
    """
    if settings.force_cpu:
        return False
    info = ffmpeg_info()
    if not info.available or not info.path:
        return False
    try:
        proc = _run([info.path, "-hide_banner", "-hwaccels"], timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    names = {
        ln.strip()
        for ln in (proc.stdout or "").splitlines()[1:]
        if ln.strip() and not ln.startswith(" ")
    }
    # `cuda`/`vaapi`/etc. being *compiled in* is not enough either, so confirm
    # by actually decoding two synthetic frames through the accelerator.
    for name in ("cuda", "d3d11va", "videotoolbox", "qsv", "vaapi", "dxva2"):
        if name not in names:
            continue
        cmd = [
            info.path, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-hwaccel", name,
            "-f", "lavfi", "-i", "color=c=black:s=320x240:r=25",
            "-frames:v", "2", "-f", "null", "-",
        ]
        try:
            if _run(cmd, timeout=25).returncode == 0:
                log.info("hardware decode via %s verified on this host", name)
                return True
        except (OSError, subprocess.SubprocessError):
            continue
    log.info("no usable hardware video decoder - decoding on CPU")
    return False


def _as_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _parse_rate(raw: Any) -> float | None:
    """Parse an ffprobe rational like '30000/1001'."""
    if raw in (None, "", "0/0"):
        return None
    text = str(raw)
    if "/" in text:
        num, _, den = text.partition("/")
        n, d = _as_float(num), _as_float(den)
        if n is None or not d:
            return None
        return n / d
    return _as_float(text)


@dataclass
class VideoMetadata:
    duration: float
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str
    codec_long: str
    pixel_format: str | None
    bitrate: int | None
    container: str
    has_audio: bool
    audio_codec: str | None
    rotation: int
    size_bytes: int
    frame_count_exact: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probe_with_ffmpeg(path: Path) -> VideoMetadata:
    """Fallback metadata probe using ffmpeg -i when ffprobe is absent."""
    ffmpeg = require_ffmpeg()
    cmd = [ffmpeg, "-hide_banner", "-i", str(path)]
    proc = _run(cmd, timeout=60)
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    
    # 1. Duration
    dur_m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    duration = 0.0
    if dur_m:
        h, m, s = float(dur_m.group(1)), float(dur_m.group(2)), float(dur_m.group(3))
        duration = h * 3600 + m * 60 + s

    # 2. Video Stream (width, height, codec, fps)
    vid_m = re.search(r"Stream #\d+:\d+.*Video:\s*([a-zA-Z0-9_-]+).*?,\s*(\d+)x(\d+)", text)
    if not vid_m:
        raise FFmpegError("No decodable video stream found in file.", stderr=text, command=cmd)
    
    codec = vid_m.group(1)
    width = int(vid_m.group(2))
    height = int(vid_m.group(3))

    fps_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:fps|tbr)", text)
    fps = float(fps_m.group(1)) if fps_m else 30.0

    # 3. Audio Stream
    aud_m = re.search(r"Stream #\d+:\d+.*Audio:\s*([a-zA-Z0-9_-]+)", text)
    has_audio = aud_m is not None
    audio_codec = aud_m.group(1) if aud_m else None

    # 4. Bitrate
    br_m = re.search(r"bitrate:\s*(\d+)\s*kb/s", text)
    bitrate = int(br_m.group(1)) * 1000 if br_m else None

    # 5. Rotation
    rot_m = re.search(r"rotate\s*:\s*(\d+)", text)
    rotation = int(rot_m.group(1)) if rot_m else 0
    rotation = ((rotation % 360) + 360) % 360
    if rotation in (90, 270):
        width, height = height, width

    frame_count = max(1, int(round(duration * fps))) if (duration > 0 and fps > 0) else 0
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0

    return VideoMetadata(
        duration=round(duration, 3),
        width=width,
        height=height,
        fps=round(fps, 4),
        frame_count=frame_count,
        codec=codec,
        codec_long=codec,
        pixel_format=None,
        bitrate=bitrate,
        container=path.suffix.lstrip("."),
        has_audio=has_audio,
        audio_codec=audio_codec,
        rotation=rotation,
        size_bytes=size_bytes,
        frame_count_exact=False,
    )


def probe_video(path: Path) -> VideoMetadata:
    """Extract real metadata. Raises FFmpegError with actionable diagnostics."""
    fp_info = ffprobe_info()
    if not fp_info.available or not fp_info.path:
        # Graceful fallback to ffmpeg metadata extraction if ffprobe is absent
        return _probe_with_ffmpeg(path)

    ffprobe = fp_info.path
    cmd = [
        ffprobe, "-hide_banner", "-loglevel", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    try:
        proc = _run(cmd, timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(
            "Timed out while reading this file's metadata.",
            stderr=str(exc), command=cmd,
        ) from exc

    if proc.returncode != 0:
        # Attempt ffmpeg fallback
        try:
            return _probe_with_ffmpeg(path)
        except Exception:
            raise FFmpegError(
                "This file could not be read as a video.",
                stderr=proc.stderr or proc.stdout, command=cmd,
            )

    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        try:
            return _probe_with_ffmpeg(path)
        except Exception:
            raise FFmpegError("FFprobe returned malformed output.", stderr=proc.stdout, command=cmd) from exc

    streams = payload.get("streams") or []
    fmt = payload.get("format") or {}
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFmpegError(
            "No video stream found in this file.",
            stderr="The container holds no decodable video track.", command=cmd,
        )

    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise FFmpegError(
            "This video reports invalid dimensions and cannot be analyzed.",
            stderr=f"width={width} height={height}", command=cmd,
        )

    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate")) or 0.0
    duration = (
        _as_float(video.get("duration"))
        or _as_float(fmt.get("duration"))
        or 0.0
    )

    # Rotation may live in a side-data block or a stream tag.
    rotation = 0
    for sd in video.get("side_data_list") or []:
        if "rotation" in sd:
            rotation = int(_as_float(sd.get("rotation")) or 0)
            break
    else:
        tag_rot = (video.get("tags") or {}).get("rotate")
        if tag_rot is not None:
            rotation = int(_as_float(tag_rot) or 0)
    rotation = ((rotation % 360) + 360) % 360

    frames_raw = video.get("nb_frames")
    frame_count = 0
    exact = False
    if frames_raw not in (None, "", "N/A"):
        try:
            frame_count = int(frames_raw)
            exact = frame_count > 0
        except (TypeError, ValueError):
            frame_count = 0
    if frame_count <= 0 and duration > 0 and fps > 0:
        frame_count = max(1, int(round(duration * fps)))
    if duration <= 0 and frame_count > 0 and fps > 0:
        duration = frame_count / fps

    if duration <= 0 or fps <= 0:
        raise FFmpegError(
            "This video has no usable duration or frame rate.",
            stderr=f"duration={duration} fps={fps}; the file may be truncated or corrupted.",
            command=cmd,
        )

    # Rotated portrait footage: report display dimensions, which is what the
    # decoder hands us and therefore what detection coordinates are relative to.
    if rotation in (90, 270):
        width, height = height, width

    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = int(_as_float(fmt.get("size")) or 0)

    return VideoMetadata(
        duration=round(duration, 3),
        width=width,
        height=height,
        fps=round(fps, 4),
        frame_count=frame_count,
        codec=str(video.get("codec_name") or "unknown"),
        codec_long=str(video.get("codec_long_name") or video.get("codec_name") or "unknown"),
        pixel_format=video.get("pix_fmt"),
        bitrate=int(_as_float(fmt.get("bit_rate")) or 0) or None,
        container=str(fmt.get("format_name") or Path(path).suffix.lstrip(".")),
        has_audio=audio is not None,
        audio_codec=str(audio.get("codec_name")) if audio else None,
        rotation=rotation,
        size_bytes=size_bytes,
        frame_count_exact=exact,
    )


def verify_decodable(path: Path) -> None:
    """Decode the first second to catch codecs ffprobe reads but can't decode."""
    ffmpeg = require_ffmpeg()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-t", "1", "-i", str(path),
        "-frames:v", "1", "-f", "null", "-",
    ]
    try:
        proc = _run(cmd, timeout=90)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError("Timed out while decoding this video.", stderr=str(exc), command=cmd) from exc
    if proc.returncode != 0:
        raise FFmpegError(
            "This video's codec could not be decoded.",
            stderr=proc.stderr, command=cmd,
        )


def extract_poster(src: Path, dest: Path, *, timestamp: float = 0.0, width: int = 640) -> bool:
    """Single still frame for job history cards. Best-effort."""
    try:
        ffmpeg = require_ffmpeg()
    except FFmpegUnavailable:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, timestamp):.3f}", "-i", str(src),
        "-frames:v", "1",
        "-vf", f"scale={width}:-2:flags=bicubic",
        "-q:v", "4", str(dest),
    ]
    try:
        proc = _run(cmd, timeout=90)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and dest.exists()


def extract_scrub_sprite(
    src: Path, dest: Path, *, duration: float, columns: int = 10, rows: int = 10, tile_width: int = 160
) -> dict[str, Any] | None:
    """Build a tiled thumbnail sheet powering timeline hover previews.

    One sprite request beats 100 individual frame fetches while scrubbing.
    """
    try:
        ffmpeg = require_ffmpeg()
    except FFmpegUnavailable:
        return None
    if duration <= 0:
        return None

    total = columns * rows
    # fps chosen so exactly `total` tiles span the video.
    rate = total / duration
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vf", (
            f"fps={rate:.6f},scale={tile_width}:-2:flags=fast_bilinear,"
            f"tile={columns}x{rows}"
        ),
        "-frames:v", "1", "-q:v", "6",
        str(dest),
    ]
    try:
        proc = _run(cmd, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not dest.exists():
        return None
    return {"columns": columns, "rows": rows, "count": total, "tileWidth": tile_width}


def build_encode_command(
    ffmpeg: str,
    *,
    width: int,
    height: int,
    fps: float,
    dest: Path,
    encoder: str,
    crf: int = 20,
    audio_source: Path | None = None,
) -> list[str]:
    """Command that consumes raw BGR24 frames on stdin and writes H.264.

    `encoder` should come from :func:`preferred_encoder` so it is known to work
    on this host; anything unrecognised degrades to libx264 rather than failing.
    """
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if encoder == "h264_vaapi":
        cmd += ["-vaapi_device", os.environ.get("VISIONTRACK_VAAPI_DEVICE", "/dev/dri/renderD128")]
    cmd += [
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "bgr24",
        "-r", f"{fps:.6f}", "-i", "-",
    ]
    if audio_source is not None:
        cmd += ["-i", str(audio_source)]

    cmd += ["-map", "0:v:0"]
    if audio_source is not None:
        cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "128k", "-shortest"]

    # yuv420p is what software encoders need; hardware paths carry their own
    # pixel format through the device filter chain instead.
    software_pix_fmt = True
    if encoder == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    elif encoder == "h264_qsv":
        cmd += ["-c:v", "h264_qsv", "-global_quality", str(crf)]
    elif encoder == "h264_amf":
        cmd += ["-c:v", "h264_amf", "-quality", "balanced", "-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf)]
    elif encoder == "h264_videotoolbox":
        cmd += ["-c:v", "h264_videotoolbox", "-q:v", str(max(1, min(100, 100 - crf * 2)))]
    elif encoder == "h264_vaapi":
        cmd += ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-qp", str(crf)]
        software_pix_fmt = False
    else:
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]

    if software_pix_fmt:
        cmd += ["-pix_fmt", "yuv420p"]
    cmd += ["-movflags", "+faststart", str(dest)]
    return cmd


def transcode(
    src: Path,
    dest: Path,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    crf: int = 20,
    on_progress=None,
    total_frames: int | None = None,
) -> None:
    """Re-encode a source file, streaming -progress output to a callback."""
    ffmpeg = require_ffmpeg()
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]

    filters: list[str] = []
    if width and height:
        filters.append(f"scale={width}:{height}:flags=bicubic")
    elif width:
        filters.append(f"scale={width}:-2:flags=bicubic")
    if fps:
        filters.append(f"fps={fps:.6f}")
    if filters:
        cmd += ["-vf", ",".join(filters)]

    encoder = preferred_encoder()
    if encoder == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    elif encoder == "h264_qsv":
        cmd += ["-c:v", "h264_qsv", "-global_quality", str(crf)]
    elif encoder == "h264_videotoolbox":
        cmd += ["-c:v", "h264_videotoolbox", "-q:v", str(max(1, min(100, 100 - crf * 2)))]
    else:
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]
    # `-c:a copy` fails outright when the source has no audio stream; `?` on the
    # map makes the audio optional so silent clips transcode fine.
    cmd += ["-map", "0:v:0", "-map", "0:a:0?", "-c:a", "copy"]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    cmd += ["-progress", "pipe:1", "-nostats", str(dest)]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        creationflags=_CREATE_NO_WINDOW,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if on_progress and line.startswith("frame=") and total_frames:
                try:
                    done = int(line.split("=", 1)[1].strip())
                except ValueError:
                    continue
                on_progress(min(1.0, done / max(1, total_frames)))
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read() if proc.stderr else ""
        if proc.stderr:
            proc.stderr.close()
        code = proc.wait()
    if code != 0:
        raise FFmpegError("Video export failed during encoding.", stderr=stderr, command=cmd)


def capabilities() -> dict[str, Any]:
    fm, fp = ffmpeg_info(), ffprobe_info()
    encoders = hardware_encoders()
    advertised = compiled_hardware_encoders()
    return {
        "ffmpeg": {
            "available": fm.available, "path": fm.path,
            "version": fm.version, "source": fm.source,
        },
        "ffprobe": {
            "available": fp.available, "path": fp.path,
            "version": fp.version, "source": fp.source,
        },
        "hardwareEncoders": encoders,
        # Advertised-but-unusable encoders: the honest explanation for why a
        # build that lists h264_nvenc still encodes on the CPU here.
        "advertisedEncoders": advertised,
        "unusableEncoders": [enc for enc in advertised if enc not in encoders],
        "preferredEncoder": encoders[0] if encoders else "libx264",
        "hardwareAccelerated": bool(encoders),
        # ffprobe is a nicety, not a requirement: probe_video falls back to
        # parsing `ffmpeg -i` output, so ffmpeg alone is a working pipeline.
        "ready": fm.available,
        "probeMode": "ffprobe" if fp.available else "ffmpeg-fallback",
    }
