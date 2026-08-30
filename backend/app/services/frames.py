"""Streaming frame extraction.

Frames arrive over an FFmpeg rawvideo pipe, read one at a time into a fixed
buffer. Nothing proportional to video length is ever held in memory, which is
what lets a 40-minute 4K file be analysed on a laptop.

Hardware decode (`-hwaccel`) and hardware encode are both used only when they
have been *functionally probed* on this host - see ffmpeg.hwaccel_decode_available
and ffmpeg.preferred_encoder. On a CPU-only box (Render, most containers) the
pipeline runs entirely in software with no behavioural difference.
"""
from __future__ import annotations

import logging
import platform
import subprocess
import threading
from pathlib import Path
from typing import Iterator

import numpy as np

from .ffmpeg import (
    FFmpegError,
    hwaccel_decode_available,
    preferred_encoder,
    require_ffmpeg,
)

log = logging.getLogger("visiontrack.frames")

_CREATE_NO_WINDOW = 0x08000000 if platform.system() == "Windows" else 0


class FrameReader:
    """Iterate decoded BGR24 frames, optionally downscaled for inference.

    Args:
        path: source video.
        width/height: decode output size. Downscaling here (rather than after)
            saves both pipe bandwidth and CPU.
        start_frame: seek target for resuming a cancelled job.
        fps: source frame rate, needed to convert start_frame to a timestamp.
        stride: emit every Nth frame. FFmpeg still decodes all of them (required
            for inter-frame compression) but we skip the copy and the inference.
        hwaccel: allow hardware-accelerated decode. Only honoured when a
            hardware decoder has actually been verified on this host, and
            dropped automatically if the accelerated attempt fails to start.
    """

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        start_frame: int = 0,
        stride: int = 1,
        hwaccel: bool = True,
    ):
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.start_frame = max(0, int(start_frame))
        self.stride = max(1, int(stride))
        # Never claim hardware we have not proven exists.
        self.hwaccel = bool(hwaccel) and hwaccel_decode_available()
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_tail: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._frame_bytes = self.width * self.height * 3
        self._closed = False

    def _command(self) -> list[str]:
        ffmpeg = require_ffmpeg()
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]

        if self.hwaccel:
            cmd += ["-hwaccel", "auto"]

        if self.start_frame > 0:
            # Input-side seek is keyframe-accurate and fast; -ss after -i would
            # decode-and-discard from zero.
            cmd += ["-ss", f"{self.start_frame / self.fps:.6f}"]

        cmd += ["-i", str(self.path)]
        cmd += [
            "-map", "0:v:0",
            "-vf", f"scale={self.width}:{self.height}:flags=bilinear",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "-fps_mode", "passthrough",
            "-an", "-sn", "-dn",
            "-",
        ]
        return cmd

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", "replace").rstrip()
                if decoded:
                    self._stderr_tail.append(decoded)
                    if len(self._stderr_tail) > 40:
                        self._stderr_tail.pop(0)
        except (ValueError, OSError):
            pass

    def open(self) -> None:
        cmd = self._command()
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._frame_bytes * 2,
                creationflags=_CREATE_NO_WINDOW,
            )
        except OSError as exc:
            if self.hwaccel:
                # A driver that refuses to load must not cost us the whole job.
                log.warning("hardware decode failed to start (%s) - retrying on CPU", exc)
                self.hwaccel = False
                self.open()
                return
            raise FFmpegError(
                "FFmpeg could not be started to read this video.",
                stderr=str(exc), command=cmd,
            ) from exc
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (absolute_frame_index, BGR frame). Honours `stride`."""
        if self._proc is None:
            self.open()
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        index = self.start_frame
        need = self._frame_bytes

        while True:
            buf = stdout.read(need)
            if not buf:
                break
            if len(buf) < need:
                # Truncated tail frame - the stream ended mid-frame.
                break
            if (index - self.start_frame) % self.stride == 0:
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)
                yield index, frame
            index += 1

        code = self._proc.wait()
        if code not in (0, None) and not self._closed:
            tail = "\n".join(self._stderr_tail)
            # 'Output file is empty' from a zero-frame seek is not a real error.
            if "Output file is empty" not in tail:
                raise FFmpegError(
                    "Video decoding stopped unexpectedly.",
                    stderr=tail or f"ffmpeg exited with code {code}",
                    command=self._command(),
                )

    def close(self) -> None:
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.kill()
        self._proc = None

    def __enter__(self) -> "FrameReader":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)


class FrameWriter:
    """Pipe annotated BGR frames into FFmpeg to produce an H.264 MP4."""

    def __init__(
        self,
        dest: Path,
        *,
        width: int,
        height: int,
        fps: float,
        crf: int = 20,
        audio_source: Path | None = None,
        encoder: str | None = None,
    ):
        from .ffmpeg import build_encode_command

        self.dest = Path(dest)
        self.width, self.height = int(width), int(height)
        self.fps = float(fps) if fps > 0 else 30.0
        self.encoder = encoder or preferred_encoder()
        self._cmd = build_encode_command(
            require_ffmpeg(), width=self.width, height=self.height, fps=self.fps,
            dest=self.dest, encoder=self.encoder, crf=crf, audio_source=audio_source,
        )
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_tail: list[str] = []
        self._stderr_thread: threading.Thread | None = None

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", "replace").rstrip()
                if decoded:
                    self._stderr_tail.append(decoded)
                    if len(self._stderr_tail) > 40:
                        self._stderr_tail.pop(0)
        except (ValueError, OSError):
            pass

    def open(self) -> None:
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._proc = subprocess.Popen(
                self._cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=_CREATE_NO_WINDOW,
            )
        except OSError as exc:
            raise FFmpegError(
                "FFmpeg could not be started to encode the export.",
                stderr=str(exc), command=self._cmd,
            ) from exc
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise FFmpegError("The encoder is not open.", command=self._cmd)
        try:
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise FFmpegError(
                "The encoder closed unexpectedly while writing frames.",
                stderr="\n".join(self._stderr_tail) or str(exc),
                command=self._cmd,
            ) from exc

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        code = proc.wait()
        if proc.stderr:
            try:
                proc.stderr.close()
            except OSError:
                pass
        self._proc = None
        if code not in (0, None):
            raise FFmpegError(
                "Video encoding failed.",
                stderr="\n".join(self._stderr_tail) or f"ffmpeg exited with code {code}",
                command=self._cmd,
            )

    def abort(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.kill()

    def __enter__(self) -> "FrameWriter":
        self.open()
        return self

    def __exit__(self, exc_type: object, *rest: object) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self.close()


def inference_dimensions(width: int, height: int, *, target_long_edge: int = 960) -> tuple[int, int]:
    """Decode size for detection.

    Downscaling before inference is a large win and costs little accuracy: YOLO
    letterboxes to 640 internally anyway. Dimensions stay even for yuv420p and
    never upscale beyond the source.
    """
    if width <= 0 or height <= 0:
        return 640, 640
    long_edge = max(width, height)
    if long_edge <= target_long_edge:
        w, h = width, height
    else:
        scale = target_long_edge / long_edge
        w = int(round(width * scale))
        h = int(round(height * scale))
    return max(2, w - (w % 2)), max(2, h - (h % 2))
