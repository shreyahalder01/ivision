"""Generate the motion fixture used by the tracking and export tests.

`sample_traffic.mp4` is a held still, which is fine for verifying counts but
exercises none of the parts of the system that depend on things actually moving:
trails, motion paths, Kalman prediction, and ID persistence across displacement.

This builds `sample_motion.mp4` by panning a crop window across the source
frame, which translates every object across the field of view at a known, steady
rate - a panning camera, in effect. The detections are still produced by real
inference; only the camera motion is synthesized, and it is synthesized with
FFmpeg rather than drawn by hand.

Run with:  python tests/make_fixture.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.services.ffmpeg import probe_video, require_ffmpeg  # noqa: E402

SOURCE = BACKEND / "data" / "tmp" / "sample_traffic.mp4"
DEST = BACKEND / "data" / "tmp" / "sample_motion.mp4"

OUT_W, OUT_H = 640, 360
DURATION = 6.0
FPS = 30.0


def build(force: bool = False) -> Path:
    if DEST.exists() and not force:
        return DEST
    if not SOURCE.exists():
        raise SystemExit(f"source fixture missing: {SOURCE}")

    ffmpeg = require_ffmpeg()
    meta = probe_video(SOURCE)
    travel = meta.width - OUT_W
    if travel <= 0:
        raise SystemExit(f"source is too narrow to pan: {meta.width}px")

    # Linear pan: x sweeps 0 -> travel over the clip, so every object moves
    # `travel` pixels leftward at a constant, checkable rate.
    crop = (
        f"crop={OUT_W}:{OUT_H}:"
        f"x='{travel}*t/{DURATION}':"
        f"y='(in_h-{OUT_H})/2'"
    )
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-stream_loop", "-1", "-i", str(SOURCE),
        "-t", f"{DURATION}",
        "-vf", f"{crop},fps={FPS}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        str(DEST),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    out = probe_video(DEST)
    print(
        f"{DEST.name}: {out.width}x{out.height} {out.fps:g}fps "
        f"{out.frame_count} frames {out.duration:.2f}s "
        f"({DEST.stat().st_size / 1024:.0f} KiB)"
    )
    print(f"pan: {travel}px over {DURATION:g}s = {travel / DURATION:.1f} px/s")
    return DEST


if __name__ == "__main__":
    build(force="--force" in sys.argv)
