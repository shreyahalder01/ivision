"""Overlay rendering for burned-in video exports.

The browser draws its own overlays on a canvas; this module reproduces them for
an exported MP4, where the annotations have to be baked into the pixels. Both
sides read the same normalised (0..1) detection geometry, so an export at any
resolution lines up with what the user saw in the workspace.

Every visual element is individually switchable because the export dialog
exposes them as separate checkboxes - boxes, class names, confidence, track IDs,
frame numbers, timestamps and motion paths.
"""
from __future__ import annotations

import colorsys
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

# Annotation styles the UI offers. Mask and skeleton need a segmentation or pose
# checkpoint; a detection run has no such data, so asking for them downgrades
# with an explicit notice rather than drawing something that merely looks like a
# mask.
STYLE_BOX_LABEL = "box_label"
STYLE_BOX_ONLY = "box_only"
STYLE_LABEL_ONLY = "label_only"
STYLE_MASK = "mask"
STYLE_SKELETON = "skeleton"

DETECTION_STYLES = (STYLE_BOX_LABEL, STYLE_BOX_ONLY, STYLE_LABEL_ONLY)


def resolve_style(requested: str | None, *, has_masks: bool = False,
                  has_keypoints: bool = False) -> tuple[str, str | None]:
    """Return (usable style, notice). Never silently substitutes."""
    style = (requested or STYLE_BOX_LABEL).strip().lower()
    if style == STYLE_MASK and not has_masks:
        return STYLE_BOX_LABEL, (
            "Segmentation masks need a segmentation checkpoint (yolo11*-seg). This "
            "analysis used a detection model, so boxes were drawn instead."
        )
    if style == STYLE_SKELETON and not has_keypoints:
        return STYLE_BOX_LABEL, (
            "Skeletons need a pose checkpoint (yolo11*-pose). This analysis used a "
            "detection model, so boxes were drawn instead."
        )
    if style not in DETECTION_STYLES:
        return STYLE_BOX_LABEL, None
    return style, None


# --------------------------------------------------------------------- palette

def _palette(n: int = 24) -> list[tuple[int, int, int]]:
    """Evenly spaced hues at fixed saturation/value, as BGR.

    Golden-ratio stepping keeps consecutive track IDs visually far apart, which
    matters when two objects overlap.
    """
    out: list[tuple[int, int, int]] = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
        out.append((int(b * 255), int(g * 255), int(r * 255)))
    return out


PALETTE = _palette()


def colour_for_track(track_id: int) -> tuple[int, int, int]:
    """Stable per-track colour, so an object keeps its colour for the whole clip."""
    return PALETTE[int(track_id) % len(PALETTE)]


# ---------------------------------------------------------------------- options

@dataclass
class OverlayOptions:
    """Which overlay elements to draw. Mirrors the export content checkboxes."""

    boxes: bool = True
    class_labels: bool = True
    confidence: bool = True
    track_ids: bool = True
    frame_numbers: bool = False
    timestamps: bool = False
    motion_paths: bool = False
    trails: bool = False
    style: str = STYLE_BOX_LABEL
    trail_seconds: float = 1.5

    @classmethod
    def from_payload(
        cls, data: dict[str, Any] | None, *, stamps_default: bool = False
    ) -> "OverlayOptions":
        """Build from an export/preview payload.

        `stamps_default` differs by destination: frame numbers and timestamps are
        essential columns in a data export but an opt-in burned-in HUD on video,
        so the caller decides.
        """
        d = data or {}

        def flag(*names: str, default: bool) -> bool:
            for n in names:
                if n in d:
                    return bool(d[n])
            return default

        return cls(
            boxes=flag("boxes", "detectionBox", default=True),
            class_labels=flag("classes", "classLabel", default=True),
            confidence=flag("confidence", "confidenceScore", default=True),
            track_ids=flag("trackIds", "trackId", default=True),
            frame_numbers=flag("frameNumbers", default=stamps_default),
            timestamps=flag("timestamps", default=stamps_default),
            motion_paths=flag("motionPaths", default=False),
            trails=flag("trails", "objectTrail", default=False),
            style=str(d.get("style") or STYLE_BOX_LABEL),
            trail_seconds=float(d.get("trailSeconds") or 1.5),
        )

    @property
    def draws_anything(self) -> bool:
        return any((
            self.boxes, self.class_labels, self.confidence, self.track_ids,
            self.frame_numbers, self.timestamps, self.motion_paths, self.trails,
        ))


# --------------------------------------------------------------------- renderer

class Annotator:
    """Draws detection overlays onto BGR frames at a fixed output size.

    Stroke weights, font size and padding are derived from the output height so
    a 480p export stays legible and a 4K export does not end up with hairlines.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        options: OverlayOptions,
        fps: float = 30.0,
        class_names: Sequence[str] | None = None,
    ):
        import cv2  # local: keeps the module importable without OpenCV present

        self._cv2 = cv2
        self.width = int(width)
        self.height = int(height)
        self.opt = options
        self.fps = float(fps) if fps > 0 else 30.0
        self.class_names = list(class_names or [])

        scale = max(0.55, min(2.4, self.height / 720.0))
        self.thickness = max(1, int(round(2 * scale)))
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 0.42 * scale
        self.text_thickness = max(1, int(round(scale)))
        self.pad = max(3, int(round(4 * scale)))
        self.trail_len = max(2, int(round(self.fps * self.opt.trail_seconds)))

        # Per-track centroid history for trails, capped so memory stays flat
        # regardless of clip length.
        self._trails: dict[int, deque[tuple[int, int]]] = {}

    # -- primitives ---------------------------------------------------------

    def _text_size(self, text: str) -> tuple[int, int]:
        (w, h), base = self._cv2.getTextSize(text, self.font, self.font_scale, self.text_thickness)
        return w, h + base

    def _label(self, frame: np.ndarray, text: str, x: int, y: int,
               colour: tuple[int, int, int], *, above: bool = True) -> None:
        """A filled pill with the box colour and auto-contrasting text."""
        cv2 = self._cv2
        tw, th = self._text_size(text)
        bw, bh = tw + self.pad * 2, th + self.pad * 2

        top = y - bh if above else y
        if top < 0:                       # flip below when it would clip the top
            top = y
        if top + bh > self.height:
            top = max(0, self.height - bh)
        left = min(max(0, x), max(0, self.width - bw))

        cv2.rectangle(frame, (left, top), (left + bw, top + bh), colour, -1)
        # Relative luminance decides black vs white text - a yellow box needs
        # black text, a deep blue one needs white.
        b, g, r = colour
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        ink = (16, 16, 16) if luma > 150 else (245, 245, 245)
        cv2.putText(
            frame, text, (left + self.pad, top + bh - self.pad - 1),
            self.font, self.font_scale, ink, self.text_thickness, cv2.LINE_AA,
        )

    def _hud(self, frame: np.ndarray, lines: Sequence[str]) -> None:
        """Frame/time stamp in the top-left, on a dark plate for legibility."""
        cv2 = self._cv2
        if not lines:
            return
        sizes = [self._text_size(t) for t in lines]
        bw = max(w for w, _ in sizes) + self.pad * 2
        bh = sum(h for _, h in sizes) + self.pad * (len(lines) + 1)
        overlay = frame[0:bh, 0:bw]
        if overlay.size:
            # Blend rather than fill so the footage stays partly visible.
            cv2.addWeighted(overlay, 0.25, np.zeros_like(overlay), 0.0, 0.0, dst=overlay)
        y = self.pad
        for text, (_, h) in zip(lines, sizes):
            y += h
            cv2.putText(
                frame, text, (self.pad, y - 2), self.font, self.font_scale,
                (235, 235, 235), self.text_thickness, cv2.LINE_AA,
            )
            y += self.pad

    def _polyline(self, frame: np.ndarray, points: Sequence[tuple[int, int]],
                  colour: tuple[int, int, int], *, fade: bool) -> None:
        cv2 = self._cv2
        if len(points) < 2:
            return
        if not fade:
            cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], False,
                          colour, max(1, self.thickness - 1), cv2.LINE_AA)
            return
        # Fading trail: older segments thinner and dimmer, so direction of
        # travel is readable at a glance.
        n = len(points) - 1
        for i in range(n):
            t = (i + 1) / n
            dim = tuple(int(c * (0.35 + 0.65 * t)) for c in colour)
            cv2.line(frame, points[i], points[i + 1], dim,
                     max(1, int(round(self.thickness * t))), cv2.LINE_AA)

    # -- public -------------------------------------------------------------

    def reset_trails(self) -> None:
        self._trails.clear()

    def static_paths(self, tracks: Iterable[dict[str, Any]]) -> dict[int, list[tuple[int, int]]]:
        """Pre-scale stored motion paths to output pixels, once per export."""
        out: dict[int, list[tuple[int, int]]] = {}
        for t in tracks:
            pts = t.get("path") or []
            if len(pts) < 2:
                continue
            out[int(t["track_id"])] = [
                (int(nx * self.width), int(ny * self.height)) for _f, nx, ny in pts
            ]
        return out

    def draw(
        self,
        frame: np.ndarray,
        detections: Sequence[dict[str, Any]],
        *,
        frame_index: int,
        timestamp: float,
        paths: dict[int, list[tuple[int, int]]] | None = None,
    ) -> np.ndarray:
        """Annotate in place and return the frame.

        `detections` carry normalised x/y/w/h so this works at any output size.
        """
        cv2 = self._cv2
        opt = self.opt

        # Full motion paths sit underneath everything else.
        if opt.motion_paths and paths:
            for det in detections:
                pts = paths.get(int(det["track_id"]))
                if pts:
                    self._polyline(frame, pts, colour_for_track(det["track_id"]), fade=False)

        for det in detections:
            tid = int(det["track_id"])
            colour = colour_for_track(tid)
            x1 = int(float(det["x"]) * self.width)
            y1 = int(float(det["y"]) * self.height)
            x2 = int((float(det["x"]) + float(det["w"])) * self.width)
            y2 = int((float(det["y"]) + float(det["h"])) * self.height)
            x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)

            if opt.trails:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                hist = self._trails.setdefault(tid, deque(maxlen=self.trail_len))
                hist.append((cx, cy))
                self._polyline(frame, list(hist), colour, fade=True)

            if opt.boxes and opt.style in (STYLE_BOX_LABEL, STYLE_BOX_ONLY):
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, self.thickness, cv2.LINE_AA)

            if opt.style == STYLE_BOX_ONLY:
                continue

            parts: list[str] = []
            if opt.class_labels:
                parts.append(str(det["class_name"]).upper())
            if opt.track_ids:
                parts.append(f"ID:{tid:02d}")
            if opt.confidence:
                parts.append(f"{float(det['confidence']) * 100:.0f}%")
            if parts:
                self._label(frame, "  ".join(parts), x1, y1, colour)

        hud: list[str] = []
        if opt.frame_numbers:
            hud.append(f"FRAME {frame_index:06d}")
        if opt.timestamps:
            hud.append(format_timecode(timestamp))
        self._hud(frame, hud)
        return frame


def format_timecode(seconds: float) -> str:
    """HH:MM:SS.mmm - the format the workspace readout uses."""
    if seconds != seconds or seconds < 0:  # NaN guard
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
