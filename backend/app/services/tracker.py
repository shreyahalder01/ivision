"""Multi-object tracking with persistent identities.

`PersistentTracker` is a from-scratch SORT-family tracker built for identity
stability rather than benchmark scores. The pieces that matter:

* A 7-state Kalman filter per track ([cx, cy, area, aspect, vx, vy, v_area])
  gives real motion prediction, so a box keeps moving through a missed detection
  instead of snapping or dying.
* Two-stage association. Stage 1 matches *confirmed* tracks against detections
  using IoU gated by predicted position. Stage 2 offers the leftovers to *lost*
  tracks using a coarse appearance descriptor plus motion-consistent distance,
  which is what recovers an ID after an occlusion.
* An explicit lifecycle (tentative -> confirmed -> lost -> removed). IDs are
  only issued on promotion to confirmed, so detector flicker cannot burn through
  the ID space, and a visible object never silently changes number.

Appearance descriptors are HSV colour histograms over a 2x2 spatial grid. That
is a genuine re-identification cue, but it is a hand-crafted descriptor, not a
learned re-ID network - the UI says so where the tracker is selected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np

try:  # optimal assignment when SciPy is present (it ships with ultralytics)
    from scipy.optimize import linear_sum_assignment as _lsa

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


# --------------------------------------------------------------------- geometry

def xyxy_to_z(box: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] -> [cx, cy, area, aspect] measurement vector."""
    w = max(1e-6, box[2] - box[0])
    h = max(1e-6, box[3] - box[1])
    return np.array([box[0] + w / 2.0, box[1] + h / 2.0, w * h, w / h], dtype=np.float64)


def z_to_xyxy(z: Sequence[float]) -> np.ndarray:
    """[cx, cy, area, aspect] -> [x1,y1,x2,y2], clamped to a sane aspect."""
    area = max(1e-6, float(z[2]))
    aspect = float(z[3])
    if not math.isfinite(aspect) or aspect <= 1e-6:
        aspect = 1.0
    w = math.sqrt(area * aspect)
    h = area / w if w > 1e-6 else math.sqrt(area)
    return np.array([z[0] - w / 2.0, z[1] - h / 2.0, z[0] + w / 2.0, z[1] + h / 2.0], dtype=np.float64)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two [N,4] / [M,4] xyxy arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = a[:, None, :]
    b = b[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return np.nan_to_num(out)


def assign(cost: np.ndarray, max_cost: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve the assignment problem, dropping pairs above `max_cost`.

    Uses Hungarian when SciPy is available and a greedy best-first pass
    otherwise; the greedy path keeps the tracker functional in a minimal install.
    """
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))

    matches: list[tuple[int, int]] = []
    if _HAS_SCIPY:
        r_idx, c_idx = _lsa(cost)
        for r, c in zip(r_idx, c_idx):
            if cost[r, c] <= max_cost:
                matches.append((int(r), int(c)))
    else:
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        used_r: set[int] = set()
        used_c: set[int] = set()
        for r, c in order:
            r, c = int(r), int(c)
            if r in used_r or c in used_c or cost[r, c] > max_cost:
                continue
            used_r.add(r)
            used_c.add(c)
            matches.append((r, c))

    matched_r = {r for r, _ in matches}
    matched_c = {c for _, c in matches}
    return (
        matches,
        [r for r in range(rows) if r not in matched_r],
        [c for c in range(cols) if c not in matched_c],
    )


# ----------------------------------------------------------------- kalman filter

class KalmanBox:
    """Constant-velocity Kalman filter over [cx, cy, area, aspect]."""

    __slots__ = ("x", "P", "F", "H", "R", "Q")

    def __init__(self, box: np.ndarray):
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 7))
        for i in range(4):
            self.H[i, i] = 1.0

        # Observation noise: centre is trustworthy, area/aspect much less so.
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])

        # Process noise: velocities drift more freely than positions.
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01
        self.Q[-1, -1] *= 0.01
        self.Q[2, 2] *= 0.01

        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0  # velocity is unobservable at birth
        self.P *= 10.0

        self.x = np.zeros((7, 1))
        self.x[:4, 0] = xyxy_to_z(box)

    def predict(self) -> np.ndarray:
        # Area must not go negative when shrinking fast.
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return z_to_xyxy(self.x[:4, 0])

    def update(self, box: np.ndarray) -> None:
        z = xyxy_to_z(box).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:  # pragma: no cover - numerically degenerate
            return
        self.x = self.x + K @ y
        I = np.eye(7)
        self.P = (I - K @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        return z_to_xyxy(self.x[:4, 0])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[4, 0]), float(self.x[5, 0])


# ---------------------------------------------------------------------- lifecycle

class TrackState(str, Enum):
    TENTATIVE = "tentative"   # seen, not yet trusted - holds no public ID
    CONFIRMED = "confirmed"   # actively matched this frame
    LOST = "lost"             # temporarily unmatched, coasting on prediction
    REMOVED = "removed"       # retired


@dataclass
class Track:
    internal_id: int
    class_id: int
    class_name: str
    kf: KalmanBox
    confidence: float
    frame: int
    timestamp: float
    public_id: int = 0
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    first_frame: int = 0
    last_frame: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    conf_sum: float = 0.0
    conf_count: int = 0
    max_confidence: float = 0.0
    descriptor: np.ndarray | None = None
    centroids: list[tuple[int, float, float]] = field(default_factory=list)
    max_speed: float = 0.0
    speed_sum: float = 0.0
    speed_count: int = 0
    distance: float = 0.0
    gap_count: int = 0
    class_votes: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.first_frame = self.last_frame = self.frame
        self.first_seen = self.last_seen = self.timestamp
        self.conf_sum = self.confidence
        self.conf_count = 1
        self.max_confidence = self.confidence
        self.class_votes = {self.class_id: 1}

    @property
    def box(self) -> np.ndarray:
        return self.kf.box

    def blend_descriptor(self, desc: np.ndarray | None, alpha: float = 0.85) -> None:
        """EMA the appearance descriptor so it tracks slow lighting changes."""
        if desc is None:
            return
        if self.descriptor is None:
            self.descriptor = desc.astype(np.float32)
        else:
            self.descriptor = (alpha * self.descriptor + (1.0 - alpha) * desc).astype(np.float32)
            n = float(np.linalg.norm(self.descriptor))
            if n > 1e-6:
                self.descriptor /= n

    def retire(self, path_samples: int = 160) -> None:
        """Compact a track that will never be observed again.

        A retired track still has to be reported - it is an object that appeared
        and left, and the timeline, inspector and class distribution all need it.
        What it no longer needs is its Kalman state, its appearance descriptor or
        every centroid it ever had, so those are released here. The centroid list
        is down-sampled to the same budget the database stores, which bounds
        memory at a few KB per object however long or crowded the footage is.
        """
        self.state = TrackState.REMOVED
        self.descriptor = None
        pts = self.centroids
        if len(pts) > path_samples:
            step = len(pts) / path_samples
            self.centroids = [
                pts[min(len(pts) - 1, int(i * step))] for i in range(path_samples)
            ]


# ------------------------------------------------------------------- descriptors

# Quantisation for the HSV histogram: 6 hue bins x 3 saturation x 3 value, per
# cell of a 2x2 spatial grid => a 48-dim descriptor.
_H_BINS, _S_BINS, _V_BINS = 6, 3, 3
_H_DIV = 180 // _H_BINS          # OpenCV hue is 0..179
_S_DIV = 256 // _S_BINS + 1
_V_DIV = 256 // _V_BINS + 1
_BINS_PER_CELL = _H_BINS + _S_BINS + _V_BINS
_DESC_DIM = 4 * _BINS_PER_CELL

# Every crop is resampled to this square before binning. That makes descriptor
# cost independent of object size - a full-frame bus box would otherwise cost
# ~200x a distant pedestrian - and it makes the grid cells contiguous, which
# matters because strided views turn each bin op into a slow gather.
_DESC_SIZE = 32
_HALF = _DESC_SIZE // 2

# Cell id (0..3) per pixel of the normalised crop, pre-multiplied into the bin
# offset so one bincount produces all four cells at once.
_side = (np.arange(_DESC_SIZE) >= _HALF).astype(np.intp)
_CELL_BASE = (np.add.outer(_side * 2, _side).ravel() * _BINS_PER_CELL)
_H_OFF = _CELL_BASE
_S_OFF = _CELL_BASE + _H_BINS
_V_OFF = _CELL_BASE + _H_BINS + _S_BINS


def prepare_appearance_frame(frame: np.ndarray) -> np.ndarray | None:
    """Convert a BGR frame to HSV once, for reuse across every detection.

    Converting per-crop instead cost ~12 ms/frame on its own.
    """
    try:
        import cv2

        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    except Exception:  # pragma: no cover
        return None


def appearance_descriptor(hsv: np.ndarray | None, box: np.ndarray) -> np.ndarray | None:
    """HSV histogram over a 2x2 grid, L2-normalised.

    `hsv` must already be HSV (see `prepare_appearance_frame`). This is a genuine
    re-identification cue but a hand-crafted one - not a learned re-ID network,
    so it is only ever used as one weighted term in the association cost.
    """
    if hsv is None:
        return None
    try:
        import cv2
    except Exception:  # pragma: no cover
        return None

    h, w = hsv.shape[:2]
    x1 = max(0, min(w - 1, int(box[0])))
    y1 = max(0, min(h - 1, int(box[1])))
    x2 = max(x1 + 1, min(w, int(box[2])))
    y2 = max(y1 + 1, min(h, int(box[3])))
    if (y2 - y1) < 4 or (x2 - x1) < 4:
        return None

    # INTER_AREA averages over each source region, so this is a box-filtered
    # downsample rather than a subsample - the histogram still reflects the
    # whole crop, not 1024 scattered pixels of it.
    small = cv2.resize(
        hsv[y1:y2, x1:x2], (_DESC_SIZE, _DESC_SIZE), interpolation=cv2.INTER_AREA
    ).reshape(-1, 3)

    idx = np.concatenate((
        _H_OFF + (small[:, 0] // _H_DIV),
        _S_OFF + (small[:, 1] // _S_DIV),
        _V_OFF + (small[:, 2] // _V_DIV),
    ))
    feats = np.bincount(idx, minlength=_DESC_DIM)[:_DESC_DIM].astype(np.float32)

    norm = float(np.linalg.norm(feats))
    return feats / norm if norm > 1e-6 else None


# ----------------------------------------------------------------- main tracker

@dataclass
class TrackerConfig:
    iou_threshold: float = 0.30          # stage-1 gate
    byte_iou_threshold: float = 0.50     # stage-1b gate (stricter: weak evidence)
    reid_distance_gate: float = 0.28     # stage-2 gate, fraction of frame diagonal
    reid_appearance_gate: float = 0.45   # max cosine distance for re-ID
    min_hits: int = 3                    # detections before an ID is issued
    max_age: int = 45                    # frames a lost track coasts (~1.5s @30fps)
    use_appearance: bool = True          # stage 2: appearance re-ID
    use_byte_association: bool = False   # stage 1b: low-confidence rescue
    low_conf_floor: float = 0.10         # BYTE's lower detection bound
    class_consistent: bool = True        # never match across classes

    @classmethod
    def for_method(cls, method: str, fps: float = 30.0) -> "TrackerConfig":
        """Tune the association strategy and lifecycle to the real frame rate.

        The four modes are genuinely different algorithms, not relabelled
        settings:

        persistent - IoU + Kalman, with appearance re-ID to recover from
            occlusion. Optimised for identifiers that never churn.
        bytetrack  - the BYTE association strategy: high-confidence detections
            match first, then leftover *low*-confidence detections are offered to
            unmatched tracks, which is what keeps partially-occluded objects
            alive instead of killing and re-issuing them.
        deepsort   - appearance-led association with a long occlusion memory.
        auto       - BYTE rescue plus appearance re-ID; the most thorough and
            the most expensive.
        """
        fps = fps if fps and fps > 0 else 30.0
        base_age = int(round(fps * 1.5))
        if method == "deepsort":
            return cls(
                iou_threshold=0.25, reid_distance_gate=0.35,
                reid_appearance_gate=0.55, min_hits=3,
                max_age=int(round(fps * 3.0)),
                use_appearance=True, use_byte_association=False,
            )
        if method == "bytetrack":
            return cls(
                iou_threshold=0.25, byte_iou_threshold=0.50, min_hits=2,
                max_age=int(round(fps * 1.0)),
                use_appearance=False, use_byte_association=True,
            )
        if method == "auto":
            return cls(
                iou_threshold=0.28, min_hits=3, max_age=int(round(fps * 2.0)),
                use_appearance=True, use_byte_association=True,
            )
        return cls(
            min_hits=3, max_age=base_age,
            use_appearance=True, use_byte_association=False,
        )


class PersistentTracker:
    """Frame-by-frame tracker producing stable public IDs."""

    def __init__(self, config: TrackerConfig | None = None, *, id_offset: int = 0,
                 frame_width: int = 1920, frame_height: int = 1080):
        self.cfg = config or TrackerConfig()
        self.tracks: list[Track] = []
        # Tracks that have left the scene for good. Held so `finalize` can report
        # every object the video contained, not just the ones still visible when
        # it ended. Compacted on retirement, so this stays small.
        self.retired: list[Track] = []
        self._next_internal = 1
        self._next_public = id_offset + 1
        self.diag = math.hypot(frame_width, frame_height) or 1.0
        self.frame_width = frame_width
        self.frame_height = frame_height

    # -- helpers ---------------------------------------------------------------

    def _new_track(self, box, class_id, class_name, conf, frame, ts, desc) -> Track:
        t = Track(
            internal_id=self._next_internal, class_id=class_id, class_name=class_name,
            kf=KalmanBox(box), confidence=conf, frame=frame, timestamp=ts,
        )
        self._next_internal += 1
        t.blend_descriptor(desc, alpha=0.0)
        t.centroids.append((frame, float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
        self.tracks.append(t)
        return t

    def _observe(self, t: Track, box, conf, class_id, class_name, frame, ts, desc) -> None:
        prev = t.centroids[-1] if t.centroids else None
        t.kf.update(box)
        t.hits += 1
        # Step 1 already bumped `time_since_update` for this frame, so a value of
        # 1 means "matched on consecutive frames". Anything higher means the
        # track coasted on prediction through at least one frame with no
        # detection - that is the gap worth counting.
        t.gap_count += 1 if t.time_since_update > 1 else 0
        t.time_since_update = 0
        t.confidence = conf
        t.conf_sum += conf
        t.conf_count += 1
        t.max_confidence = max(t.max_confidence, conf)
        t.last_frame = frame
        t.last_seen = ts
        t.class_votes[class_id] = t.class_votes.get(class_id, 0) + 1
        # Majority class over the track's life; suppresses single-frame flips.
        best = max(t.class_votes.items(), key=lambda kv: kv[1])[0]
        if best != t.class_id:
            t.class_id = best
        t.class_name = class_name if best == class_id else t.class_name
        t.blend_descriptor(desc)

        cx, cy = float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)
        if prev is not None:
            dframes = max(1, frame - prev[0])
            step = math.hypot(cx - prev[1], cy - prev[2])
            speed = step / dframes            # px per frame
            t.distance += step
            t.max_speed = max(t.max_speed, speed)
            t.speed_sum += speed
            t.speed_count += 1
        t.centroids.append((frame, cx, cy))

        if t.state is TrackState.TENTATIVE and t.hits >= self.cfg.min_hits:
            t.state = TrackState.CONFIRMED
            t.public_id = self._next_public
            self._next_public += 1
        elif t.state is TrackState.LOST:
            t.state = TrackState.CONFIRMED  # ID preserved across the gap

    def _appearance_cost(self, track: Track, desc: np.ndarray | None) -> float:
        if track.descriptor is None or desc is None:
            return 0.5  # neutral - neither evidence for nor against
        return float(np.clip(1.0 - float(np.dot(track.descriptor, desc)), 0.0, 1.0))

    # -- main step -------------------------------------------------------------

    def update(
        self,
        boxes: np.ndarray,
        class_ids: Sequence[int],
        class_names: Sequence[str],
        confidences: Sequence[float],
        frame_index: int,
        timestamp: float,
        image: np.ndarray | None = None,
        low_boxes: np.ndarray | None = None,
        low_class_ids: Sequence[int] | None = None,
        low_class_names: Sequence[str] | None = None,
        low_confidences: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Advance the tracker one frame and return this frame's live tracks.

        `boxes` are the high-confidence detections (above the user's threshold).
        The optional `low_*` arrays carry sub-threshold detections used only by
        the BYTE rescue stage - they can extend an existing track but never
        create one, so a weak detection can never invent an object.
        """
        n = len(boxes)
        boxes = np.asarray(boxes, dtype=np.float64).reshape(n, 4) if n else np.zeros((0, 4))

        # 1. Predict every existing track forward.
        for t in self.tracks:
            t.kf.predict()
            t.age += 1
            t.time_since_update += 1

        descriptors: list[np.ndarray | None] = [None] * n
        hsv: np.ndarray | None = None
        if self.cfg.use_appearance and image is not None:
            # One colour-space conversion per frame, shared by every crop.
            hsv = prepare_appearance_frame(image)
            for i in range(n):
                descriptors[i] = appearance_descriptor(hsv, boxes[i])

        active = [t for t in self.tracks if t.state in (TrackState.CONFIRMED, TrackState.TENTATIVE)]
        lost = [t for t in self.tracks if t.state is TrackState.LOST]

        unmatched_dets = list(range(n))
        unmatched_active: list[Track] = []

        # 2. Stage 1 - IoU against actively-tracked objects.
        if active and unmatched_dets:
            pred = np.array([t.box for t in active])
            dets = boxes[unmatched_dets]
            ious = iou_matrix(pred, dets)
            cost = 1.0 - ious
            if self.cfg.class_consistent:
                for ti, t in enumerate(active):
                    for di, d in enumerate(unmatched_dets):
                        if class_ids[d] != t.class_id:
                            cost[ti, di] = 1e6
            matches, un_t, un_d = assign(cost, 1.0 - self.cfg.iou_threshold)
            for ti, di in matches:
                d = unmatched_dets[di]
                self._observe(
                    active[ti], boxes[d], float(confidences[d]), int(class_ids[d]),
                    class_names[d], frame_index, timestamp, descriptors[d],
                )
            unmatched_dets = [unmatched_dets[i] for i in un_d]
            unmatched_active = [active[i] for i in un_t]
        else:
            unmatched_active = list(active)

        # 3. Stage 1b - BYTE rescue. Offer sub-threshold detections to tracks
        #    that found no strong match. This is what carries an object through
        #    a partial occlusion where the detector loses confidence but not
        #    the object. Confirmed tracks only: a tentative track should not be
        #    promoted on weak evidence.
        if (
            self.cfg.use_byte_association
            and low_boxes is not None
            and len(low_boxes)
            and unmatched_active
        ):
            candidates = [t for t in unmatched_active if t.state is TrackState.CONFIRMED]
            if candidates:
                lb = np.asarray(low_boxes, dtype=np.float64).reshape(len(low_boxes), 4)
                l_cls = list(low_class_ids or [])
                l_names = list(low_class_names or [])
                l_conf = list(low_confidences or [])
                pred = np.array([t.box for t in candidates])
                cost = 1.0 - iou_matrix(pred, lb)
                if self.cfg.class_consistent:
                    for ti, t in enumerate(candidates):
                        for di in range(len(lb)):
                            if di < len(l_cls) and l_cls[di] != t.class_id:
                                cost[ti, di] = 1e6
                matches, un_t, _ = assign(cost, 1.0 - self.cfg.byte_iou_threshold)
                for ti, di in matches:
                    desc = appearance_descriptor(hsv, lb[di]) if hsv is not None else None
                    self._observe(
                        candidates[ti], lb[di],
                        float(l_conf[di]) if di < len(l_conf) else self.cfg.low_conf_floor,
                        int(l_cls[di]) if di < len(l_cls) else candidates[ti].class_id,
                        l_names[di] if di < len(l_names) else candidates[ti].class_name,
                        frame_index, timestamp, desc,
                    )
                rescued = {id(candidates[ti]) for ti, _ in matches}
                unmatched_active = [t for t in unmatched_active if id(t) not in rescued]

        # 4. Stage 2 - re-identify lost tracks (this is the occlusion recovery).
        if lost and unmatched_dets:
            cost = np.full((len(lost), len(unmatched_dets)), 1e6, dtype=np.float64)
            for ti, t in enumerate(lost):
                tb = t.box
                tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
                t_area = max(1e-6, (tb[2] - tb[0]) * (tb[3] - tb[1]))
                for di, d in enumerate(unmatched_dets):
                    if self.cfg.class_consistent and class_ids[d] != t.class_id:
                        continue
                    db = boxes[d]
                    dcx, dcy = (db[0] + db[2]) / 2, (db[1] + db[3]) / 2
                    dist = math.hypot(dcx - tcx, dcy - tcy) / self.diag
                    if dist > self.cfg.reid_distance_gate:
                        continue
                    d_area = max(1e-6, (db[2] - db[0]) * (db[3] - db[1]))
                    # Scale must be plausible: things don't double in size mid-occlusion.
                    ratio = max(t_area, d_area) / min(t_area, d_area)
                    if ratio > 4.0:
                        continue
                    app = self._appearance_cost(t, descriptors[d])
                    if self.cfg.use_appearance and app > self.cfg.reid_appearance_gate:
                        continue
                    # Weighted blend: appearance dominates, motion/scale gate it.
                    cost[ti, di] = (
                        0.55 * app
                        + 0.30 * (dist / max(1e-6, self.cfg.reid_distance_gate))
                        + 0.15 * min(1.0, (ratio - 1.0) / 3.0)
                    )
            matches, _, un_d = assign(cost, 0.75)
            for ti, di in matches:
                d = unmatched_dets[di]
                self._observe(
                    lost[ti], boxes[d], float(confidences[d]), int(class_ids[d]),
                    class_names[d], frame_index, timestamp, descriptors[d],
                )
            unmatched_dets = [unmatched_dets[i] for i in un_d]

        # 5. Birth new tracks from the remaining high-confidence detections only.
        for d in unmatched_dets:
            self._new_track(
                boxes[d], int(class_ids[d]), class_names[d], float(confidences[d]),
                frame_index, timestamp, descriptors[d],
            )

        # 6. Age out: confirmed -> lost -> removed. Tentatives die immediately.
        # A removed track that earned a public ID is kept in `retired`: the
        # object existed, its detections are already in the database, and
        # dropping it here would leave those rows without a track - a broken
        # inspector, a missing timeline lane and a class distribution that omits
        # anything that left before the video ended.
        survivors: list[Track] = []
        for t in self.tracks:
            if t.time_since_update == 0:
                survivors.append(t)
                continue
            if t.state is TrackState.TENTATIVE:
                continue  # unconfirmed flicker, discard silently
            if t.time_since_update <= self.cfg.max_age:
                t.state = TrackState.LOST
                survivors.append(t)
                continue
            if t.public_id > 0:
                t.retire()
                self.retired.append(t)
        self.tracks = survivors

        # 7. Report only tracks matched on this frame and holding a public ID.
        out: list[dict[str, Any]] = []
        for t in self.tracks:
            if t.time_since_update != 0 or t.state is not TrackState.CONFIRMED:
                continue
            box = t.box
            out.append({
                "trackId": t.public_id,
                "classId": t.class_id,
                "className": t.class_name,
                "confidence": t.confidence,
                "box": (
                    float(max(0.0, box[0])), float(max(0.0, box[1])),
                    float(min(self.frame_width, box[2])), float(min(self.frame_height, box[3])),
                ),
                "velocity": t.kf.velocity,
                "age": t.age,
                "hits": t.hits,
            })
        return out

    # -- reporting -------------------------------------------------------------

    def active_count(self) -> int:
        return sum(1 for t in self.tracks if t.state is TrackState.CONFIRMED)

    def finalize(self) -> list[Track]:
        """Every track that ever earned a public ID, in ID order.

        Includes objects that left the scene mid-video, so the stored track set
        always accounts for every detection row that was written.
        """
        tracks = [t for t in self.retired + self.tracks if t.public_id > 0]
        tracks.sort(key=lambda t: t.public_id)
        return tracks
