"""Identity-stability tests for the tracking layer.

The product requirement these guard is specific: "Track IDs should not randomly
change while an object remains visible." So each test builds synthetic footage
with known ground truth, runs the tracker, and asserts that every ground-truth
object collected exactly one public ID.

Detections are attributed to ground-truth objects by IoU, never by proximity -
an earlier proximity-based version of this test reported false failures on the
crossing-paths scenario because both objects sat in the same neighbourhood.

Run with:  python -m pytest tests/test_tracker.py -v
       or:  python tests/test_tracker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.tracker import (  # noqa: E402
    PersistentTracker,
    TrackerConfig,
    appearance_descriptor,
    iou_matrix,
    prepare_appearance_frame,
)

FPS = 30.0
W, H = 960, 540
MODES = ("persistent", "bytetrack", "deepsort", "auto")


# ----------------------------------------------------------------- scene helpers

class Actor:
    """A ground-truth object with a known path, size and colour."""

    def __init__(self, name, x0, y0, dx, dy, w, h, colour):
        self.name = name
        self.x0, self.y0, self.dx, self.dy = x0, y0, dx, dy
        self.w, self.h = w, h
        self.colour = colour  # BGR, so the appearance descriptor has real signal

    def box(self, frame: int) -> np.ndarray:
        x = self.x0 + self.dx * frame
        y = self.y0 + self.dy * frame
        return np.array([x, y, x + self.w, y + self.h], dtype=np.float64)


def render(actors_boxes) -> np.ndarray:
    """Paint the visible boxes onto a frame so appearance re-ID has something
    to work with. A flat grey background keeps the descriptor discriminative."""
    img = np.full((H, W, 3), 60, dtype=np.uint8)
    for box, colour in actors_boxes:
        x1, y1, x2, y2 = (int(max(0, v)) for v in box)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 > x1 and y2 > y1:
            img[y1:y2, x1:x2] = colour
    return img


def simulate(actors, frames, *, mode, visible=None, conf=None, class_id=0,
             class_name="person", low_conf_cut=0.30):
    """Run `frames` steps of tracking and return {actor name: set(public ids)}.

    `visible(actor, f) -> bool` gates whether the detector sees an actor at all.
    `conf(actor, f) -> float` sets detection confidence; anything under
    `low_conf_cut` is routed to the tracker as a BYTE low-confidence detection,
    exactly as the real pipeline does.
    """
    cfg = TrackerConfig.for_method(mode, FPS)
    trk = PersistentTracker(cfg)
    seen: dict[str, set[int]] = {a.name: set() for a in actors}

    for f in range(frames):
        present = [a for a in actors if (visible(a, f) if visible else True)]
        truth = {a.name: a.box(f) for a in present}
        img = render([(a.box(f), a.colour) for a in present])

        hi_b, hi_c, hi_n, hi_conf = [], [], [], []
        lo_b, lo_c, lo_n, lo_conf = [], [], [], []
        for a in present:
            c = conf(a, f) if conf else 0.90
            if c >= low_conf_cut:
                hi_b.append(a.box(f)); hi_c.append(class_id)
                hi_n.append(class_name); hi_conf.append(c)
            else:
                lo_b.append(a.box(f)); lo_c.append(class_id)
                lo_n.append(class_name); lo_conf.append(c)

        reported = trk.update(
            np.array(hi_b) if hi_b else np.zeros((0, 4)),
            hi_c, hi_n, hi_conf, f, f / FPS, image=img,
            low_boxes=np.array(lo_b) if lo_b else None,
            low_class_ids=lo_c, low_class_names=lo_n, low_confidences=lo_conf,
        )

        # Attribute each reported track to a ground-truth actor by best IoU.
        if reported and truth:
            names = list(truth)
            gt = np.array([truth[n] for n in names])
            rb = np.array([r["box"] for r in reported], dtype=np.float64)
            ious = iou_matrix(gt, rb)
            for ri, rep in enumerate(reported):
                gi = int(np.argmax(ious[:, ri]))
                if ious[gi, ri] >= 0.5:
                    seen[names[gi]].add(int(rep["trackId"]))
    return seen


def check(seen, label):
    problems = [
        f"{name}: {sorted(ids) or 'never tracked'}"
        for name, ids in seen.items()
        if len(ids) != 1
    ]
    assert not problems, f"{label} - unstable identities: {'; '.join(problems)}"
    return {n: next(iter(i)) for n, i in seen.items()}


# ------------------------------------------------------------------------ tests

def _two_actors_apart():
    return [
        Actor("left",  100, 200,  4.0, 0.0, 60, 140, (40, 60, 200)),
        Actor("right", 800, 260, -3.0, 0.0, 70, 120, (200, 140, 40)),
    ]


def _two_actors_crossing():
    # Paths intersect around frame 67 - the case that broke naive attribution.
    return [
        Actor("a", 100, 220,  8.0, 0.0, 60, 140, (40, 60, 200)),
        Actor("b", 900, 220, -4.0, 0.0, 60, 140, (200, 140, 40)),
    ]


def test_separate_paths_keep_one_id_each():
    for mode in MODES:
        check(simulate(_two_actors_apart(), 90, mode=mode), f"[{mode}] separate paths")


def test_crossing_paths_keep_one_id_each():
    for mode in MODES:
        check(simulate(_two_actors_crossing(), 100, mode=mode), f"[{mode}] crossing paths")


def test_id_survives_full_occlusion():
    """Vanish an actor for 12 frames. Appearance re-ID must recover the same ID."""
    def visible(actor, f):
        return not (actor.name == "left" and 40 <= f < 52)

    for mode in ("persistent", "deepsort", "auto"):
        check(simulate(_two_actors_apart(), 100, mode=mode, visible=visible),
              f"[{mode}] full occlusion")


def test_byte_rescues_low_confidence_window():
    """Confidence dips below threshold for 20 frames.

    BYTE-enabled modes should keep reporting the object; plain IoU tracking
    without the rescue stage should lose it. This asserts the modes are
    genuinely different algorithms, not relabelled settings.
    """
    def conf(actor, f):
        return 0.14 if 30 <= f < 50 else 0.90

    counts = {}
    for mode in MODES:
        cfg = TrackerConfig.for_method(mode, FPS)
        trk = PersistentTracker(cfg)
        actors = _two_actors_apart()
        hits = 0
        for f in range(80):
            hi_b, hi_c, hi_n, hi_conf = [], [], [], []
            lo_b, lo_c, lo_n, lo_conf = [], [], [], []
            for a in actors:
                c = conf(a, f)
                (hi_b if c >= 0.30 else lo_b).append(a.box(f))
                (hi_c if c >= 0.30 else lo_c).append(0)
                (hi_n if c >= 0.30 else lo_n).append("person")
                (hi_conf if c >= 0.30 else lo_conf).append(c)
            img = render([(a.box(f), a.colour) for a in actors])
            rep = trk.update(
                np.array(hi_b) if hi_b else np.zeros((0, 4)),
                hi_c, hi_n, hi_conf, f, f / FPS, image=img,
                low_boxes=np.array(lo_b) if lo_b else None,
                low_class_ids=lo_c, low_class_names=lo_n, low_confidences=lo_conf,
            )
            if rep:
                hits += 1
        counts[mode] = hits

    assert counts["bytetrack"] > counts["persistent"], (
        f"BYTE rescue gave no advantage: {counts}"
    )
    assert counts["auto"] > counts["persistent"], f"auto should include BYTE: {counts}"


def test_descriptor_is_size_invariant_and_colour_sensitive():
    """The descriptor must survive scale change but distinguish colour."""
    red = np.full((H, W, 3), 60, dtype=np.uint8)
    red[100:300, 100:200] = (40, 60, 200)
    blue = np.full((H, W, 3), 60, dtype=np.uint8)
    blue[100:300, 100:200] = (200, 140, 40)

    small = np.full((H, W, 3), 60, dtype=np.uint8)
    small[100:200, 100:150] = (40, 60, 200)   # same colour, half the size

    hsv_red, hsv_blue, hsv_small = (prepare_appearance_frame(i) for i in (red, blue, small))
    d_red = appearance_descriptor(hsv_red, np.array([100, 100, 200, 300]))
    d_blue = appearance_descriptor(hsv_blue, np.array([100, 100, 200, 300]))
    d_small = appearance_descriptor(hsv_small, np.array([100, 100, 150, 200]))

    assert d_red is not None and d_blue is not None and d_small is not None
    same_scale = float(np.dot(d_red, d_small))
    diff_colour = float(np.dot(d_red, d_blue))
    assert same_scale > 0.97, f"scale changed the descriptor too much: {same_scale:.4f}"
    assert diff_colour < 0.80, f"descriptor cannot tell the colours apart: {diff_colour:.4f}"


def test_low_confidence_cannot_create_a_track():
    """A weak detection may extend a track but must never invent an object."""
    actor = Actor("ghost", 300, 200, 2.0, 0.0, 60, 140, (40, 60, 200))
    trk = PersistentTracker(TrackerConfig.for_method("auto", FPS))
    for f in range(40):
        img = render([(actor.box(f), actor.colour)])
        rep = trk.update(
            np.zeros((0, 4)), [], [], [], f, f / FPS, image=img,
            low_boxes=np.array([actor.box(f)]),
            low_class_ids=[0], low_class_names=["person"], low_confidences=[0.15],
        )
        assert not rep, f"frame {f}: low-confidence detection created a track {rep}"


if __name__ == "__main__":
    test_separate_paths_keep_one_id_each()
    print("PASS  separate paths           one ID per object, all 4 modes")
    test_crossing_paths_keep_one_id_each()
    print("PASS  crossing paths           one ID per object, all 4 modes")
    test_id_survives_full_occlusion()
    print("PASS  12-frame full occlusion  ID recovered by appearance re-ID")
    counts = test_byte_rescues_low_confidence_window()
    print(f"PASS  BYTE low-conf rescue     frames reported {counts}")
    same, diff = test_descriptor_is_size_invariant_and_colour_sensitive()
    print(f"PASS  descriptor               scale-invariance {same:.4f}, colour separation {diff:.4f}")
    test_low_confidence_cannot_create_a_track()
    print("PASS  weak detections          never create a track")
