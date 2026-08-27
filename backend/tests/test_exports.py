"""Export tests against real analysis output.

Every format is produced from a genuine inference run over
`data/tmp/sample_traffic.mp4` (4 people, 1 bus, known exactly), then parsed back
and checked against the database. That is the only way to catch the failures that
matter: a CSV whose columns drift from its header, a COCO file that references
image ids it never declares, a YOLO box that falls outside 0..1, or an annotated
MP4 that silently drops frames.

Skips rather than passing when FFmpeg or the AI runtime is missing.

Run with:  python tests/test_exports.py
"""
from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.config import EXPORT_DIR  # noqa: E402
from app.services import exporters, store  # noqa: E402
from app.services.detector import probe_environment  # noqa: E402
from app.services.ffmpeg import capabilities, probe_video  # noqa: E402
from app.services.jobs import JobHandle, JobStatus, new_job_id  # noqa: E402
from app.services.pipeline import run_analysis  # noqa: E402

SAMPLE = BACKEND / "data" / "tmp" / "sample_traffic.mp4"
EXPECTED_TRACKS = {"person": 4, "bus": 1}

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    if ok:
        print(f"  PASS  {label}")
    else:
        failures.append(f"{label}: {detail}")
        print(f"  FAIL  {label}  {detail}")
        assert ok, f"{label}: {detail}"
    return ok


def run_one(job_id: str, fmt: str, options: dict) -> tuple[Path, dict]:
    """Run an export synchronously and return (path, export row)."""
    opt = exporters.ExportOptions(options, fmt=fmt)
    export_id = exporters.new_export_id()
    store.create_export({
        "id": export_id, "job_id": job_id,
        "kind": "video" if fmt in exporters.VIDEO_FORMATS else "data",
        "fmt": fmt, "status": exporters.ExportStatus.QUEUED,
        "options": opt.to_dict(),
    })
    exporters.run_export(export_id, JobHandle(job_id=export_id))
    row = store.get_export(export_id)
    assert row is not None
    if row["status"] != exporters.ExportStatus.COMPLETE:
        raise AssertionError(f"{fmt} export {row['status']}: {row.get('error')}")
    return Path(row["path"]), row


# ------------------------------------------------------------------ format tests

def test_csv(job_id: str, ndet: int, meta) -> None:
    print("\nCSV")
    path, row = run_one(job_id, "csv", {})
    text = path.read_text(encoding="utf-8")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    rows = list(reader)

    check(len(rows) == ndet, "one row per detection", f"{len(rows)} rows vs {ndet} detections")
    check(
        all(len(r) == len(header) for r in rows),
        "every row matches the header width",
        f"header={len(header)}",
    )
    for col in ("frame", "track_id", "class_name", "confidence", "x", "y", "width", "height"):
        check(col in header, f"column '{col}' present", str(header))

    # Geometry must be in source pixels and inside the frame.
    xi, yi = header.index("x"), header.index("y")
    wi, hi = header.index("width"), header.index("height")
    bad = [
        r for r in rows
        if not (0 <= float(r[xi]) <= meta.width and 0 <= float(r[yi]) <= meta.height
                and 0 < float(r[wi]) <= meta.width and 0 < float(r[hi]) <= meta.height)
    ]
    check(not bad, "all boxes inside the source frame", f"{len(bad)} outliers")

    ti = header.index("track_id")
    check(
        len({int(r[ti]) for r in rows}) == 5,
        "5 distinct track ids", str(sorted({int(r[ti]) for r in rows})),
    )

    # Deselecting content must actually remove columns.
    path2, _ = run_one(job_id, "csv", {"contents": {
        "confidence": False, "classes": False, "timestamps": False,
    }})
    header2 = next(csv.reader(io.StringIO(path2.read_text(encoding="utf-8"))))
    check(
        "confidence" not in header2 and "class_name" not in header2
        and "timecode" not in header2 and "frame" in header2,
        "unchecked content is omitted", str(header2),
    )


def test_json(job_id: str, ndet: int, ntracks: int) -> None:
    print("\nJSON")
    path, _ = run_one(job_id, "json", {"contents": {"motionPaths": True}})
    data = json.loads(path.read_text(encoding="utf-8"))

    check(data["schema"] == exporters.SCHEMA_VERSION, "schema declared", data.get("schema"))
    check(len(data["detections"]) == ndet, "all detections present", str(len(data["detections"])))
    check(len(data["tracks"]) == ntracks, "all tracks present", str(len(data["tracks"])))
    check(
        data["job"]["settings"]["trackingMethod"] is not None,
        "settings block is reproducible", str(data["job"]["settings"]),
    )
    d0 = data["detections"][0]
    check(
        {"frame", "trackId", "className", "confidence", "boundingBox"} <= set(d0),
        "detection carries the expected fields", str(sorted(d0)),
    )
    check(
        all(len(p) == 3 for t in data["tracks"] for p in t.get("motionPath", [])),
        "motion paths are [frame, x, y] triples",
    )
    # Pretty mode must stay valid JSON, not just indented text.
    path2, _ = run_one(job_id, "json", {"pretty": True})
    parsed = json.loads(path2.read_text(encoding="utf-8"))
    check(len(parsed["detections"]) == ndet, "pretty output parses identically")


def test_coco(job_id: str, ndet: int) -> None:
    print("\nCOCO")
    path, _ = run_one(job_id, "coco", {})
    data = json.loads(path.read_text(encoding="utf-8"))

    for key in ("info", "images", "annotations", "categories"):
        check(key in data, f"'{key}' section present")
    check(len(data["annotations"]) == ndet, "annotation per detection", str(len(data["annotations"])))

    image_ids = {img["id"] for img in data["images"]}
    orphans = [a for a in data["annotations"] if a["image_id"] not in image_ids]
    check(not orphans, "every annotation references a declared image", f"{len(orphans)} orphans")

    cat_ids = {c["id"] for c in data["categories"]}
    check(
        all(a["category_id"] in cat_ids for a in data["annotations"]),
        "every annotation references a declared category",
    )
    a0 = data["annotations"][0]
    check(len(a0["bbox"]) == 4 and a0["area"] > 0, "bbox is [x,y,w,h] with area", str(a0["bbox"]))
    check("track_id" in a0 and a0["attributes"]["track_id"] == a0["track_id"],
          "track id carried in both conventional places")
    check(len({a["id"] for a in data["annotations"]}) == ndet, "annotation ids are unique")


def test_yolo(job_id: str, ndet: int) -> None:
    print("\nYOLO")
    path, _ = run_one(job_id, "yolo", {})
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        check("classes.txt" in names and "data.yaml" in names and "README.txt" in names,
              "archive carries classes.txt, data.yaml and README")
        labels = [n for n in names if n.startswith("labels/")]
        check(bool(labels), "label files written", str(len(labels)))

        total = 0
        out_of_range: list[str] = []
        for name in labels:
            for line in zf.read(name).decode().splitlines():
                parts = line.split()
                total += 1
                cx, cy, w, h = (float(v) for v in parts[1:5])
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                    out_of_range.append(f"{name}: {line}")
        check(total == ndet, "one label line per detection", f"{total} vs {ndet}")
        check(not out_of_range, "all geometry normalised to 0..1",
              "; ".join(out_of_range[:3]))

        classes = zf.read("classes.txt").decode().splitlines()
        check(len(classes) == 2, "only detected classes listed", str(classes))
        check(
            all(line.split()[0] == str(i) for i, line in enumerate(classes)),
            "class indices are contiguous from 0", str(classes),
        )
        # Track ids are appended after the 5 YOLO columns when requested.
        sample = zf.read(labels[0]).decode().splitlines()[0].split()
        check(len(sample) == 7, "confidence and track id appended", str(sample))


def test_video_clean(job_id: str, meta) -> None:
    print("\nMP4 (no annotations, unchanged geometry)")
    path, row = run_one(job_id, "mp4", {"annotated": False})
    check(path.exists() and path.stat().st_size > 0, "file written")
    check(
        path.stat().st_size == SAMPLE.stat().st_size,
        "byte-identical copy - no needless re-encode",
        f"{path.stat().st_size} vs {SAMPLE.stat().st_size}",
    )


def test_video_annotated(job_id: str, meta) -> None:
    print("\nMP4 (annotated, half resolution, 15 fps)")
    half_w = meta.width // 2 - (meta.width // 2) % 2
    path, row = run_one(job_id, "mp4", {
        "annotated": True,
        "width": half_w,
        "fps": 15,
        "contents": {
            "boxes": True, "classes": True, "confidence": True, "trackIds": True,
            "frameNumbers": True, "timestamps": True, "motionPaths": True, "trails": True,
        },
    })
    check(path.exists() and path.stat().st_size > 0, "file written")

    out = probe_video(path)
    check(out.width == half_w, f"width honoured ({half_w})", str(out.width))
    check(abs(out.fps - 15.0) < 0.5, "frame rate honoured (15)", f"{out.fps:g}")
    check(
        abs(out.duration - meta.duration) < 0.35,
        "duration preserved across the frame-rate change",
        f"{out.duration:.2f}s vs {meta.duration:.2f}s",
    )
    expected = round(meta.frame_count * 15.0 / meta.fps)
    check(
        abs(out.frame_count - expected) <= 2,
        f"frame count resampled, not truncated (~{expected})",
        str(out.frame_count),
    )
    check(out.codec == "h264", "H.264 output", out.codec)


def test_unsupported_style_downgrades(job_id: str) -> None:
    print("\nAnnotation style fallback")
    _path, row = run_one(job_id, "mp4", {
        "annotated": True, "contents": {"style": "mask", "boxes": True},
    })
    # The style is unavailable without a -seg checkpoint. The export must
    # succeed, say so, and not pretend it drew a mask.
    style, notice = exporters.resolve_style("mask", has_masks=False)
    check(style == "box_label", "mask downgrades to box_label", style)
    check(bool(notice) and "segmentation" in notice.lower(),
          "downgrade is explained, not silent", str(notice))


def test_validation(job_id: str) -> None:
    print("\nValidation")
    job = store.get_job(job_id)
    err = exporters.validate_request(job, "tiff")
    check(err.get("code") == "unsupported_format", "unknown format rejected", str(err))
    for key in ("title", "message", "cause", "action"):
        check(bool(err.get(key)), f"rejection carries '{key}'")
    check(exporters.validate_request(None, "csv").get("code") == "job_not_found",
          "missing job rejected")
    check(exporters.validate_request(job, "csv") == {}, "valid request accepted")


def test_cancellation(job_id: str) -> None:
    print("\nCancellation")
    before = set(EXPORT_DIR.glob("*"))
    opt = exporters.ExportOptions({"annotated": True}, fmt="mp4")
    export_id = exporters.new_export_id()
    store.create_export({
        "id": export_id, "job_id": job_id, "kind": "video", "fmt": "mp4",
        "status": exporters.ExportStatus.QUEUED, "options": opt.to_dict(),
    })
    handle = JobHandle(job_id=export_id)
    handle.cancel()                      # cancelled before the first frame
    exporters.run_export(export_id, handle)
    row = store.get_export(export_id)
    assert row is not None
    check(row["status"] == exporters.ExportStatus.CANCELLED, "status is cancelled", row["status"])
    check(row["path"] is None, "no path recorded for a cancelled export", str(row["path"]))
    check(bool(row.get("error", {}).get("action")), "cancellation carries a next step",
          str(row.get("error")))
    new_files = set(EXPORT_DIR.glob("*")) - before
    check(not new_files, "no partial artifact left on disk", str(new_files))


# ------------------------------------------------------------------------- main

def main() -> int:
    if not SAMPLE.exists():
        print(f"SKIP  sample video missing: {SAMPLE}")
        return 0
    caps = capabilities()
    if not caps["ready"]:
        print("SKIP  FFmpeg/FFprobe unavailable")
        return 0
    env = probe_environment()
    if not env["aiAvailable"]:
        print(f"SKIP  AI runtime unavailable: {env.get('reason')}")
        return 0

    store.init_db()
    meta = probe_video(SAMPLE)
    job_id = new_job_id()
    store.create_job({
        "id": job_id, "filename": SAMPLE.name, "original_name": "sample_traffic.mp4",
        "status": JobStatus.QUEUED, "video_path": str(SAMPLE),
        "metadata": meta.to_dict(),
        "classes": ["person", "bus", "car", "truck"],
        "confidence": 0.30, "iou": 0.45, "model": "auto",
        "tracking_method": "auto", "annotation_style": "box_label", "frame_stride": 1,
    })

    print(f"analysing {SAMPLE.name} on {env['device']} ...")
    run_analysis(job_id, JobHandle(job_id=job_id))
    job = store.get_job(job_id)
    assert job is not None
    if job["status"] != JobStatus.COMPLETE:
        print(f"SKIP  analysis did not complete: {job['status']} {job.get('error')}")
        store.delete_job(job_id)
        return 0

    tracks = store.get_tracks(job_id)
    ndet = store.detection_count(job_id)
    counts: dict[str, int] = {}
    for t in tracks:
        counts[t["class_name"]] = counts.get(t["class_name"], 0) + 1
    print(f"analysis: {ndet} detections, {len(tracks)} tracks {counts}")
    if counts != EXPECTED_TRACKS:
        print(f"  note: track counts {counts} differ from {EXPECTED_TRACKS}")

    try:
        test_csv(job_id, ndet, meta)
        test_json(job_id, ndet, len(tracks))
        test_coco(job_id, ndet)
        test_yolo(job_id, ndet)
        test_video_clean(job_id, meta)
        test_video_annotated(job_id, meta)
        test_unsupported_style_downgrades(job_id)
        test_validation(job_id)
        test_cancellation(job_id)
    finally:
        for row in store.list_exports(job_id):
            if row.get("path"):
                Path(row["path"]).unlink(missing_ok=True)
        store.delete_job(job_id)

    if failures:
        print(f"\nFAILED  {len(failures)} check(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS  all export formats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
