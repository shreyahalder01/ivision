"""End-to-end pipeline test against a real video with known content.

`data/tmp/sample_traffic.mp4` is built from ultralytics' `bus.jpg` with a
zoom/pan applied, so the ground truth is known exactly: 4 people and 1 bus,
present for the whole clip. A correct run must therefore report 5 unique
objects - one stable track each - not 5-per-frame or a churn of new IDs.

This is a real inference run: it needs torch + ultralytics + FFmpeg. If the AI
runtime is unavailable the test skips rather than pretending to pass, matching
the product's own Demo Mode rule about never presenting simulated results.

Run with:  python tests/test_pipeline_e2e.py
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.services import store  # noqa: E402
from app.services.detector import probe_environment  # noqa: E402
from app.services.ffmpeg import capabilities, probe_video  # noqa: E402
from app.services.jobs import JobHandle, JobStatus, new_job_id  # noqa: E402
from app.services.pipeline import load_results_file, run_analysis  # noqa: E402

SAMPLE = BACKEND / "data" / "tmp" / "sample_traffic.mp4"
EXPECTED = {"person": 4, "bus": 1}


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

    print(f"device   : {env['device']} ({env.get('deviceName') or 'cpu'})")
    print(f"ffmpeg   : {caps['ffmpeg']['version']} via {caps['ffmpeg']['source']}")
    print(f"encoder  : {caps['preferredEncoder']}")

    store.init_db()
    meta = probe_video(SAMPLE)
    job_id = new_job_id()
    store.create_job({
        "id": job_id,
        "filename": SAMPLE.name,
        "original_name": SAMPLE.name,
        "status": JobStatus.QUEUED,
        "video_path": str(SAMPLE),
        "metadata": meta.to_dict(),
        "classes": ["person", "bus", "car", "truck"],
        "confidence": 0.30,
        "iou": 0.45,
        "model": "auto",
        "tracking_method": "auto",
        "annotation_style": "box_label",
        "frame_stride": 1,
    })

    handle = JobHandle(job_id=job_id)
    t0 = time.perf_counter()
    run_analysis(job_id, handle)
    wall = time.perf_counter() - t0

    job = store.get_job(job_id)
    assert job is not None
    status = job["status"]
    if status != JobStatus.COMPLETE:
        print(f"FAIL  status={status}")
        print(f"      error={job.get('error')}")
        return 1

    tracks = store.get_tracks(job_id)
    ndet = store.detection_count(job_id)
    counts = Counter(t["class_name"] for t in tracks)
    fps = meta.frame_count / wall if wall > 0 else 0.0

    print(f"\nframes   : {job['processed_frames']}/{job['total_frames']}"
          f"  ({meta.width}x{meta.height} @ {meta.fps:g}fps)")
    print(f"wall     : {wall:.2f}s  ->  {fps:.1f} FPS end-to-end")
    print(f"reported : processing_fps={job.get('processing_fps')}")
    print(f"detections: {ndet}")
    print(f"tracks   : {len(tracks)}  {dict(counts)}")

    ok = True

    if dict(counts) != EXPECTED:
        print(f"FAIL  expected {EXPECTED}, got {dict(counts)}")
        ok = False

    # Every object is on screen the whole clip, so each track should span
    # essentially the full duration - that is the real identity-stability check.
    for t in tracks:
        span = float(t["duration"]) / meta.duration
        if span < 0.80:
            print(f"FAIL  track {t['track_id']} ({t['class_name']}) covers only "
                  f"{span:.0%} of the clip - identity likely broke mid-way")
            ok = False

    # The results file the UI streams overlays from must agree with the database.
    results = load_results_file(job_id)
    if results is None:
        print("FAIL  results file missing")
        ok = False
    else:
        file_dets = sum(len(v) for v in results["frames"].values())
        if file_dets != ndet:
            print(f"FAIL  results file has {file_dets} detections, database has {ndet}")
            ok = False
        else:
            print(f"results  : {results['schema']}, {len(results['frames'])} frames, "
                  f"{file_dets} detections - matches database")

    # store returns the stored summary under "results" (summary_json -> results).
    summary = job.get("results") or {}
    if summary.get("uniqueObjects") != len(tracks):
        print(f"FAIL  results.uniqueObjects={summary.get('uniqueObjects')} "
              f"but {len(tracks)} tracks stored")
        ok = False
    else:
        classes = summary.get("classDistribution") or []
        shares = ", ".join(f"{c['className']} {c['share']:.0%}" for c in classes)
        print(f"summary  : {summary['uniqueObjects']} unique objects, "
              f"{summary.get('totalDetections')} detections, "
              f"avg conf {summary.get('averageConfidence'):.3f}, {shares}")

    store.delete_job(job_id)
    print("\nPASS  end-to-end pipeline" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
