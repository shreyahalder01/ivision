import sys
from pathlib import Path
import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.services import store
from app.services.detector import probe_environment
from app.services.ffmpeg import capabilities, probe_video
from app.services.jobs import JobHandle, JobStatus, new_job_id
from app.services.pipeline import run_analysis

SAMPLE = BACKEND / "data" / "tmp" / "sample_traffic.mp4"


@pytest.fixture(scope="session")
def analysis_session():
    """Runs real analysis on sample_traffic.mp4 once for the test session."""
    if not SAMPLE.exists():
        pytest.skip(f"Sample video missing: {SAMPLE}")
    caps = capabilities()
    if not caps["ready"]:
        pytest.skip("FFmpeg/FFprobe unavailable")
    env = probe_environment()
    if not env["aiAvailable"]:
        pytest.skip(f"AI runtime unavailable: {env.get('reason')}")

    store.init_db()
    meta = probe_video(SAMPLE)
    job_id = new_job_id()
    store.create_job({
        "id": job_id,
        "filename": SAMPLE.name,
        "original_name": "sample_traffic.mp4",
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

    run_analysis(job_id, JobHandle(job_id=job_id))
    job = store.get_job(job_id)
    if not job or job["status"] != JobStatus.COMPLETE:
        store.delete_job(job_id)
        pytest.skip(f"Analysis did not complete: {job.get('status') if job else 'not found'}")

    tracks = store.get_tracks(job_id)
    ndet = store.detection_count(job_id)

    data = {
        "job_id": job_id,
        "ndet": ndet,
        "ntracks": len(tracks),
        "meta": meta,
    }

    yield data

    # Cleanup session exports and job
    for row in store.list_exports(job_id):
        if row.get("path"):
            Path(row["path"]).unlink(missing_ok=True)
    store.delete_job(job_id)


@pytest.fixture
def job_id(analysis_session):
    return analysis_session["job_id"]


@pytest.fixture
def ndet(analysis_session):
    return analysis_session["ndet"]


@pytest.fixture
def ntracks(analysis_session):
    return analysis_session["ntracks"]


@pytest.fixture
def meta(analysis_session):
    return analysis_session["meta"]
