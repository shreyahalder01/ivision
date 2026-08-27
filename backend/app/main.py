"""VisionTrack AI - Main FastAPI Application Server."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import mimetypes
import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import (
    CLASS_GROUPS,
    COCO_CLASSES,
    EXPORT_DIR,
    FEATURED_CLASSES,
    MODEL_DIR,
    RESULT_DIR,
    THUMB_DIR,
    TMP_DIR,
    UPLOAD_DIR,
    VERSION,
    settings,
)
from .errors import error_payload
from .services import analytics, detector, exporters, ffmpeg, jobs, pipeline, store
from .services.detector import MODEL_ORDER, MODEL_SPECS, probe_environment
from .services.exporters import ALL_FORMATS, DATA_FORMATS, VIDEO_FORMATS, start_export
from .services.ffmpeg import capabilities, extract_poster, extract_scrub_sprite, probe_video
from .services.jobs import JobStatus, manager, new_job_id
from .services.pipeline import load_results_file, run_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("visiontrack.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle."""
    log.info("Starting VisionTrack AI backend server (v%s)...", VERSION)
    store.init_db()
    manager.bus.bind_loop(asyncio.get_running_loop())
    yield
    log.info("Shutting down VisionTrack AI backend...")
    manager.shutdown(wait=False)


app = FastAPI(
    title="VisionTrack AI",
    version=VERSION,
    description="Real-Time Computer Vision Detection & Tracking Engine",
    lifespan=lifespan,
)

# CORS middleware
origins = list(settings.cors_origins) or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in origins else origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models & Request Payloads
# ---------------------------------------------------------------------------

class CreateJobRequest(BaseModel):
    video_path: str = Field(..., description="Absolute path or relative path to video file")
    original_name: str | None = None
    classes: list[str] = Field(default_factory=lambda: ["person", "car", "truck", "bus"])
    confidence: float = Field(default=0.30, ge=0.01, le=1.0)
    iou: float = Field(default=0.45, ge=0.01, le=1.0)
    model: str = Field(default="auto")
    tracking_method: str = Field(default="auto")
    annotation_style: str = Field(default="box_label")
    frame_stride: int = Field(default=1, ge=1, le=30)


class CreateExportRequest(BaseModel):
    format: str = Field(..., description="Export format: mp4, json, csv, coco, yolo")
    options: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# System & Discovery Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/system/capabilities")
async def get_capabilities() -> dict[str, Any]:
    """Inspect environment, GPU/CUDA hardware, FFmpeg binaries and presets."""
    env = probe_environment()
    caps = capabilities()
    return {
        "version": VERSION,
        "ready": caps.get("ready", False) and env.get("aiAvailable", False),
        "ai": env,
        "ffmpeg": caps,
        "models": [
            {
                "key": spec.key,
                "label": spec.label,
                "speed": spec.speed,
                "accuracy": spec.accuracy,
                "paramsM": spec.params_m,
                "sizeMb": spec.size_mb,
                "minVramGb": spec.min_vram_gb,
            }
            for spec in (MODEL_SPECS[k] for k in MODEL_ORDER)
        ],
        "cocoClasses": COCO_CLASSES,
        "featuredClasses": list(FEATURED_CLASSES),
        "classGroups": CLASS_GROUPS,
        "trackingMethods": [
            {
                "key": "auto",
                "label": "Auto (Optimized)",
                "description": "Selects best tracking strategy based on hardware and detection density.",
            },
            {
                "key": "persistent",
                "label": "Persistent Re-ID",
                "description": "Combines Kalman motion filter with HSV colour appearance descriptors for occlusion recovery.",
            },
            {
                "key": "bytetrack",
                "label": "ByteTrack",
                "description": "Two-stage association keeping low-confidence detections to prevent ID churn in occlusions.",
            },
            {
                "key": "deepsort",
                "label": "DeepSORT",
                "description": "IoU distance matching with cascade confirmation for stable identity trajectories.",
            },
        ],
        "annotationStyles": [
            {"key": "box_label", "label": "Standard Box & Label"},
            {"key": "corner_brackets", "label": "Corner Brackets"},
            {"key": "cyber_hud", "label": "Cyber HUD"},
            {"key": "minimal_dot", "label": "Minimal Center Dot"},
            {"key": "mask", "label": "Alpha Shaded Box"},
        ],
    }


@app.get("/api/videos/samples")
async def list_sample_videos() -> list[dict[str, Any]]:
    """Discover built-in sample videos available for 1-click test runs."""
    samples = []
    # Check data/tmp or sample directories
    candidate_dirs = [TMP_DIR, UPLOAD_DIR, Path(__file__).resolve().parent.parent / "data" / "tmp"]
    seen_names = set()

    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            for sample_file in directory.glob(f"*{ext}"):
                if sample_file.name in seen_names:
                    continue
                seen_names.add(sample_file.name)
                try:
                    meta = probe_video(sample_file)
                    samples.append({
                        "name": sample_file.name,
                        "filename": sample_file.name,
                        "videoPath": str(sample_file.resolve()),
                        "sizeBytes": sample_file.stat().st_size,
                        "metadata": meta.to_dict(),
                    })
                except Exception:
                    continue
    return samples


# ---------------------------------------------------------------------------
# Video Ingestion & Media Streaming
# ---------------------------------------------------------------------------

@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload video footage, run ffprobe validation, and generate thumbnails."""
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail=error_payload(
                "invalid_filename", "INVALID FILENAME",
                "No filename was supplied with the upload.",
                "The browser sent an empty filename.",
                "Select a valid video file to upload.",
            ),
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=error_payload(
                "unsupported_format", "UNSUPPORTED VIDEO FORMAT",
                f"The extension '{ext}' is not supported.",
                f"Supported formats: {', '.join(sorted(settings.allowed_extensions))}",
                "Upload an MP4, MOV, MKV, or WebM video.",
            ),
        )

    temp_id = new_job_id()
    dest_path = UPLOAD_DIR / f"{temp_id}_{file.filename}"

    total_bytes = 0
    try:
        with dest_path.open("wb") as buffer:
            while chunk := await file.read(settings.upload_chunk_bytes):
                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=error_payload(
                            "file_too_large", "FILE TOO LARGE",
                            f"Uploaded file exceeds {settings.max_upload_bytes // (1024*1024)} MB limit.",
                            "The file is larger than the configured maximum upload size.",
                            "Upload a smaller file or split the video into shorter segments.",
                        ),
                    )
                buffer.write(chunk)
    except Exception as exc:
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail=error_payload(
                "upload_failed", "UPLOAD FAILED",
                "An error occurred while saving the uploaded file.",
                str(exc),
                "Try uploading again.",
            ),
        )

    # Probe video
    try:
        meta = probe_video(dest_path)
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=error_payload(
                "probe_failed", "UNREADABLE VIDEO FILE",
                "The uploaded file could not be decoded as a valid video.",
                str(exc),
                "Ensure the file is not corrupted and uses standard H.264, H.265 or VP9 codecs.",
            ),
        )

    # Generate poster and sprite sheet
    poster_path_str: str | None = None
    sprite_path_str: str | None = None
    sprite_meta: dict[str, Any] | None = None

    try:
        poster_dest = THUMB_DIR / f"{temp_id}_poster.jpg"
        if extract_poster(dest_path, poster_dest):
            poster_path_str = str(poster_dest.resolve())
    except Exception as exc:
        log.warning("Poster extraction failed: %s", exc)

    try:
        sprite_dest = THUMB_DIR / f"{temp_id}_sprites.jpg"
        sprite_meta = extract_scrub_sprite(dest_path, sprite_dest, duration=meta.duration)
        if sprite_meta:
            sprite_path_str = str(sprite_dest.resolve())
    except Exception as exc:
        log.warning("Sprite sheet extraction failed: %s", exc)

    return {
        "tempId": temp_id,
        "filename": file.filename,
        "videoPath": str(dest_path.resolve()),
        "sizeBytes": total_bytes,
        "metadata": meta.to_dict(),
        "posterPath": poster_path_str,
        "spritePath": sprite_path_str,
        "spriteMeta": sprite_meta,
    }


@app.get("/api/media/{file_path:path}")
async def stream_media(file_path: str, request: Request) -> Response:
    """Stream media files (video/images) with full HTTP 206 Range seeking support."""
    # Resolve file path safely
    path = Path(file_path)
    if not path.is_absolute():
        # Try finding in backend data dirs
        for root in (UPLOAD_DIR, RESULT_DIR, EXPORT_DIR, THUMB_DIR, TMP_DIR):
            cand = root / file_path
            if cand.exists():
                path = cand
                break

    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    file_size = path.stat().st_size
    content_type, _ = mimetypes.guess_type(str(path))
    content_type = content_type or "application/octet-stream"

    # Non-video files can be served directly
    if not content_type.startswith("video/"):
        return FileResponse(path=str(path), media_type=content_type)

    range_header = request.headers.get("Range")
    if not range_header:
        return FileResponse(
            path=str(path),
            media_type=content_type,
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        )

    # Parse Range: bytes=start-end
    try:
        range_str = range_header.replace("bytes=", "").strip()
        parts = range_str.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
        content_length = end - start + 1
    except Exception:
        raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable")

    def _file_chunk_generator(file_p: Path, offset: int, length: int):
        with open(file_p, "rb") as f:
            f.seek(offset)
            remaining = length
            chunk_size = 64 * 1024
            while remaining > 0:
                read_len = min(remaining, chunk_size)
                data = f.read(read_len)
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": content_type,
    }
    return StreamingResponse(
        _file_chunk_generator(path, start, content_length),
        status_code=206,
        headers=headers,
        media_type=content_type,
    )


# ---------------------------------------------------------------------------
# Jobs Management Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/jobs")
async def create_job(req: CreateJobRequest) -> dict[str, Any]:
    """Create and trigger a background video analysis job."""
    video_p = Path(req.video_path)
    if not video_p.exists():
        raise HTTPException(
            status_code=400,
            detail=error_payload(
                "video_not_found", "VIDEO NOT FOUND",
                f"Source file not found: {req.video_path}",
                "The requested video path does not exist on disk.",
                "Ensure the video is uploaded before starting analysis.",
            ),
        )

    # Validate video metadata
    try:
        meta = probe_video(video_p)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=error_payload(
                "probe_failed", "INVALID VIDEO FILE",
                str(exc),
                "FFprobe failed to inspect video attributes.",
                "Provide a valid video file.",
            ),
        )

    job_id = new_job_id()
    original_name = req.original_name or video_p.name

    # Extract poster if not existing
    poster_dest = THUMB_DIR / f"{job_id}_poster.jpg"
    poster_path_str: str | None = None
    if not poster_dest.exists():
        try:
            if extract_poster(video_p, poster_dest):
                poster_path_str = str(poster_dest.resolve())
        except Exception:
            pass

    store.create_job({
        "id": job_id,
        "filename": video_p.name,
        "original_name": original_name,
        "status": JobStatus.QUEUED,
        "video_path": str(video_p.resolve()),
        "poster_path": poster_path_str,
        "metadata": meta.to_dict(),
        "classes": req.classes,
        "confidence": req.confidence,
        "iou": req.iou,
        "model": req.model,
        "tracking_method": req.tracking_method,
        "annotation_style": req.annotation_style,
        "frame_stride": req.frame_stride,
        "total_frames": meta.frame_count,
        "stage": "Queued for analysis",
    })

    # Trigger async analysis on worker thread
    manager.submit(job_id, lambda handle: run_analysis(job_id, handle))

    job = store.get_job(job_id)
    return job or {"id": job_id, "status": JobStatus.QUEUED}


@app.get("/api/jobs")
async def list_jobs_endpoint(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    status: str | None = None,
) -> dict[str, Any]:
    """List analysis jobs with pagination and status filters."""
    jobs_list = store.list_jobs(limit=limit, offset=offset, status=status)
    total = store.count_jobs(status=status)
    return {
        "items": jobs_list,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/jobs/{job_id}")
async def get_job_endpoint(job_id: str) -> dict[str, Any]:
    """Retrieve full job status, metadata, progress, and results summary."""
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Add active runtime handle progress if running
    handle = manager.handle(job_id)
    if handle and handle.last_progress:
        job["liveProgress"] = handle.last_progress

    return job


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job_endpoint(job_id: str) -> dict[str, Any]:
    """Cancel a running analysis job."""
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    cancelled = manager.cancel(job_id)
    return {"jobId": job_id, "cancelled": cancelled}


@app.post("/api/jobs/{job_id}/resume")
async def resume_job_endpoint(job_id: str) -> dict[str, Any]:
    """Resume an interrupted or cancelled analysis job."""
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if manager.is_running(job_id):
        raise HTTPException(status_code=409, detail="Job is already running")

    manager.submit(job_id, lambda handle: run_analysis(job_id, handle, resume=True))
    return {"jobId": job_id, "status": JobStatus.EXTRACTING, "resumed": True}


@app.delete("/api/jobs/{job_id}")
async def delete_job_endpoint(job_id: str) -> dict[str, Any]:
    """Delete a job and remove its results and exports from disk."""
    if manager.is_running(job_id):
        manager.cancel(job_id)

    job = store.delete_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Clean up results file
    res_path = RESULT_DIR / f"{job_id}.json"
    res_path.unlink(missing_ok=True)

    return {"jobId": job_id, "deleted": True}


@app.get("/api/jobs/{job_id}/results")
async def get_job_results_endpoint(job_id: str) -> dict[str, Any]:
    """Return compact frame overlay data for client-side bounding box playback."""
    results = load_results_file(job_id)
    if results is None:
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(status_code=404, detail="Results not ready or unavailable")
    return results


@app.get("/api/jobs/{job_id}/tracks")
async def get_job_tracks_endpoint(job_id: str) -> list[dict[str, Any]]:
    """Retrieve full tracks with spatial motion trajectories and velocity stats."""
    return store.get_tracks(job_id)


@app.get("/api/jobs/{job_id}/tracks/{track_id}")
async def get_track_detail_endpoint(job_id: str, track_id: int) -> dict[str, Any]:
    """Retrieve a single track by ID."""
    track = store.get_track(job_id, track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    return track


# ---------------------------------------------------------------------------
# Exports Management Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/jobs/{job_id}/exports")
async def create_export_endpoint(job_id: str, req: CreateExportRequest) -> dict[str, Any]:
    """Initiate an export job (MP4, CSV, JSON, COCO, YOLO)."""
    try:
        export_record = start_export(job_id, req.format, req.options)
        return export_record
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=getattr(exc, "args", [str(exc)])[0] if getattr(exc, "args", None) else str(exc),
        )


@app.get("/api/jobs/{job_id}/exports")
async def list_job_exports_endpoint(job_id: str) -> list[dict[str, Any]]:
    """List all exports created for a specific job."""
    return store.list_exports(job_id)


@app.get("/api/exports/{export_id}")
async def get_export_endpoint(export_id: str) -> dict[str, Any]:
    """Get status and details of an export."""
    exp = store.get_export(export_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Export not found")
    return exp


@app.get("/api/exports/{export_id}/download")
async def download_export_endpoint(export_id: str) -> Response:
    """Download the completed export artifact file."""
    exp = store.get_export(export_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Export not found")

    if exp["status"] != "complete" or not exp.get("path"):
        raise HTTPException(status_code=400, detail="Export is not complete or file missing")

    path = Path(exp["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export file missing from disk")

    filename = path.name
    media_type, _ = mimetypes.guess_type(filename)
    media_type = media_type or "application/octet-stream"

    return FileResponse(
        path=str(path),
        filename=filename,
        media_type=media_type,
    )


# ---------------------------------------------------------------------------
# WebSockets Real-time Event Streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_events(websocket: WebSocket, job_id: str) -> None:
    """Stream live frame progress, stage updates, FPS throughput, and exports."""
    await websocket.accept()
    queue = manager.bus.subscribe(job_id)
    try:
        while True:
            # Check for events from the bus
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("WebSocket exception for %s: %s", job_id, exc)
    finally:
        manager.bus.unsubscribe(job_id, queue)


# ---------------------------------------------------------------------------
# Static frontend serving (Production build)
# ---------------------------------------------------------------------------

dist_dir = Path(__file__).resolve().parent.parent.parent / "dist"
if not dist_dir.exists():
    dist_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

if dist_dir.exists() and (dist_dir / "index.html").exists():
    from fastapi.staticfiles import StaticFiles

    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str) -> Response:
        """Serve SPA index.html or static build assets for non-API routes."""
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            raise HTTPException(status_code=404, detail="Endpoint not found")
        file_path = dist_dir / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        index_file = dist_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        raise HTTPException(status_code=404, detail="Frontend build index.html not found")

