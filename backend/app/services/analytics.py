"""Analytics derived from stored detections and tracks.

Everything here is computed from what the detector actually produced. When a
figure cannot be derived (no detections, zero-length track) it is reported as
zero or null rather than back-filled with a plausible-looking number.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from ..config import CLASS_GROUPS


def _group_for(class_name: str) -> str:
    for group, members in CLASS_GROUPS.items():
        if class_name in members:
            return group
    return "other"


def build_summary(
    tracks: Sequence[dict[str, Any]],
    *,
    total_detections: int,
    metadata: dict[str, Any],
    selected_classes: Sequence[str],
    frame_stride: int = 1,
) -> dict[str, Any]:
    """Headline metrics for the job card and the analytics panel."""
    duration = float(metadata.get("duration") or 0.0)
    fps = float(metadata.get("fps") or 30.0)

    class_counter: Counter[str] = Counter()
    group_counter: Counter[str] = Counter()
    conf_weighted = 0.0
    conf_weight = 0
    longest: list[dict[str, Any]] = []

    for t in tracks:
        name = t["class_name"]
        class_counter[name] += 1
        group_counter[_group_for(name)] += 1
        conf_weighted += float(t["avg_confidence"]) * int(t["detection_count"])
        conf_weight += int(t["detection_count"])
        longest.append({
            "trackId": t["track_id"],
            "className": name,
            "duration": round(float(t["duration"]), 3),
            "detectionCount": int(t["detection_count"]),
            "avgConfidence": round(float(t["avg_confidence"]), 4),
            "firstSeen": round(float(t["first_seen"]), 3),
            "lastSeen": round(float(t["last_seen"]), 3),
        })

    longest.sort(key=lambda x: x["duration"], reverse=True)
    unique = len(tracks)
    avg_conf = (conf_weighted / conf_weight) if conf_weight else 0.0

    total_classes = sum(class_counter.values()) or 1
    distribution = [
        {
            "className": name,
            "count": count,
            "share": round(count / total_classes, 4),
        }
        for name, count in class_counter.most_common()
    ]

    # Detections per second of footage, honest about the sampling rate.
    effective_fps = fps / max(1, frame_stride)

    return {
        "totalDetections": total_detections,
        "uniqueObjects": unique,
        "classCounts": dict(class_counter),
        "groupCounts": dict(group_counter),
        "people": group_counter.get("people", 0),
        "vehicles": group_counter.get("vehicles", 0),
        "animals": group_counter.get("animals", 0),
        "averageConfidence": round(avg_conf, 4),
        "classDistribution": distribution,
        "longestTracks": longest[:12],
        "classesRequested": list(selected_classes),
        "classesDetected": sorted(class_counter.keys()),
        "detectionsPerSecond": round(total_detections / duration, 2) if duration > 0 else 0.0,
        "effectiveAnalysisFps": round(effective_fps, 2),
        "frameStride": frame_stride,
        "avgTrackDuration": (
            round(sum(t["duration"] for t in longest) / len(longest), 3) if longest else 0.0
        ),
    }


def build_timeseries(
    tracks: Sequence[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    buckets: int = 120,
) -> dict[str, Any]:
    """Detection frequency and per-class activity over time.

    Bucketed so the charts stay light regardless of video length.
    """
    duration = float(metadata.get("duration") or 0.0)
    if duration <= 0 or not tracks:
        return {"buckets": [], "bucketSeconds": 0.0, "classSeries": {}, "peak": None}

    buckets = max(10, min(buckets, 600))
    step = duration / buckets

    counts = [0] * buckets
    unique_per_bucket: list[set[int]] = [set() for _ in range(buckets)]
    class_series: dict[str, list[int]] = defaultdict(lambda: [0] * buckets)

    for t in tracks:
        start = max(0, min(buckets - 1, int(float(t["first_seen"]) / step)))
        end = max(0, min(buckets - 1, int(float(t["last_seen"]) / step)))
        per_bucket = max(1, int(t["detection_count"]) // max(1, end - start + 1))
        for b in range(start, end + 1):
            counts[b] += per_bucket
            unique_per_bucket[b].add(int(t["track_id"]))
            class_series[t["class_name"]][b] += 1

    peak_idx = max(range(buckets), key=lambda i: counts[i]) if buckets else 0
    return {
        "bucketSeconds": round(step, 4),
        "buckets": [
            {
                "t": round(i * step, 3),
                "detections": counts[i],
                "concurrent": len(unique_per_bucket[i]),
            }
            for i in range(buckets)
        ],
        "classSeries": {k: v for k, v in class_series.items()},
        "peak": {
            "t": round(peak_idx * step, 3),
            "detections": counts[peak_idx],
            "concurrent": len(unique_per_bucket[peak_idx]),
        } if counts else None,
    }


def build_track_lanes(tracks: Sequence[dict[str, Any]], *, limit: int = 60) -> list[dict[str, Any]]:
    """Timeline lane rows: one bar per tracked object, longest first."""
    ordered = sorted(tracks, key=lambda t: (-float(t["duration"]), int(t["track_id"])))
    lanes = []
    for t in ordered[:limit]:
        lanes.append({
            "trackId": int(t["track_id"]),
            "className": t["class_name"],
            "classId": int(t["class_id"]),
            "firstSeen": round(float(t["first_seen"]), 3),
            "lastSeen": round(float(t["last_seen"]), 3),
            "duration": round(float(t["duration"]), 3),
            "detectionCount": int(t["detection_count"]),
            "avgConfidence": round(float(t["avg_confidence"]), 4),
            "gapCount": int(t.get("gap_count") or 0),
        })
    return lanes


def track_detail(track: dict[str, Any], *, metadata: dict[str, Any]) -> dict[str, Any]:
    """Object Inspector payload for a single track."""
    fps = float(metadata.get("fps") or 30.0)
    duration = float(track["duration"])
    detections = int(track["detection_count"])
    expected = max(1, int(track["last_frame"]) - int(track["first_frame"]) + 1)

    return {
        "trackId": int(track["track_id"]),
        "className": track["class_name"],
        "classId": int(track["class_id"]),
        "firstSeen": round(float(track["first_seen"]), 3),
        "lastSeen": round(float(track["last_seen"]), 3),
        "firstFrame": int(track["first_frame"]),
        "lastFrame": int(track["last_frame"]),
        "screenTime": round(duration, 3),
        "detectionCount": detections,
        "avgConfidence": round(float(track["avg_confidence"]), 4),
        "maxConfidence": round(float(track["max_confidence"]), 4),
        "maxSpeed": round(float(track["max_speed"]), 2),
        "avgSpeed": round(float(track["avg_speed"]), 2),
        "distance": round(float(track["distance"]), 1),
        "gapCount": int(track.get("gap_count") or 0),
        # How continuously the object was actually detected while on screen.
        "continuity": round(min(1.0, detections / expected), 4),
        "path": track.get("path") or [],
        "fps": fps,
    }


def summarize_tracks_for_storage(
    tracks: Iterable[Any],
    *,
    fps: float,
    frame_stride: int,
    width: int,
    height: int,
    job_id: str,
    path_samples: int = 160,
) -> list[tuple]:
    """Convert live tracker objects into `tracks` table rows.

    Motion paths are down-sampled to `path_samples` points and stored
    normalised, which keeps the payload small enough to ship to the browser for
    trail rendering.
    """
    rows: list[tuple] = []
    import json as _json

    for t in tracks:
        if t.public_id <= 0 or not t.centroids:
            continue
        first_frame = t.first_frame
        last_frame = t.last_frame
        duration = max(0.0, (last_frame - first_frame) / fps) if fps > 0 else 0.0

        pts = t.centroids
        if len(pts) > path_samples:
            step = len(pts) / path_samples
            pts = [pts[min(len(pts) - 1, int(i * step))] for i in range(path_samples)]
        path = [
            [f, round(cx / width, 5), round(cy / height, 5)]
            for (f, cx, cy) in pts
        ]

        avg_conf = t.conf_sum / t.conf_count if t.conf_count else 0.0
        avg_speed = t.speed_sum / t.speed_count if t.speed_count else 0.0

        rows.append((
            job_id, t.public_id, t.class_id, t.class_name,
            first_frame, last_frame,
            round(first_frame / fps, 4) if fps > 0 else 0.0,
            round(last_frame / fps, 4) if fps > 0 else 0.0,
            round(duration, 4),
            t.conf_count,
            round(avg_conf, 5), round(t.max_confidence, 5),
            round(t.max_speed, 3), round(avg_speed, 3), round(t.distance, 2),
            t.gap_count,
            _json.dumps(path),
        ))
    return rows


def humanize_seconds(seconds: float) -> str:
    if seconds != seconds or seconds in (math.inf, -math.inf) or seconds < 0:
        return "--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
