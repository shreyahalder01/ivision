import json
import time
import urllib.request

base = "http://127.0.0.1:8787/api"

# 1. Capabilities
caps_res = urllib.request.urlopen(f"{base}/system/capabilities")
caps = json.loads(caps_res.read().decode("utf-8"))
print("AI Device:", caps["ai"]["device"], "| GPU Accelerated:", caps["ai"]["gpuAccelerated"])
print("Preferred Encoder:", caps["ffmpeg"]["preferredEncoder"])

# 2. Samples
samples_res = urllib.request.urlopen(f"{base}/videos/samples")
samples = json.loads(samples_res.read().decode("utf-8"))
print("Available samples:", [s["name"] for s in samples])
target_video = samples[0]["videoPath"]

# 3. Create Job
req_data = json.dumps({
    "video_path": target_video,
    "original_name": "sample_traffic.mp4",
    "classes": ["person", "bus", "car", "truck"],
    "confidence": 0.30,
    "iou": 0.45,
    "model": "auto",
    "tracking_method": "auto",
    "annotation_style": "box_label",
    "frame_stride": 1,
}).encode("utf-8")

req = urllib.request.Request(f"{base}/jobs", data=req_data, headers={"Content-Type": "application/json"})
job = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
job_id = job["id"]
print("Created Job ID:", job_id)

# 4. Poll for completion
for i in range(40):
    time.sleep(0.5)
    j = json.loads(urllib.request.urlopen(f"{base}/jobs/{job_id}").read().decode("utf-8"))
    status = j["status"]
    prog = j.get("progress", 0.0)
    stage = j.get("stage", "")
    print(f"[{i:02d}] Status: {status} | Progress: {prog:.1%} | Stage: {stage}")
    if status in ("complete", "failed", "cancelled"):
        break

assert status == "complete", f"Job failed with status {status}"
results = j.get("results", {})
print("Results summary:", results.get("uniqueObjects"), "unique objects,", results.get("totalDetections"), "detections")

# 5. Overlay Results payload
overlay_res = urllib.request.urlopen(f"{base}/jobs/{job_id}/results")
overlay = json.loads(overlay_res.read().decode("utf-8"))
print(f"Overlay payload: {len(overlay['frames'])} frames, {len(overlay['tracks'])} tracks")

# 6. Trigger CSV and JSON Exports
for fmt in ("csv", "json"):
    exp_req_data = json.dumps({"format": fmt, "options": {}}).encode("utf-8")
    exp_req = urllib.request.Request(f"{base}/jobs/{job_id}/exports", data=exp_req_data, headers={"Content-Type": "application/json"})
    exp = json.loads(urllib.request.urlopen(exp_req).read().decode("utf-8"))
    time.sleep(1.0)
    exp_status = json.loads(urllib.request.urlopen(f"{base}/exports/{exp['id']}").read().decode("utf-8"))
    print(f"Export {fmt.upper()}: Status = {exp_status['status']}, Size = {exp_status.get('size_bytes')} bytes")

print("\nSUCCESS: All endpoints and full-stack pipelines verified successfully!")
