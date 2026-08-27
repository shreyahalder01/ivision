import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

os.environ["VISIONTRACK_FORCE_CPU"] = "1"

from app.services.detector import probe_environment, Detector

env = probe_environment(refresh=True)
print("Device:", env["device"])
print("Device Name:", env["deviceName"])
print("GPU Accelerated:", env["gpuAccelerated"])
print("AI Available:", env["aiAvailable"])
print("Reason:", env["reason"])

assert env["device"] == "cpu"
assert not env["gpuAccelerated"]
assert env["aiAvailable"]

detector = Detector("yolo11n", device="cpu")
detector.load()
print("YOLO11n CPU model loaded successfully.")
print("Class count:", len(detector.class_names))

print("\nSUCCESS: Server-side CPU fallback operates reliably and gracefully!")
