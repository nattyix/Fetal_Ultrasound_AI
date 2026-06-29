"""
quantize_model.py — Static INT8 quantization for MONAI DenseNet121
Run once locally, then push the output .pth to GitHub.

Usage:
    cd D:\fetal_ultrasound_ai
    python quantize_model.py
"""
import torch
import numpy as np
import os
from monai.transforms import Compose, EnsureChannelFirst, Resize, ScaleIntensity
from monai.networks.nets import DenseNet121

INPUT_PATH  = "models/fetal_ultrasound_model.pth"
OUTPUT_PATH = "models/fetal_ultrasound_model_int8.pth"
IMAGE_SIZE  = (224, 224)

transform = Compose([
    EnsureChannelFirst(channel_dim=-1),
    Resize(IMAGE_SIZE),
    ScaleIntensity(),
])

# ── Load float32 model ────────────────────────────────────────────────────
print("Loading float32 model...")
model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=4)
model.load_state_dict(torch.load(INPUT_PATH, map_location="cpu", weights_only=True))
model.eval()

# ── Calibration data (random dummy — no real images needed) ──────────────
def calibration_loader(n=8):
    for _ in range(n):
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        yield transform(dummy_img).unsqueeze(0)

# ── Static PTQ via torch.quantization ────────────────────────────────────
print("Preparing static quantization...")
model.qconfig = torch.quantization.get_default_qconfig("x86")

# Fuse Conv+BN+ReLU patterns where possible
try:
    torch.quantization.fuse_modules(model, [["features.0", "features.1", "features.2"]], inplace=True)
    print("  Fused initial Conv+BN+ReLU")
except Exception as e:
    print(f"  Fusion skipped: {e}")

torch.quantization.prepare(model, inplace=True)

print("Calibrating with dummy data...")
with torch.no_grad():
    for i, batch in enumerate(calibration_loader(16)):
        model(batch)
        print(f"  Calibration batch {i+1}/16", end="\r")
print()

print("Converting to INT8...")
torch.quantization.convert(model, inplace=True)

# ── Save ──────────────────────────────────────────────────────────────────
print("Saving INT8 model state dict...")
torch.save(model.state_dict(), OUTPUT_PATH)

# ── Verify ────────────────────────────────────────────────────────────────
print("Verifying INT8 model loads and runs...")
model2 = DenseNet121(spatial_dims=2, in_channels=3, out_channels=4)
model2.qconfig = torch.quantization.get_default_qconfig("x86")
torch.quantization.prepare(model2, inplace=True)
torch.quantization.convert(model2, inplace=True)
model2.load_state_dict(torch.load(OUTPUT_PATH, map_location="cpu", weights_only=True))
model2.eval()

dummy = transform(np.random.randint(0,255,(224,224,3),dtype=np.uint8)).unsqueeze(0)
with torch.no_grad():
    out = model2(dummy)
    probs = torch.softmax(out, dim=1)
print(f"Output shape : {out.shape}")
print(f"Probabilities: {probs[0].tolist()}")

orig_mb = os.path.getsize(INPUT_PATH)  / 1024 / 1024
new_mb  = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
print(f"\nOriginal : {orig_mb:.1f} MB")
print(f"INT8     : {new_mb:.1f} MB")
print(f"Reduction: {(1 - new_mb/orig_mb)*100:.0f}%")
print(f"\nDone! Now push models/fetal_ultrasound_model_int8.pth to GitHub.")