import torch
import os
import sys
import numpy as np
from PIL import Image

from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Resize,
    ScaleIntensity
)
from monai.networks.nets import DenseNet121   # ← updated from resnet18


# -------------------------
# Configuration
# -------------------------
MODEL_PATH = "models/fetal_ultrasound_model.pth"
IMAGE_SIZE = (224, 224)

CLASSES = [
    "Trans-thalamic",
    "Trans-ventricular",
    "Trans-cerebellum",
    "Diverse / Other"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Load model
# -------------------------
model = DenseNet121(
    spatial_dims=2,
    in_channels=3,
    out_channels=4,
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


# -------------------------
# Transforms
# -------------------------
transform = Compose([
    EnsureChannelFirst(channel_dim=-1),
    Resize(IMAGE_SIZE),
    ScaleIntensity()
])


# -------------------------
# Prediction function
# -------------------------
def predict(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    return (
        predicted_class.item(),
        confidence.item(),
        probabilities.squeeze().cpu().tolist()
    )


# -------------------------
# Risk interpretation logic
# -------------------------
def risk_indicator(confidence):
    if confidence >= 0.85:
        return "LOW", 1 - confidence
    elif confidence >= 0.6:
        return "MODERATE", 1 - confidence
    else:
        return "HIGH", 1 - confidence


# -------------------------
# CLI entry point
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        pred, conf, probs = predict(image_path)
        risk, score = risk_indicator(conf)

        print("\nUltrasound Analysis Result")
        print("---------------------------")
        print(f"Detected Plane   : {CLASSES[pred]}")
        print(f"Confidence       : {conf*100:.2f}%")
        print(f"Risk Indicator   : {risk}")
        print(f"Risk Score       : {score:.2f}")
        print("\nAll Class Probabilities:")
        for i, cls in enumerate(CLASSES):
            print(f"  {cls:25s}: {probs[i]*100:.2f}%")
        print("\nDisclaimer: This is an AI-based assessment, not a medical diagnosis.")

    except Exception as e:
        print(f"\nError: {e}")