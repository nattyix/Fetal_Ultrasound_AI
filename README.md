# 🧬 FetalGuard-AI

> **AI-powered fetal ultrasound plane classification and preeclampsia risk screening using Deep Learning and Explainable AI.**

FetalGuard-AI is an intelligent clinical decision-support system that classifies fetal brain ultrasound images into standard anatomical planes using a **DenseNet121** deep learning model while providing **Score-CAM explainability** to visualize the regions influencing the prediction. The system also integrates a **rule-based preeclampsia risk assessment module** based on maternal clinical parameters and generates downloadable diagnostic reports.

> **⚠️ Medical Disclaimer**
>
> This project is intended for **educational and clinical decision-support purposes only**. It is **not** a replacement for professional medical diagnosis or treatment. All predictions should be interpreted by qualified healthcare professionals.

---

# ✨ Features

### 🧠 AI-Based Fetal Ultrasound Classification
Classifies fetal brain ultrasound images into four standard anatomical planes:

- Trans-thalamic
- Trans-ventricular
- Trans-cerebellum
- Diverse / Other

---

### 🔥 Explainable AI with Score-CAM

Unlike Grad-CAM, Score-CAM performs only forward passes, eliminating expensive gradient computation.

Benefits:

- Memory efficient
- No backpropagation required
- Ideal for low-memory cloud deployments
- Visualizes regions influencing model predictions

---

### 📷 Automated Image Quality Assessment

Evaluates uploaded ultrasound images using multiple quality metrics:

- Blur Detection
- Brightness
- Contrast
- Signal-to-Noise Ratio
- Edge Definition

The system also provides practical recommendations to improve scan quality.

---

### 📊 Confidence-Based Risk Flagging

Each prediction is categorized into:

- 🟢 LOW Risk
- 🟡 MODERATE Risk
- 🔴 HIGH Risk

Low-confidence predictions are automatically flagged for manual clinical review.

---

### ❤️ Preeclampsia Risk Assessment

A rule-based clinical scoring engine evaluates maternal risk using:

- Blood Pressure
- Proteinuria
- Gestational Age
- Symptoms
- Laboratory Findings
- Medical History

The module returns:

- Risk Score
- Risk Category
- Triggered Clinical Criteria
- Recommended Monitoring Plan

---

### 📄 Downloadable Diagnostic Report

Generates a professional HTML report containing:

- Original Ultrasound Image
- Score-CAM Visualization
- Plane Prediction
- Confidence Scores
- Image Quality Metrics
- Clinical Interpretation

---

# 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Backend | Flask, Gunicorn |
| Deep Learning | PyTorch, MONAI (DenseNet121) |
| Image Processing | OpenCV, Pillow, NumPy |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render |
| Explainable AI | Score-CAM |

---

# 📁 Project Structure

```text
Fetal_Ultrasound_AI/
│
├── app.py
├── quantize_model.py
├── models/
│   └── fetal_ultrasound_model.pth
│
├── scripts/
│   └── preeclampsia_risk.py
│
├── templates/
├── static/
│
├── requirements.txt
├── Procfile
├── render.yaml
└── .gitignore
```

---

# 🚀 Getting Started

## Prerequisites

- Python 3.10+
- pip

---

## Installation

```bash
git clone https://github.com/nattyix/Fetal_Ultrasound_AI.git

cd Fetal_Ultrasound_AI

pip install -r requirements.txt
```

Place the trained model inside:

```text
models/
└── fetal_ultrasound_model.pth
```

---

## Run Locally

```bash
python app.py
```

Application starts at:

```
http://localhost:5000
```

For production:

```bash
gunicorn app:app
```

---

# 🌐 API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Web Interface |
| GET | `/health` | Health Check |
| GET | `/image/<key>` | Retrieve stored image |
| POST | `/analyze` | Analyze fetal ultrasound image |
| POST | `/preeclampsia` | Clinical risk assessment |
| POST | `/report/html` | Generate downloadable report |

---

## Example: Analyze Ultrasound

```bash
curl -X POST http://localhost:5000/analyze \
-F "image=@scan.jpg"
```

Example Response

```json
{
  "success": true,
  "plane": "Trans-thalamic",
  "confidence": 92.4,
  "risk": "LOW",
  "quality": {
    "score": 81.2,
    "level": "EXCELLENT"
  }
}
```

---

## Example: Preeclampsia Assessment

```bash
curl -X POST http://localhost:5000/preeclampsia \
-H "Content-Type: application/json" \
-d '{
  "systolic_bp":145,
  "diastolic_bp":95,
  "gestational_age_weeks":32,
  "proteinuria":"moderate",
  "severe_headache":true
}'
```

Returns:

- Risk Score
- Risk Category
- Triggered Criteria
- Monitoring Recommendation

---

# 🧠 How Score-CAM Works

Traditional Grad-CAM requires gradient computation using a backward pass, which significantly increases memory consumption.

FetalGuard-AI instead employs **Score-CAM**, which follows these steps:

1. Perform a forward pass to obtain prediction probabilities.
2. Extract feature maps from the final convolutional layer.
3. Select the most informative activation channels.
4. Mask the input image using each activation map.
5. Perform additional forward passes.
6. Weight each activation map based on confidence improvement.
7. Combine weighted maps into the final heatmap.

Because Score-CAM relies solely on forward inference (`torch.inference_mode()`), it maintains low and stable memory usage, making it ideal for deployment on resource-constrained environments such as Render's free tier.

---

# ☁️ Deployment

The project includes:

- `render.yaml`
- `Procfile`

for seamless deployment on **Render**.

Optimizations include:

- CPU-only PyTorch
- Single-thread inference
- Score-CAM (no gradient computation)
- Memory-efficient image caching
- Automatic garbage collection
- JPEG compression before storage

These optimizations allow the application to run comfortably within **512 MB RAM**.

---

# 👩‍💻 Authors

- **Natalia Mathews** — Project Lead
- **Limnisha Changkakati** — AI/ML Developer
- **Dr. Geethu S Kumar** — Research Supervisor

---

# 📄 License

This repository currently does not include a license.

If you intend to distribute or open-source the project, consider adding an appropriate LICENSE file such as:

- MIT License
- Apache 2.0
- GNU GPL v3

---

# 🌍 Live Demo

**https://fetal-ultrasound-ai.onrender.com/**

> **Note:** The application is hosted on Render's free tier. Initial startup may take **30–60 seconds** if the service has been idle.

---

# ⭐ Highlights

- Deep Learning using **DenseNet121**
- Explainable AI with **Score-CAM**
- Rule-Based Clinical Risk Assessment
- Automated Ultrasound Image Quality Analysis
- Confidence-Aware Predictions
- Downloadable Diagnostic Reports
- Flask REST API
- Render Cloud Deployment
- Memory Optimized for Low-Resource Environments
