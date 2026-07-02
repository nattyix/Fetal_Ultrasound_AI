<div align="center">

# 🧬 FetalGuard-AI

### AI-Powered Fetal Ultrasound Plane Classification & Preeclampsia Risk Screening

<p>
An intelligent clinical decision-support system that combines <b>DenseNet121</b>, <b>Explainable AI</b>, and a rule-based maternal risk assessment model to assist fetal ultrasound analysis.
</p>

<p>

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask)
![DenseNet121](https://img.shields.io/badge/DenseNet121-Transfer%20Learning-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-Educational-orange?style=for-the-badge)

</p>

<img src="assets/banner.png" width="900">

> Replace the banner above with your project banner or dashboard screenshot.

</div>

---

# 📑 Table of Contents

- 🌍 Overview
- ✨ Features
- 🧠 AI Pipeline
- 🏗️ Architecture
- 📸 Demo
- 📂 Project Structure
- 🧬 Deep Learning Model
- 📊 Dataset
- 📈 Model Performance
- ⚙️ Tech Stack
- 🖼️ Screenshots
- 🚀 Installation
- 💻 Usage
- 🔮 Future Improvements
- 🤝 Contributing
- 📜 Disclaimer

---

# 🌍 Overview

**FetalGuard-AI** is a deep learning-powered clinical decision-support application that automatically classifies fetal brain ultrasound images into standard anatomical planes while providing **Score-CAM visual explanations** and **maternal preeclampsia risk screening**.

The platform combines computer vision, transfer learning, image quality assessment, and clinical rule-based analysis into a single lightweight web application for educational and research use.

---

# ✨ Key Features

| 🚀 Feature | Description |
|------------|-------------|
| 🧠 DenseNet121 | Transfer learning for fetal ultrasound classification |
| 🔥 Score-CAM | Explainable AI visualization |
| 👶 Plane Classification | Trans-thalamic, Trans-ventricular, Trans-cerebellum & Diverse |
| ❤️ Preeclampsia Screening | Maternal clinical risk assessment |
| 📷 Image Quality Analysis | Blur, brightness, contrast & SNR evaluation |
| 📊 Confidence Scores | Probability-based prediction confidence |
| 📄 PDF Report | Downloadable clinical report |
| 🌐 Flask Web App | Interactive browser interface |

---

# 🧠 AI Pipeline

```text
Ultrasound Image
        │
        ▼
Image Quality Assessment
        │
        ▼
DenseNet121
        │
        ▼
Plane Classification
        │
 ┌──────┼──────────────┐
 ▼      ▼              ▼
Score-CAM      Confidence     Risk Flag
        │
        ▼
Clinical Report Generation
```

---

# 🏗️ Project Architecture

```text
User Upload
      │
      ▼
Ultrasound Image
      │
      ▼
Preprocessing
      │
      ▼
DenseNet121
      │
      ▼
Prediction
      │
 ┌────┴──────────────┐
 ▼                   ▼
Score-CAM      Confidence Score
      │
      ▼
Preeclampsia Assessment
      │
      ▼
Diagnostic Report
```

---

# 📸 Demo

> Add GIF or deployed application screenshot here.

```md
<img src="assets/demo.gif" width="900">
```

---

# 📂 Project Structure

```text
FetalGuard-AI
│
├── app.py
├── models/
│   ├── fetal_ultrasound_model.pth
│   ├── fold_1_best.pth
│   └── ...
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── preeclampsia_risk.py
├── static/
├── templates/
├── requirements.txt
└── README.md
```

---

# 🧬 Deep Learning Model

| Component | Details |
|-----------|---------|
| Backbone | DenseNet121 |
| Framework | PyTorch |
| Explainability | Score-CAM |
| Classes | 4 |
| Deployment | Flask |

---

# 📊 Dataset

| Dataset | Purpose |
|---------|---------|
| Fetal Brain Ultrasound Images | Plane Classification |
| Maternal Clinical Parameters | Preeclampsia Risk Screening |

**Classes**

- Trans-thalamic
- Trans-ventricular
- Trans-cerebellum
- Diverse

---

# 📈 Model Performance

| Metric | Value |
|---------|------:|
| Accuracy | 97%+ |
| Backbone | DenseNet121 |
| Explainability | Score-CAM |
| Deployment | Flask |

---

# ⚙️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV, Pillow |
| Explainability | Score-CAM |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |

---

# 🖼️ Screenshots

| Dashboard | Prediction |
|-----------|------------|
| ![](assets/home.png) | ![](assets/prediction.png) |

| Score-CAM | PDF Report |
|-----------|------------|
| ![](assets/scorecam.png) | ![](assets/report.png) |

> Replace placeholders with actual screenshots from your application.

---

# 🚀 Installation

```bash
git clone https://github.com/yourusername/FetalGuard-AI.git
cd FetalGuard-AI
pip install -r requirements.txt
python app.py
```

---

# 💻 Usage

1. Upload a fetal ultrasound image.
2. Review image quality metrics.
3. View predicted anatomical plane.
4. Inspect Score-CAM heatmap.
5. Enter maternal clinical parameters.
6. Generate preeclampsia risk assessment.
7. Download the clinical report.

---

# 🔮 Future Improvements

- Vision Transformers (ViT)
- DICOM Support
- Multi-fetal Analysis
- Cloud Deployment
- REST API
- 3D Ultrasound Support
- Mobile Application

---

# 🤝 Contributing

Contributions are welcome! Fork the repository, create a feature branch, commit your changes, and submit a Pull Request.

---

# 📜 Medical Disclaimer

This project is intended for **educational and research purposes only**.

It is **not a certified medical diagnostic system** and should always be used alongside professional clinical evaluation.

---

<div align="center">

## ⭐ Star this repository if you found it useful!

Made with ❤️ using PyTorch, Flask & Explainable AI.

</div>
