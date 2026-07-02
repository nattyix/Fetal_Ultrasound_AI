<div align="center">

# 🧬 FetalGuard-AI

### AI-Powered Fetal Ultrasound Plane Classification & Preeclampsia Risk Assessment

<p>

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![MONAI](https://img.shields.io/badge/MONAI-Medical%20AI-green?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-Educational-orange?style=for-the-badge)

</p>

An intelligent clinical decision support system that combines fetal ultrasound image analysis, image quality assessment, explainable AI, and preeclampsia risk screening.

</div>

---

# 📑 Table of Contents

- Overview
- Key Features
- AI Pipeline
- System Architecture
- Project Structure
- Deep Learning Model
- Clinical Risk Assessment
- Explainable AI
- Dataset
- Tech Stack
- Performance
- Installation
- Usage
- Future Improvements
- Contributing
- Disclaimer
- Authors

---

# 🌍 Overview

**FetalGuard-AI** is an AI-assisted prenatal healthcare application designed to classify fetal ultrasound planes and provide an early screening score for **preeclampsia risk**.

The application combines a **DenseNet121** deep learning model with clinical indicators, image quality assessment, and explainable AI to support faster and more transparent prenatal screening.

---

# ✨ Key Features

| Feature | Description |
|--------|-------------|
| 🧠 Ultrasound Plane Classification | Detects fetal brain ultrasound plane |
| ❤️ Preeclampsia Risk Screening | Clinical scoring engine |
| 📊 Image Quality Assessment | Evaluates uploaded scan quality |
| 🔥 Explainable AI | Score-CAM visualization |
| 📋 Clinical Report | Generates structured screening report |
| 🌐 Flask Dashboard | Interactive web application |
| ⚡ Real-Time Inference | Fast prediction pipeline |

---

# 🧠 AI Pipeline

```text
Ultrasound Image
        │
        ▼
Image Preprocessing
        │
        ▼
DenseNet121 (MONAI)
        │
        ▼
Plane Classification
        │
        ├──────────────┐
        ▼              ▼
Image Quality    Clinical Inputs
        │              │
        └──────┬───────┘
               ▼
Preeclampsia Risk Engine
               │
               ▼
Clinical Report + Score-CAM
```

---

# 🏗️ System Architecture

```text
User Upload
     │
     ▼
Ultrasound Scan
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
 ┌───┴────────────┐
 ▼                ▼
Quality      Score-CAM
Assessment
      │
      ▼
Clinical Risk Analysis
      │
      ▼
Final Diagnostic Report
```

---

# 📂 Project Structure

```text
FetalGuard-AI/
│
├── app.py
├── models/
├── templates/
├── static/
├── utils/
├── requirements.txt
├── README.md
└── trained_model.pth
```

---

# 🧬 Deep Learning Model

| Component | Details |
|----------|---------|
| Backbone | DenseNet121 |
| Framework | PyTorch + MONAI |
| Task | Multi-class Ultrasound Plane Classification |
| Explainability | Score-CAM |
| Deployment | Flask |

---

# ❤️ Clinical Risk Assessment

The application combines image predictions with maternal clinical indicators to estimate preeclampsia risk.

Clinical indicators include:

- Blood Pressure
- Maternal Age
- BMI
- Medical History
- Proteinuria
- Additional prenatal factors

The final score is presented as a structured clinical screening report.

---

# 🔥 Explainable AI

To improve transparency, **Score-CAM** highlights image regions influencing the model's prediction, helping users understand the reasoning behind each classification.

---

# 📊 Dataset

The model is trained on fetal ultrasound images representing multiple standard fetal brain planes.

The project includes preprocessing, augmentation, and validation strategies to improve robustness across varying image quality.

---

# ⚙️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Deep Learning | PyTorch, MONAI |
| Backend | Flask |
| Computer Vision | OpenCV, Pillow |
| Explainability | Score-CAM |
| Data Science | NumPy, Pandas |

---

# 📈 Performance

| Metric | Result |
|--------|-------:|
| Backbone | DenseNet121 |
| Explainability | Score-CAM |
| Deployment | Flask |
| Inference | Real-Time |

---

# 🚀 Installation

```bash
git clone https://github.com/nattyix/Fetal_Ultrasound_AI.git
cd Fetal_Ultrasound_AI
pip install -r requirements.txt
python app.py
```

---

# 💻 Usage

1. Launch the Flask application.
2. Upload a fetal ultrasound image.
3. Enter clinical information.
4. Review the predicted ultrasound plane.
5. Analyze image quality.
6. View Score-CAM visualization.
7. Generate the preeclampsia screening report.

---

# 🔮 Future Improvements

- Vision Transformer models
- Multi-organ fetal assessment
- DICOM integration
- Cloud deployment
- REST API
- Electronic Health Record integration
- Mobile application

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push your branch
5. Open a Pull Request

---

# 📜 Medical Disclaimer

This project is intended for **research, educational, and demonstration purposes only**.

It is **not** a certified medical device and should not replace professional clinical diagnosis or decision-making.

---

# 👨‍💻 Authors

**FetalGuard-AI Team**

- **Limnisha Changkakati**
- **Natalia Mathews**

---

<div align="center">

### ⭐ If you found this project helpful, please consider giving it a star!

Built with ❤️ using PyTorch, MONAI and Flask.

</div>
