FetalGuard-AI 🧬

AI-powered fetal ultrasound plane classification and preeclampsia risk screening, deployed as a lightweight Flask web app.

FetalGuard-AI takes a fetal brain ultrasound image, classifies it into one of four standard imaging planes using a DenseNet121 model, and overlays a Score-CAM heatmap to visualize which regions drove the prediction — all optimized to run within Render's free-tier 512MB memory limit. It also includes a rule-based preeclampsia risk assessment tool based on maternal clinical inputs.


⚠️ Medical Disclaimer: This tool is intended for decision-support and educational purposes only. It is not a substitute for professional medical diagnosis. All outputs should be reviewed by a qualified clinician.




Features


Fetal plane classification — Classifies ultrasound images into:

Trans-thalamic
Trans-ventricular
Trans-cerebellum
Diverse / Other



Score-CAM visual explanations — A gradient-free, forward-pass-only variant of Class Activation Mapping that highlights the regions the model focused on, without the memory overhead of backpropagation-based methods (e.g. Grad-CAM). This keeps the app runnable on constrained/free hosting tiers.
Automated image quality assessment — Scores uploaded scans on blur, contrast, brightness, signal-to-noise, and edge definition, and surfaces actionable recommendations (e.g. "adjust ultrasound gain settings").
Confidence-based risk flagging — Classifies each prediction as LOW / MODERATE / HIGH confidence risk and recommends manual review when appropriate.
Preeclampsia risk assessment — A separate rule-based module (scripts/preeclampsia_risk.py) that scores maternal risk from blood pressure, proteinuria, symptoms, and clinical history/labs, and returns a risk classification, triggered criteria, and a monitoring plan.
Downloadable HTML report — Generates a shareable, styled HTML report combining the original scan, the Score-CAM overlay, classification results, and confidence breakdown.



Tech Stack

LayerToolsBackendFlask, GunicornModelPyTorch (CPU), MONAI (DenseNet121)Image processingOpenCV (headless), Pillow, NumPyFrontendHTML/CSS/JS (Flask templates + static assets)DeploymentRender (render.yaml, Procfile)


Project Structure

Fetal_Ultrasound_AI/
├── app.py                     # Flask app: routes, inference, Score-CAM, HTML report builder
├── quantize_model.py          # Model quantization utility (for smaller/faster inference)
├── models/                    # Trained model weights (fetal_ultrasound_model.pth)
├── scripts/
│   └── preeclampsia_risk.py   # Rule-based preeclampsia risk scoring logic
├── templates/                 # Flask HTML templates (index.html, etc.)
├── static/                    # CSS/JS/static assets
├── requirements.txt
├── render.yaml                # Render deployment config
├── Procfile                   # Process file for Render/Heroku-style deployment
└── .gitignore


Getting Started

Prerequisites


Python 3.10+
pip


Installation

bashgit clone https://github.com/nattyix/Fetal_Ultrasound_AI.git
cd Fetal_Ultrasound_AI
pip install -r requirements.txt


Note: requirements.txt pins CPU-only PyTorch (torch==2.1.0+cpu) via the PyTorch CPU wheel index, so no GPU/CUDA setup is required.



Place your trained model weights at:

models/fetal_ultrasound_model.pth

Running locally

bashpython app.py

The app will start on http://0.0.0.0:5000.

For production-style serving:

bashgunicorn app:app


API Endpoints

MethodRouteDescriptionGET/Serves the main web UIGET/healthHealth check ({"status": "ok"})GET/image/<key>Serves a stored (original or overlay) image by keyPOST/analyzeUpload an ultrasound image (multipart/form-data, field image); returns plane classification, confidence, risk level, image quality metrics, and keys to the original + Score-CAM overlay imagesPOST/preeclampsiaSubmit maternal clinical data (JSON) for preeclampsia risk scoringPOST/report/htmlGenerates a downloadable/shareable HTML report from a prior /analyze result

Example: /analyze

bashcurl -X POST http://localhost:5000/analyze \
  -F "image=@/path/to/scan.jpg"

Response (abridged):

json{
  "success": true,
  "plane": "Trans-thalamic",
  "confidence": 92.4,
  "risk": "LOW",
  "verdict": "STANDARD",
  "quality": { "score": 81.2, "level": "EXCELLENT" },
  "all_probs": { "Trans-thalamic": 92.4, "Trans-ventricular": 4.1, ... },
  "orig_key": "…",
  "overlay_key": "…"
}

Example: /preeclampsia

bashcurl -X POST http://localhost:5000/preeclampsia \
  -H "Content-Type: application/json" \
  -d '{
    "systolic_bp": 145,
    "diastolic_bp": 95,
    "gestational_age_weeks": 32,
    "proteinuria": "moderate",
    "severe_headache": true
  }'

Returns a risk level, risk score, triggered clinical criteria, and a recommended monitoring plan.


How Score-CAM Works Here

Standard Grad-CAM requires a backward pass to compute gradients, which spikes memory usage — a problem on free-tier hosting. This app instead uses Score-CAM:


Run a single forward pass to get a baseline prediction confidence.
Extract the feature maps from the model's last convolutional block.
Select the top-N most active channels.
For each channel, mask the input image with that channel's activation map and re-run a forward pass to see how much it increases the target class's confidence.
Combine the weighted channel maps into a final heatmap.


Because every step is a forward pass (torch.inference_mode()), memory stays low and constant — no .backward() call is ever made.


Deployment

This project ships with render.yaml and a Procfile for one-click deployment to Render. The app is tuned for Render's free tier (512MB RAM):


Single-threaded PyTorch (torch.set_num_threads(1))
Score-CAM instead of Grad-CAM (no backprop memory spike)
In-memory image cache capped at 10 images, with aggressive gc.collect() calls after each request
JPEG compression/thumbnailing before storing images



Authors 


Natalia Mathews
Limnisha Changkakati




License

No license file is currently included in this repository. Add a LICENSE file to clarify usage terms if you intend to share or open-source this work.
