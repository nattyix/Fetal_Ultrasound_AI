"""
FetalGuard-AI — Flask Application (Render-optimised)
Memory budget: < 512 MB (Render free tier)

Optimisations applied
---------------------
- Model loaded once at module level (gunicorn pre-load via --preload)
- Grad-CAM disabled in low-memory mode (env DISABLE_GRADCAM=1)
- Images resized to 224x224 before base64 encoding (smaller payload)
- torch.inference_mode() instead of no_grad (lower overhead)
- weights_only=True on torch.load (safer + faster)
"""
import traceback
import os
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import base64
from io import BytesIO

from monai.transforms import Compose, EnsureChannelFirst, Resize, ScaleIntensity
from monai.networks.nets import DenseNet121
from scripts.preeclampsia_risk import PreeclampsiaInputs, assess_preeclampsia_risk

# ── Config ────────────────────────────────────────────────────────────────
MODEL_PATH      = "models/fetal_ultrasound_model.pth"
IMAGE_SIZE      = (224, 224)
# Set env var DISABLE_GRADCAM=1 on Render to skip Grad-CAM and save ~200MB
DISABLE_GRADCAM = os.environ.get("DISABLE_GRADCAM", "0") == "1"

CLASSES = ["Trans-thalamic", "Trans-ventricular", "Trans-cerebellum", "Diverse / Other"]
PLANE_INFO = {
    "Trans-thalamic": {
        "description": "Axial view at the level of the thalami",
        "structures": "Thalami, cavum septi pellucidi, falx cerebri",
        "measurements": "Biparietal diameter (BPD), head circumference (HC)",
        "clinical_significance": "Essential for biometry and detection of midline anomalies",
        "common_findings": "Evaluation of brain symmetry, ventricle size, midline shift",
        "color": "#2563eb"
    },
    "Trans-ventricular": {
        "description": "Axial view showing the lateral ventricles",
        "structures": "Lateral ventricles, choroid plexus, anterior and posterior horns",
        "measurements": "Atrial width, ventricular dimensions",
        "clinical_significance": "Critical for detecting ventriculomegaly and CNS anomalies",
        "common_findings": "Ventricular enlargement, choroid plexus cysts, brain development",
        "color": "#7c3aed"
    },
    "Trans-cerebellum": {
        "description": "Axial view at the level of the cerebellum",
        "structures": "Cerebellum, cisterna magna, nuchal fold",
        "measurements": "Transcerebellar diameter (TCD), cisterna magna depth",
        "clinical_significance": "Key for posterior fossa evaluation and dating",
        "common_findings": "Cerebellar hypoplasia, Dandy-Walker malformation, spina bifida",
        "color": "#dc2626"
    },
    "Diverse / Other": {
        "description": "Non-standard or unclear plane identification",
        "structures": "Variable — depends on specific view obtained",
        "measurements": "May require repositioning for standard biometry",
        "clinical_significance": "Requires expert review and potential re-scanning",
        "common_findings": "Image quality assessment, probe repositioning needed",
        "color": "#ea580c"
    }
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB (reduced from 16)

# ── Device & transform ────────────────────────────────────────────────────
device = torch.device("cpu")   # Render free tier is CPU only

transform = Compose([
    EnsureChannelFirst(channel_dim=-1),
    Resize(IMAGE_SIZE),
    ScaleIntensity(),
])

# ── Model — loaded once at module level ───────────────────────────────────
# Gunicorn must be started with --preload so this runs once in the master
# process and workers inherit the already-loaded model via fork.
print("[FetalGuard] Loading model...", flush=True)
_m = DenseNet121(spatial_dims=2, in_channels=3, out_channels=4)
_m.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)
_m.eval()
# Convert to half precision to cut memory from ~330MB → ~165MB
# Comment this out if you see NaN predictions
_m = _m.half()
model = _m
print("[FetalGuard] Model ready.", flush=True)


# ── Grad-CAM ──────────────────────────────────────────────────────────────
def grad_cam_generate(mdl, image_np, class_idx):
    """Hook-free Grad-CAM compatible with MONAI DenseNet121 in-place ops."""
    img_h, img_w = image_np.shape[:2]

    # Use float32 for grad-cam regardless of model precision
    inp = transform(image_np).unsqueeze(0).float()

    feature_holder = {}

    def _fwd_hook(module, inp_, out_):
        feat = out_.clone().float()   # ensure float32
        feat.retain_grad()
        feature_holder["feat"] = feat
        return feat

    hook = mdl.features.register_forward_hook(_fwd_hook)
    try:
        mdl.zero_grad()
        # Temporarily cast model to float32 for backward pass
        mdl_f = mdl.float()
        output = mdl_f(inp)
        feat   = feature_holder.get("feat")
        if feat is None:
            return np.zeros((img_h, img_w), dtype=np.float32)

        output[0, class_idx].backward()
        grads = feat.grad
        if grads is None:
            return np.zeros((img_h, img_w), dtype=np.float32)

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * feat).sum(dim=1)).squeeze()
        cam     = cam.detach().cpu().numpy()
        c_min, c_max = cam.min(), cam.max()
        if c_max - c_min < 1e-8:
            return np.zeros((img_h, img_w), dtype=np.float32)
        cam = (cam - c_min) / (c_max - c_min)
        return cv2.resize(cam, (img_w, img_h)).astype(np.float32)
    finally:
        hook.remove()
        mdl.zero_grad()
        # Cast back to half to restore memory savings
        mdl.half()


def blank_heatmap(image_np):
    """Return a neutral grey overlay when Grad-CAM is disabled."""
    overlay = image_np.copy()
    cv2.putText(overlay, "Grad-CAM disabled (low-memory mode)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return overlay


# ── Image quality scoring ─────────────────────────────────────────────────
def assess_image_quality(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur_score       = min(100.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 10)
    contrast_score   = min(100.0, float(gray.std()) * 2)
    brightness       = float(gray.mean())
    brightness_score = 100.0 - abs(brightness - 127) / 1.27
    snr_score        = min(100.0, brightness / (float(gray.std()) + 1e-6) * 10)
    edges            = cv2.Canny(gray, 50, 150)
    edge_score       = min(100.0, (edges > 0).sum() / edges.size * 500)
    quality_score    = (blur_score*0.30 + contrast_score*0.25 +
                        brightness_score*0.15 + snr_score*0.15 + edge_score*0.15)
    issues, recommendations = [], []
    if blur_score < 40:
        issues.append("Blurry image")
        recommendations.append("Ensure probe is in good contact with skin")
    if contrast_score < 30:
        issues.append("Low contrast")
        recommendations.append("Adjust ultrasound gain settings")
    if brightness_score < 50:
        if brightness < 100:
            issues.append("Image too dark"); recommendations.append("Increase brightness/gain")
        else:
            issues.append("Image too bright"); recommendations.append("Decrease brightness/gain")
    if edge_score < 30:
        issues.append("Poor definition"); recommendations.append("Adjust focus depth")
    if   quality_score >= 75: ql, qc = "EXCELLENT", "#22c55e"
    elif quality_score >= 60: ql, qc = "GOOD",      "#3b82f6"
    elif quality_score >= 40: ql, qc = "MODERATE",  "#f59e0b"
    else:                     ql, qc = "POOR",       "#ef4444"
    return {
        "score": quality_score, "level": ql, "color": qc,
        "metrics": {"blur": blur_score, "contrast": contrast_score,
                    "brightness": brightness_score, "snr": snr_score, "edge": edge_score},
        "issues": issues, "recommendations": recommendations,
    }


# ── Inference ─────────────────────────────────────────────────────────────
def predict(image_np):
    inp = transform(image_np).unsqueeze(0).half()   # match model dtype
    with torch.inference_mode():
        outputs    = model(inp)
        probs      = torch.softmax(outputs.float(), dim=1)
        conf, pred = torch.max(probs, dim=1)
    return pred.item(), conf.item(), probs[0].cpu().numpy()


def risk_indicator(confidence):
    if confidence >= 0.85:   return "LOW",      "Highly confident classification"
    elif confidence >= 0.60: return "MODERATE", "Review recommended for clinical correlation"
    else:                    return "HIGH",     "Manual expert review strongly recommended"


def image_to_b64(img, is_pil=False):
    buf = BytesIO()
    if is_pil:
        # Resize to 512px wide to reduce payload size
        img.thumbnail((512, 512), Image.LANCZOS)
        img.save(buf, format="JPEG", quality=80)
    else:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil.thumbnail((512, 512), Image.LANCZOS)
        pil.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "gradcam": not DISABLE_GRADCAM})


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        image_pil = Image.open(file.stream).convert("RGB")
        image_np  = np.array(image_pil)

        quality_data = assess_image_quality(image_np)
        pred, conf, all_probs = predict(image_np)

        if DISABLE_GRADCAM:
            overlay = blank_heatmap(image_np)
        else:
            cam_map = grad_cam_generate(model, image_np, pred)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

        risk, risk_message = risk_indicator(conf)
        plane_name = CLASSES[pred]
        plane_data = PLANE_INFO[plane_name]

        is_standard = plane_name != "Diverse / Other"
        high_conf   = conf >= 0.65
        if is_standard and high_conf:       verdict = "STANDARD"
        elif is_standard and not high_conf: verdict = "UNCERTAIN"
        else:                               verdict = "NON_STANDARD"

        return jsonify({
            "success":      True,
            "plane":        plane_name,
            "plane_color":  plane_data["color"],
            "confidence":   round(conf * 100, 2),
            "risk":         risk,
            "risk_message": risk_message,
            "verdict":      verdict,
            "plane_info":   plane_data,
            "all_probs":    {CLASSES[i]: round(float(all_probs[i]) * 100, 2) for i in range(4)},
            "quality": {
                "score":           round(quality_data["score"], 1),
                "level":           quality_data["level"],
                "color":           quality_data["color"],
                "metrics":         {k: round(v, 1) for k, v in quality_data["metrics"].items()},
                "issues":          quality_data["issues"],
                "recommendations": quality_data["recommendations"],
            },
            "original_b64": image_to_b64(image_pil, is_pil=True),
            "heatmap_b64":  image_to_b64(overlay),
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename":     file.filename,
            "gradcam_active": not DISABLE_GRADCAM,
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/preeclampsia", methods=["POST"])
def preeclampsia():
    try:
        d = request.json

        def opt_float(key):
            v = d.get(key, "")
            try:   return float(v) if str(v).strip() else None
            except: return None

        def opt_bool(key):
            v = d.get(key)
            return bool(v) if v is not None else None

        inputs = PreeclampsiaInputs(
            systolic_bp           = float(d["systolic_bp"]),
            diastolic_bp          = float(d["diastolic_bp"]),
            gestational_age_weeks = int(d["gestational_age_weeks"]),
            proteinuria           = d.get("proteinuria", "none"),
            severe_headache       = bool(d.get("severe_headache",      False)),
            visual_disturbances   = bool(d.get("visual_disturbances",  False)),
            epigastric_pain       = bool(d.get("epigastric_pain",      False)),
            sudden_edema          = bool(d.get("sudden_edema",         False)),
            nulliparous           = bool(d.get("nulliparous",          False)),
            multiple_gestation    = bool(d.get("multiple_gestation",   False)),
            prior_preeclampsia    = bool(d.get("prior_preeclampsia",   False)),
            chronic_hypertension  = bool(d.get("chronic_hypertension", False)),
            diabetes              = bool(d.get("diabetes",             False)),
            kidney_disease        = bool(d.get("kidney_disease",       False)),
            autoimmune_disease    = bool(d.get("autoimmune_disease",   False)),
            obesity_bmi_over_30   = bool(d.get("obesity_bmi_over_30",  False)),
            age_over_35           = bool(d.get("age_over_35",          False)),
            ivf_conception        = bool(d.get("ivf_conception",       False)),
            platelet_count        = opt_float("platelet_count"),
            alt_ast_elevated      = opt_bool("alt_ast_elevated"),
            serum_creatinine      = opt_float("serum_creatinine"),
            uric_acid_elevated    = opt_bool("uric_acid_elevated"),
            sflt1_plgf_ratio      = opt_float("sflt1_plgf_ratio"),
        )
        result = assess_preeclampsia_risk(inputs)
        return jsonify({
            "success":            True,
            "risk_level":         result.risk_level,
            "risk_score":         result.risk_score,
            "risk_color":         result.risk_color,
            "classification":     result.classification,
            "triggered_criteria": result.triggered_criteria,
            "severe_features":    result.severe_features,
            "recommendations":    result.recommendations,
            "monitoring_plan":    result.monitoring_plan,
            "summary":            result.summary,
            "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/report/html", methods=["POST"])
def report_html():
    try:
        return jsonify({"html": _build_html_report(request.json)})
    except Exception:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500


def _build_html_report(d):
    ts    = d.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    plane = d.get("plane", "—")
    pd_   = PLANE_INFO.get(plane, {})
    conf  = d.get("confidence", 0)
    risk  = d.get("risk", "—")
    probs = d.get("all_probs", {})
    orig  = d.get("original_b64", "")
    ov    = d.get("heatmap_b64", "")
    fname = d.get("filename", "—")
    rc    = "#22c55e" if risk == "LOW" else "#f59e0b" if risk == "MODERATE" else "#ef4444"
    color = pd_.get("color", "#3b82f6")
    bars  = "".join([
        f'<div style="margin-bottom:12px;">'
        f'<div style="display:flex;justify-content:space-between;">'
        f'<span style="font-weight:600;">{cn}</span>'
        f'<span style="color:#6b7280;">{probs.get(cn,0):.1f}%</span></div>'
        f'<div style="background:#e5e7eb;height:8px;border-radius:4px;overflow:hidden;margin-top:4px;">'
        f'<div style="background:{PLANE_INFO[cn]["color"]};height:100%;width:{probs.get(cn,0)}%;"></div>'
        f'</div></div>'
        for cn in CLASSES
    ])
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>FetalGuard-AI Report</title>
<style>
body{{font-family:sans-serif;background:#667eea;padding:30px;color:#1f2937;}}
.box{{max-width:900px;margin:0 auto;background:white;border-radius:16px;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,0.3);}}
.hdr{{background:linear-gradient(135deg,#1e3a8a,#3b82f6);color:white;padding:32px;}}
.body{{padding:32px;}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;}}
img{{width:100%;border-radius:8px;}}
.lbl{{background:#f3f4f6;padding:10px;text-align:center;font-weight:600;font-size:13px;}}
.card{{background:{color}15;border:2px solid {color};border-radius:12px;padding:24px;margin-bottom:16px;}}
.pname{{font-size:26px;font-weight:700;color:{color};margin-bottom:8px;}}
.cnum{{font-size:44px;font-weight:700;color:{color};font-family:monospace;}}
.badge{{background:{rc};color:white;padding:6px 18px;border-radius:16px;font-weight:700;font-size:13px;display:inline-block;margin:8px 0;}}
.row{{background:white;padding:12px;border-radius:8px;border-left:4px solid {color};margin-bottom:8px;}}
.disc{{background:#fef3c7;border-left:4px solid #f59e0b;padding:16px;border-radius:8px;margin-top:24px;}}
.ftr{{background:#f9fafb;padding:20px 32px;text-align:center;color:#6b7280;font-size:12px;border-top:1px solid #e5e7eb;}}
</style></head>
<body><div class="box">
<div class="hdr"><h1>🧬 FetalGuard-AI Screening Report</h1><p>{ts}</p></div>
<div class="body">
<div class="grid">
  <div><img src="data:image/jpeg;base64,{orig}"><div class="lbl">Original Scan</div></div>
  <div><img src="data:image/jpeg;base64,{ov}"><div class="lbl">Grad-CAM Heatmap</div></div>
</div>
<div class="card">
  <div class="pname">{plane}</div><div class="cnum">{conf:.1f}%</div>
  <div class="badge">Risk: {risk}</div>
  <div class="row"><b>Description:</b> {pd_.get("description","—")}</div>
  <div class="row"><b>Structures:</b> {pd_.get("structures","—")}</div>
  <div class="row"><b>Key Measurements:</b> {pd_.get("measurements","—")}</div>
  <div class="row"><b>Clinical Significance:</b> {pd_.get("clinical_significance","—")}</div>
</div>
<h3>Confidence Breakdown</h3>{bars}
<div class="disc"><strong>⚠️ Medical Disclaimer:</strong> AI decision-support only. Not a clinical diagnosis.</div>
</div>
<div class="ftr">FetalGuard-AI · DenseNet121 · {fname}</div>
</div></body></html>"""


# ── Local dev entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)