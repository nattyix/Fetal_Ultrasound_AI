"""
FetalGuard-AI — Flask (Render free tier, Score-CAM — no backprop)
Score-CAM uses only forward passes → no backward memory spike → fits 512MB
"""
import traceback, os, gc, uuid
from flask import Flask, render_template, request, jsonify, send_file
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
MODEL_PATH   = "models/fetal_ultrasound_model.pth"
IMAGE_SIZE   = (224, 224)
THUMB_SIZE   = (256, 256)
JPEG_QUALITY = 60

torch.set_num_threads(1)

CLASSES = ["Trans-thalamic", "Trans-ventricular", "Trans-cerebellum", "Diverse / Other"]
PLANE_INFO = {
    "Trans-thalamic":    {"description":"Axial view at the level of the thalami","structures":"Thalami, cavum septi pellucidi, falx cerebri","measurements":"Biparietal diameter (BPD), head circumference (HC)","clinical_significance":"Essential for biometry and detection of midline anomalies","common_findings":"Evaluation of brain symmetry, ventricle size, midline shift","color":"#2563eb"},
    "Trans-ventricular": {"description":"Axial view showing the lateral ventricles","structures":"Lateral ventricles, choroid plexus, anterior and posterior horns","measurements":"Atrial width, ventricular dimensions","clinical_significance":"Critical for detecting ventriculomegaly and CNS anomalies","common_findings":"Ventricular enlargement, choroid plexus cysts, brain development","color":"#7c3aed"},
    "Trans-cerebellum":  {"description":"Axial view at the level of the cerebellum","structures":"Cerebellum, cisterna magna, nuchal fold","measurements":"Transcerebellar diameter (TCD), cisterna magna depth","clinical_significance":"Key for posterior fossa evaluation and dating","common_findings":"Cerebellar hypoplasia, Dandy-Walker malformation, spina bifida","color":"#dc2626"},
    "Diverse / Other":   {"description":"Non-standard or unclear plane identification","structures":"Variable — depends on specific view obtained","measurements":"May require repositioning for standard biometry","clinical_significance":"Requires expert review and potential re-scanning","common_findings":"Image quality assessment, probe repositioning needed","color":"#ea580c"},
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# ── Image store ───────────────────────────────────────────────────────────
_image_store: dict = {}

def _store_image(b: bytes) -> str:
    key = uuid.uuid4().hex
    if len(_image_store) >= 10:
        del _image_store[next(iter(_image_store))]
    _image_store[key] = b
    return key

# ── Transform ─────────────────────────────────────────────────────────────
transform = Compose([EnsureChannelFirst(channel_dim=-1), Resize(IMAGE_SIZE), ScaleIntensity()])

# ── Model ─────────────────────────────────────────────────────────────────
print("[FetalGuard] Loading model...", flush=True)
model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()
gc.collect()
print("[FetalGuard] Ready.", flush=True)


# ── Score-CAM (NO backprop — only forward passes) ─────────────────────────
# Memory cost = 1 forward pass per channel subset → constant, low RAM
# Works perfectly with MONAI's in-place ops since no .backward() is called
def score_cam(mdl, image_np, class_idx, max_channels=32):
    """
    Gradient-free Score-CAM.
    Uses forward passes only — safe for MONAI, no OOM from backward graph.
    max_channels: subset of feature channels to use (lower = faster/less RAM)
    """
    img_h, img_w = image_np.shape[:2]
    inp = transform(image_np).unsqueeze(0)   # [1,3,224,224]

    # Step 1: get feature maps via forward hook (no grad needed)
    feat_holder = {}
    def _hook(m, i, o):
        feat_holder["feat"] = o.detach().cpu()
    hook = mdl.features.register_forward_hook(_hook)

    with torch.inference_mode():
        base_out = mdl(inp)                          # [1,4]
        base_score = torch.softmax(base_out, dim=1)[0, class_idx].item()
    hook.remove()

    feat = feat_holder["feat"]                       # [1, C, h, w]
    C = feat.shape[1]

    # Step 2: pick top-K channels by activation magnitude (saves time+RAM)
    channel_scores = feat[0].abs().mean(dim=(1, 2)) # [C]
    top_idx = channel_scores.argsort(descending=True)[:max_channels]

    # Step 3: for each selected channel, mask input and forward pass
    cam = np.zeros((img_h, img_w), dtype=np.float32)

    for idx in top_idx:
        ch_map = feat[0, idx].numpy()               # [h, w]
        # Normalise channel map to [0,1]
        mn, mx = ch_map.min(), ch_map.max()
        if mx - mn < 1e-8:
            continue
        ch_norm = (ch_map - mn) / (mx - mn)
        # Upsample to input size
        ch_up = cv2.resize(ch_norm, (img_w, img_h)) # [224,224]
        # Mask the input
        mask = torch.tensor(ch_up, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        # Convert to plain tensor to avoid MONAI MetaTensor incompatibility
        inp_plain  = inp.as_tensor() if hasattr(inp, "as_tensor") else torch.as_tensor(inp)
        masked_inp = inp_plain * mask
        with torch.inference_mode():
            out   = mdl(masked_inp)
            score = torch.softmax(out, dim=1)[0, class_idx].item()
        # Weight this channel's map by how much it helps the target class
        weight = max(score - base_score, 0)
        cam   += weight * ch_up
        del masked_inp, out, mask
        gc.collect()

    # Normalise final CAM
    mn, mx = cam.min(), cam.max()
    if mx - mn < 1e-8:
        return np.zeros((img_h, img_w), dtype=np.float32)
    return ((cam - mn) / (mx - mn)).astype(np.float32)


# ── Image quality ─────────────────────────────────────────────────────────
def assess_image_quality(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur  = min(100.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 10)
    cont  = min(100.0, float(gray.std()) * 2)
    bri   = float(gray.mean())
    bri_s = 100.0 - abs(bri - 127) / 1.27
    snr   = min(100.0, bri / (float(gray.std()) + 1e-6) * 10)
    edges = cv2.Canny(gray, 50, 150)
    edge  = min(100.0, (edges > 0).sum() / edges.size * 500)
    q     = blur*0.30 + cont*0.25 + bri_s*0.15 + snr*0.15 + edge*0.15
    iss, rec = [], []
    if blur < 40:  iss.append("Blurry image");    rec.append("Ensure probe is in good contact with skin")
    if cont < 30:  iss.append("Low contrast");    rec.append("Adjust ultrasound gain settings")
    if bri_s < 50:
        if bri < 100: iss.append("Image too dark");   rec.append("Increase brightness/gain")
        else:         iss.append("Image too bright");  rec.append("Decrease brightness/gain")
    if edge < 30:  iss.append("Poor definition"); rec.append("Adjust focus depth")
    if   q >= 75: ql,qc = "EXCELLENT","#22c55e"
    elif q >= 60: ql,qc = "GOOD",     "#3b82f6"
    elif q >= 40: ql,qc = "MODERATE", "#f59e0b"
    else:         ql,qc = "POOR",     "#ef4444"
    return {"score":q,"level":ql,"color":qc,
            "metrics":{"blur":blur,"contrast":cont,"brightness":bri_s,"snr":snr,"edge":edge},
            "issues":iss,"recommendations":rec}


# ── Inference ─────────────────────────────────────────────────────────────
def predict(image_np):
    inp = transform(image_np).unsqueeze(0)
    with torch.inference_mode():
        out  = model(inp)
        prob = torch.softmax(out, dim=1)
        conf, pred = torch.max(prob, dim=1)
    res = pred.item(), conf.item(), prob[0].cpu().numpy()
    del inp, out, prob; gc.collect()
    return res

def risk_indicator(c):
    if c >= 0.85:   return "LOW",      "Highly confident classification"
    elif c >= 0.60: return "MODERATE", "Review recommended for clinical correlation"
    else:           return "HIGH",     "Manual expert review strongly recommended"

def compress_jpeg(rgb_np):
    pil = Image.fromarray(rgb_np)
    pil.thumbnail(THUMB_SIZE, Image.LANCZOS)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()


# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

@app.route("/health")
def health(): return jsonify({"status":"ok"})

@app.route("/image/<key>")
def serve_image(key):
    data = _image_store.get(key)
    if not data: return "", 404
    return send_file(BytesIO(data), mimetype="image/jpeg")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    try:
        image_pil = Image.open(file.stream).convert("RGB")
        image_np  = np.array(image_pil)
        del image_pil; gc.collect()

        quality_data         = assess_image_quality(image_np)
        pred, conf, all_prob = predict(image_np)

        # Score-CAM — forward passes only, no backprop, no OOM
        cam_map     = score_cam(model, image_np, pred, max_channels=32)
        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        overlay_bgr = cv2.addWeighted(image_np, 0.6, heatmap_bgr, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        del cam_map, heatmap_bgr, overlay_bgr; gc.collect()

        orig_key    = _store_image(compress_jpeg(image_np))
        overlay_key = _store_image(compress_jpeg(overlay_rgb))
        del image_np, overlay_rgb; gc.collect()

        risk, risk_msg = risk_indicator(conf)
        plane = CLASSES[pred]
        pi    = PLANE_INFO[plane]
        is_std = plane != "Diverse / Other"
        hc     = conf >= 0.65
        verdict = "STANDARD" if (is_std and hc) else "UNCERTAIN" if is_std else "NON_STANDARD"

        return jsonify({
            "success":True,"plane":plane,"plane_color":pi["color"],
            "confidence":round(conf*100,2),"risk":risk,"risk_message":risk_msg,
            "verdict":verdict,"plane_info":pi,
            "all_probs":{CLASSES[i]:round(float(all_prob[i])*100,2) for i in range(4)},
            "quality":{"score":round(quality_data["score"],1),"level":quality_data["level"],
                       "color":quality_data["color"],
                       "metrics":{k:round(v,1) for k,v in quality_data["metrics"].items()},
                       "issues":quality_data["issues"],"recommendations":quality_data["recommendations"]},
            "orig_key":orig_key,"overlay_key":overlay_key,
            "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename":file.filename,"gradcam_active":True,
        })
    except Exception:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/preeclampsia", methods=["POST"])
def preeclampsia():
    try:
        d = request.json
        def opt_float(k):
            v=d.get(k,"")
            try: return float(v) if str(v).strip() else None
            except: return None
        def opt_bool(k):
            v=d.get(k)
            return bool(v) if v is not None else None
        inputs = PreeclampsiaInputs(
            systolic_bp=float(d["systolic_bp"]),diastolic_bp=float(d["diastolic_bp"]),
            gestational_age_weeks=int(d["gestational_age_weeks"]),proteinuria=d.get("proteinuria","none"),
            severe_headache=bool(d.get("severe_headache",False)),visual_disturbances=bool(d.get("visual_disturbances",False)),
            epigastric_pain=bool(d.get("epigastric_pain",False)),sudden_edema=bool(d.get("sudden_edema",False)),
            nulliparous=bool(d.get("nulliparous",False)),multiple_gestation=bool(d.get("multiple_gestation",False)),
            prior_preeclampsia=bool(d.get("prior_preeclampsia",False)),chronic_hypertension=bool(d.get("chronic_hypertension",False)),
            diabetes=bool(d.get("diabetes",False)),kidney_disease=bool(d.get("kidney_disease",False)),
            autoimmune_disease=bool(d.get("autoimmune_disease",False)),obesity_bmi_over_30=bool(d.get("obesity_bmi_over_30",False)),
            age_over_35=bool(d.get("age_over_35",False)),ivf_conception=bool(d.get("ivf_conception",False)),
            platelet_count=opt_float("platelet_count"),alt_ast_elevated=opt_bool("alt_ast_elevated"),
            serum_creatinine=opt_float("serum_creatinine"),uric_acid_elevated=opt_bool("uric_acid_elevated"),
            sflt1_plgf_ratio=opt_float("sflt1_plgf_ratio"),
        )
        r = assess_preeclampsia_risk(inputs)
        return jsonify({"success":True,"risk_level":r.risk_level,"risk_score":r.risk_score,
            "risk_color":r.risk_color,"classification":r.classification,
            "triggered_criteria":r.triggered_criteria,"severe_features":r.severe_features,
            "recommendations":r.recommendations,"monitoring_plan":r.monitoring_plan,
            "summary":r.summary,"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    except Exception:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/report/html", methods=["POST"])
def report_html():
    try:
        d = request.json
        d["original_b64"] = base64.b64encode(_image_store.get(d.get("orig_key",""), b"")).decode()
        d["heatmap_b64"]  = base64.b64encode(_image_store.get(d.get("overlay_key",""), b"")).decode()
        return jsonify({"html": _build_html_report(d)})
    except Exception:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500


def _build_html_report(d):
    ts=d.get("timestamp",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    plane=d.get("plane","—"); pd_=PLANE_INFO.get(plane,{}); conf=d.get("confidence",0)
    risk=d.get("risk","—"); probs=d.get("all_probs",{}); orig=d.get("original_b64","")
    ov=d.get("heatmap_b64",""); fname=d.get("filename","—")
    rc="#22c55e" if risk=="LOW" else "#f59e0b" if risk=="MODERATE" else "#ef4444"
    color=pd_.get("color","#3b82f6")
    bars="".join([f'<div style="margin-bottom:12px;"><div style="display:flex;justify-content:space-between;"><span style="font-weight:600;">{cn}</span><span style="color:#6b7280;">{probs.get(cn,0):.1f}%</span></div><div style="background:#e5e7eb;height:8px;border-radius:4px;overflow:hidden;margin-top:4px;"><div style="background:{PLANE_INFO[cn]["color"]};height:100%;width:{probs.get(cn,0)}%;"></div></div></div>' for cn in CLASSES])
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>FetalGuard-AI Report</title>
<style>body{{font-family:sans-serif;background:#667eea;padding:30px;color:#1f2937;}}.box{{max-width:900px;margin:0 auto;background:white;border-radius:16px;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,0.3);}}.hdr{{background:linear-gradient(135deg,#1e3a8a,#3b82f6);color:white;padding:32px;}}.body{{padding:32px;}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;}}img{{width:100%;border-radius:8px;}}.lbl{{background:#f3f4f6;padding:10px;text-align:center;font-weight:600;font-size:13px;}}.card{{background:{color}15;border:2px solid {color};border-radius:12px;padding:24px;margin-bottom:16px;}}.pname{{font-size:26px;font-weight:700;color:{color};margin-bottom:8px;}}.cnum{{font-size:44px;font-weight:700;color:{color};font-family:monospace;}}.badge{{background:{rc};color:white;padding:6px 18px;border-radius:16px;font-weight:700;font-size:13px;display:inline-block;margin:8px 0;}}.row{{background:white;padding:12px;border-radius:8px;border-left:4px solid {color};margin-bottom:8px;}}.disc{{background:#fef3c7;border-left:4px solid #f59e0b;padding:16px;border-radius:8px;margin-top:24px;}}.ftr{{background:#f9fafb;padding:20px 32px;text-align:center;color:#6b7280;font-size:12px;border-top:1px solid #e5e7eb;}}</style></head>
<body><div class="box"><div class="hdr"><h1>🧬 FetalGuard-AI Screening Report</h1><p>{ts}</p></div>
<div class="body"><div class="grid"><div><img src="data:image/jpeg;base64,{orig}"><div class="lbl">Original Scan</div></div><div><img src="data:image/jpeg;base64,{ov}"><div class="lbl">Score-CAM Heatmap</div></div></div>
<div class="card"><div class="pname">{plane}</div><div class="cnum">{conf:.1f}%</div><div class="badge">Risk: {risk}</div>
<div class="row"><b>Description:</b> {pd_.get("description","—")}</div><div class="row"><b>Structures:</b> {pd_.get("structures","—")}</div>
<div class="row"><b>Key Measurements:</b> {pd_.get("measurements","—")}</div><div class="row"><b>Clinical Significance:</b> {pd_.get("clinical_significance","—")}</div></div>
<h3>Confidence Breakdown</h3>{bars}
<div class="disc"><strong>⚠️ Medical Disclaimer:</strong> AI decision-support only. Not a clinical diagnosis.</div>
</div><div class="ftr">FetalGuard-AI · DenseNet121 · Score-CAM · {fname}</div></div></body></html>"""


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)