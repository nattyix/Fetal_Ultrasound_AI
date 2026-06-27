/* ── State ───────────────────────────────────────────────────────────────── */
const state = {
  usHistory: [],
  peHistory: [],
  lastAnalysis: null,
};

/* ── Tab switching ────────────────────────────────────────────────────────── */
document.querySelectorAll(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

/* ── Drag & Drop + File input ────────────────────────────────────────────── */
const uploadZone = document.getElementById("upload-zone");
const fileInput  = document.getElementById("file-input");

uploadZone.addEventListener("click", (e) => {
  if (e.target.tagName !== "BUTTON") fileInput.click();
});
uploadZone.addEventListener("dragover",  e => { e.preventDefault(); uploadZone.classList.add("dragging"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragging"));
uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("dragging");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  const allowed = ["image/png", "image/jpeg", "image/jpg"];
  if (!allowed.includes(file.type)) {
    alert("Please upload a PNG or JPG image.");
    return;
  }
  runAnalysis(file);
}

/* ── Analysis ────────────────────────────────────────────────────────────── */
async function runAnalysis(file) {
  document.getElementById("how-to").style.display   = "none";
  document.getElementById("results").style.display  = "none";
  document.getElementById("spinner").style.display  = "block";

  const formData = new FormData();
  formData.append("image", file);

  try {
    const res  = await fetch("/analyze", { method: "POST", body: formData });
    const data = await res.json();

    if (!data.success) throw new Error(data.error || "Analysis failed");

    state.lastAnalysis = data;
    renderAnalysis(data);
    addToUSHistory(data);
    updateSessionStats();

    document.getElementById("spinner").style.display  = "none";
    document.getElementById("results").style.display  = "block";
    document.getElementById("upload-inner").querySelector(".upload-sub").textContent = `✅ ${file.name}`;
  } catch (err) {
    document.getElementById("spinner").style.display  = "none";
    document.getElementById("how-to").style.display   = "block";
    alert("Error: " + err.message);
  }
}

/* ── Render analysis ─────────────────────────────────────────────────────── */
function renderAnalysis(d) {
  // Images
  document.getElementById("orig-img").src  = "data:image/png;base64," + d.original_b64;
  document.getElementById("heat-img").src  = "data:image/png;base64," + d.heatmap_b64;

  // Quality
  const qs = document.getElementById("q-score");
  qs.textContent = d.quality.score;
  qs.style.color = d.quality.color;
  const ql = document.getElementById("q-level");
  ql.textContent = d.quality.level;
  ql.style.color = d.quality.color;

  const metrics = d.quality.metrics;
  document.getElementById("q-metrics").innerHTML = Object.entries(metrics).map(([k, v]) =>
    `<div class="q-metric">
      <span class="q-metric-label">${k}</span>
      <span class="q-metric-value">${v}/100</span>
    </div>`
  ).join("");

  let notesHtml = "";
  if (d.quality.issues.length)         notesHtml += `<p style="font-size:12px;color:var(--amber);font-weight:700;margin-bottom:6px;">⚠️ Issues</p>` + d.quality.issues.map(i => `<div class="q-note issue">• ${i}</div>`).join("");
  if (d.quality.recommendations.length) notesHtml += `<p style="font-size:12px;color:#60a5fa;font-weight:700;margin:10px 0 6px;">💡 Recommendations</p>` + d.quality.recommendations.map(r => `<div class="q-note rec">• ${r}</div>`).join("");
  document.getElementById("q-notes").innerHTML = notesHtml || `<div class="q-note">No issues detected.</div>`;

  // Verdict
  renderVerdict(d);

  // Result card
  document.getElementById("r-plane").textContent    = "🎯 " + d.plane;
  document.getElementById("r-plane").style.color    = d.plane_color;
  document.getElementById("r-conf").textContent     = d.confidence + "%";
  document.getElementById("r-risk-msg").textContent = d.risk_message;
  const rb = document.getElementById("r-risk");
  rb.textContent  = "⚠️ " + d.risk + " RISK";
  rb.className    = "risk-badge risk-" + d.risk;

  // Clinical info
  const pi = d.plane_info;
  document.getElementById("clinical-info").innerHTML = [
    ["Description",           pi.description],
    ["Anatomical Structures",  pi.structures],
    ["Key Measurements",       pi.measurements],
    ["Clinical Significance",  pi.clinical_significance],
    ["Common Findings",        pi.common_findings],
  ].map(([label, val]) =>
    `<div class="info-block">
      <div class="info-block-label">${label}</div>
      <div class="info-block-value">${val}</div>
    </div>`
  ).join("");

  // Confidence bars
  const COLORS = {
    "Trans-thalamic":    "#2563eb",
    "Trans-ventricular": "#7c3aed",
    "Trans-cerebellum":  "#dc2626",
    "Diverse / Other":   "#ea580c",
  };
  document.getElementById("conf-bars").innerHTML = Object.entries(d.all_probs).map(([name, prob]) =>
    `<div class="prob-item">
      <div class="prob-row">
        <span class="prob-name">${name}</span>
        <span class="prob-val">${prob.toFixed(2)}%</span>
      </div>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width:${prob}%;background:${COLORS[name] || "#3b82f6"};box-shadow:0 0 10px ${COLORS[name] || "#3b82f6"}66;"></div>
      </div>
    </div>`
  ).join("");
}

function renderVerdict(d) {
  const box = document.getElementById("verdict-box");
  const v   = d.verdict;

  let cls, icon, title, color, lines;

  if (v === "STANDARD") {
    cls   = "standard";
    icon  = "✅";
    title = "STANDARD PLANE DETECTED";
    color = "#22c55e";
    lines = [
      `<strong>Identified Plane:</strong> ${d.plane}`,
      `<strong>AI Confidence:</strong> ${d.confidence}% — reliable classification`,
      `<strong>Brain Position:</strong> Anatomically standard view confirmed`,
      `<strong>Measurements Available:</strong> ${d.plane_info.measurements}`,
      `<strong>Structures Visible:</strong> ${d.plane_info.structures}`,
    ];
  } else if (v === "UNCERTAIN") {
    cls   = "uncertain";
    icon  = "⚠️";
    title = "UNCERTAIN — EXPERT REVIEW NEEDED";
    color = "#f59e0b";
    lines = [
      `<strong>Possible Plane:</strong> ${d.plane} (low confidence)`,
      `<strong>AI Confidence:</strong> ${d.confidence}% — below reliable threshold`,
      "<strong>Brain Position:</strong> Cannot confirm standard plane",
      "<strong>Action Required:</strong> Expert sonographer review",
      "<strong>Suggested:</strong> Rescan with adjusted probe angle",
    ];
  } else {
    cls   = "non-std";
    icon  = "🔄";
    title = "NON-STANDARD PLANE DETECTED";
    color = "#f97316";
    lines = [
      "<strong>Identified:</strong> Diverse / Non-standard view",
      `<strong>AI Confidence:</strong> ${d.confidence}%`,
      "<strong>Brain Position:</strong> Standard plane not confirmed",
      "<strong>Action Required:</strong> Probe repositioning recommended",
      "<strong>Note:</strong> Standard biometric measurements cannot be taken",
    ];
  }

  box.className = "verdict-box " + cls;
  box.innerHTML = `
    <div class="verdict-icon">${icon}</div>
    <div class="verdict-title" style="color:${color};">${title}</div>
    <div class="verdict-sub">FetalGuard-AI Brain Position Assessment</div>
    ${lines.map(l => `<div class="verdict-detail">${l}</div>`).join("")}
  `;
}

/* ── Report ──────────────────────────────────────────────────────────────── */
async function downloadHTMLReport() {
  if (!state.lastAnalysis) return;
  try {
    const res  = await fetch("/report/html", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(state.lastAnalysis),
    });
    const data = await res.json();
    const blob = new Blob([data.html], { type: "text/html" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `fetalgaurd_report_${timestamp()}.html`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert("Report error: " + err.message);
  }
}

/* ── Preeclampsia assessment ─────────────────────────────────────────────── */
async function runPEAssessment() {
  document.getElementById("pe-results").style.display = "none";
  document.getElementById("pe-spinner").style.display = "block";
  document.getElementById("btn-pe-assess").disabled   = true;

  const payload = {
    systolic_bp:           parseFloat(document.getElementById("sbp").value),
    diastolic_bp:          parseFloat(document.getElementById("dbp").value),
    gestational_age_weeks: parseInt(document.getElementById("ga").value),
    proteinuria:           document.getElementById("proteinuria").value,
    severe_headache:       document.getElementById("severe_headache").checked,
    visual_disturbances:   document.getElementById("visual_dist").checked,
    epigastric_pain:       document.getElementById("epigastric").checked,
    sudden_edema:          document.getElementById("edema").checked,
    nulliparous:           document.getElementById("nullip").checked,
    multiple_gestation:    document.getElementById("mult_gest").checked,
    prior_preeclampsia:    document.getElementById("prior_pe").checked,
    chronic_hypertension:  document.getElementById("chronic_htn").checked,
    diabetes:              document.getElementById("diabetes").checked,
    kidney_disease:        document.getElementById("kidney").checked,
    autoimmune_disease:    document.getElementById("autoimmune").checked,
    obesity_bmi_over_30:   document.getElementById("obesity").checked,
    age_over_35:           document.getElementById("age35").checked,
    ivf_conception:        document.getElementById("ivf").checked,
    platelet_count:        document.getElementById("platelets").value.trim(),
    serum_creatinine:      document.getElementById("creatinine").value.trim(),
    sflt1_plgf_ratio:      document.getElementById("sflt1").value.trim(),
    alt_ast_elevated:      document.getElementById("alt_ast").checked,
    uric_acid_elevated:    document.getElementById("uric_acid").checked,
  };

  try {
    const res  = await fetch("/preeclampsia", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.error);

    renderPEResults(data, payload);
    addToPEHistory(data, payload);
    updateSessionStats();
  } catch (err) {
    alert("PE assessment error: " + err.message);
  } finally {
    document.getElementById("pe-spinner").style.display = "none";
    document.getElementById("btn-pe-assess").disabled   = false;
  }
}

function renderPEResults(d, payload) {
  const rc = `risk-${d.risk_level}`;

  const severeSet = new Set(d.severe_features.map(s => s.slice(0, 15)));
  const criteriaHtml = d.triggered_criteria.length
    ? d.triggered_criteria.map(c => {
        const isSev = [...severeSet].some(sf => c.startsWith(sf));
        return `<div class="pe-criterion ${isSev ? "severe" : ""}">• ${c}</div>`;
      }).join("")
    : `<div class="pe-criterion">No significant criteria triggered.</div>`;

  const severeHtml = d.severe_features.length
    ? `<div class="card-title" style="margin-top:20px;">🚨 Severe Feature Flags</div>` +
      d.severe_features.map(sf => `<div class="pe-criterion severe">⚠️ ${sf}</div>`).join("")
    : "";

  const recsHtml    = d.recommendations.map(r  => `<div class="pe-rec">→ ${r}</div>`).join("");
  const monitorHtml = d.monitoring_plan.map(m  => `<div class="pe-monitor">◆ ${m}</div>`).join("");

  const html = `
    <div class="glass-card" style="text-align:center;">
      <div class="pe-score-ring" style="color:${d.risk_color};">${Math.round(d.risk_score)}<span style="font-size:28px;color:var(--dim);">/100</span></div>
      <div class="pe-class-label">${d.classification}</div>
      <div style="margin-top:16px;"><span class="risk-badge ${rc}">${d.risk_level} RISK</span></div>
      <div class="pe-summary-box">${d.summary}</div>
    </div>

    <div class="pe-results-grid">
      <div class="glass-card">
        <div class="card-title">📋 Triggered Clinical Criteria</div>
        ${criteriaHtml}
        ${severeHtml}
      </div>
      <div class="glass-card">
        <div class="card-title">💊 Clinical Recommendations</div>
        ${recsHtml}
        <div class="card-title" style="margin-top:20px;">📅 Monitoring Plan</div>
        ${monitorHtml}
      </div>
    </div>

    <div class="glass-card">
      <div class="card-title">📊 Risk Score Breakdown</div>
      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
        <span style="color:var(--muted);font-size:13px;text-transform:uppercase;letter-spacing:1px;">Risk Score</span>
        <span style="color:${d.risk_color};font-weight:700;font-family:var(--font-mono);">${d.risk_score} / 100</span>
      </div>
      <div class="score-bar-wrap">
        <div class="score-bar-fill" style="width:${d.risk_score}%;background:${d.risk_color};box-shadow:0 0 16px ${d.risk_color}99;"></div>
      </div>
      <div class="score-bar-labels">
        <span style="color:var(--green);">LOW</span>
        <span style="color:var(--amber);">MODERATE</span>
        <span style="color:var(--orange);">HIGH</span>
        <span style="color:var(--red);">SEVERE</span>
      </div>
    </div>

    <div class="disclaimer">
      ⚠️ <strong>Medical Disclaimer:</strong> This preeclampsia risk score is a clinical decision-support tool only.
      It does <em>not</em> constitute a diagnosis. All outputs must be interpreted by a qualified obstetrician or midwife.
    </div>
  `;

  const el = document.getElementById("pe-results");
  el.innerHTML = html;
  el.style.display = "block";
  el.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ── History ─────────────────────────────────────────────────────────────── */
function addToUSHistory(d) {
  state.usHistory.unshift({
    timestamp:     d.timestamp,
    filename:      d.filename,
    plane:         d.plane,
    confidence:    d.confidence,
    quality_score: d.quality.score,
    quality_level: d.quality.level,
    risk:          d.risk,
  });
  if (state.usHistory.length > 50) state.usHistory = state.usHistory.slice(0, 50);
  renderUSHistory();
}

function addToPEHistory(d, payload) {
  state.peHistory.unshift({
    timestamp:      d.timestamp,
    ga_weeks:       payload.gestational_age_weeks,
    bp:             `${payload.systolic_bp}/${payload.diastolic_bp}`,
    risk_level:     d.risk_level,
    risk_score:     d.risk_score,
    classification: d.classification,
  });
  if (state.peHistory.length > 50) state.peHistory = state.peHistory.slice(0, 50);
  renderPEHistory();
}

function renderUSHistory() {
  const list  = document.getElementById("us-history-list");
  const stats = document.getElementById("us-history-stats");
  const btn   = document.getElementById("btn-clear-us");
  const h     = state.usHistory;

  if (!h.length) {
    list.innerHTML  = `<p class="empty-msg">No ultrasound analyses yet.</p>`;
    stats.style.display = "none";
    btn.style.display   = "none";
    return;
  }

  const avgConf  = (h.reduce((s, x) => s + x.confidence, 0) / h.length).toFixed(1);
  const avgQ     = (h.reduce((s, x) => s + x.quality_score, 0) / h.length).toFixed(1);
  const lowRisk  = h.filter(x => x.risk === "LOW").length;

  stats.innerHTML = `
    <div class="stat-box"><div class="stat-box-label">Total</div><div class="stat-box-value">${h.length}</div></div>
    <div class="stat-box"><div class="stat-box-label">Avg Confidence</div><div class="stat-box-value">${avgConf}%</div></div>
    <div class="stat-box"><div class="stat-box-label">Avg Quality</div><div class="stat-box-value">${avgQ}</div></div>
    <div class="stat-box"><div class="stat-box-label">Low Risk</div><div class="stat-box-value">${lowRisk}/${h.length}</div></div>
  `;
  stats.style.display = "flex";
  btn.style.display   = "inline-block";

  list.innerHTML = h.map(e => `
    <div class="history-item">
      <div><div class="hist-ts">${e.timestamp}</div><div class="hist-sub">${e.filename}</div></div>
      <div style="color:#60a5fa;font-weight:700;">${e.plane}</div>
      <div class="hist-mono" style="color:var(--green);">${e.confidence.toFixed(1)}%</div>
      <div class="hist-mono" style="color:var(--blue);">${e.quality_score.toFixed(1)}</div>
      <div><span class="risk-badge risk-${e.risk}" style="font-size:11px;padding:5px 12px;">${e.risk}</span></div>
    </div>
  `).join("");
}

function renderPEHistory() {
  const list  = document.getElementById("pe-history-list");
  const stats = document.getElementById("pe-history-stats");
  const btns  = document.getElementById("pe-hist-btns");
  const h     = state.peHistory;

  if (!h.length) {
    list.innerHTML  = `<p class="empty-msg">No preeclampsia assessments yet.</p>`;
    stats.style.display = "none";
    btns.style.display  = "none";
    return;
  }

  const avgScore  = (h.reduce((s, x) => s + x.risk_score, 0) / h.length).toFixed(1);
  const highSev   = h.filter(x => ["HIGH","SEVERE"].includes(x.risk_level)).length;
  const sevOnly   = h.filter(x => x.risk_level === "SEVERE").length;

  stats.innerHTML = `
    <div class="stat-box"><div class="stat-box-label">Total</div><div class="stat-box-value">${h.length}</div></div>
    <div class="stat-box"><div class="stat-box-label">Avg Score</div><div class="stat-box-value">${avgScore}</div></div>
    <div class="stat-box"><div class="stat-box-label">High/Severe</div><div class="stat-box-value">${highSev}</div></div>
    <div class="stat-box"><div class="stat-box-label">Severe</div><div class="stat-box-value">${sevOnly}</div></div>
  `;
  stats.style.display = "flex";
  btns.style.display  = "flex";

  list.innerHTML = h.map(e => `
    <div class="history-item">
      <div><div class="hist-ts">${e.timestamp}</div><div class="hist-sub">${e.classification}</div></div>
      <div style="color:#60a5fa;font-weight:700;">${e.ga_weeks} wks</div>
      <div class="hist-mono">${e.bp} mmHg</div>
      <div class="hist-mono" style="color:var(--amber);">${Math.round(e.risk_score)}/100</div>
      <div><span class="risk-badge risk-${e.risk_level}" style="font-size:11px;padding:5px 12px;">${e.risk_level}</span></div>
    </div>
  `).join("");
}

function clearHistory(type) {
  if (type === "us") { state.usHistory = []; renderUSHistory(); }
  else               { state.peHistory = []; renderPEHistory(); }
  updateSessionStats();
}

function exportPEHistory() {
  const blob = new Blob([JSON.stringify(state.peHistory, null, 2)], { type: "application/json" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = `pe_history_${timestamp()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/* ── Session stats sidebar ───────────────────────────────────────────────── */
function updateSessionStats() {
  const wrap = document.getElementById("session-stats");
  const h    = state.usHistory;
  const p    = state.peHistory;

  if (!h.length && !p.length) { wrap.style.display = "none"; return; }
  wrap.style.display = "block";

  document.getElementById("stat-analyses").textContent = h.length;
  document.getElementById("stat-pe").textContent       = p.length;
  document.getElementById("stat-conf").textContent     = h.length
    ? (h.reduce((s, x) => s + x.confidence, 0) / h.length).toFixed(1) + "%"
    : "—";
}

/* ── Utils ───────────────────────────────────────────────────────────────── */
function timestamp() {
  return new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
}
