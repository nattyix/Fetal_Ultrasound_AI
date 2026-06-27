"""
preeclampsia_risk.py
────────────────────
Rule-based + weighted clinical scoring engine for early preeclampsia
risk detection, designed to integrate with the Fetal Ultrasound AI app.

Risk stratification follows ACOG / NICE guidelines.
THIS IS A DECISION-SUPPORT TOOL — NOT A DIAGNOSTIC INSTRUMENT.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreeclampsiaInputs:
    # ── Vitals ──────────────────────────────────────────────────────────────
    systolic_bp: float          # mmHg
    diastolic_bp: float         # mmHg
    gestational_age_weeks: int  # 0–42

    # ── Proteinuria ─────────────────────────────────────────────────────────
    proteinuria: str            # "none" | "trace" | "1+" | "2+" | "3+" | "massive"

    # ── Symptoms (bool) ─────────────────────────────────────────────────────
    severe_headache: bool = False
    visual_disturbances: bool = False
    epigastric_pain: bool = False
    sudden_edema: bool = False

    # ── Maternal risk factors ────────────────────────────────────────────────
    nulliparous: bool = False
    multiple_gestation: bool = False
    prior_preeclampsia: bool = False
    chronic_hypertension: bool = False
    diabetes: bool = False
    kidney_disease: bool = False
    autoimmune_disease: bool = False
    obesity_bmi_over_30: bool = False
    age_over_35: bool = False
    ivf_conception: bool = False

    # ── Lab markers (optional) ───────────────────────────────────────────────
    platelet_count: Optional[float] = None       # ×10⁹/L; normal ≥150
    alt_ast_elevated: Optional[bool] = None      # liver enzymes
    serum_creatinine: Optional[float] = None     # mg/dL; normal <0.9 in pregnancy
    uric_acid_elevated: Optional[bool] = None
    sflt1_plgf_ratio: Optional[float] = None     # >38 high risk; >85 very high


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreeclampsiaResult:
    risk_level: str                          # "LOW" | "MODERATE" | "HIGH" | "SEVERE"
    risk_score: float                        # 0–100
    risk_color: str
    classification: str                      # descriptive label
    triggered_criteria: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    monitoring_plan: list[str] = field(default_factory=list)
    severe_features: list[str] = field(default_factory=list)
    summary: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Scoring engine
# ─────────────────────────────────────────────────────────────────────────────

_PROTEINURIA_GRADE = {
    "none": 0,
    "trace": 1,
    "1+": 2,
    "2+": 3,
    "3+": 4,
    "massive": 5,
}


def assess_preeclampsia_risk(p: PreeclampsiaInputs) -> PreeclampsiaResult:
    """
    Evaluate preeclampsia risk from clinical inputs.
    Returns a PreeclampsiaResult with risk level, score, and guidance.
    """
    score: float = 0.0
    criteria: list[str] = []
    severe_features: list[str] = []

    # ── 1. Blood pressure ─────────────────────────────────────────────────
    map_mmhg = (p.systolic_bp + 2 * p.diastolic_bp) / 3

    if p.systolic_bp >= 160 or p.diastolic_bp >= 110:
        score += 35
        criteria.append(f"Severe-range hypertension (SBP {p.systolic_bp} / DBP {p.diastolic_bp} mmHg)")
        severe_features.append("BP ≥ 160/110 mmHg — meets severe criteria")
    elif p.systolic_bp >= 140 or p.diastolic_bp >= 90:
        score += 20
        criteria.append(f"Hypertension detected (SBP {p.systolic_bp} / DBP {p.diastolic_bp} mmHg)")
    elif p.systolic_bp >= 130 or p.diastolic_bp >= 80:
        score += 8
        criteria.append(f"Elevated BP (SBP {p.systolic_bp} / DBP {p.diastolic_bp} mmHg) — borderline")

    # ── 2. Proteinuria ────────────────────────────────────────────────────
    pgrade = _PROTEINURIA_GRADE.get(p.proteinuria.lower().strip(), 0)
    if pgrade >= 4:           # 3+ or massive
        score += 20
        criteria.append(f"Significant proteinuria ({p.proteinuria})")
        severe_features.append(f"Heavy proteinuria ({p.proteinuria})")
    elif pgrade == 3:         # 2+
        score += 12
        criteria.append(f"Moderate proteinuria ({p.proteinuria})")
    elif pgrade >= 1:
        score += 5
        criteria.append(f"Trace / mild proteinuria ({p.proteinuria})")

    # ── 3. Gestational age context ────────────────────────────────────────
    if p.gestational_age_weeks < 34:
        score += 10
        criteria.append(f"Preterm gestation (GA {p.gestational_age_weeks} wks) — early-onset risk")
    elif p.gestational_age_weeks >= 20:
        score += 2
        criteria.append(f"≥20 weeks gestation (GA {p.gestational_age_weeks} wks)")
    else:
        # <20 weeks: PE very unlikely; flag for differential
        criteria.append(f"GA {p.gestational_age_weeks} wks — PE unlikely at this gestational age; consider molar pregnancy if BP elevated")

    # ── 4. Neurological / end-organ symptoms ─────────────────────────────
    if p.severe_headache:
        score += 12
        criteria.append("Severe / persistent headache")
        severe_features.append("Severe headache — CNS involvement possible")
    if p.visual_disturbances:
        score += 12
        criteria.append("Visual disturbances (scotomata, blurred vision, photopsia)")
        severe_features.append("Visual symptoms — possible cerebral or retinal involvement")
    if p.epigastric_pain:
        score += 10
        criteria.append("Epigastric / RUQ pain — possible HELLP/liver involvement")
        severe_features.append("Epigastric pain — HELLP syndrome must be excluded")
    if p.sudden_edema:
        score += 5
        criteria.append("Sudden onset or rapidly worsening edema")

    # ── 5. Lab markers ────────────────────────────────────────────────────
    if p.platelet_count is not None:
        if p.platelet_count < 100:
            score += 15
            criteria.append(f"Thrombocytopenia (platelets {p.platelet_count:.0f} ×10⁹/L)")
            severe_features.append(f"Platelets < 100 ×10⁹/L — HELLP criterion")
        elif p.platelet_count < 150:
            score += 6
            criteria.append(f"Low-normal platelets ({p.platelet_count:.0f} ×10⁹/L)")

    if p.alt_ast_elevated:
        score += 10
        criteria.append("Elevated liver enzymes (ALT/AST)")
        severe_features.append("Elevated liver enzymes — hepatic involvement, possible HELLP")

    if p.serum_creatinine is not None and p.serum_creatinine > 1.1:
        score += 10
        criteria.append(f"Elevated creatinine ({p.serum_creatinine:.2f} mg/dL) — renal impairment")
        severe_features.append(f"Creatinine > 1.1 mg/dL — renal involvement")

    if p.uric_acid_elevated:
        score += 5
        criteria.append("Elevated uric acid (hyperuricemia)")

    if p.sflt1_plgf_ratio is not None:
        if p.sflt1_plgf_ratio > 85:
            score += 18
            criteria.append(f"sFlt-1/PlGF ratio severely elevated ({p.sflt1_plgf_ratio:.1f} > 85)")
            severe_features.append(f"sFlt-1/PlGF > 85 — high predictive value for PE")
        elif p.sflt1_plgf_ratio > 38:
            score += 10
            criteria.append(f"sFlt-1/PlGF ratio elevated ({p.sflt1_plgf_ratio:.1f} > 38)")

    # ── 6. Maternal risk factors ──────────────────────────────────────────
    rf_score = 0
    if p.prior_preeclampsia:
        rf_score += 8
        criteria.append("Prior preeclampsia (highest single risk factor)")
    if p.chronic_hypertension:
        rf_score += 6
        criteria.append("Chronic / pre-existing hypertension")
    if p.multiple_gestation:
        rf_score += 5
        criteria.append("Multiple gestation (twins/higher)")
    if p.nulliparous:
        rf_score += 4
        criteria.append("Nulliparous (first pregnancy)")
    if p.diabetes:
        rf_score += 4
        criteria.append("Diabetes (type 1, 2, or gestational)")
    if p.kidney_disease:
        rf_score += 5
        criteria.append("Chronic kidney disease")
    if p.autoimmune_disease:
        rf_score += 5
        criteria.append("Autoimmune disease (SLE, APS, etc.)")
    if p.obesity_bmi_over_30:
        rf_score += 3
        criteria.append("Obesity (BMI > 30)")
    if p.age_over_35:
        rf_score += 2
        criteria.append("Advanced maternal age (> 35 years)")
    if p.ivf_conception:
        rf_score += 3
        criteria.append("IVF/ART conception")

    score += min(rf_score, 20)  # cap risk-factor contribution at 20

    # ── 7. Clamp score ────────────────────────────────────────────────────
    score = min(score, 100.0)

    # ── 8. Classify ───────────────────────────────────────────────────────
    # Severe features override numeric score
    has_severe = len(severe_features) > 0 and (
        p.systolic_bp >= 160 or p.diastolic_bp >= 110
        or p.platelet_count is not None and p.platelet_count < 100
        or p.alt_ast_elevated
        or p.serum_creatinine is not None and p.serum_creatinine > 1.1
        or p.visual_disturbances
        or p.severe_headache
        or p.epigastric_pain
    )

    if has_severe or score >= 70:
        risk_level = "SEVERE"
        risk_color = "#ef4444"
        classification = "Preeclampsia with Severe Features"
    elif score >= 45:
        risk_level = "HIGH"
        risk_color = "#f97316"
        classification = "High Risk — Possible Preeclampsia"
    elif score >= 25:
        risk_level = "MODERATE"
        risk_color = "#f59e0b"
        classification = "Moderate Risk — Preeclampsia Screening Warranted"
    else:
        risk_level = "LOW"
        risk_color = "#22c55e"
        classification = "Low Risk — Routine Monitoring"

    # ── 9. Recommendations ────────────────────────────────────────────────
    recs, monitoring = _build_guidance(risk_level, p, severe_features)

    # ── 10. Summary ───────────────────────────────────────────────────────
    summary = _build_summary(risk_level, score, p, criteria)

    return PreeclampsiaResult(
        risk_level=risk_level,
        risk_score=round(score, 1),
        risk_color=risk_color,
        classification=classification,
        triggered_criteria=criteria,
        recommendations=recs,
        monitoring_plan=monitoring,
        severe_features=severe_features,
        summary=summary,
    )


def _build_guidance(
    risk_level: str,
    p: PreeclampsiaInputs,
    severe_features: list[str],
) -> tuple[list[str], list[str]]:
    recs: list[str] = []
    monitoring: list[str] = []

    if risk_level == "SEVERE":
        recs += [
            "Immediate obstetric review — same-day hospitalisation advised",
            "Initiate antihypertensive therapy if SBP ≥ 160 or DBP ≥ 110 mmHg",
            "Magnesium sulfate for seizure prophylaxis (eclampsia prevention)",
            "Corticosteroids if GA < 34 weeks for fetal lung maturity",
            "Continuous fetal monitoring (CTG)",
            "Assess for HELLP syndrome: FBC, LFTs, LDH, peripheral smear",
            "Discuss delivery timing with senior obstetric team",
        ]
        monitoring += [
            "BP every 15–30 minutes until stable, then hourly",
            "Strict fluid balance / hourly urine output",
            "Repeat labs (FBC, LFTs, creatinine, uric acid) every 6–12 hours",
            "Daily fetal biophysical profile / Doppler",
        ]

    elif risk_level == "HIGH":
        recs += [
            "Urgent obstetric review within 24 hours",
            "Lab workup: FBC, LFTs, creatinine, uric acid, urine protein:creatinine ratio",
            "Consider sFlt-1/PlGF angiogenic ratio if not yet done",
            "Antihypertensive therapy if BP ≥ 140/90 mmHg",
            "Low-dose aspirin (if not already started) — discuss with physician",
            "Fetal growth scan and umbilical artery Doppler",
        ]
        monitoring += [
            "BP every 4–6 hours minimum",
            "Daily urine protein check",
            "Repeat labs every 2–3 days",
            "Weekly fetal monitoring",
        ]

    elif risk_level == "MODERATE":
        recs += [
            "Schedule obstetric review within 1–2 weeks",
            "Baseline labs: FBC, renal function, LFTs, urine dipstick/PCR",
            "Commence low-dose aspirin (75–150 mg/day) before 16 weeks if high-risk factors present",
            "Patient education: warning signs to seek immediate care",
            "Lifestyle: sodium restriction, adequate hydration, rest",
        ]
        monitoring += [
            "BP checks twice weekly",
            "Urine dipstick at each visit",
            "Growth scan at 28 and 34 weeks",
        ]

    else:  # LOW
        recs += [
            "Continue routine antenatal care",
            "BP and urine protein at every antenatal visit",
            "Consider aspirin if ≥2 moderate risk factors co-exist",
        ]
        monitoring += [
            "Routine antenatal BP monitoring",
            "Urine dipstick at each visit",
            "Standard growth scans per local protocol",
        ]

    return recs, monitoring


def _build_summary(
    risk_level: str,
    score: float,
    p: PreeclampsiaInputs,
    criteria: list[str],
) -> str:
    ga = p.gestational_age_weeks
    bp = f"{p.systolic_bp:.0f}/{p.diastolic_bp:.0f}"
    n = len(criteria)
    return (
        f"Assessment at {ga} weeks gestation: BP {bp} mmHg, "
        f"proteinuria {p.proteinuria}. "
        f"{n} clinical criterion/criteria met. "
        f"Overall risk score {score:.1f}/100 — classified as {risk_level} risk."
    )