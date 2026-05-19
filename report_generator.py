"""
report_generator.py — Clinical EEG PDF Report Generator
Produces a structured, professional clinical report from real-world
patient EEG data and the Digital Twin prediction results.

Usage (standalone):
    python report_generator.py

Usage (from api.py):
    from report_generator import generate_clinical_report
    pdf_bytes = generate_clinical_report(patient_info, eeg_features, twin_result)
"""

import io
import time
from datetime import datetime


# ── ReportLab imports ─────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import Flowable

# ── Colour palette ─────────────────────────────────────────────────────────────
GOLD      = colors.HexColor("#D4A017")
DARK_GOLD = colors.HexColor("#B8860B")
DARK_BG   = colors.HexColor("#0D1117")
CARD_BG   = colors.HexColor("#161B22")
GREEN     = colors.HexColor("#2ECC71")
RED       = colors.HexColor("#E74C3C")
BLUE      = colors.HexColor("#3498DB")
ORANGE    = colors.HexColor("#F39C12")
GRAY_DARK = colors.HexColor("#555555")
GRAY_MID  = colors.HexColor("#888888")
GRAY_LIGHT= colors.HexColor("#CCCCCC")
WHITE     = colors.white
BLACK     = colors.black

# ── EEG Feature metadata (name, label, unit, typical range) ───────────────────
FEATURE_META = {
    "psd_delta":        ("Delta PSD",       "0.5–4 Hz",    "µV²/Hz", "0.5–5.0"),
    "psd_theta":        ("Theta PSD",       "4–8 Hz",      "µV²/Hz", "0.3–3.0"),
    "psd_alpha":        ("Alpha PSD",       "8–13 Hz",     "µV²/Hz", "0.5–4.0"),
    "psd_beta":         ("Beta PSD",        "13–30 Hz",    "µV²/Hz", "0.3–3.0"),
    "psd_gamma":        ("Gamma PSD",       "30–50 Hz",    "µV²/Hz", "0.1–1.5"),
    "alpha_asymmetry":  ("Alpha Asymmetry", "Frontal L–R", "ratio",  "−0.3–0.3"),
    "coh_Fp1_Fp2":      ("Coh Fp1↔Fp2",    "Prefrontal",  "0–1",    "0.50–0.85"),
    "coh_F3_F4":        ("Coh F3↔F4",      "Frontal",     "0–1",    "0.50–0.85"),
    "coh_C3_C4":        ("Coh C3↔C4",      "Central",     "0–1",    "0.50–0.85"),
    "coh_P3_P4":        ("Coh P3↔P4",      "Parietal",    "0–1",    "0.50–0.85"),
    "coh_O1_O2":        ("Coh O1↔O2",      "Occipital",   "0–1",    "0.50–0.85"),
    "coh_Fp1_F3":       ("Coh Fp1→F3",     "Pre-Frontal", "0–1",    "0.55–0.90"),
    "coh_Fp2_F4":       ("Coh Fp2→F4",     "Pre-Frontal", "0–1",    "0.55–0.90"),
    "coh_F3_C3":        ("Coh F3→C3",      "Frontal-Cen", "0–1",    "0.55–0.90"),
    "coh_F4_C4":        ("Coh F4→C4",      "Frontal-Cen", "0–1",    "0.55–0.90"),
    "coh_C3_P3":        ("Coh C3→P3",      "Centro-Par.", "0–1",    "0.55–0.90"),
    "age":              ("Patient Age",     "—",           "years",  "18–80"),
    "gender":           ("Gender",          "—",           "0=F/1=M","0 or 1"),
}

FEATURE_ORDER = [
    "psd_delta","psd_theta","psd_alpha","psd_beta","psd_gamma",
    "alpha_asymmetry",
    "coh_Fp1_Fp2","coh_F3_F4","coh_C3_C4","coh_P3_P4","coh_O1_O2",
    "coh_Fp1_F3","coh_Fp2_F4","coh_F3_C3","coh_F4_C4","coh_C3_P3",
    "age","gender",
]


# ── Horizontal rule Flowable ──────────────────────────────────────────────────
class ColorHR(Flowable):
    def __init__(self, width, color=GOLD, thickness=1.0, spaceAfter=4):
        super().__init__()
        self.width = width
        self.color = color
        self.thickness = thickness
        self.spaceAfter = spaceAfter
        self.height = thickness + spaceAfter

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.spaceAfter, self.width, self.spaceAfter)


# ── Style factory ─────────────────────────────────────────────────────────────
def _make_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["title"] = ParagraphStyle(
        "title", parent=base["Normal"],
        fontSize=22, textColor=GOLD, fontName="Helvetica-Bold",
        spaceAfter=2, alignment=TA_CENTER,
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle", parent=base["Normal"],
        fontSize=10, textColor=GRAY_MID, fontName="Helvetica",
        spaceAfter=6, alignment=TA_CENTER,
    )
    styles["section"] = ParagraphStyle(
        "section", parent=base["Normal"],
        fontSize=11, textColor=GOLD, fontName="Helvetica-Bold",
        spaceBefore=10, spaceAfter=4, leftIndent=0,
    )
    styles["body"] = ParagraphStyle(
        "body", parent=base["Normal"],
        fontSize=9.5, textColor=colors.HexColor("#333333"),
        fontName="Helvetica", leading=14, alignment=TA_JUSTIFY,
    )
    styles["small"] = ParagraphStyle(
        "small", parent=base["Normal"],
        fontSize=8, textColor=GRAY_MID, fontName="Helvetica",
        leading=11,
    )
    styles["footer"] = ParagraphStyle(
        "footer", parent=base["Normal"],
        fontSize=7.5, textColor=GRAY_MID, fontName="Helvetica",
        alignment=TA_CENTER,
    )
    styles["alert_mdd"] = ParagraphStyle(
        "alert_mdd", parent=base["Normal"],
        fontSize=14, textColor=RED, fontName="Helvetica-Bold",
        alignment=TA_CENTER, spaceBefore=4, spaceAfter=4,
    )
    styles["alert_healthy"] = ParagraphStyle(
        "alert_healthy", parent=base["Normal"],
        fontSize=14, textColor=GREEN, fontName="Helvetica-Bold",
        alignment=TA_CENTER, spaceBefore=4, spaceAfter=4,
    )
    styles["label"] = ParagraphStyle(
        "label", parent=base["Normal"],
        fontSize=8, textColor=GRAY_DARK, fontName="Helvetica-Bold",
    )
    styles["value"] = ParagraphStyle(
        "value", parent=base["Normal"],
        fontSize=9.5, textColor=BLACK, fontName="Helvetica",
    )
    return styles


# ── Band‐power risk flag ───────────────────────────────────────────────────────
def _band_risk_flag(key, value):
    """Return (flag_text, colour) based on clinical EEG heuristics."""
    if key == "psd_delta":
        if value > 3.5:  return "↑ Elevated", ORANGE
        if value < 0.3:  return "↓ Low",      BLUE
        return "Normal", GREEN
    if key == "psd_alpha":
        if value < 0.8:  return "↓ Reduced (MDD marker)", RED
        if value > 4.0:  return "↑ High",     BLUE
        return "Normal", GREEN
    if key == "psd_theta":
        if value > 2.5:  return "↑ Elevated", ORANGE
        return "Normal", GREEN
    if key == "alpha_asymmetry":
        if value > 0.15: return "↑ R-dominant (MDD risk)", RED
        if value < -0.15:return "↓ L-dominant",            BLUE
        return "Balanced", GREEN
    if key.startswith("coh_"):
        if value < 0.45: return "↓ Low coherence",   RED
        if value > 0.90: return "↑ High coherence",  BLUE
        return "Normal", GREEN
    return "—", GRAY_MID


# ── Main PDF generator ────────────────────────────────────────────────────────

def generate_clinical_report(patient_info: dict, eeg_features: dict, twin_result: dict) -> bytes:
    """
    Generate a structured clinical EEG PDF report.

    Parameters
    ----------
    patient_info : dict
        Keys: patient_name, patient_id, doctor_name, hospital, report_date,
              clinical_notes (optional)
    eeg_features : dict
        18 EEG feature key→value pairs (raw, un-normalised).
    twin_result : dict
        Output of EEGDigitalTwin.update_state() — prediction, mdd_probability,
        confidence, inference_ms, rolling_risk, alert.

    Returns
    -------
    bytes  — PDF file contents
    """
    buf = io.BytesIO()
    W, H = A4
    MARGIN = 20 * mm

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=18*mm, bottomMargin=18*mm,
        title="EEG Clinical Report",
        author="EEG Digital Twin System",
    )

    S = _make_styles()
    PAGE_W = W - 2 * MARGIN
    story  = []

    # ── HEADER ────────────────────────────────────────────────────────────────
    story.append(Paragraph("EEG DIGITAL TWIN", S["title"]))
    story.append(Paragraph("Clinical Mental Health Assessment Report", S["subtitle"]))
    story.append(Paragraph(
        "IEEE Framework · Cloud-Based Real-Time MDD Detection · SRM Institute of Science and Technology",
        S["subtitle"]
    ))
    story.append(Spacer(1, 3*mm))
    story.append(ColorHR(PAGE_W, GOLD, 1.5, 2))
    story.append(Spacer(1, 2*mm))

    # ── SECTION 1 — PATIENT INFORMATION ────────────────────────────────────────
    story.append(Paragraph("SECTION 1 — PATIENT INFORMATION", S["section"]))
    story.append(ColorHR(PAGE_W, GRAY_LIGHT, 0.5, 2))

    report_date = patient_info.get("report_date") or datetime.now().strftime("%Y-%m-%d %H:%M")
    gender_str  = "Female" if str(eeg_features.get("gender", "")) == "0" else \
                  "Male"   if str(eeg_features.get("gender", "")) == "1" else \
                  patient_info.get("gender_display", "—")

    pi_data = [
        ["Patient Name",  patient_info.get("patient_name", "—"),
         "Patient ID",    patient_info.get("patient_id", "—")],
        ["Referring Doctor", patient_info.get("doctor_name", "—"),
         "Date of Report",   report_date],
        ["Hospital / Clinic", patient_info.get("hospital", "—"),
         "Age / Gender",
         f"{int(eeg_features.get('age', 0))} yrs / {gender_str}"],
    ]

    pi_table = Table(pi_data, colWidths=[PAGE_W*0.22, PAGE_W*0.28, PAGE_W*0.22, PAGE_W*0.28])
    pi_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,-1), colors.HexColor("#F8F9FA")),
        ("BACKGROUND",  (0,0),(0,-1),  colors.HexColor("#EEEEEE")),
        ("BACKGROUND",  (2,0),(2,-1),  colors.HexColor("#EEEEEE")),
        ("FONTNAME",    (0,0),(0,-1),  "Helvetica-Bold"),
        ("FONTNAME",    (2,0),(2,-1),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 8.5),
        ("TEXTCOLOR",   (0,0),(0,-1),  GRAY_DARK),
        ("TEXTCOLOR",   (2,0),(2,-1),  GRAY_DARK),
        ("TEXTCOLOR",   (1,0),(1,-1),  BLACK),
        ("TEXTCOLOR",   (3,0),(3,-1),  BLACK),
        ("GRID",        (0,0),(-1,-1), 0.4, GRAY_LIGHT),
        ("PADDING",     (0,0),(-1,-1), 7),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#F8F9FA"), colors.HexColor("#F0F2F4")]),
    ]))
    story.append(pi_table)
    story.append(Spacer(1, 5*mm))

    # ── SECTION 2 — DIAGNOSIS RESULT ──────────────────────────────────────────
    story.append(Paragraph("SECTION 2 — DIGITAL TWIN DIAGNOSIS", S["section"]))
    story.append(ColorHR(PAGE_W, GRAY_LIGHT, 0.5, 2))

    prediction   = twin_result.get("prediction", "Unknown")
    mdd_prob     = twin_result.get("mdd_probability", 0.0)
    confidence   = twin_result.get("confidence", 0.0)
    inference_ms = twin_result.get("inference_ms", 0.0)
    rolling_risk = twin_result.get("rolling_risk", 0.0)
    alert        = twin_result.get("alert", False)
    is_mdd       = prediction == "MDD"

    result_bg = colors.HexColor("#FFF5F5") if is_mdd else colors.HexColor("#F0FFF4")
    result_border = RED if is_mdd else GREEN
    diag_text = "⚠  Major Depressive Disorder (MDD) Pattern Detected" \
                if is_mdd else \
                "✓  No Depressive Pattern Detected — Healthy EEG Signature"

    dx_style = S["alert_mdd"] if is_mdd else S["alert_healthy"]
    diag_para = Paragraph(diag_text, dx_style)

    prob_pct    = f"{mdd_prob*100:.1f}%"
    conf_pct    = f"{confidence*100:.1f}%"
    roll_pct    = f"{rolling_risk:.1f}%"
    lat_str     = f"{inference_ms:.2f} ms"
    alert_str   = "YES — Refer to Psychiatry" if alert else "No"
    alert_color = RED if alert else GREEN

    diag_data = [
        ["Metric", "Value", "Interpretation"],
        ["Diagnosis",         prediction,   "MDD = Major Depressive Disorder"],
        ["MDD Probability",   prob_pct,     "Likelihood of depressive pattern (0–100%)"],
        ["Confidence",        conf_pct,     "Model certainty in prediction"],
        ["Rolling MDD Risk",  roll_pct,     "% of recent 20 samples classified as MDD"],
        ["Inference Latency", lat_str,      "Target < 120 ms (IEEE Table II)"],
        ["Clinical Alert",    alert_str,    "Triggered when MDD probability > 70%"],
    ]

    col_w = [PAGE_W*0.30, PAGE_W*0.22, PAGE_W*0.48]
    diag_table = Table(diag_data, colWidths=col_w)
    diag_style = TableStyle([
        # Header
        ("BACKGROUND",  (0,0),(-1,0),  DARK_GOLD),
        ("TEXTCOLOR",   (0,0),(-1,0),  WHITE),
        ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,0),  9),
        ("ALIGN",       (0,0),(-1,0),  "CENTER"),
        # Body
        ("FONTSIZE",    (0,1),(-1,-1), 9),
        ("FONTNAME",    (0,1),(0,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,1),(0,-1),  GRAY_DARK),
        ("GRID",        (0,0),(-1,-1), 0.4, GRAY_LIGHT),
        ("PADDING",     (0,0),(-1,-1), 7),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, colors.HexColor("#F8F8F8")]),
        # Highlight diagnosis row
        ("BACKGROUND",  (1,1),(1,1),   result_bg),
        ("TEXTCOLOR",   (1,1),(1,1),   result_border),
        ("FONTNAME",    (1,1),(1,1),   "Helvetica-Bold"),
        ("FONTSIZE",    (1,1),(1,1),   11),
    ])
    # Colour alert cell
    if alert:
        diag_style.add("TEXTCOLOR",  (1,6),(1,6),  RED)
        diag_style.add("FONTNAME",   (1,6),(1,6),  "Helvetica-Bold")

    diag_table.setStyle(diag_style)

    result_box = Table(
        [[diag_para], [Spacer(1, 3*mm)], [diag_table]],
        colWidths=[PAGE_W],
    )
    result_box.setStyle(TableStyle([
        ("BOX",         (0,0),(-1,-1), 1.5, result_border),
        ("BACKGROUND",  (0,0),(-1,-1), result_bg),
        ("PADDING",     (0,0),(-1,-1), 8),
        ("TOPPADDING",  (0,0),(-1,0),  10),
    ]))
    story.append(result_box)
    story.append(Spacer(1, 5*mm))

    # ── SECTION 3 — EEG FEATURE ANALYSIS ──────────────────────────────────────
    story.append(Paragraph("SECTION 3 — EEG FEATURE ANALYSIS", S["section"]))
    story.append(ColorHR(PAGE_W, GRAY_LIGHT, 0.5, 2))
    story.append(Paragraph(
        "All 18 EEG features submitted for analysis. Values are raw (un-normalised); "
        "normalization (IEEE Eq. 6: x′ = (x−μ)/σ) is applied internally before inference.",
        S["body"]
    ))
    story.append(Spacer(1, 3*mm))

    feat_header = ["Feature", "Band / Region", "Submitted Value", "Unit", "Ref. Range", "Status"]
    feat_rows = [feat_header]
    for key in FEATURE_ORDER:
        label, band, unit, ref = FEATURE_META.get(key, (key, "—", "—", "—"))
        val = eeg_features.get(key)
        if val is None:
            val_str = "—"
            flag, fcol = "Missing", ORANGE
        else:
            if key == "gender":
                val_str = "Female (0)" if float(val) == 0 else "Male (1)"
            elif key == "age":
                val_str = str(int(float(val)))
            else:
                val_str = f"{float(val):.4f}"
            flag, fcol = _band_risk_flag(key, float(val))

        feat_rows.append([label, band, val_str, unit, ref, flag])

    feat_col_w = [PAGE_W*0.22, PAGE_W*0.16, PAGE_W*0.15, PAGE_W*0.10, PAGE_W*0.14, PAGE_W*0.23]
    feat_table = Table(feat_rows, colWidths=feat_col_w, repeatRows=1)

    ft_style = TableStyle([
        # Header
        ("BACKGROUND",  (0,0),(-1,0),  DARK_GOLD),
        ("TEXTCOLOR",   (0,0),(-1,0),  WHITE),
        ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,0),  8),
        ("ALIGN",       (0,0),(-1,0),  "CENTER"),
        # Body
        ("FONTSIZE",    (0,1),(-1,-1), 8),
        ("FONTNAME",    (0,1),(0,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,1),(0,-1),  GRAY_DARK),
        ("GRID",        (0,0),(-1,-1), 0.3, GRAY_LIGHT),
        ("PADDING",     (0,0),(-1,-1), 5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, colors.HexColor("#F8F8F8")]),
        ("ALIGN",       (2,0),(5,-1),  "CENTER"),
    ])
    # Colour status column per row
    for row_idx, row in enumerate(feat_rows[1:], start=1):
        flag_val = row[5]
        if "MDD" in flag_val or "Elevated" in flag_val or "Low coher" in flag_val or "R-dominant" in flag_val:
            fcol = RED
        elif "Reduced" in flag_val or "↓" in flag_val:
            fcol = ORANGE
        elif "High" in flag_val or "↑" in flag_val:
            fcol = BLUE
        elif "Normal" in flag_val or "Balanced" in flag_val:
            fcol = GREEN
        else:
            fcol = GRAY_MID
        ft_style.add("TEXTCOLOR", (5,row_idx),(5,row_idx), fcol)
        ft_style.add("FONTNAME",  (5,row_idx),(5,row_idx), "Helvetica-Bold")

    feat_table.setStyle(ft_style)
    story.append(feat_table)
    story.append(Spacer(1, 5*mm))

    # ── SECTION 4 — BAND POWER SUMMARY ────────────────────────────────────────
    story.append(Paragraph("SECTION 4 — BRAINWAVE BAND POWER SUMMARY", S["section"]))
    story.append(ColorHR(PAGE_W, GRAY_LIGHT, 0.5, 2))

    bands = ["psd_delta","psd_theta","psd_alpha","psd_beta","psd_gamma"]
    band_labels = {"psd_delta":"Delta (0.5–4 Hz)","psd_theta":"Theta (4–8 Hz)",
                   "psd_alpha":"Alpha (8–13 Hz)","psd_beta":"Beta (13–30 Hz)",
                   "psd_gamma":"Gamma (30–50 Hz)"}

    # Simple bar via table cell width proportional to value
    band_vals = {b: float(eeg_features.get(b, 0)) for b in bands}
    max_val   = max(band_vals.values()) or 1.0

    bp_header = ["Band", "Power (µV²/Hz)", "Relative Strength", "Clinical Significance"]
    bp_rows   = [bp_header]
    clin_notes = {
        "psd_delta": "Sleep regulation, restorative processes",
        "psd_theta": "Memory, emotion, drowsiness",
        "psd_alpha": "Relaxation, inhibition — ↓ linked to depression",
        "psd_beta":  "Active thinking, focus, alertness",
        "psd_gamma": "Higher cognitive function, perception",
    }
    for b in bands:
        v = band_vals[b]
        pct = int((v / max_val) * 100)
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
        bp_rows.append([band_labels[b], f"{v:.4f}", f"{bar} {pct}%", clin_notes[b]])

    bp_col_w = [PAGE_W*0.22, PAGE_W*0.18, PAGE_W*0.22, PAGE_W*0.38]
    bp_table = Table(bp_rows, colWidths=bp_col_w)
    bp_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0),  DARK_GOLD),
        ("TEXTCOLOR",   (0,0),(-1,0),  WHITE),
        ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,0),  8),
        ("ALIGN",       (0,0),(-1,0),  "CENTER"),
        ("FONTSIZE",    (0,1),(-1,-1), 8),
        ("FONTNAME",    (0,1),(0,-1),  "Helvetica-Bold"),
        ("GRID",        (0,0),(-1,-1), 0.3, GRAY_LIGHT),
        ("PADDING",     (0,0),(-1,-1), 5),
        ("FONTNAME",    (2,1),(2,-1),  "Courier"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, colors.HexColor("#F8F8F8")]),
    ]))
    story.append(bp_table)
    story.append(Spacer(1, 5*mm))

    # ── SECTION 5 — CLINICAL INTERPRETATION ───────────────────────────────────
    story.append(Paragraph("SECTION 5 — CLINICAL INTERPRETATION", S["section"]))
    story.append(ColorHR(PAGE_W, GRAY_LIGHT, 0.5, 2))

    alpha_val = float(eeg_features.get("psd_alpha", 0))
    delta_val = float(eeg_features.get("psd_delta", 0))
    asym_val  = float(eeg_features.get("alpha_asymmetry", 0))
    avg_coh   = sum(float(eeg_features.get(k, 0)) for k in FEATURE_ORDER if k.startswith("coh_")) / 10

    interp_lines = []

    # Prediction interpretation
    if is_mdd:
        interp_lines.append(
            f"<b>Primary Finding:</b> The EEG Digital Twin model (SVM, RBF kernel; IEEE Eq. 7–8) "
            f"has classified this EEG sample as indicative of <b>Major Depressive Disorder (MDD)</b> "
            f"with a probability of <b>{mdd_prob*100:.1f}%</b> and model confidence of "
            f"<b>{confidence*100:.1f}%</b>. This classification is based on learned feature patterns "
            f"from the training dataset and corroborated by the following biomarkers:"
        )
    else:
        interp_lines.append(
            f"<b>Primary Finding:</b> The EEG Digital Twin model has classified this EEG sample as "
            f"<b>Healthy (No MDD)</b> with a confidence of <b>{confidence*100:.1f}%</b>. "
            f"MDD probability is <b>{mdd_prob*100:.1f}%</b>, below the clinical alert threshold of 70%."
        )

    # Alpha band
    if alpha_val < 1.0:
        interp_lines.append(
            f"<b>Alpha Power ({alpha_val:.3f} µV²/Hz — Reduced):</b> Diminished alpha activity is a "
            f"well-documented neurophysiological biomarker of depression, associated with reduced "
            f"cortical inhibition and altered frontal lobe function."
        )
    elif alpha_val > 3.5:
        interp_lines.append(
            f"<b>Alpha Power ({alpha_val:.3f} µV²/Hz — Elevated):</b> High alpha power suggests "
            f"cortical idling or relaxation states, inconsistent with active depressive rumination."
        )
    else:
        interp_lines.append(
            f"<b>Alpha Power ({alpha_val:.3f} µV²/Hz — Normal):</b> Within the expected reference range. "
            f"Alpha activity reflects normal cortical inhibitory function."
        )

    # Delta band
    if delta_val > 3.5:
        interp_lines.append(
            f"<b>Delta Power ({delta_val:.3f} µV²/Hz — Elevated):</b> High delta activity at rest "
            f"may indicate sleep disruption, cognitive slowing, or cortical hyperactivation — "
            f"commonly observed in depressive disorders."
        )

    # Alpha asymmetry
    if asym_val > 0.15:
        interp_lines.append(
            f"<b>Alpha Asymmetry ({asym_val:.3f} — Right-dominant):</b> Frontal alpha asymmetry "
            f"with right-side dominance is an established neural correlate of depression, associated "
            f"with reduced approach motivation and heightened withdrawal behaviour."
        )
    elif abs(asym_val) <= 0.15:
        interp_lines.append(
            f"<b>Alpha Asymmetry ({asym_val:.3f} — Balanced):</b> Frontal alpha symmetry is within "
            f"normal limits, suggesting balanced hemispheric activation patterns."
        )

    # Coherence
    if avg_coh < 0.55:
        interp_lines.append(
            f"<b>Inter-hemispheric Coherence (avg {avg_coh:.3f} — Low):</b> Reduced coherence between "
            f"electrode pairs may indicate disrupted long-range cortical connectivity, a feature "
            f"consistently reported in MDD neuroimaging literature."
        )
    else:
        interp_lines.append(
            f"<b>Inter-hemispheric Coherence (avg {avg_coh:.3f} — Normal):</b> Adequate connectivity "
            f"between electrode pairs. No significant coherence reduction detected."
        )

    # Rolling risk
    if float(rolling_risk) > 60:
        interp_lines.append(
            f"<b>Rolling MDD Risk ({rolling_risk:.1f}%):</b> A sustained MDD classification rate "
            f"above 60% across the monitoring window is clinically significant and warrants "
            f"immediate psychiatric evaluation."
        )

    for line in interp_lines:
        story.append(Paragraph(line, S["body"]))
        story.append(Spacer(1, 2.5*mm))

    story.append(Spacer(1, 2*mm))

    # ── SECTION 6 — RECOMMENDATIONS ───────────────────────────────────────────
    story.append(Paragraph("SECTION 6 — CLINICAL RECOMMENDATIONS", S["section"]))
    story.append(ColorHR(PAGE_W, GRAY_LIGHT, 0.5, 2))

    if is_mdd:
        recs = [
            ("Urgent Psychiatric Referral",
             "MDD probability exceeds clinical threshold. Immediate referral to a licensed "
             "psychiatrist or clinical psychologist is strongly advised for formal DSM-5/ICD-11 evaluation."),
            ("Confirmatory Clinical Interview",
             "EEG-based screening should be supplemented with a structured clinical interview "
             "(e.g., PHQ-9, HAMD) and comprehensive medical history review."),
            ("Longitudinal EEG Monitoring",
             "Repeat EEG assessment in 4–6 weeks to monitor trajectory. The digital twin's "
             "rolling risk feature enables continuous longitudinal tracking."),
            ("Pharmacotherapy Evaluation",
             "If clinical diagnosis is confirmed, evidence-based antidepressant therapy "
             "(SSRIs, SNRIs) should be considered per local prescribing guidelines."),
            ("Psychosocial Intervention",
             "Cognitive Behavioural Therapy (CBT) or Mindfulness-Based Cognitive Therapy (MBCT) "
             "should be integrated into the management plan."),
        ] if alert else [
            ("Psychiatric Referral",
             "MDD probability is moderate. Clinical review is recommended to rule out subsyndromal "
             "depressive symptoms."),
            ("Follow-Up EEG Assessment",
             "Schedule repeat EEG monitoring in 8–12 weeks to establish trend baseline."),
            ("Supportive Psychotherapy",
             "Consider psychosocial support and stress management interventions as preventive measures."),
            ("Patient Education",
             "Educate patient on sleep hygiene, physical activity, and mood monitoring techniques."),
        ]
    else:
        recs = [
            ("Routine Follow-Up",
             "No depressive EEG pattern detected. Routine follow-up as per standard clinical protocol."),
            ("Annual EEG Screening",
             "Consider annual EEG digital twin screening as part of preventive mental health monitoring "
             "especially for high-risk populations."),
            ("Wellness Maintenance",
             "Maintain current lifestyle factors: sleep quality, physical activity, social engagement — "
             "all positively correlated with healthy EEG signatures."),
            ("Patient Counselling",
             "Reinforce positive mental health behaviours and provide patient with self-monitoring resources."),
        ]

    rec_data = [["#", "Recommendation", "Details"]]
    for i, (title, detail) in enumerate(recs, 1):
        rec_data.append([str(i), title, detail])

    rec_col_w = [PAGE_W*0.04, PAGE_W*0.24, PAGE_W*0.72]
    rec_table = Table(rec_data, colWidths=rec_col_w)
    rec_style = TableStyle([
        ("BACKGROUND",  (0,0),(-1,0),  DARK_GOLD),
        ("TEXTCOLOR",   (0,0),(-1,0),  WHITE),
        ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,0),  8),
        ("ALIGN",       (0,0),(-1,0),  "CENTER"),
        ("FONTSIZE",    (0,1),(-1,-1), 8),
        ("FONTNAME",    (1,1),(1,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (1,1),(1,-1),  DARK_GOLD),
        ("VALIGN",      (0,0),(-1,-1), "TOP"),
        ("GRID",        (0,0),(-1,-1), 0.3, GRAY_LIGHT),
        ("PADDING",     (0,0),(-1,-1), 6),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, colors.HexColor("#F8F8F8")]),
        ("ALIGN",       (0,1),(0,-1),  "CENTER"),
    ])
    rec_table.setStyle(rec_style)
    story.append(rec_table)
    story.append(Spacer(1, 5*mm))

    # ── SECTION 7 — CLINICAL NOTES (OPTIONAL) ─────────────────────────────────
    notes = patient_info.get("clinical_notes", "").strip()
    if notes:
        story.append(Paragraph("SECTION 7 — CLINICIAN NOTES", S["section"]))
        story.append(ColorHR(PAGE_W, GRAY_LIGHT, 0.5, 2))
        notes_table = Table(
            [[Paragraph(notes, S["body"])]],
            colWidths=[PAGE_W],
        )
        notes_table.setStyle(TableStyle([
            ("BOX",        (0,0),(-1,-1), 0.5, GRAY_LIGHT),
            ("BACKGROUND", (0,0),(-1,-1), colors.HexColor("#FAFAFA")),
            ("PADDING",    (0,0),(-1,-1), 10),
        ]))
        story.append(notes_table)
        story.append(Spacer(1, 5*mm))

    # ── DISCLAIMER + FOOTER ────────────────────────────────────────────────────
    story.append(ColorHR(PAGE_W, GOLD, 1.0, 3))
    disclaimer = (
        "<b>IMPORTANT DISCLAIMER:</b> This report is generated by an AI-based EEG Digital Twin system "
        "for research and screening purposes only. It does <b>NOT</b> constitute a formal clinical "
        "diagnosis. All findings must be interpreted by a qualified medical professional. "
        "The model is based on the IEEE paper: <i>'Cloud-Based EEG Digital Twin Framework for Real-Time "
        "Mental Monitoring Using Machine Learning'</i> — Salaria, Verma, Deepika D. — "
        "SRM Institute of Science and Technology."
    )
    story.append(Paragraph(disclaimer, S["small"]))
    story.append(Spacer(1, 3*mm))

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(
        f"Report generated: {gen_time} · EEG Digital Twin v1.0 · "
        f"Algorithm: SVM RBF Kernel · Features: 18 · Inference: {inference_ms:.1f} ms",
        S["footer"]
    ))

    # ── BUILD ──────────────────────────────────────────────────────────────────
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    BASE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, BASE)

    # Try to load the real twin; otherwise use mock data
    try:
        from digital_twin import EEGDigitalTwin
        from eeg_data import generate_eeg_features
        twin = EEGDigitalTwin()
        row  = generate_eeg_features(1).drop(columns=["label"]).iloc[0].to_dict()
        result = twin.update_state(row)
    except Exception as e:
        print(f"[WARNING] Could not load twin ({e}) — using mock data.")
        row = {
            "psd_delta":1.2,"psd_theta":0.9,"psd_alpha":0.7,"psd_beta":1.4,"psd_gamma":0.5,
            "alpha_asymmetry":0.22,
            "coh_Fp1_Fp2":0.42,"coh_F3_F4":0.40,"coh_C3_C4":0.44,"coh_P3_P4":0.41,
            "coh_O1_O2":0.43,"coh_Fp1_F3":0.50,"coh_Fp2_F4":0.51,"coh_F3_C3":0.49,
            "coh_F4_C4":0.50,"coh_C3_P3":0.48,"age":42,"gender":0,
        }
        result = {
            "timestamp":"2026-04-15T10:00:00Z","prediction":"MDD",
            "mdd_probability":0.83,"confidence":0.83,
            "inference_ms":12.5,"rolling_risk":65.0,"alert":True,
        }

    patient = {
        "patient_name":   "John Doe",
        "patient_id":     "PT-20260415-001",
        "doctor_name":    "Dr. Priya Sharma",
        "hospital":       "SRM Medical Centre, Chennai",
        "report_date":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "clinical_notes": "Patient reports 6-week history of persistent low mood, "
                          "poor sleep, and reduced concentration. No prior psychiatric history.",
    }

    pdf_bytes = generate_clinical_report(patient, row, result)
    out_path  = os.path.join(BASE, "test_clinical_report.pdf")
    with open(out_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"\n✅  Clinical report saved → {out_path}")
    print(f"   Size: {len(pdf_bytes):,} bytes")
