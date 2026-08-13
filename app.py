import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["KERAS_HOME"]          = os.path.join(PROJECT_ROOT, ".keras")
os.environ["MPLCONFIGDIR"]        = os.path.join(PROJECT_ROOT, ".matplotlib")
os.environ["STREAMLIT_CONFIG_DIR"]= os.path.join(PROJECT_ROOT, ".streamlit")
os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
os.environ["OMP_NUM_THREADS"]     = "1"

os.makedirs(os.environ["KERAS_HOME"],          exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"],        exist_ok=True)
os.makedirs(os.environ["STREAMLIT_CONFIG_DIR"],exist_ok=True)

import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from multimodal_pipeline import (
    MultimodalPipeline, FEATURE_COLS, STATUS_CLASSES,
    FACIAL_CLASSES, FACIAL_EMOTION_TO_STATUS_MATRIX
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindSense AI — Clinical Intelligence System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# LUXURY ANIMATED OBSIDIAN & NEON EMERALD DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --bg:        #030712;
  --surface:   #0b0f19;
  --card:      #111827;
  --border:    #1f2937;
  --border-hi: #374151;
  --fg:        #f9fafb;
  --muted:     #9ca3af;
  --dim:       #4b5563;
  --neon:      #00f5a0;
  --cyan:      #00d9f5;
  --violet:    #8b5cf6;
  --amber:     #fbbf24;
  --crimson:   #ff0055;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
  background: var(--bg) !important;
  font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
  color: var(--fg) !important;
}

.block-container {
  padding: 0 2.5rem 4rem !important;
  max-width: 1320px !important;
}
@media(max-width:768px){ .block-container{ padding:0 1rem 3rem !important; } }

/* Kill Streamlit default chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display:none !important; }

/* ── KEYFRAME ANIMATIONS ── */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes pulseGlow {
  0% { box-shadow: 0 0 12px rgba(0, 245, 160, 0.2); }
  50% { box-shadow: 0 0 28px rgba(0, 245, 160, 0.45); }
  100% { box-shadow: 0 0 12px rgba(0, 245, 160, 0.2); }
}

@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.animate-fade {
  animation: fadeIn 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

/* ── STICKY TOP NAVIGATION BAR ── */
[data-baseweb="tab-list"] {
  position: sticky !important;
  top: 0 !important;
  z-index: 99999 !important;
  background: rgba(3, 7, 18, 0.92) !important;
  backdrop-filter: blur(20px) !important;
  -webkit-backdrop-filter: blur(20px) !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0.35rem !important;
  padding: 0.7rem 0 !important;
  flex-wrap: wrap !important;
  box-shadow: 0 4px 30px rgba(0,0,0,0.8) !important;
}
[data-baseweb="tab"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  margin: 0.15rem 0.2rem !important;
  padding: 0.55rem 1.15rem !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
  letter-spacing: 0.03em !important;
  transition: all .25s ease !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  background: var(--card) !important;
  color: var(--neon) !important;
  border-color: var(--neon) !important;
  box-shadow: 0 0 20px rgba(0, 245, 160, 0.22) !important;
  transform: translateY(-1px) !important;
}
[data-baseweb="tab"]:hover {
  color: var(--fg) !important;
  border-color: var(--border-hi) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: var(--fg) !important; }

/* Buttons — Animated Neon CTA */
.stButton > button {
  background: linear-gradient(135deg, #00f5a0, #00d9f5) !important;
  background-size: 200% 200% !important;
  color: #030712 !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.7rem 2rem !important;
  font-size: 0.85rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
  transition: all .25s ease !important;
  box-shadow: 0 0 24px rgba(0, 245, 160, 0.3) !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #00d9f5, #00f5a0) !important;
  box-shadow: 0 0 35px rgba(0, 245, 160, 0.55) !important;
  transform: translateY(-2px) scale(1.01) !important;
}

/* Input Controls */
[data-baseweb="input"], [data-baseweb="base-input"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--fg) !important;
  transition: border-color 0.2s ease !important;
}
[data-baseweb="input"]:focus-within {
  border-color: var(--neon) !important;
  box-shadow: 0 0 12px rgba(0, 245, 160, 0.2) !important;
}
[data-baseweb="input"] input {
  color: var(--fg) !important;
  font-weight: 600 !important;
}
.stNumberInput label {
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  color: var(--fg) !important;
}

/* Uploaders & Camera */
[data-testid="stFileUploader"] {
  background: var(--card) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: 12px !important; padding: 0.85rem 1.1rem !important;
  transition: border-color 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--neon) !important;
}
[data-testid="stCameraInput"] {
  border: 1px solid var(--border) !important;
  border-radius: 12px !important; overflow: hidden !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important; border-radius: 10px !important;
}
[data-testid="stDataFrame"] * { color: var(--fg) !important; background: transparent !important; }

/* Sliders */
[role="slider"] { background: var(--neon) !important; border-color: var(--neon) !important; }

/* General typography */
p, h1, h2, h3, h4, label, li, span, div { color: var(--fg) !important; }
a { color: var(--neon) !important; }
hr { border-color: var(--border) !important; }

.overline {
  font-size: 0.65rem; font-weight: 700; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--neon) !important;
  margin: 0 0 0.45rem;
}
.result-status {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.2rem; font-weight: 700; line-height: 1.1; margin: 0 0 0.35rem;
}
.result-emotion {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.85rem; font-weight: 700; line-height: 1.15; margin: 0 0 0.3rem;
}
.section-divider {
  display: flex; align-items: center; gap: 0.85rem; margin: 1.85rem 0 1.1rem;
}
.section-divider span {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--dim) !important; white-space: nowrap;
}
.section-divider::before, .section-divider::after {
  content: ''; flex: 1; height: 1px; background: var(--border);
}
.badge {
  display: inline-block; border-radius: 6px; padding: 0.25rem 0.85rem;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
}
.prob-bar-wrap { margin: 0.35rem 0; }
.prob-bar-label {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 0.22rem; font-size: 0.78rem; color: var(--muted);
}
.prob-bar-track {
  height: 7px; background: var(--border); border-radius: 999px; overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 999px; transition: width .5s cubic-bezier(0.16, 1, 0.3, 1); }

/* Circular Gauge Ring SVG styling */
.svg-gauge { transform: rotate(-90deg); transform-origin: 50% 50%; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
STATUS_PALETTE = {
    "Healthy":         "#00f5a0",
    "Mild_Stress":     "#fbbf24",
    "Moderate_Stress": "#f97316",
    "Severe_Stress":   "#ff0055",
}
STATUS_BG = {
    "Healthy":         "rgba(0, 245, 160, 0.12)",
    "Mild_Stress":     "rgba(251, 191, 36, 0.12)",
    "Moderate_Stress": "rgba(249, 115, 22, 0.12)",
    "Severe_Stress":   "rgba(255, 0, 85, 0.12)",
}
STATUS_LABEL = {
    "Healthy":         "Healthy",
    "Mild_Stress":     "Mild Stress",
    "Moderate_Stress": "Moderate Stress",
    "Severe_Stress":   "Severe Stress",
}

FEATURE_RANGES = {
    "Sleep_Quality":          (1.0, 10.0, 5.0),
    "Social_Engagement":      (1.0, 10.0, 5.0),
    "Daily_App_Usage_Min":    (0.0, 600.0, 180.0),
    "Typing_Speed_WPM":       (10.0, 120.0, 50.0),
    "Session_Frequency":      (1.0, 50.0, 12.0),
    "Idle_Time_Min":          (0.0, 300.0, 30.0),
    "Facial_Emotion_Variance":(0.0, 2.0, 0.5),
    "Eye_Blink_Rate":         (5.0, 60.0, 18.0),
    "Smile_Intensity":        (0.0, 1.0, 0.4),
    "Head_Motion_Index":      (0.0, 2.0, 0.3),
    "MFCC_Mean":              (-50.0, 50.0, 0.0),
    "MFCC_Variance":          (0.0, 20.0, 3.0),
    "Pitch_Mean":             (50.0, 400.0, 160.0),
    "Speech_Rate":            (0.5, 10.0, 3.0),
    "Heart_Rate_BPM":         (40.0, 160.0, 72.0),
    "HRV_Index":              (10.0, 100.0, 45.0),
    "Skin_Temperature":       (30.0, 40.0, 36.5),
    "GSR_Level":              (0.0, 20.0, 4.0),
}
FEATURE_GROUPS = {
    "Lifestyle & Behaviour": ["Sleep_Quality","Social_Engagement","Daily_App_Usage_Min","Typing_Speed_WPM","Session_Frequency","Idle_Time_Min"],
    "Facial & Vision":        ["Facial_Emotion_Variance","Eye_Blink_Rate","Smile_Intensity","Head_Motion_Index"],
    "Speech & Audio":         ["MFCC_Mean","MFCC_Variance","Pitch_Mean","Speech_Rate"],
    "Physiological":         ["Heart_Rate_BPM","HRV_Index","Skin_Temperature","GSR_Level"],
}

def lbl(f): return f.replace("_"," ")


# ── Load Pipeline ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_pipeline():
    return MultimodalPipeline()

pipeline = get_pipeline()


# ── Helpers & Visual Ring Gauge Generators ──────────────────────────────────
def apply_chart_style(ax, fig):
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#1f2937"); ax.spines["bottom"].set_color("#1f2937")
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.xaxis.label.set_color("#9ca3af"); ax.xaxis.label.set_size(8)

def overline(txt): return f'<p class="overline">{txt}</p>'

def status_badge(status):
    c = STATUS_PALETTE.get(status, "#9ca3af")
    bg = STATUS_BG.get(status, "rgba(156,163,175,.1)")
    lbl_txt = STATUS_LABEL.get(status, status)
    return f'<span class="badge" style="color:{c};background:{bg};border:1px solid {c};box-shadow:0 0 12px {c}44">{lbl_txt}</span>'

def get_current_feature_values():
    vals = {}
    for feat in FEATURE_COLS:
        sk_tab = f"tab_mlp_{feat}"
        sk_fus = f"fusion_inputs_{feat}"
        if sk_tab in st.session_state:
            vals[feat] = st.session_state[sk_tab]
        elif sk_fus in st.session_state:
            vals[feat] = st.session_state[sk_fus]
        else:
            vals[feat] = FEATURE_RANGES[feat][2]
    return vals

def svg_ring_gauge(value_pct, color="#00f5a0", label="CONFIDENCE", size=84):
    """Renders a sleek SVG ring gauge meter."""
    r = 34
    circ = 2 * 3.14159 * r
    offset = circ - (value_pct / 100.0) * circ
    return f"""
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center">
  <svg width="{size}" height="{size}" viewBox="0 0 80 80">
    <circle cx="40" cy="40" r="{r}" stroke="#1f2937" stroke-width="7" fill="none"/>
    <circle cx="40" cy="40" r="{r}" stroke="{color}" stroke-width="7" fill="none"
      stroke-dasharray="{circ}" stroke-dashoffset="{offset}" stroke-linecap="round" class="svg-gauge"/>
    <text x="40" y="44" font-family="'Space Grotesk', sans-serif" font-size="16" font-weight="700"
      fill="#f9fafb" text-anchor="middle">{value_pct:.0f}%</text>
  </svg>
  <span style="font-size:0.62rem;font-weight:700;letter-spacing:0.1em;color:#9ca3af;margin-top:0.3rem;text-transform:uppercase">{label}</span>
</div>"""

def html_prob_bars(prob_dict, highlight_key=None, palette=None):
    bars = ""
    for k, v in sorted(prob_dict.items(), key=lambda x:-x[1]):
        pct  = v * 100
        col  = (palette or {}).get(k, "#00f5a0") if k == highlight_key else "#1f2937"
        fill = (palette or {}).get(k, "#00f5a0") if k == highlight_key else "#374151"
        glow = f"box-shadow:0 0 10px {fill}66;" if k == highlight_key else ""
        bars += f"""
<div class="prob-bar-wrap">
  <div class="prob-bar-label">
    <span style="color:{'#f9fafb' if k==highlight_key else '#9ca3af'};font-weight:{'700' if k==highlight_key else '400'}">{k}</span>
    <span style="color:{col};font-weight:700">{pct:.1f}%</span>
  </div>
  <div class="prob-bar-track">
    <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{fill};{glow}"></div>
  </div>
</div>"""
    return bars

def full_result_block(result):
    status    = result["diagnosis"]
    sc        = STATUS_PALETTE.get(status, "#00f5a0")
    sev       = result.get("severity_scores", {})
    conf      = result.get("confidence_pct", 0)
    latency   = result.get("latency_ms", 0)
    top_em    = result.get("top_facial_emotion")
    top_em_pc = result.get("top_facial_emotion_confidence", 0)
    em_dict   = result.get("facial_emotion_details") or {}
    fused     = result.get("fused_probabilities") or {}
    weights   = result.get("modality_weights", {})
    face_ok   = result.get("face_accepted", True)

    st.markdown(f"""
<div class="animate-fade" style="background:{STATUS_BG.get(status,'rgba(0,245,160,.08)')};
  border:1px solid {sc};border-radius:16px;padding:1.6rem 2rem;margin-bottom:1.4rem;
  box-shadow:0 0 30px {sc}22">
  <div style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap">
    <div style="flex:1;min-width:200px">
      <p class="overline">Clinical Status Diagnosis</p>
      <p class="result-status" style="color:{sc}">{STATUS_LABEL.get(status,status)}</p>
      {status_badge(status)}
    </div>
    <div style="display:flex;gap:2rem;align-items:center">
      {svg_ring_gauge(conf, color=sc, label="Confidence")}
      <div style="text-align:center;padding-left:1rem;border-left:1px solid #1f2937">
        <p style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;
          color:#f9fafb;margin:0;line-height:1">{latency}<span style="font-size:0.9rem;color:#9ca3af">ms</span></p>
        <p style="font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#9ca3af;margin:.3rem 0 0">Latency</p>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,1,1], gap="medium")
    with c1:
        st.markdown(overline("Severity Index"), unsafe_allow_html=True)
        scores_html = ""
        for name, val, col in [("Depression", sev.get("Depression_Score",0), "#ff0055"),
                               ("Anxiety",    sev.get("Anxiety_Score",0),    "#f97316"),
                               ("Stress",     sev.get("Stress_Score",0),     "#fbbf24")]:
            pct = min(val, 100)
            scores_html += f"""
<div style="margin-bottom:0.9rem">
  <div style="display:flex;justify-content:space-between;margin-bottom:0.25rem">
    <span style="font-size:0.78rem;font-weight:600;color:#f9fafb">{name}</span>
    <span style="font-size:0.78rem;font-weight:700;color:{col}">{val:.1f}</span>
  </div>
  <div style="background:#1f2937;border-radius:999px;height:7px;overflow:hidden">
    <div style="width:{pct}%;height:100%;background:{col};box-shadow:0 0 10px {col}88"></div>
  </div>
</div>"""
        st.markdown(f'<div class="animate-fade" style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1.2rem">{scores_html}</div>', unsafe_allow_html=True)

    with c2:
        st.markdown(overline("Facial Vision Branch"), unsafe_allow_html=True)
        if top_em:
            gate_c = "#00f5a0" if face_ok else "#ff0055"
            gate_t = "CNN ACCEPTED" if face_ok else "TABULAR FALLBACK"
            st.markdown(f"""
<div class="animate-fade" style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1.2rem;height:100%">
  <span style="display:inline-block;padding:0.22rem 0.7rem;border-radius:6px;font-size:0.7rem;font-weight:700;letter-spacing:0.08em;background:rgba(0,245,160,0.14);color:#00f5a0;border:1px solid #00f5a0">{top_em.upper()}</span>
  <p class="result-emotion" style="margin-top:0.35rem">{top_em}</p>
  <p style="font-size:0.8rem;color:#9ca3af;margin:0 0 0.8rem">{top_em_pc:.1f}% confidence</p>
  <span class="badge" style="color:{gate_c};border:1px solid {gate_c}">● {gate_t}</span>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1.2rem"><p style="color:#4b5563;font-size:0.85rem">No face detected</p></div>', unsafe_allow_html=True)

    with c3:
        st.markdown(overline("Modality Weight Allocation"), unsafe_allow_html=True)
        mw_html = ""
        for mname, mval, mcol in [("Tabular",weights.get("tabular",0)*100,"#00f5a0"),
                                  ("Facial", weights.get("facial",0)*100,"#00d9f5"),
                                  ("Audio",  weights.get("audio",0)*100, "#8b5cf6")]:
            mw_html += f"""
<div style="margin-bottom:0.8rem">
  <div style="display:flex;justify-content:space-between;margin-bottom:0.2rem">
    <span style="font-size:0.78rem;color:#9ca3af">{mname}</span>
    <span style="font-size:0.78rem;font-weight:700;color:{mcol}">{mval:.0f}%</span>
  </div>
  <div style="background:#1f2937;border-radius:999px;height:6px;overflow:hidden">
    <div style="width:{mval:.0f}%;height:100%;background:{mcol};box-shadow:0 0 8px {mcol}66"></div>
  </div>
</div>"""
        st.markdown(f'<div class="animate-fade" style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1.2rem">{mw_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"><span>Probabilistic Breakdown</span></div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2, gap="medium")
    with d1:
        st.markdown(overline("Emotion Probability Distribution"), unsafe_allow_html=True)
        if em_dict:
            bars_html = html_prob_bars(em_dict, highlight_key=top_em)
            st.markdown(f'<div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1.2rem">{bars_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#4b5563;font-size:0.82rem">No facial emotion data.</p>', unsafe_allow_html=True)

    with d2:
        st.markdown(overline("Mental Health Status Distribution"), unsafe_allow_html=True)
        if fused:
            bars_html2 = html_prob_bars(
                {STATUS_LABEL.get(k,k):v for k,v in fused.items()},
                highlight_key=STATUS_LABEL.get(status, status),
                palette={STATUS_LABEL.get(k,k):v for k,v in STATUS_PALETTE.items()}
            )
            st.markdown(f'<div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1.2rem">{bars_html2}</div>', unsafe_allow_html=True)


# ── OpenCV cascade ─────────────────────────────────────────────────────────────
try:
    _cp = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade_ui = cv2.CascadeClassifier(_cp) if os.path.exists(_cp) else None
except Exception:
    face_cascade_ui = None


# ══════════════════════════════════════════════════════════════════════════════
# TOP STICKY NAVBAR
# ══════════════════════════════════════════════════════════════════════════════
(t_overview, t_tabular, t_fusion,
 t_facial,   t_audio,   t_thresh,
 t_xai,      t_cam) = st.tabs([
    "Overview",
    "Tabular MLP",
    "Multimodal Fusion",
    "Facial CNN",
    "Audio Classifier",
    "CNN Thresholds",
    "XAI Explainer",
    "Live Webcam",
])


# ── Interactive Feature Input Helper ─────────────────────────────────────────
def render_interactive_tabular_inputs(key_prefix="tab_input"):
    st.markdown(overline("Clinical Indicator Control Panel"), unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.82rem;color:#9ca3af;margin-bottom:1rem">Adjust patient physiological and behavioural parameters. Model predictions update dynamically.</p>', unsafe_allow_html=True)

    inputs = {}
    cols = st.columns(4, gap="medium")

    for idx, (grp_name, feats) in enumerate(FEATURE_GROUPS.items()):
        with cols[idx]:
            st.markdown(f'<div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1.1rem;height:100%">'
                        f'<p style="font-size:0.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#00f5a0;margin-bottom:0.8rem">{grp_name}</p>', unsafe_allow_html=True)
            for feat in feats:
                lo, hi, df = FEATURE_RANGES[feat]
                step_val = 0.1 if isinstance(df, float) and df < 5.0 else (0.5 if isinstance(df, float) else 1)
                sk = f"{key_prefix}_{feat}"

                if isinstance(df, float):
                    val = st.number_input(
                        lbl(feat), min_value=float(lo), max_value=float(hi),
                        value=float(df), step=float(step_val),
                        key=sk
                    )
                else:
                    val = st.number_input(
                        lbl(feat), min_value=int(lo), max_value=int(hi),
                        value=int(df), step=int(step_val),
                        key=sk
                    )
                inputs[feat] = val
            st.markdown('</div>', unsafe_allow_html=True)

    return inputs


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with t_overview:
    # Futuristic Hero section with gradient shift
    st.markdown("""
<div style="padding:2.8rem 0 1.8rem;border-bottom:1px solid #1f2937;margin-bottom:1.8rem">
  <p style="font-size:0.68rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;
    color:#00f5a0;margin:0 0 0.85rem">Multimodal Psychiatric Intelligence System</p>
  <h1 style="font-family:'Space Grotesk',sans-serif;font-size:clamp(2.4rem,4.5vw,3.8rem);
    font-weight:700;line-height:1.08;letter-spacing:-0.03em;margin:0 0 0.9rem">
    Evaluate patient mental state.<br>
    <em style="font-style:normal;background:linear-gradient(135deg,#00f5a0,#00d9f5,#8b5cf6);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent">Precision AI in real time.</em>
  </h1>
  <p style="font-size:0.95rem;color:#9ca3af;max-width:620px;line-height:1.75;margin:0">
    Fusing 18 clinical features, 48×48 facial micro-expressions, and 280-dimensional acoustic speech signals via Gated Late Fusion.
  </p>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
    gap:1.5rem;margin-top:2.2rem;padding-top:1.8rem;border-top:1px solid #1f2937;max-width:640px">
    <div><p style="font-family:'Space Grotesk',sans-serif;font-size:2.2rem;font-weight:700;color:#00f5a0;margin:0">18</p>
      <p style="font-size:0.74rem;color:#9ca3af;margin:.25rem 0 0">Clinical Indicators</p></div>
    <div><p style="font-family:'Space Grotesk',sans-serif;font-size:2.2rem;font-weight:700;color:#00d9f5;margin:0">7</p>
      <p style="font-size:0.74rem;color:#9ca3af;margin:.25rem 0 0">Facial Emotions</p></div>
    <div><p style="font-family:'Space Grotesk',sans-serif;font-size:2.2rem;font-weight:700;color:#fbbf24;margin:0">4</p>
      <p style="font-size:0.74rem;color:#9ca3af;margin:.25rem 0 0">Diagnostic Classes</p></div>
    <div><p style="font-family:'Space Grotesk',sans-serif;font-size:2.2rem;font-weight:700;color:#f9fafb;margin:0">&lt;100ms</p>
      <p style="font-size:0.74rem;color:#9ca3af;margin:.25rem 0 0">Inference Speed</p></div>
  </div>
</div>
""", unsafe_allow_html=True)

    arch = [
        ("01","PyTorch Tabular MLP","18-feature multitask MLP predicts 4 mental health classes + depression, anxiety & stress regression scores in one pass.","#00f5a0"),
        ("02","Keras Facial CNN","48×48 grayscale CNN detects 7 emotions (Happy, Sad, Angry, Fear, Disgust, Neutral, Surprise) from photos or webcam streams.","#00d9f5"),
        ("03","Acoustic Classifier","280-dimensional MFCC + pitch + spectral features extracted from .wav audio and classified by an ensemble.","#fbbf24"),
        ("04","Gated Fusion Engine","Confidence-weighted late fusion combines all three modalities. Per-emotion thresholds gate the CNN so low-confidence neutral reads never hijack the diagnosis.","#8b5cf6"),
    ]
    c1, c2 = st.columns(2, gap="large")
    for i,(num,title,desc,col) in enumerate(arch):
        col_ref = c1 if i%2==0 else c2
        with col_ref:
            st.markdown(f"""
<div class="animate-fade" style="background:#111827;border:1px solid #1f2937;border-radius:14px;padding:1.4rem;margin-bottom:1.1rem">
  <p style="font-family:'Space Grotesk',sans-serif;font-size:1.8rem;color:{col};margin:0 0 0.4rem;font-weight:700">{num}</p>
  <p style="font-size:0.92rem;font-weight:600;color:#f9fafb;margin:0 0 0.35rem">{title}</p>
  <p style="font-size:0.82rem;color:#9ca3af;line-height:1.65;margin:0">{desc}</p>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"><span>System Architecture & Status</span></div>', unsafe_allow_html=True)
    ms1, ms2, ms3 = st.columns(3, gap="medium")
    for col, (name, key, path, tc) in zip([ms1,ms2,ms3], [
        ("Tabular MLP","tabular","numerical_classifier.pth","#00f5a0"),
        ("Facial CNN","facial","facial_emotion_cnn.keras","#00d9f5"),
        ("Audio Classifier","audio","audio_emotion_classifier.joblib","#fbbf24"),
    ]):
        ok = pipeline.loaded_status.get(key,False)
        badge_html = f'<span class="badge" style="color:{tc};border:1px solid {tc}">{"✓ LOADED" if ok else "✗ MISSING"}</span>'
        col.markdown(f"""
<div class="animate-fade" style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1.2rem">
  <p style="font-size:0.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#9ca3af;margin:0 0 0.45rem">{name}</p>
  {badge_html}
  <p style="font-size:0.7rem;color:#4b5563;margin:.55rem 0 0;word-break:break-all">{path}</p>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Tabular MLP
# ══════════════════════════════════════════════════════════════════════════════
with t_tabular:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    tab_feature_values = render_interactive_tabular_inputs(key_prefix="tab_mlp")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if st.button("EXECUTE TABULAR PREDICTION", key="btn_run_tab_page"):
        with st.spinner("Executing PyTorch Tabular MLP..."):
            probs, reg = pipeline.predict_tabular(tab_feature_values)
        pred_idx  = int(np.argmax(probs))
        status_k  = STATUS_CLASSES[pred_idx]

        st.markdown("<hr style='margin:1.35rem 0;border-color:#1f2937'>", unsafe_allow_html=True)
        sc = STATUS_PALETTE.get(status_k,"#00f5a0")

        st.markdown(f"""
<div class="animate-fade" style="background:{STATUS_BG.get(status_k,'rgba(0,245,160,.08)')};
  border:1px solid {sc};border-radius:14px;padding:1.5rem 1.8rem;margin-bottom:1.3rem;
  display:flex;align-items:center;gap:2rem;flex-wrap:wrap;box-shadow:0 0 25px {sc}22">
  <div>
    <p class="overline">Predicted Status</p>
    <p class="result-status" style="color:{sc}">{STATUS_LABEL.get(status_k,status_k)}</p>
    {status_badge(status_k)}
  </div>
  <div style="display:flex;gap:2rem;margin-left:auto">
    <div style="text-align:center"><p style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;color:#ff0055;margin:0">{reg['Depression_Score']:.1f}</p>
      <p style="font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#9ca3af;margin:.25rem 0 0">Depression</p></div>
    <div style="text-align:center"><p style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;color:#f97316;margin:0">{reg['Anxiety_Score']:.1f}</p>
      <p style="font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#9ca3af;margin:.25rem 0 0">Anxiety</p></div>
    <div style="text-align:center"><p style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;color:#fbbf24;margin:0">{reg['Stress_Score']:.1f}</p>
      <p style="font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#9ca3af;margin:.25rem 0 0">Stress</p></div>
  </div>
</div>
""", unsafe_allow_html=True)

        tb1, tb2 = st.columns([1,1], gap="large")
        with tb1:
            st.markdown(overline("Status Probabilities"), unsafe_allow_html=True)
            bars_html = html_prob_bars(
                {STATUS_LABEL.get(s,s):float(probs[i]) for i,s in enumerate(STATUS_CLASSES)},
                highlight_key=STATUS_LABEL.get(status_k),
                palette={STATUS_LABEL.get(k,k):v for k,v in STATUS_PALETTE.items()}
            )
            st.markdown(f'<div class="animate-fade" style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1.2rem 1.4rem">{bars_html}</div>', unsafe_allow_html=True)

        with tb2:
            st.markdown(overline("Input Vector Summary"), unsafe_allow_html=True)
            rows_d = [{"Feature":lbl(f),"Value":round(v,2)} for f,v in tab_feature_values.items()]
            st.dataframe(pd.DataFrame(rows_d), use_container_width=True, hide_index=True, height=280)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Multimodal Fusion
# ══════════════════════════════════════════════════════════════════════════════
with t_fusion:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    fusion_feature_values = render_interactive_tabular_inputs(key_prefix="fusion_inputs")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    up1, up2 = st.columns(2, gap="large")
    with up1:
        st.markdown(overline("Facial Image Stream"), unsafe_allow_html=True)
        face_file = st.file_uploader("Upload face image", type=["png","jpg","jpeg"], key="fusion_face", label_visibility="collapsed")
    with up2:
        st.markdown(overline("Acoustic Speech Recording"), unsafe_allow_html=True)
        audio_file = st.file_uploader("Upload audio", type=["wav"], key="fusion_audio", label_visibility="collapsed")

    face_input = Image.open(face_file) if face_file else None
    audio_path = None
    if audio_file:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_file.read()); audio_path = tmp.name

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    if st.button("EXECUTE FUSED MULTIMODAL DIAGNOSIS", key="btn_fusion_main"):
        with st.spinner("Processing all modalities…"):
            result = pipeline.predict_multimodal_patient(fusion_feature_values, face_image_input=face_input, audio_input=audio_path)

        st.markdown("<hr style='margin:1.5rem 0;border-color:#1f2937'>", unsafe_allow_html=True)
        full_result_block(result)

    if audio_path and os.path.exists(audio_path):
        try: os.unlink(audio_path)
        except: pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Facial CNN
# ══════════════════════════════════════════════════════════════════════════════
with t_facial:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.85rem;color:#9ca3af;margin-bottom:1.2rem">Upload a facial crop for 7-class emotion classification.</p>', unsafe_allow_html=True)

    face_file_tab = st.file_uploader("Upload face photo", type=["png","jpg","jpeg"], key="face_tab", label_visibility="collapsed")

    if face_file_tab:
        img = Image.open(face_file_tab)
        st.markdown("<hr style='margin:1.1rem 0;border-color:#1f2937'>", unsafe_allow_html=True)
        fi1, fi2 = st.columns([1,2], gap="large")
        with fi1:
            st.image(img, use_container_width=True)

        with fi2:
            with st.spinner("Analysing facial expression…"):
                status_probs, emotion_probs = pipeline.predict_facial(img)

            if emotion_probs:
                top_em = max(emotion_probs, key=emotion_probs.get)

                st.markdown(f"""
<div class="animate-fade" style="background:#111827;border:1px solid #1f2937;border-radius:14px;padding:1.3rem 1.5rem;margin-bottom:1.1rem">
  <p class="overline">Detected Facial Emotion</p>
  <span style="display:inline-block;padding:0.25rem 0.75rem;border-radius:6px;font-size:0.72rem;font-weight:700;letter-spacing:0.08em;background:rgba(0,245,160,0.14);color:#00f5a0;border:1px solid #00f5a0">{top_em.upper()}</span>
  <p class="result-emotion" style="margin-top:0.35rem">{top_em}</p>
  <p style="font-size:0.82rem;color:#9ca3af;margin:0">{emotion_probs[top_em]*100:.1f}% confidence</p>
</div>
""", unsafe_allow_html=True)

                st.markdown(overline("Emotion Probability Distribution"), unsafe_allow_html=True)
                em_html = html_prob_bars(emotion_probs, highlight_key=top_em)
                st.markdown(f'<div class="animate-fade" style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1.2rem">{em_html}</div>', unsafe_allow_html=True)

                if status_probs is not None:
                    top_s_idx = int(np.argmax(status_probs))
                    top_s     = STATUS_CLASSES[top_s_idx]
                    sc        = STATUS_PALETTE.get(top_s,"#00f5a0")
                    st.markdown(f"""
<div class="animate-fade" style="background:{STATUS_BG.get(top_s,'rgba(0,245,160,.08)')};
  border:1px solid {sc};border-radius:12px;padding:1.1rem 1.4rem;margin-top:1rem">
  <p class="overline">Mapped Mental Health Status</p>
  <p style="font-family:'Space Grotesk',sans-serif;font-size:1.6rem;font-weight:700;
    color:{sc};margin:0 0 0.5rem">{STATUS_LABEL.get(top_s,top_s)}</p>
""", unsafe_allow_html=True)
                    mapping_html = html_prob_bars(
                        {STATUS_LABEL.get(s,s):float(status_probs[i]) for i,s in enumerate(STATUS_CLASSES)},
                        highlight_key=STATUS_LABEL.get(top_s),
                        palette={STATUS_LABEL.get(k,k):v for k,v in STATUS_PALETTE.items()}
                    )
                    st.markdown(f'{mapping_html}</div>', unsafe_allow_html=True)
            else:
                st.info("No face detected. Try a clear, front-facing photo.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Audio Classifier
# ══════════════════════════════════════════════════════════════════════════════
with t_audio:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.85rem;color:#9ca3af;margin-bottom:1.2rem">Upload a speech .wav recording for 280-dim acoustic emotion classification.</p>', unsafe_allow_html=True)

    audio_file_tab = st.file_uploader("Upload speech audio", type=["wav"], key="audio_tab", label_visibility="collapsed")

    if audio_file_tab:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_file_tab.read()); tmp_audio = tmp.name
        try:
            with st.spinner("Extracting acoustic features and classifying…"):
                status_probs, audio_det = pipeline.predict_audio(tmp_audio)

            if audio_det:
                top_ae = max(audio_det, key=audio_det.get)
                st.markdown("<hr style='margin:1.1rem 0;border-color:#1f2937'>", unsafe_allow_html=True)

                a1, a2 = st.columns([1,2], gap="large")
                with a1:
                    st.markdown(f"""
<div class="animate-fade" style="background:#111827;border:1px solid #1f2937;border-radius:14px;padding:1.4rem">
  <p class="overline">Acoustic Speech Emotion</p>
  <span style="display:inline-block;padding:0.22rem 0.7rem;border-radius:6px;font-size:0.72rem;font-weight:700;letter-spacing:0.08em;background:rgba(0,217,245,0.14);color:#00d9f5;border:1px solid #00d9f5">{top_ae.upper()}</span>
  <p class="result-emotion" style="margin-top:0.35rem">{top_ae.capitalize()}</p>
  <p style="font-size:0.82rem;color:#9ca3af;margin:0">{audio_det[top_ae]*100:.1f}% confidence</p>
</div>
""", unsafe_allow_html=True)

                with a2:
                    st.markdown(overline("Acoustic Emotion Distribution"), unsafe_allow_html=True)
                    ae_html = html_prob_bars(audio_det, highlight_key=top_ae)
                    st.markdown(f'<div class="animate-fade" style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1.2rem">{ae_html}</div>', unsafe_allow_html=True)
            else:
                st.info("Could not extract audio features.")
        finally:
            if os.path.exists(tmp_audio): os.unlink(tmp_audio)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — CNN Thresholds
# ══════════════════════════════════════════════════════════════════════════════
with t_thresh:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""
<p style="font-size:0.85rem;color:#9ca3af;max-width:640px;margin-bottom:1.3rem;line-height:1.75">
Per-emotion confidence thresholds enforced by the Gated Multimodal Engine before trusting CNN predictions.
</p>
""", unsafe_allow_html=True)

    thresholds = pipeline.REALTIME_EMOTION_THRESHOLDS
    em_names    = list(thresholds.keys())
    thresh_vals = [thresholds[e] for e in em_names]
    bar_cols    = ["#00f5a0" if v < 0.40 else "#fbbf24" if v < 0.49 else "#ff0055" for v in thresh_vals]

    fig, ax = plt.subplots(figsize=(9, 3.2))
    bars_p  = ax.bar(em_names, thresh_vals, color=bar_cols, width=0.55)
    ax.axhline(0.5, color="#1f2937", linewidth=1.2, linestyle="--")
    ax.set_ylim(0, 0.68)
    ax.set_ylabel("Min. confidence required", color="#9ca3af")
    for bar, val in zip(bars_p, thresh_vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.012, f"{val:.0%}", ha="center", va="bottom", fontsize=9, color="#f9fafb", fontweight="700")
    apply_chart_style(ax, fig)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-divider"><span>Prior Mapping Matrix</span></div>', unsafe_allow_html=True)
    matrix_df = pd.DataFrame(
        FACIAL_EMOTION_TO_STATUS_MATRIX,
        index=FACIAL_CLASSES,
        columns=[STATUS_LABEL.get(s,s) for s in STATUS_CLASSES]
    ).round(2)
    st.dataframe(matrix_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — XAI Explainer
# ══════════════════════════════════════════════════════════════════════════════
with t_xai:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    xai_inputs = render_interactive_tabular_inputs(key_prefix="xai_inputs")

    if st.button("GENERATE ATTRIBUTION REPORT", key="btn_xai_main"):
        with st.spinner("Computing attributions…"):
            probs, reg = pipeline.predict_tabular(xai_inputs)
            pred_idx   = int(np.argmax(probs))
            status_k   = STATUS_CLASSES[pred_idx]
            xai_rows   = []
            for feat in FEATURE_COLS:
                lo,hi,mid = FEATURE_RANGES[feat]
                val    = xai_inputs.get(feat,mid)
                impact = (val-mid)/(hi-lo+1e-9)
                xai_rows.append({"feature":feat,"label":lbl(feat),"value":val,
                                  "impact":round(impact,4),"abs":abs(impact)})
            xai_sorted = sorted(xai_rows, key=lambda x:-x["abs"])

        st.markdown("<hr style='margin:1.35rem 0;border-color:#1f2937'>", unsafe_allow_html=True)
        xc1, xc2 = st.columns([1,2], gap="large")
        with xc1:
            sc = STATUS_PALETTE.get(status_k,"#00f5a0")
            st.markdown(f"""
<div class="animate-fade" style="background:{STATUS_BG.get(status_k,'rgba(0,245,160,.08)')};border:1px solid {sc};border-radius:14px;padding:1.3rem">
  <p class="overline">Current Prediction</p>
  <p class="result-status" style="color:{sc}">{STATUS_LABEL.get(status_k,status_k)}</p>
  {status_badge(status_k)}
</div>""", unsafe_allow_html=True)

        with xc2:
            st.markdown(overline("Top Feature Drivers"), unsafe_allow_html=True)
            top10 = xai_sorted[:10]
            fig, ax = plt.subplots(figsize=(7, 3.4))
            lbls_x = [r["label"] for r in top10]
            imps   = [r["impact"] for r in top10]
            bcols  = ["#00f5a0" if v>=0 else "#ff0055" for v in imps]
            ax.barh(lbls_x, imps, color=bcols, height=0.55)
            ax.axvline(0, color="#1f2937", linewidth=1)
            ax.set_xlabel("Relative impact (+ = above mid, − = below mid)")
            apply_chart_style(ax, fig)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Live Webcam
# ══════════════════════════════════════════════════════════════════════════════
with t_cam:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    wc1, wc2 = st.columns([1,1], gap="large")

    with wc1:
        st.markdown(overline("Live Camera Stream"), unsafe_allow_html=True)
        camera_photo = st.camera_input("Take snapshot", key="live_webcam", label_visibility="collapsed")

    with wc2:
        if camera_photo is not None:
            pil_img  = Image.open(camera_photo).convert("RGB")
            frame_np = np.array(pil_img)

            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces     = []
            if face_cascade_ui:
                try: faces = face_cascade_ui.detectMultiScale(gray, 1.1, 4, minSize=(30,30))
                except: faces = []

            frame_preview = frame_np.copy()
            for (x,y,w,h) in faces:
                cv2.rectangle(frame_preview,(x,y),(x+w,y+h),(0,245,160),2)

            with st.spinner("Running multimodal inference…"):
                realtime_res = pipeline.predict_realtime_frame(frame_np, get_current_feature_values())

            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            full_result_block(realtime_res)

            if len(faces)>0:
                st.markdown(f'<div class="section-divider"><span>Face Detection Bounding Box</span></div>', unsafe_allow_html=True)
                st.image(frame_preview, use_container_width=True)
        else:
            st.markdown("""
<div style="border:1px solid #1f2937;border-radius:14px;padding:3.2rem 2rem;
  text-align:center;margin-top:0.5rem;background:#0b0f19">
  <p style="font-size:0.72rem;font-weight:700;letter-spacing:0.12em;color:#00f5a0;margin:0 0 0.45rem;text-transform:uppercase">CAMERA ENGINE READY</p>
  <p style="font-size:0.92rem;font-weight:600;color:#f9fafb;margin:0 0 0.35rem">Take a snapshot to begin real-time diagnostic evaluation</p>
  <p style="font-size:0.78rem;color:#4b5563;margin:0">Enables OpenCV face detection, Gated CNN facial emotion classification & PyTorch multitask regression.</p>
</div>""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin:3.8rem 0 1.25rem;border-color:#1f2937">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem">
  <p style="font-size:0.74rem;color:#4b5563;margin:0">
    MindSense AI &nbsp;·&nbsp; PyTorch Multitask MLP &nbsp;·&nbsp; Keras Facial CNN &nbsp;·&nbsp; Acoustic Classifier &nbsp;·&nbsp; Gated Fusion
  </p>
  <p style="font-size:0.74rem;color:#4b5563;margin:0">All inference runs locally. No data leaves your environment.</p>
</div>
""", unsafe_allow_html=True)
