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
    page_title="MindSense AI — Multimodal Psychiatric System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM & STICKY TOP NAVBAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,600;1,500&display=swap');

:root {
  --bg:        #0d1117;
  --surface:   #161b22;
  --card:      #1c2230;
  --border:    #2d3748;
  --border-hi: #3d4f6a;
  --fg:        #e8edf5;
  --muted:     #8892a4;
  --dim:       #555f72;
  --teal:      #22d3b0;
  --teal-dim:  #1a9e86;
  --teal-glow: rgba(34,211,176,.12);
  --amber:     #f59e0b;
  --rose:      #f87171;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
  background: var(--bg) !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  color: var(--fg) !important;
}
.block-container {
  padding: 0 2.5rem 4rem !important;
  max-width: 1240px !important;
}
@media(max-width:768px){ .block-container{ padding:0 1rem 3rem !important; } }

/* Kill Streamlit default chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display:none !important; }

/* ── STICKY TOP NAVBAR ── */
[data-baseweb="tab-list"] {
  position: sticky !important;
  top: 0 !important;
  z-index: 99999 !important;
  background: #0d1117 !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0.2rem !important;
  padding: 0.6rem 0 !important;
  flex-wrap: wrap !important;
  box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
}
[data-baseweb="tab"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  margin: 0.2rem 0.25rem !important;
  padding: 0.55rem 1.1rem !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
  letter-spacing: 0.03em !important;
  transition: all .2s ease !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  background: var(--card) !important;
  color: var(--teal) !important;
  border-color: var(--teal) !important;
  box-shadow: 0 0 15px rgba(34,211,176,0.2) !important;
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

/* Buttons */
.stButton > button {
  background: var(--teal) !important;
  color: #0d1117 !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.65rem 1.8rem !important;
  font-size: 0.85rem !important; font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  transition: all .18s !important;
  box-shadow: 0 0 20px rgba(34,211,176,.2) !important;
}
.stButton > button:hover {
  background: #1dc9a8 !important;
  box-shadow: 0 0 30px rgba(34,211,176,.35) !important;
  transform: translateY(-1px) !important;
}

/* Input Boxes & Number Controls */
[data-baseweb="input"], [data-baseweb="base-input"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--fg) !important;
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

/* Metric Cards */
[data-testid="metric-container"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 1.1rem 1.3rem !important;
}
[data-testid="stMetricLabel"] {
  font-size: 0.68rem !important; font-weight: 600 !important;
  text-transform: uppercase !important; letter-spacing: 0.1em !important;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Playfair Display', serif !important;
  font-size: 1.6rem !important; font-weight: 600 !important;
  color: var(--fg) !important; line-height: 1.2 !important;
}

/* Uploaders & Camera */
[data-testid="stFileUploader"] {
  background: var(--card) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: 12px !important; padding: 0.75rem 1rem !important;
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
[role="slider"] { background: var(--teal) !important; border-color: var(--teal) !important; }

/* General typography */
p, h1, h2, h3, h4, label, li, span, div { color: var(--fg) !important; }
a { color: var(--teal) !important; }
hr { border-color: var(--border) !important; }

.overline {
  font-size: 0.67rem; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--teal) !important;
  margin: 0 0 0.45rem;
}
.result-status {
  font-family: 'Playfair Display', serif;
  font-size: 2.1rem; font-weight: 600; line-height: 1.15; margin: 0 0 0.3rem;
}
.result-emotion {
  font-family: 'Playfair Display', serif;
  font-size: 1.9rem; font-weight: 600; line-height: 1.15; margin: 0 0 0.3rem;
}
.section-divider {
  display: flex; align-items: center; gap: 0.75rem; margin: 1.75rem 0 1rem;
}
.section-divider span {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--dim) !important; white-space: nowrap;
}
.section-divider::before, .section-divider::after {
  content: ''; flex: 1; height: 1px; background: var(--border);
}
.badge {
  display: inline-block; border-radius: 999px; padding: 0.2rem 0.9rem;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
}
.prob-bar-wrap { margin: 0.3rem 0; }
.prob-bar-label {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 0.2rem; font-size: 0.78rem; color: var(--muted);
}
.prob-bar-track {
  height: 6px; background: var(--border); border-radius: 999px; overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 999px; transition: width .4s ease; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
STATUS_PALETTE = {
    "Healthy":         "#22d3b0",
    "Mild_Stress":     "#f59e0b",
    "Moderate_Stress": "#fb923c",
    "Severe_Stress":   "#f87171",
}
STATUS_BG = {
    "Healthy":         "rgba(34,211,176,.12)",
    "Mild_Stress":     "rgba(245,158,11,.12)",
    "Moderate_Stress": "rgba(251,146,60,.12)",
    "Severe_Stress":   "rgba(248,113,113,.12)",
}
STATUS_LABEL = {
    "Healthy":         "Healthy",
    "Mild_Stress":     "Mild Stress",
    "Moderate_Stress": "Moderate Stress",
    "Severe_Stress":   "Severe Stress",
}
EMOTION_ICONS = {
    "Angry":"😠","Disgust":"🤢","Fear":"😨",
    "Happy":"😄","Neutral":"😐","Sad":"😢","Surprise":"😲",
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def apply_chart_style(ax, fig):
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#2d3748"); ax.spines["bottom"].set_color("#2d3748")
    ax.tick_params(colors="#8892a4", labelsize=8)
    ax.xaxis.label.set_color("#8892a4"); ax.xaxis.label.set_size(8)

def overline(txt): return f'<p class="overline">{txt}</p>'

def status_badge(status):
    c = STATUS_PALETTE.get(status, "#8892a4")
    bg = STATUS_BG.get(status, "rgba(136,146,164,.1)")
    lbl_txt = STATUS_LABEL.get(status, status)
    return f'<span class="badge" style="color:{c};background:{bg};border:1px solid {c}">{lbl_txt}</span>'

def html_prob_bars(prob_dict, highlight_key=None, palette=None):
    bars = ""
    for k, v in sorted(prob_dict.items(), key=lambda x:-x[1]):
        pct  = v * 100
        col  = (palette or {}).get(k, "#22d3b0") if k == highlight_key else "#2d3748"
        fill = (palette or {}).get(k, "#22d3b0") if k == highlight_key else "#3d4f6a"
        bars += f"""
<div class="prob-bar-wrap">
  <div class="prob-bar-label">
    <span style="color:{'#e8edf5' if k==highlight_key else '#8892a4'};font-weight:{'600' if k==highlight_key else '400'}">{k}</span>
    <span style="color:{col};font-weight:600">{pct:.1f}%</span>
  </div>
  <div class="prob-bar-track">
    <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{fill}"></div>
  </div>
</div>"""
    return bars

def full_result_block(result):
    status    = result["diagnosis"]
    sc        = STATUS_PALETTE.get(status, "#22d3b0")
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
<div style="background:{STATUS_BG.get(status,'rgba(34,211,176,.08)')};
  border:1px solid {sc};border-radius:16px;padding:1.5rem 1.8rem;margin-bottom:1.2rem">
  <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap">
    <div style="flex:1;min-width:180px">
      <p class="overline">Mental Health Status</p>
      <p class="result-status" style="color:{sc}">{STATUS_LABEL.get(status,status)}</p>
      {status_badge(status)}
    </div>
    <div style="display:flex;gap:1.2rem;flex-wrap:wrap">
      <div style="text-align:center">
        <p style="font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:600;
          color:{sc};margin:0;line-height:1">{conf:.1f}<span style="font-size:1rem;color:#8892a4">%</span></p>
        <p style="font-size:0.68rem;text-transform:uppercase;letter-spacing:.08em;color:#8892a4;margin:.2rem 0 0">Confidence</p>
      </div>
      <div style="text-align:center">
        <p style="font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:600;
          color:#e8edf5;margin:0;line-height:1">{latency}<span style="font-size:0.9rem;color:#8892a4">ms</span></p>
        <p style="font-size:0.68rem;text-transform:uppercase;letter-spacing:.08em;color:#8892a4;margin:.2rem 0 0">Latency</p>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,1,1], gap="medium")
    with c1:
        st.markdown(overline("Severity Scores"), unsafe_allow_html=True)
        scores_html = ""
        for name, val, col in [("Depression", sev.get("Depression_Score",0), "#f87171"),
                               ("Anxiety",    sev.get("Anxiety_Score",0),    "#fb923c"),
                               ("Stress",     sev.get("Stress_Score",0),     "#f59e0b")]:
            pct = min(val, 100)
            scores_html += f"""
<div style="margin-bottom:0.9rem">
  <div style="display:flex;justify-content:space-between;margin-bottom:0.25rem">
    <span style="font-size:0.8rem;font-weight:500;color:#e8edf5">{name}</span>
    <span style="font-size:0.8rem;font-weight:700;color:{col}">{val:.1f}</span>
  </div>
  <div style="background:#2d3748;border-radius:999px;height:7px;overflow:hidden">
    <div style="width:{pct}%;height:100%;background:{col};border-radius:999px"></div>
  </div>
</div>"""
        st.markdown(f'<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.2rem">{scores_html}</div>', unsafe_allow_html=True)

    with c2:
        st.markdown(overline("Facial Emotion"), unsafe_allow_html=True)
        if top_em:
            icon = EMOTION_ICONS.get(top_em,"")
            gate_c = "#22d3b0" if face_ok else "#f87171"
            gate_t = "CNN accepted" if face_ok else "Tabular driving"
            st.markdown(f"""
<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.2rem;height:100%">
  <p style="font-size:2.2rem;margin:0 0 0.2rem">{icon}</p>
  <p class="result-emotion">{top_em}</p>
  <p style="font-size:0.82rem;color:#8892a4;margin:0 0 0.75rem">{top_em_pc:.1f}% confidence</p>
  <span class="badge" style="color:{gate_c};border:1px solid {gate_c}">● {gate_t}</span>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.2rem"><p style="color:#555f72;font-size:0.85rem">No face detected</p></div>', unsafe_allow_html=True)

    with c3:
        st.markdown(overline("Modality Weights"), unsafe_allow_html=True)
        mw_html = ""
        for mname, mval, mcol in [("Tabular",weights.get("tabular",0)*100,"#22d3b0"),
                                  ("Facial", weights.get("facial",0)*100,"#6ee7f7"),
                                  ("Audio",  weights.get("audio",0)*100, "#f59e0b")]:
            mw_html += f"""
<div style="margin-bottom:0.75rem">
  <div style="display:flex;justify-content:space-between;margin-bottom:0.2rem">
    <span style="font-size:0.78rem;color:#8892a4">{mname}</span>
    <span style="font-size:0.78rem;font-weight:600;color:{mcol}">{mval:.0f}%</span>
  </div>
  <div style="background:#2d3748;border-radius:999px;height:5px;overflow:hidden">
    <div style="width:{mval:.0f}%;height:100%;background:{mcol}"></div>
  </div>
</div>"""
        st.markdown(f'<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.2rem">{mw_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"><span>Probabilities</span></div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2, gap="medium")
    with d1:
        st.markdown(overline("Emotion Probabilities"), unsafe_allow_html=True)
        if em_dict:
            bars_html = html_prob_bars(em_dict, highlight_key=top_em)
            st.markdown(f'<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.2rem">{bars_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#555f72;font-size:0.82rem">No face data.</p>', unsafe_allow_html=True)

    with d2:
        st.markdown(overline("Fused Status Probabilities"), unsafe_allow_html=True)
        if fused:
            bars_html2 = html_prob_bars(
                {STATUS_LABEL.get(k,k):v for k,v in fused.items()},
                highlight_key=STATUS_LABEL.get(status, status),
                palette={STATUS_LABEL.get(k,k):v for k,v in STATUS_PALETTE.items()}
            )
            st.markdown(f'<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.2rem">{bars_html2}</div>', unsafe_allow_html=True)


# ── OpenCV cascade ─────────────────────────────────────────────────────────────
try:
    _cp = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade_ui = cv2.CascadeClassifier(_cp) if os.path.exists(_cp) else None
except Exception:
    face_cascade_ui = None


# ══════════════════════════════════════════════════════════════════════════════
# TOP STICKY NAVBAR  (FIRST THING ON PAGE)
# ══════════════════════════════════════════════════════════════════════════════
(t_overview, t_tabular, t_fusion,
 t_facial,   t_audio,   t_thresh,
 t_xai,      t_cam) = st.tabs([
    "🏠 Overview",
    "📊 Tabular MLP (Inputs & Model)",
    "🌐 Multimodal Fusion",
    "📷 Facial CNN",
    "🎵 Audio Classifier",
    "⚙️ CNN Thresholds",
    "🔍 XAI Explainer",
    "🎥 Live Webcam",
])


# ── Interactive Feature Input Helper (Used in Tabular MLP & Multimodal) ──────
def render_interactive_tabular_inputs(key_prefix="tab_input"):
    """Renders clean 4-column numerical input controls for all 18 patient indicators."""
    st.markdown(overline("Interactive Patient Indicator Inputs"), unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.82rem;color:#8892a4;margin-bottom:1rem">Enter or adjust patient values below. All inputs directly feed the PyTorch Multitask MLP and Gated Fusion Engine.</p>', unsafe_allow_html=True)

    inputs = {}
    cols = st.columns(4, gap="medium")

    for idx, (grp_name, feats) in enumerate(FEATURE_GROUPS.items()):
        with cols[idx]:
            st.markdown(f'<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1rem;height:100%">'
                        f'<p style="font-size:0.75rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#22d3b0;margin-bottom:0.75rem">{grp_name}</p>', unsafe_allow_html=True)
            for feat in feats:
                lo, hi, df = FEATURE_RANGES[feat]
                step_val = 0.1 if isinstance(df, float) and df < 5.0 else (0.5 if isinstance(df, float) else 1)
                
                # Check session state default
                sk = f"{key_prefix}_{feat}"
                if sk not in st.session_state:
                    st.session_state[sk] = float(df) if isinstance(df, float) else int(df)

                if isinstance(df, float):
                    val = st.number_input(
                        lbl(feat), min_value=float(lo), max_value=float(hi),
                        value=float(st.session_state[sk]), step=float(step_val),
                        key=sk
                    )
                else:
                    val = st.number_input(
                        lbl(feat), min_value=int(lo), max_value=int(hi),
                        value=int(st.session_state[sk]), step=int(step_val),
                        key=sk
                    )
                inputs[feat] = val
            st.markdown('</div>', unsafe_allow_html=True)

    return inputs


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with t_overview:
    # Hero section
    st.markdown("""
<div style="padding:2.5rem 0 1.5rem;border-bottom:1px solid #2d3748;margin-bottom:1.5rem">
  <p style="font-size:0.7rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase;
    color:#22d3b0;margin:0 0 0.8rem">Multimodal Psychiatric Intelligence</p>
  <h1 style="font-family:'Playfair Display',serif;font-size:clamp(2.2rem,4vw,3.5rem);
    font-weight:600;line-height:1.1;letter-spacing:-0.02em;margin:0 0 0.8rem">
    Understand what you feel.<br>
    <em style="font-style:italic;background:linear-gradient(135deg,#22d3b0,#6ee7f7);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent">In real time.</em>
  </h1>
  <p style="font-size:0.92rem;color:#8892a4;max-width:580px;line-height:1.75;margin:0">
    Three AI models — PyTorch MLP, Keras Facial CNN, and Acoustic Classifier —
    fused with gated confidence-weighted attention for precise psychiatric evaluation.
  </p>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
    gap:1.5rem;margin-top:2rem;padding-top:1.5rem;border-top:1px solid #2d3748;max-width:600px">
    <div><p style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:600;color:#22d3b0;margin:0">18</p>
      <p style="font-size:0.75rem;color:#8892a4;margin:.2rem 0 0">Input features</p></div>
    <div><p style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:600;color:#6ee7f7;margin:0">7</p>
      <p style="font-size:0.75rem;color:#8892a4;margin:.2rem 0 0">Emotion classes</p></div>
    <div><p style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:600;color:#f59e0b;margin:0">4</p>
      <p style="font-size:0.75rem;color:#8892a4;margin:.2rem 0 0">Health categories</p></div>
    <div><p style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:600;color:#e8edf5;margin:0">&lt;100ms</p>
      <p style="font-size:0.75rem;color:#8892a4;margin:.2rem 0 0">Inference time</p></div>
  </div>
</div>
""", unsafe_allow_html=True)

    arch = [
        ("01","PyTorch Tabular MLP","18-feature multitask MLP predicts 4 mental health classes + depression, anxiety & stress regression scores in one pass.","#22d3b0"),
        ("02","Keras Facial CNN","48×48 grayscale CNN detects 7 emotions (Happy, Sad, Angry, Fear, Disgust, Neutral, Surprise) from uploaded photos or webcam frames.","#6ee7f7"),
        ("03","Acoustic Classifier","280-dimensional MFCC + pitch + spectral features extracted from .wav audio and classified by a scikit-learn ensemble.","#f59e0b"),
        ("04","Gated Fusion Engine","Confidence-weighted late fusion combines all three modalities. Per-emotion thresholds gate the CNN so a weak 'Neutral' never hijacks the diagnosis.","#a78bfa"),
    ]
    c1, c2 = st.columns(2, gap="large")
    for i,(num,title,desc,col) in enumerate(arch):
        col_ref = c1 if i%2==0 else c2
        with col_ref:
            st.markdown(f"""
<div style="background:#1c2230;border:1px solid #2d3748;border-radius:14px;padding:1.3rem;margin-bottom:1rem">
  <p style="font-family:'Playfair Display',serif;font-size:1.8rem;color:{col};margin:0 0 0.4rem;font-weight:600">{num}</p>
  <p style="font-size:0.9rem;font-weight:600;color:#e8edf5;margin:0 0 0.3rem">{title}</p>
  <p style="font-size:0.8rem;color:#8892a4;line-height:1.6;margin:0">{desc}</p>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"><span>Model Status</span></div>', unsafe_allow_html=True)
    ms1, ms2, ms3 = st.columns(3, gap="medium")
    for col, (name, key, path, tc) in zip([ms1,ms2,ms3], [
        ("Tabular MLP","tabular","numerical_classifier.pth","#22d3b0"),
        ("Facial CNN","facial","facial_emotion_cnn.keras","#6ee7f7"),
        ("Audio Classifier","audio","audio_emotion_classifier.joblib","#f59e0b"),
    ]):
        ok = pipeline.loaded_status.get(key,False)
        badge_html = f'<span class="badge" style="color:{tc};border:1px solid {tc}">{"✓ Loaded" if ok else "✗ Missing"}</span>'
        col.markdown(f"""
<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.2rem">
  <p style="font-size:0.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#8892a4;margin:0 0 0.4rem">{name}</p>
  {badge_html}
  <p style="font-size:0.72rem;color:#555f72;margin:.5rem 0 0;word-break:break-all">{path}</p>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Tabular MLP (Inputs & Model)
# ══════════════════════════════════════════════════════════════════════════════
with t_tabular:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    
    # Render interactive input boxes directly on the page!
    tab_feature_values = render_interactive_tabular_inputs(key_prefix="tab_mlp")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if st.button("🚀 Run PyTorch Tabular Prediction", key="btn_run_tab_page"):
        with st.spinner("Executing PyTorch Tabular MLP..."):
            probs, reg = pipeline.predict_tabular(tab_feature_values)
        pred_idx  = int(np.argmax(probs))
        status_k  = STATUS_CLASSES[pred_idx]

        st.markdown("<hr style='margin:1.25rem 0;border-color:#2d3748'>", unsafe_allow_html=True)
        sc = STATUS_PALETTE.get(status_k,"#22d3b0")
        
        st.markdown(f"""
<div style="background:{STATUS_BG.get(status_k,'rgba(34,211,176,.08)')};
  border:1px solid {sc};border-radius:14px;padding:1.4rem 1.8rem;margin-bottom:1.2rem;
  display:flex;align-items:center;gap:2rem;flex-wrap:wrap">
  <div>
    <p class="overline">Predicted Mental Health Status</p>
    <p class="result-status" style="color:{sc}">{STATUS_LABEL.get(status_k,status_k)}</p>
    {status_badge(status_k)}
  </div>
  <div style="display:flex;gap:1.8rem;margin-left:auto">
    <div style="text-align:center"><p style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:600;color:#f87171;margin:0">{reg['Depression_Score']:.1f}</p>
      <p style="font-size:0.68rem;text-transform:uppercase;letter-spacing:.08em;color:#8892a4;margin:.2rem 0 0">Depression</p></div>
    <div style="text-align:center"><p style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:600;color:#fb923c;margin:0">{reg['Anxiety_Score']:.1f}</p>
      <p style="font-size:0.68rem;text-transform:uppercase;letter-spacing:.08em;color:#8892a4;margin:.2rem 0 0">Anxiety</p></div>
    <div style="text-align:center"><p style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:600;color:#f59e0b;margin:0">{reg['Stress_Score']:.1f}</p>
      <p style="font-size:0.68rem;text-transform:uppercase;letter-spacing:.08em;color:#8892a4;margin:.2rem 0 0">Stress</p></div>
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
            st.markdown(f'<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.4rem">{bars_html}</div>', unsafe_allow_html=True)

        with tb2:
            st.markdown(overline("Input Summary Table"), unsafe_allow_html=True)
            rows_d = [{"Feature":lbl(f),"Value":round(v,2)} for f,v in tab_feature_values.items()]
            st.dataframe(pd.DataFrame(rows_d), use_container_width=True, hide_index=True, height=280)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Multimodal Fusion
# ══════════════════════════════════════════════════════════════════════════════
with t_fusion:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    
    # Feature Inputs on fusion page
    fusion_feature_values = render_interactive_tabular_inputs(key_prefix="fusion_inputs")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    up1, up2 = st.columns(2, gap="large")
    with up1:
        st.markdown(overline("Face photo (optional)"), unsafe_allow_html=True)
        face_file = st.file_uploader("Upload face image", type=["png","jpg","jpeg"], key="fusion_face", label_visibility="collapsed")
    with up2:
        st.markdown(overline("Speech audio .wav (optional)"), unsafe_allow_html=True)
        audio_file = st.file_uploader("Upload audio", type=["wav"], key="fusion_audio", label_visibility="collapsed")

    face_input = Image.open(face_file) if face_file else None
    audio_path = None
    if audio_file:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_file.read()); audio_path = tmp.name

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    if st.button("🔥 Run Comprehensive Multimodal Diagnosis", key="btn_fusion_main"):
        with st.spinner("Processing all modalities…"):
            result = pipeline.predict_multimodal_patient(fusion_feature_values, face_image_input=face_input, audio_input=audio_path)

        st.markdown("<hr style='margin:1.5rem 0;border-color:#2d3748'>", unsafe_allow_html=True)
        full_result_block(result)

    if audio_path and os.path.exists(audio_path):
        try: os.unlink(audio_path)
        except: pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Facial CNN
# ══════════════════════════════════════════════════════════════════════════════
with t_facial:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.88rem;color:#8892a4;margin-bottom:1.2rem">Upload a face image — Keras CNN predicts all 7 facial emotions (Happy, Sad, Angry, Fear, Disgust, Neutral, Surprise).</p>', unsafe_allow_html=True)

    face_file_tab = st.file_uploader("Upload face photo", type=["png","jpg","jpeg"], key="face_tab", label_visibility="collapsed")

    if face_file_tab:
        img = Image.open(face_file_tab)
        st.markdown("<hr style='margin:1.1rem 0;border-color:#2d3748'>", unsafe_allow_html=True)
        fi1, fi2 = st.columns([1,2], gap="large")
        with fi1:
            st.image(img, use_container_width=True)

        with fi2:
            with st.spinner("Analysing facial expression…"):
                status_probs, emotion_probs = pipeline.predict_facial(img)

            if emotion_probs:
                top_em = max(emotion_probs, key=emotion_probs.get)
                icon   = EMOTION_ICONS.get(top_em,"")

                st.markdown(f"""
<div style="background:#1c2230;border:1px solid #2d3748;border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:1rem">
  <p class="overline">Detected Emotion</p>
  <p style="font-size:2.5rem;margin:0 0 0.1rem">{icon}</p>
  <p class="result-emotion">{top_em}</p>
  <p style="font-size:0.82rem;color:#8892a4;margin:0">{emotion_probs[top_em]*100:.1f}% confidence</p>
</div>
""", unsafe_allow_html=True)

                st.markdown(overline("All 7 Emotion Probabilities"), unsafe_allow_html=True)
                em_html = html_prob_bars(emotion_probs, highlight_key=top_em)
                st.markdown(f'<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.4rem">{em_html}</div>', unsafe_allow_html=True)

                if status_probs is not None:
                    top_s_idx = int(np.argmax(status_probs))
                    top_s     = STATUS_CLASSES[top_s_idx]
                    sc        = STATUS_PALETTE.get(top_s,"#22d3b0")
                    st.markdown(f"""
<div style="background:{STATUS_BG.get(top_s,'rgba(34,211,176,.08)')};
  border:1px solid {sc};border-radius:12px;padding:1rem 1.3rem;margin-top:0.9rem">
  <p class="overline">Mapped Mental Health Status</p>
  <p style="font-family:'Playfair Display',serif;font-size:1.55rem;font-weight:600;
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
    st.markdown('<p style="font-size:0.88rem;color:#8892a4;margin-bottom:1.2rem">Upload a .wav speech recording for acoustic emotion analysis.</p>', unsafe_allow_html=True)

    audio_file_tab = st.file_uploader("Upload speech audio", type=["wav"], key="audio_tab", label_visibility="collapsed")

    if audio_file_tab:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_file_tab.read()); tmp_audio = tmp.name
        try:
            with st.spinner("Extracting features and classifying…"):
                status_probs, audio_det = pipeline.predict_audio(tmp_audio)

            if audio_det:
                top_ae = max(audio_det, key=audio_det.get)
                st.markdown("<hr style='margin:1.1rem 0;border-color:#2d3748'>", unsafe_allow_html=True)

                a1, a2 = st.columns([1,2], gap="large")
                with a1:
                    st.markdown(f"""
<div style="background:#1c2230;border:1px solid #2d3748;border-radius:14px;padding:1.3rem 1.4rem">
  <p class="overline">Speech Emotion</p>
  <p class="result-emotion">{top_ae.capitalize()}</p>
  <p style="font-size:0.82rem;color:#8892a4;margin:0 0 1rem">{audio_det[top_ae]*100:.1f}% confidence</p>
</div>
""", unsafe_allow_html=True)

                with a2:
                    st.markdown(overline("Acoustic Emotion Probabilities"), unsafe_allow_html=True)
                    ae_html = html_prob_bars(audio_det, highlight_key=top_ae)
                    st.markdown(f'<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.4rem">{ae_html}</div>', unsafe_allow_html=True)
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
<p style="font-size:0.88rem;color:#8892a4;max-width:640px;margin-bottom:1.3rem;line-height:1.75">
Per-emotion confidence thresholds used by the Gated Multimodal Pipeline to accept or reject CNN facial predictions.
</p>
""", unsafe_allow_html=True)

    thresholds = pipeline.REALTIME_EMOTION_THRESHOLDS
    em_names    = list(thresholds.keys())
    thresh_vals = [thresholds[e] for e in em_names]
    bar_cols    = ["#22d3b0" if v < 0.42 else "#f59e0b" if v < 0.49 else "#f87171" for v in thresh_vals]

    fig, ax = plt.subplots(figsize=(9, 3.2))
    bars_p  = ax.bar(em_names, thresh_vals, color=bar_cols, width=0.55)
    ax.axhline(0.5, color="#2d3748", linewidth=1.2, linestyle="--")
    ax.set_ylim(0, 0.68)
    ax.set_ylabel("Min. confidence required", color="#8892a4")
    for bar, val in zip(bars_p, thresh_vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.012, f"{val:.0%}", ha="center", va="bottom", fontsize=9, color="#e8edf5", fontweight="600")
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

    if st.button("🔍 Generate Explainability Attribution Report", key="btn_xai_main"):
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

        st.markdown("<hr style='margin:1.25rem 0;border-color:#2d3748'>", unsafe_allow_html=True)
        xc1, xc2 = st.columns([1,2], gap="large")
        with xc1:
            sc = STATUS_PALETTE.get(status_k,"#22d3b0")
            st.markdown(f"""
<div style="background:{STATUS_BG.get(status_k,'rgba(34,211,176,.08)')};border:1px solid {sc};border-radius:14px;padding:1.2rem">
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
            bcols  = ["#22d3b0" if v>=0 else "#f87171" for v in imps]
            ax.barh(lbls_x, imps, color=bcols, height=0.55)
            ax.axvline(0, color="#2d3748", linewidth=1)
            ax.set_xlabel("Relative impact (+ = above mid, − = below mid)")
            apply_chart_style(ax, fig)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Live Webcam  (always last)
# ══════════════════════════════════════════════════════════════════════════════
with t_cam:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    wc1, wc2 = st.columns([1,1], gap="large")

    with wc1:
        st.markdown(overline("Live Camera Input"), unsafe_allow_html=True)
        camera_photo = st.camera_input("Take a photo", key="live_webcam", label_visibility="collapsed")

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
                cv2.rectangle(frame_preview,(x,y),(x+w,y+h),(34,211,176),2)

            with st.spinner("Running multimodal inference…"):
                realtime_res = pipeline.predict_realtime_frame(frame_np, feature_values)

            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            full_result_block(realtime_res)

            if len(faces)>0:
                st.markdown(f'<div class="section-divider"><span>Face Detection Preview</span></div>', unsafe_allow_html=True)
                st.image(frame_preview, use_container_width=True)
        else:
            st.markdown("""
<div style="border:1.5px dashed #2d3748;border-radius:14px;padding:3rem 2rem;
  text-align:center;margin-top:0.5rem;background:#161b22">
  <p style="font-size:2rem;margin:0 0 0.5rem">📸</p>
  <p style="font-size:0.9rem;font-weight:500;color:#e8edf5;margin:0 0 0.3rem">Click to capture</p>
  <p style="font-size:0.8rem;color:#555f72;margin:0">Enable your webcam and take a snapshot to begin real-time analysis.</p>
</div>""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin:3.5rem 0 1.25rem;border-color:#2d3748">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem">
  <p style="font-size:0.75rem;color:#555f72;margin:0">
    MindSense AI &nbsp;·&nbsp; PyTorch MLP &nbsp;·&nbsp; Keras CNN &nbsp;·&nbsp; Acoustic Classifier &nbsp;·&nbsp; Gated Fusion
  </p>
  <p style="font-size:0.75rem;color:#555f72;margin:0">All inference runs locally. No data leaves your device.</p>
</div>
""", unsafe_allow_html=True)
