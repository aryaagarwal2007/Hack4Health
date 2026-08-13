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
import matplotlib.patches as mpatches
from PIL import Image
import cv2

from multimodal_pipeline import (
    MultimodalPipeline, FEATURE_COLS, STATUS_CLASSES,
    FACIAL_CLASSES, FACIAL_EMOTION_TO_STATUS_MATRIX
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM  (deep slate + electric teal + amber accent)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,600;1,500&display=swap');

/* ── Palette ── */
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
  --healthy:   #22d3b0;
  --mild:      #f59e0b;
  --moderate:  #fb923c;
  --severe:    #f87171;
}

/* ── Reset ── */
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

/* ── Kill chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display:none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: var(--fg) !important; }
[data-testid="stSidebar"] .stExpander { border-color: var(--border) !important; }

/* ── Tab bar ── */
[data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important; padding: 0 !important;
  flex-wrap: wrap !important;
}
[data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -1px !important;
  padding: 0.7rem 1.2rem !important;
  font-size: 0.8rem !important; font-weight: 500 !important;
  color: var(--muted) !important; letter-spacing: 0.04em !important;
  transition: all .2s !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  color: var(--teal) !important;
  border-bottom-color: var(--teal) !important;
}
[data-baseweb="tab"]:hover { color: var(--fg) !important; }

/* ── Buttons ── */
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
.stButton > button:active { transform: translateY(0) !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 1.1rem 1.3rem !important;
  transition: border-color .2s !important;
}
[data-testid="metric-container"]:hover { border-color: var(--border-hi) !important; }
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

/* ── Uploaders ── */
[data-testid="stFileUploader"] {
  background: var(--card) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: 12px !important; padding: 0.75rem 1rem !important;
}
[data-testid="stFileUploader"] *,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderDropzoneInstructions"] p { color: var(--muted) !important; }

/* ── Camera input ── */
[data-testid="stCameraInput"] {
  border: 1px solid var(--border) !important;
  border-radius: 12px !important; overflow: hidden !important;
}

/* ── Info / warning ── */
[data-testid="stInfo"] {
  background: rgba(34,211,176,.08) !important;
  border: none !important; border-left: 3px solid var(--teal) !important;
  border-radius: 6px !important; color: var(--fg) !important;
}
[data-testid="stInfo"] p { color: var(--fg) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important; border-radius: 10px !important;
}
[data-testid="stDataFrame"] * { color: var(--fg) !important; background: transparent !important; }

/* ── Sliders ── */
[role="slider"] { background: var(--teal) !important; border-color: var(--teal) !important; }

/* ── Images ── */
[data-testid="stImage"] img { border-radius: 10px !important; }

/* ── Plots ── */
.stPyplot, .stPyplot > div { background: transparent !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important; margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] summary { color: var(--fg) !important; }

/* ── General text colour fix ── */
p, h1, h2, h3, h4, label, li, span, div { color: var(--fg) !important; }
a { color: var(--teal) !important; }
hr { border-color: var(--border) !important; }

/* ── Glass card helper ── */
.glass-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.4rem 1.6rem;
  transition: border-color .2s, box-shadow .2s;
}
.glass-card:hover {
  border-color: var(--border-hi);
  box-shadow: 0 4px 24px rgba(0,0,0,.35);
}

/* ── Status badges ── */
.badge {
  display: inline-block;
  border-radius: 999px;
  padding: 0.2rem 0.9rem;
  font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.06em;
}

/* ── Progress bars ── */
.prob-bar-wrap { margin: 0.3rem 0; }
.prob-bar-label {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 0.2rem;
  font-size: 0.78rem; color: var(--muted);
}
.prob-bar-track {
  height: 6px; background: var(--border);
  border-radius: 999px; overflow: hidden;
}
.prob-bar-fill {
  height: 100%; border-radius: 999px;
  transition: width .4s ease;
}

/* ── Hero gradient headline ── */
.hero-headline {
  font-family: 'Playfair Display', serif;
  font-size: clamp(2.2rem, 5vw, 3.8rem);
  font-weight: 600; line-height: 1.1;
  letter-spacing: -0.02em;
  color: var(--fg) !important;
}
.hero-headline em {
  font-style: italic;
  background: linear-gradient(135deg, var(--teal), #6ee7f7);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* ── Stat pill ── */
.stat-pill {
  border-top: 2px solid var(--teal);
  padding-top: 0.85rem; margin-top: 0.5rem;
}
.stat-num {
  font-family: 'Playfair Display', serif;
  font-size: 2.2rem; font-weight: 600; color: var(--fg) !important;
  line-height: 1; margin: 0;
}
.stat-label { font-size: 0.78rem; font-weight: 500; color: var(--muted) !important; margin: 0.2rem 0 0; }

/* ── Section overline ── */
.overline {
  font-size: 0.67rem; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--teal) !important;
  margin: 0 0 0.45rem;
}

/* ── Result block ── */
.result-status {
  font-family: 'Playfair Display', serif;
  font-size: 2.1rem; font-weight: 600; line-height: 1.15;
  margin: 0 0 0.3rem;
}
.result-emotion {
  font-family: 'Playfair Display', serif;
  font-size: 1.9rem; font-weight: 600; line-height: 1.15;
  margin: 0 0 0.3rem; color: var(--fg) !important;
}

/* ── Divider with label ── */
.section-divider {
  display: flex; align-items: center; gap: 0.75rem;
  margin: 1.75rem 0 1rem;
}
.section-divider span {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--dim) !important; white-space: nowrap;
}
.section-divider::before, .section-divider::after {
  content: ''; flex: 1; height: 1px; background: var(--border);
}

/* Architecture step card */
.step-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.4rem;
  height: 100%; transition: all .2s;
}
.step-card:hover { border-color: var(--teal); box-shadow: 0 0 20px var(--teal-glow); }
.step-num {
  font-family: 'Playfair Display', serif; font-size: 2rem;
  color: var(--teal) !important; margin: 0 0 0.5rem; font-weight: 600;
}
.step-title { font-size: 0.92rem; font-weight: 600; color: var(--fg) !important; margin: 0 0 0.4rem; }
.step-desc  { font-size: 0.8rem; color: var(--muted) !important; line-height: 1.65; margin: 0; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
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
    "Sleep_Quality":(1.,10.,5.),"Social_Engagement":(1.,10.,5.),
    "Daily_App_Usage_Min":(0.,600.,180.),"Typing_Speed_WPM":(10.,120.,50.),
    "Session_Frequency":(1.,50.,12.),"Idle_Time_Min":(0.,300.,30.),
    "Facial_Emotion_Variance":(0.,2.,.5),"Eye_Blink_Rate":(5.,60.,18.),
    "Smile_Intensity":(0.,1.,.4),"Head_Motion_Index":(0.,2.,.3),
    "MFCC_Mean":(-50.,50.,0.),"MFCC_Variance":(0.,20.,3.),
    "Pitch_Mean":(50.,400.,160.),"Speech_Rate":(.5,10.,3.),
    "Heart_Rate_BPM":(40.,160.,72.),"HRV_Index":(10.,100.,45.),
    "Skin_Temperature":(30.,40.,36.5),"GSR_Level":(0.,20.,4.),
}
FEATURE_GROUPS = {
    "Lifestyle & Behaviour":["Sleep_Quality","Social_Engagement","Daily_App_Usage_Min","Typing_Speed_WPM","Session_Frequency","Idle_Time_Min"],
    "Facial & Vision":["Facial_Emotion_Variance","Eye_Blink_Rate","Smile_Intensity","Head_Motion_Index"],
    "Speech & Audio":["MFCC_Mean","MFCC_Variance","Pitch_Mean","Speech_Rate"],
    "Physiological":["Heart_Rate_BPM","HRV_Index","Skin_Temperature","GSR_Level"],
}

def lbl(f): return f.replace("_"," ")


# ── Helpers ────────────────────────────────────────────────────────────────────
def apply_chart_style(ax, fig):
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#2d3748"); ax.spines["bottom"].set_color("#2d3748")
    ax.tick_params(colors="#8892a4", labelsize=8)
    ax.xaxis.label.set_color("#8892a4"); ax.xaxis.label.set_size(8)

def overline(txt):
    return f'<p class="overline">{txt}</p>'

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

def status_badge(status):
    c = STATUS_PALETTE.get(status, "#8892a4")
    bg = STATUS_BG.get(status, "rgba(136,146,164,.1)")
    lbl_txt = STATUS_LABEL.get(status, status)
    return f'<span class="badge" style="color:{c};background:{bg};border:1px solid {c}">{lbl_txt}</span>'

def full_result_block(result, show_modality_chart=True):
    """Renders a complete, rich results panel from a prediction dict."""
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

    # ── Row 1: headline ──────────────────────────────────────────────────────
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

    # ── Row 2: Severity + Emotion + Weights ──────────────────────────────────
    c1, c2, c3 = st.columns([1,1,1], gap="medium")

    with c1:
        st.markdown(overline("Severity Scores"), unsafe_allow_html=True)
        scores_html = ""
        score_info = [
            ("Depression", sev.get("Depression_Score",0), "#f87171"),
            ("Anxiety",    sev.get("Anxiety_Score",0),    "#fb923c"),
            ("Stress",     sev.get("Stress_Score",0),     "#f59e0b"),
        ]
        for name, val, col in score_info:
            pct = min(val, 100)
            scores_html += f"""
<div style="margin-bottom:0.9rem">
  <div style="display:flex;justify-content:space-between;margin-bottom:0.25rem">
    <span style="font-size:0.8rem;font-weight:500;color:#e8edf5">{name}</span>
    <span style="font-size:0.8rem;font-weight:700;color:{col}">{val:.1f}</span>
  </div>
  <div style="background:#2d3748;border-radius:999px;height:7px;overflow:hidden">
    <div style="width:{pct}%;height:100%;background:{col};border-radius:999px;
      box-shadow:0 0 8px {col}66"></div>
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
<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;
  padding:1.1rem 1.2rem;height:100%;box-sizing:border-box">
  <p style="font-size:2.2rem;margin:0 0 0.2rem">{icon}</p>
  <p class="result-emotion">{top_em}</p>
  <p style="font-size:0.82rem;color:#8892a4;margin:0 0 0.75rem">{top_em_pc:.1f}% confidence</p>
  <span class="badge" style="color:{gate_c};background:transparent;
    border:1px solid {gate_c};font-size:0.65rem">● {gate_t}</span>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.2rem"><p style="color:#555f72;font-size:0.85rem">No face detected</p></div>', unsafe_allow_html=True)

    with c3:
        st.markdown(overline("Modality Weights"), unsafe_allow_html=True)
        w_tab  = weights.get("tabular",0)*100
        w_face = weights.get("facial",0)*100
        w_aud  = weights.get("audio",0)*100
        mw_html = ""
        for mname, mval, mcol in [("Tabular",w_tab,"#22d3b0"),("Facial",w_face,"#6ee7f7"),("Audio",w_aud,"#f59e0b")]:
            mw_html += f"""
<div style="margin-bottom:0.75rem">
  <div style="display:flex;justify-content:space-between;margin-bottom:0.2rem">
    <span style="font-size:0.78rem;color:#8892a4">{mname}</span>
    <span style="font-size:0.78rem;font-weight:600;color:{mcol}">{mval:.0f}%</span>
  </div>
  <div style="background:#2d3748;border-radius:999px;height:5px;overflow:hidden">
    <div style="width:{mval:.0f}%;height:100%;background:{mcol};border-radius:999px"></div>
  </div>
</div>"""
        st.markdown(f'<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.2rem">{mw_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"><span>Detailed Probabilities</span></div>', unsafe_allow_html=True)

    # ── Row 3: Emotion bars + Fused status bars ───────────────────────────────
    d1, d2 = st.columns(2, gap="medium")

    with d1:
        st.markdown(overline("All Emotion Probabilities"), unsafe_allow_html=True)
        if em_dict:
            bars_html = html_prob_bars(em_dict, highlight_key=top_em)
            st.markdown(f'<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.2rem">{bars_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#555f72;font-size:0.82rem">No facial emotion data.</p>', unsafe_allow_html=True)

    with d2:
        st.markdown(overline("Fused Status Probabilities"), unsafe_allow_html=True)
        if fused:
            bars_html2 = html_prob_bars(
                {STATUS_LABEL.get(k,k):v for k,v in fused.items()},
                highlight_key=STATUS_LABEL.get(status, status),
                palette={STATUS_LABEL.get(k,k):v for k,v in STATUS_PALETTE.items()}
            )
            st.markdown(f'<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.2rem">{bars_html2}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"><span>Feature Drivers</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE LOAD
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_pipeline():
    return MultimodalPipeline()

pipeline = get_pipeline()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:0.5rem 0 0.25rem">
  <p style="font-family:'Playfair Display',serif;font-size:1.15rem;font-weight:600;
    color:#e8edf5;margin:0">MindSense AI</p>
  <p style="font-size:0.72rem;color:#8892a4;margin:0.15rem 0 0">Control Panel</p>
</div>
""", unsafe_allow_html=True)

    def dot(ok): return "●" if ok else "○"
    for name, key in [("Tabular MLP","tabular"),("Facial CNN","facial"),("Audio Clf","audio")]:
        ok = pipeline.loaded_status.get(key,False)
        c  = "#22d3b0" if ok else "#f87171"
        st.markdown(f'<p style="font-size:0.78rem;color:{c};margin:0.15rem 0">{dot(ok)}&nbsp; {name}</p>', unsafe_allow_html=True)

    st.markdown("<hr style='margin:0.85rem 0;border-color:#2d3748'>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#8892a4;margin-bottom:0.5rem">Patient Indicators</p>', unsafe_allow_html=True)

    feature_values = {}
    for grp, feats in FEATURE_GROUPS.items():
        with st.expander(grp, expanded=False):
            for feat in feats:
                lo, hi, df = FEATURE_RANGES[feat]
                feature_values[feat] = st.slider(lbl(feat), float(lo), float(hi), float(df), key=f"sl_{feat}")


# ── OpenCV cascade ─────────────────────────────────────────────────────────────
try:
    _cp = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade_ui = cv2.CascadeClassifier(_cp) if os.path.exists(_cp) else None
except Exception:
    face_cascade_ui = None


# ══════════════════════════════════════════════════════════════════════════════
# HERO  (no tab — always visible)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:3.5rem 0 2rem;border-bottom:1px solid #2d3748;margin-bottom:0">
  <p style="font-size:0.7rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase;
    color:#22d3b0;margin:0 0 1rem">Multimodal Psychiatric Intelligence</p>
  <h1 class="hero-headline">
    Understand what you feel.<br><em>In real time.</em>
  </h1>
  <p style="font-size:0.95rem;color:#8892a4;max-width:560px;line-height:1.8;margin:1rem 0 0">
    Three AI models — PyTorch MLP, Keras Facial CNN, and Acoustic Classifier —
    fused with gated confidence-weighted attention for precise psychiatric evaluation.
  </p>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
    gap:0;margin-top:2.5rem;padding-top:2rem;border-top:1px solid #2d3748;max-width:600px">
    <div class="stat-pill">
      <p class="stat-num">18</p>
      <p class="stat-label">Input features</p>
    </div>
    <div class="stat-pill">
      <p class="stat-num">7</p>
      <p class="stat-label">Emotion classes</p>
    </div>
    <div class="stat-pill">
      <p class="stat-num">4</p>
      <p class="stat-label">Health categories</p>
    </div>
    <div class="stat-pill">
      <p class="stat-num">&lt;100ms</p>
      <p class="stat-label">Inference time</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
(t_overview, t_fusion, t_tabular,
 t_facial,   t_audio,  t_thresh,
 t_xai,      t_cam) = st.tabs([
    "Overview",
    "Multimodal Fusion",
    "Tabular MLP",
    "Facial CNN",
    "Audio Classifier",
    "CNN Thresholds",
    "XAI Explainer",
    "Live Webcam",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with t_overview:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Architecture cards
    arch = [
        ("01","PyTorch Tabular MLP","18-feature multitask MLP predicts 4 mental health classes + depression, anxiety & stress regression scores in one forward pass.","#22d3b0"),
        ("02","Keras Facial CNN","48×48 grayscale CNN detects 7 emotions (Happy, Sad, Angry, Fear, Disgust, Neutral, Surprise) from uploaded photos or webcam frames.","#6ee7f7"),
        ("03","Acoustic Classifier","280-dimensional MFCC + pitch + spectral features extracted from .wav audio and classified by a scikit-learn ensemble.","#f59e0b"),
        ("04","Gated Fusion Engine","Confidence-weighted late fusion combines all three modalities. Per-emotion thresholds gate the CNN so a weak 'Neutral' never hijacks the diagnosis.","#a78bfa"),
    ]
    c1, c2 = st.columns(2, gap="large")
    for i,(num,title,desc,col) in enumerate(arch):
        col_ref = c1 if i%2==0 else c2
        with col_ref:
            st.markdown(f"""
<div class="step-card" style="border-color:#2d3748">
  <p class="step-num" style="color:{col}">{num}</p>
  <p class="step-title">{title}</p>
  <p class="step-desc">{desc}</p>
</div>
""", unsafe_allow_html=True)
        if i%2==0 and i>0: pass

    st.markdown('<div class="section-divider"><span>Model Status</span></div>', unsafe_allow_html=True)

    ms1, ms2, ms3 = st.columns(3, gap="medium")
    status_info = [
        ("Tabular MLP","tabular","numerical_classifier.pth","#22d3b0"),
        ("Facial CNN","facial","facial_emotion_cnn.keras","#6ee7f7"),
        ("Audio Classifier","audio","audio_emotion_classifier.joblib","#f59e0b"),
    ]
    for col, (name, key, path, tc) in zip([ms1,ms2,ms3], status_info):
        ok = pipeline.loaded_status.get(key,False)
        badge_html = f'<span class="badge" style="color:{tc};background:rgba(34,211,176,.08);border:1px solid {tc}">{"✓ Loaded" if ok else "✗ Missing"}</span>'
        col.markdown(f"""
<div class="glass-card">
  <p style="font-size:0.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
    color:#8892a4;margin:0 0 0.4rem">{name}</p>
  {badge_html}
  <p style="font-size:0.72rem;color:#555f72;margin:.5rem 0 0;word-break:break-all">{path}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"><span>Quick Start</span></div>', unsafe_allow_html=True)
    steps_qs = [
        ("Adjust sliders","Open the sidebar → tune all 18 patient indicators."),
        ("Run Multimodal Fusion","Go to Multimodal Fusion and optionally attach a face photo + audio clip, then click Run."),
        ("Explore individual branches","Use Tabular MLP, Facial CNN, Audio Classifier tabs to inspect each modality independently."),
        ("Check CNN Thresholds","Visit CNN Thresholds to see the per-emotion confidence gates and the emotion→status matrix."),
        ("Explain the model","XAI Explainer generates a full feature attribution waterfall showing exactly what is driving the prediction."),
        ("Live webcam","Last tab — real-time frame analysis with full output panel on every snapshot."),
    ]
    g1, g2 = st.columns(2, gap="large")
    for i,(h,b) in enumerate(steps_qs):
        col_ref = g1 if i%2==0 else g2
        with col_ref:
            st.markdown(f"""
<div style="display:flex;gap:0.85rem;margin-bottom:1rem;align-items:flex-start">
  <span style="font-family:'Playfair Display',serif;font-size:1.2rem;color:#22d3b0;
    font-weight:600;min-width:1.8rem;padding-top:0.05rem">{i+1:02d}</span>
  <div>
    <p style="font-size:0.85rem;font-weight:600;color:#e8edf5;margin:0">{h}</p>
    <p style="font-size:0.78rem;color:#8892a4;margin:.15rem 0 0;line-height:1.6">{b}</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Multimodal Fusion
# ══════════════════════════════════════════════════════════════════════════════
with t_fusion:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.88rem;color:#8892a4;max-width:580px;margin-bottom:1.4rem;line-height:1.75">Consolidate evidence across all three modalities — tabular indicators, facial expressions, and speech audio — into a single fused diagnosis.</p>', unsafe_allow_html=True)

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
    if st.button("Run Multimodal Diagnosis", key="btn_fusion"):
        with st.spinner("Processing all modalities…"):
            result = pipeline.predict_multimodal_patient(feature_values, face_image_input=face_input, audio_input=audio_path)

        st.markdown("<hr style='margin:1.5rem 0;border-color:#2d3748'>", unsafe_allow_html=True)
        full_result_block(result)

        # Per-modality breakdown
        mod_probs = result.get("modality_probabilities",{})
        p1,p2,p3 = st.columns(3, gap="medium")
        for col,(mname,pkey) in zip([p1,p2,p3],[("Tabular","tabular"),("Facial","facial"),("Audio","audio")]):
            with col:
                st.markdown(overline(f"{mname} probabilities"), unsafe_allow_html=True)
                pd_ = mod_probs.get(pkey)
                if pd_:
                    bars = html_prob_bars(
                        {STATUS_LABEL.get(k,k):v for k,v in pd_.items()},
                        highlight_key=STATUS_LABEL.get(result["diagnosis"]),
                        palette={STATUS_LABEL.get(k,k):v for k,v in STATUS_PALETTE.items()}
                    )
                    st.markdown(f'<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1rem 1.2rem">{bars}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="font-size:0.8rem;color:#555f72;font-style:italic">Not used in this run</p>', unsafe_allow_html=True)

    if audio_path and os.path.exists(audio_path):
        try: os.unlink(audio_path)
        except: pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Tabular MLP
# ══════════════════════════════════════════════════════════════════════════════
with t_tabular:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.88rem;color:#8892a4;margin-bottom:1.2rem;line-height:1.75">Direct inference on 18 physiological & behavioural indicators using the PyTorch Multitask MLP. Adjust the sidebar sliders first.</p>', unsafe_allow_html=True)

    if st.button("Run Tabular Prediction", key="btn_tab"):
        with st.spinner("Running MLP inference…"):
            probs, reg = pipeline.predict_tabular(feature_values)
        pred_idx  = int(np.argmax(probs))
        status_k  = STATUS_CLASSES[pred_idx]

        st.markdown("<hr style='margin:1.25rem 0;border-color:#2d3748'>", unsafe_allow_html=True)

        # Status headline
        sc = STATUS_PALETTE.get(status_k,"#22d3b0")
        st.markdown(f"""
<div style="background:{STATUS_BG.get(status_k,'rgba(34,211,176,.08)')};
  border:1px solid {sc};border-radius:14px;padding:1.3rem 1.6rem;margin-bottom:1.2rem;
  display:flex;align-items:center;gap:2rem;flex-wrap:wrap">
  <div>
    <p class="overline">Predicted Status</p>
    <p class="result-status" style="color:{sc}">{STATUS_LABEL.get(status_k,status_k)}</p>
    {status_badge(status_k)}
  </div>
  <div style="display:flex;gap:1.5rem">
    <div><p style="font-family:'Playfair Display',serif;font-size:1.8rem;font-weight:600;
      color:#e8edf5;margin:0">{reg['Depression_Score']:.0f}</p>
      <p style="font-size:0.68rem;text-transform:uppercase;letter-spacing:.08em;color:#f87171;margin:.2rem 0 0">Depression</p></div>
    <div><p style="font-family:'Playfair Display',serif;font-size:1.8rem;font-weight:600;
      color:#e8edf5;margin:0">{reg['Anxiety_Score']:.0f}</p>
      <p style="font-size:0.68rem;text-transform:uppercase;letter-spacing:.08em;color:#fb923c;margin:.2rem 0 0">Anxiety</p></div>
    <div><p style="font-family:'Playfair Display',serif;font-size:1.8rem;font-weight:600;
      color:#e8edf5;margin:0">{reg['Stress_Score']:.0f}</p>
      <p style="font-size:0.68rem;text-transform:uppercase;letter-spacing:.08em;color:#f59e0b;margin:.2rem 0 0">Stress</p></div>
  </div>
</div>
""", unsafe_allow_html=True)

        tb1, tb2 = st.columns([1,1], gap="large")
        with tb1:
            st.markdown(overline("Status Probability Distribution"), unsafe_allow_html=True)
            bars_html = html_prob_bars(
                {STATUS_LABEL.get(s,s):float(probs[i]) for i,s in enumerate(STATUS_CLASSES)},
                highlight_key=STATUS_LABEL.get(status_k),
                palette={STATUS_LABEL.get(k,k):v for k,v in STATUS_PALETTE.items()}
            )
            st.markdown(f'<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.4rem">{bars_html}</div>', unsafe_allow_html=True)

        with tb2:
            st.markdown(overline("Current Input Values"), unsafe_allow_html=True)
            rows_d = []
            for feat in FEATURE_COLS:
                lo,hi,_ = FEATURE_RANGES[feat]
                val = feature_values.get(feat,0)
                pct = (val-lo)/(hi-lo)*100 if hi>lo else 0
                rows_d.append({"Feature":lbl(feat),"Value":round(val,2),"Range %":f"{pct:.0f}%"})
            st.dataframe(pd.DataFrame(rows_d), use_container_width=True, hide_index=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Facial CNN
# ══════════════════════════════════════════════════════════════════════════════
with t_facial:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.88rem;color:#8892a4;margin-bottom:1.2rem;line-height:1.75">Upload a face image. The Keras CNN detects the face, crops it to 48×48 grayscale, and outputs all 7 emotion probabilities plus the mapped mental health status.</p>', unsafe_allow_html=True)

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

                # headline
                st.markdown(f"""
<div style="background:#1c2230;border:1px solid #2d3748;border-radius:14px;
  padding:1.2rem 1.4rem;margin-bottom:1rem">
  <p class="overline">Detected Emotion</p>
  <p style="font-size:2.5rem;margin:0 0 0.1rem">{icon}</p>
  <p class="result-emotion">{top_em}</p>
  <p style="font-size:0.82rem;color:#8892a4;margin:0">{emotion_probs[top_em]*100:.1f}% confidence</p>
</div>
""", unsafe_allow_html=True)

                # All emotion probabilities
                st.markdown(overline("All 7 Emotion Probabilities"), unsafe_allow_html=True)
                em_html = html_prob_bars(emotion_probs, highlight_key=top_em)
                st.markdown(f'<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.4rem">{em_html}</div>', unsafe_allow_html=True)

                # Mapped mental health status
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
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.88rem;color:#8892a4;margin-bottom:1.2rem;line-height:1.75">Upload a .wav speech recording. 280 acoustic features (MFCCs, pitch, spectral centroid, chroma) are extracted and classified by a trained ensemble.</p>', unsafe_allow_html=True)

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
""", unsafe_allow_html=True)
                    if status_probs is not None:
                        top_s_idx = int(np.argmax(status_probs))
                        top_s     = STATUS_CLASSES[top_s_idx]
                        sc_       = STATUS_PALETTE.get(top_s,"#22d3b0")
                        st.markdown(f"""
  <hr style="margin:.75rem 0;border-color:#2d3748">
  <p class="overline">Mapped Status</p>
  <p style="font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:600;
    color:{sc_};margin:0">{STATUS_LABEL.get(top_s,top_s)}</p>
""", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with a2:
                    st.markdown(overline("All Acoustic Emotion Probabilities"), unsafe_allow_html=True)
                    ae_html = html_prob_bars(audio_det, highlight_key=top_ae)
                    st.markdown(f'<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.4rem">{ae_html}</div>', unsafe_allow_html=True)

                    if status_probs is not None:
                        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
                        st.markdown(overline("Mental Health Probability Mapping"), unsafe_allow_html=True)
                        top_s = STATUS_CLASSES[int(np.argmax(status_probs))]
                        mh_html = html_prob_bars(
                            {STATUS_LABEL.get(s,s):float(status_probs[i]) for i,s in enumerate(STATUS_CLASSES)},
                            highlight_key=STATUS_LABEL.get(top_s),
                            palette={STATUS_LABEL.get(k,k):v for k,v in STATUS_PALETTE.items()}
                        )
                        st.markdown(f'<div style="background:#1c2230;border:1px solid #2d3748;border-radius:12px;padding:1.1rem 1.4rem">{mh_html}</div>', unsafe_allow_html=True)
            else:
                st.info("Could not extract audio features. Ensure the file is a valid .wav recording.")
        finally:
            if os.path.exists(tmp_audio): os.unlink(tmp_audio)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — CNN Thresholds
# ══════════════════════════════════════════════════════════════════════════════
with t_thresh:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("""
<p style="font-size:0.88rem;color:#8892a4;max-width:640px;margin-bottom:1.3rem;line-height:1.75">
The live webcam path gates the CNN output before trusting it. If the model's top emotion
probability doesn't clear its per-class threshold, the face branch is dropped and the
tabular model drives the fused diagnosis — preventing a weakly-confident Neutral from
overriding clear expressions.
</p>
""", unsafe_allow_html=True)

    thresholds = pipeline.REALTIME_EMOTION_THRESHOLDS
    rationale  = {
        "Happy":   "Distinctive smile — lower bar acceptable.",
        "Sad":     "Unmistakable drooping expression.",
        "Angry":   "Strong brow contraction.",
        "Fear":    "Wide eyes + open mouth — slightly elevated.",
        "Disgust": "Often confused with Angry — higher bar.",
        "Surprise":"Short-lived expression — elevated bar.",
        "Neutral": "Default fallback class — needs clear majority.",
    }

    # Visual chart
    em_names    = list(thresholds.keys())
    thresh_vals = [thresholds[e] for e in em_names]
    bar_cols    = []
    for v in thresh_vals:
        if v < 0.42: bar_cols.append("#22d3b0")
        elif v < 0.49: bar_cols.append("#f59e0b")
        else: bar_cols.append("#f87171")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    bars_p  = ax.bar(em_names, thresh_vals, color=bar_cols, width=0.55)
    ax.axhline(0.5, color="#2d3748", linewidth=1.2, linestyle="--", label="50% line")
    ax.set_ylim(0, 0.68)
    ax.set_ylabel("Min. confidence required", color="#8892a4")
    for bar, val in zip(bars_p, thresh_vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.012,
                f"{val:.0%}", ha="center", va="bottom", fontsize=9, color="#e8edf5", fontweight="600")
    apply_chart_style(ax, fig)
    ax.yaxis.label.set_color("#8892a4")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig); plt.close()

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # Threshold table
    rows_t = []
    for em in em_names:
        v = thresholds[em]
        level = "Low" if v < 0.42 else "Medium" if v < 0.50 else "High"
        rows_t.append({"Emotion":em,"Threshold":f"{v:.0%}","Strictness":level,"Rationale":rationale.get(em,"")})
    st.dataframe(pd.DataFrame(rows_t), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-divider"><span>Emotion → Status Mapping Matrix</span></div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.82rem;color:#8892a4;margin-bottom:0.75rem;line-height:1.65">Each accepted facial emotion is projected onto the 4-class mental health probability space using this fixed mapping matrix. Values represent prior probabilities — row-normalised.</p>', unsafe_allow_html=True)

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
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.88rem;color:#8892a4;margin-bottom:1.2rem;line-height:1.75">Explainability report showing which patient indicator features are driving the current model prediction relative to their midpoint values. Uses current sidebar slider values.</p>', unsafe_allow_html=True)

    if st.button("Generate XAI Report", key="btn_xai"):
        with st.spinner("Computing attributions…"):
            probs, reg = pipeline.predict_tabular(feature_values)
            pred_idx   = int(np.argmax(probs))
            status_k   = STATUS_CLASSES[pred_idx]
            xai_rows   = []
            for feat in FEATURE_COLS:
                lo,hi,mid = FEATURE_RANGES[feat]
                val    = feature_values.get(feat,mid)
                impact = (val-mid)/(hi-lo+1e-9)
                xai_rows.append({"feature":feat,"label":lbl(feat),"value":val,
                                  "impact":round(impact,4),"abs":abs(impact)})
            xai_sorted = sorted(xai_rows, key=lambda x:-x["abs"])

        st.markdown("<hr style='margin:1.25rem 0;border-color:#2d3748'>", unsafe_allow_html=True)

        xc1, xc2 = st.columns([1,2], gap="large")
        with xc1:
            sc = STATUS_PALETTE.get(status_k,"#22d3b0")
            st.markdown(f"""
<div style="background:{STATUS_BG.get(status_k,'rgba(34,211,176,.08)')};
  border:1px solid {sc};border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:1rem">
  <p class="overline">Current Prediction</p>
  <p class="result-status" style="color:{sc}">{STATUS_LABEL.get(status_k,status_k)}</p>
  {status_badge(status_k)}
</div>
""", unsafe_allow_html=True)
            st.metric("Depression Score", f"{reg['Depression_Score']:.1f}")
            st.metric("Anxiety Score",    f"{reg['Anxiety_Score']:.1f}")
            st.metric("Stress Score",     f"{reg['Stress_Score']:.1f}")

        with xc2:
            st.markdown(overline("Top 10 Feature Drivers"), unsafe_allow_html=True)
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

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        st.markdown(overline("Full Attribution Table"), unsafe_allow_html=True)
        tbl = pd.DataFrame([{
            "Feature":lbl(r["feature"]),
            "Value":round(r["value"],2),
            "Impact":f"{r['impact']:+.3f}",
            "Direction":"↑ Above mid" if r["impact"]>0 else "↓ Below mid",
        } for r in xai_sorted])
        st.dataframe(tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Live Webcam  (always last)
# ══════════════════════════════════════════════════════════════════════════════
with t_cam:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    wc1, wc2 = st.columns([1,1], gap="large")

    with wc1:
        st.markdown(overline("Live Capture"), unsafe_allow_html=True)
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

            # Full result panel
            full_result_block(realtime_res, show_modality_chart=False)

            # Face preview + bounding box
            if len(faces)>0 or realtime_res.get("facial_emotion_details"):
                st.markdown(f"""
<div class="section-divider"><span>Face Detection Preview</span></div>
""", unsafe_allow_html=True)
                pv1, pv2 = st.columns([1,1], gap="medium")
                with pv1:
                    st.image(frame_preview, use_container_width=True)
                    st.markdown(f'<p style="font-size:0.72rem;color:#8892a4;margin:.35rem 0;text-align:center">{len(faces)} face(s) detected</p>', unsafe_allow_html=True)
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
  <p style="font-size:0.75rem;color:#555f72;margin:0;font-family:'Plus Jakarta Sans',sans-serif">
    MindSense AI &nbsp;·&nbsp; PyTorch MLP &nbsp;·&nbsp; Keras CNN &nbsp;·&nbsp; Acoustic Classifier &nbsp;·&nbsp; Gated Fusion
  </p>
  <p style="font-size:0.75rem;color:#555f72;margin:0">All inference runs locally. No data leaves your device.</p>
</div>
""", unsafe_allow_html=True)
