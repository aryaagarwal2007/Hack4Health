# MindSense AI — Hack4Health

> **Multimodal Psychiatric Evaluation System**  
> Fusing facial expression, speech acoustics, and physiological indicators to predict mental health status in real time.

---

## Overview

MindSense AI is a three-branch multimodal AI system built for the Hack4Health hackathon. It combines:

| Branch | Model | Task |
|--------|-------|------|
| **Tabular MLP** | PyTorch multitask MLP | 4-class mental health status + Depression / Anxiety / Stress regression |
| **Facial CNN** | Keras 48×48 grayscale CNN | 7-class facial emotion detection |
| **Audio Classifier** | scikit-learn ensemble | Acoustic emotion classification from speech |

All three branches are fused using **Gated Confidence-Weighted Late Fusion** with per-emotion thresholds to prevent ambiguous "Neutral" predictions from overriding clearly expressive frames.

---

## Project Structure

```
Hack4Health/
├── app.py                          # Streamlit frontend (8-tab premium UI)
├── multimodal_pipeline.py          # Core inference engine & gated fusion
├── requirements.txt                # Python dependencies
│
├── training/                       # All model training scripts
│   ├── phase1_csv_baseline.py      # Baseline tabular classifier
│   ├── phase2_facial_cnn.py        # Facial emotion CNN training
│   ├── phase2b_facial_cnn_transfer.py  # Transfer learning variant
│   ├── phase3_audio_classifier.py  # Initial audio model
│   ├── phase3b_audio_improved.py   # Improved 280-dim feature extractor
│   ├── phase4_fusion.py            # Multimodal fusion experiments
│   ├── phase5_explainability.py    # SHAP explainability
│   └── eval_on_original_val.py     # Validation evaluation
│
├── mental_health_model_package/    # PyTorch Tabular MLP package
│   ├── train_best_mlp.py           # Training script
│   ├── inference.py                # Standalone inference helper
│   ├── models/
│   │   └── numerical_classifier.pth   # Trained MLP weights
│   ├── artifacts/
│   │   ├── feature_scaler.pkl      # StandardScaler for 18 features
│   │   ├── target_scaler.pkl       # Scaler for regression targets
│   │   ├── label_encoder.pkl       # Label encoder for status classes
│   │   └── model_config.json       # Architecture configuration
│   └── README.md
│
├── models/                         # Keras + scikit-learn saved models
│   ├── facial_emotion_cnn.keras    # Keras CNN (48×48 → 7 emotions)
│   ├── audio_emotion_classifier.joblib   # Acoustic ensemble classifier
│   ├── audio_label_encoder.joblib  # Label encoder for audio classes
│   └── audio_class_names.joblib    # Class name list for audio
│
└── data/
    ├── mental_health_multimodal.csv   # Tabular training dataset
    └── images_quarantine/             # FER-2013 quarantine subset
        └── {emotion}/                 # Angry, Disgust, Fear, Happy, ...
```

---

## Mental Health Categories

| Status | Description |
|--------|-------------|
| **Healthy** | No significant stress indicators |
| **Mild Stress** | Early-stage stress signals |
| **Moderate Stress** | Elevated physiological & behavioural markers |
| **Severe Stress** | High-risk indicators across multiple modalities |

---

## Facial Emotion CNN — Confidence Thresholds

To prevent false positives in live webcam mode, the CNN output is gated per emotion:

| Emotion | Threshold | Reason |
|---------|:---------:|--------|
| Happy | 38% | Distinctive smile |
| Sad / Angry | 40% | Clear expressions |
| Fear | 42% | Slightly ambiguous |
| Surprise | 44% | Short-lived |
| Disgust | 48% | Often confused with Angry |
| **Neutral** | **52%** | Default fallback — needs clear majority |

---

## Input Features (18 total)

**Lifestyle & Behaviour** — Sleep Quality, Social Engagement, Daily App Usage, Typing Speed, Session Frequency, Idle Time  
**Facial & Vision** — Emotion Variance, Eye Blink Rate, Smile Intensity, Head Motion Index  
**Speech & Audio** — MFCC Mean, MFCC Variance, Pitch Mean, Speech Rate  
**Physiological** — Heart Rate BPM, HRV Index, Skin Temperature, GSR Level  

---

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/aryaagarwal2007/Hack4Health.git
cd Hack4Health

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## App Tabs

| Tab | Description |
|-----|-------------|
| **Overview** | Architecture cards, model status, quick-start guide |
| **Multimodal Fusion** | Upload face + audio → full gated diagnosis |
| **Tabular MLP** | Run tabular-only prediction with probability distribution |
| **Facial CNN** | Upload face image → all 7 emotion probabilities |
| **Audio Classifier** | Upload .wav → acoustic emotion + mental health mapping |
| **CNN Thresholds** | Per-emotion gate chart + emotion→status matrix |
| **XAI Explainer** | Feature attribution waterfall + ranked table |
| **Live Webcam** | Real-time capture with full prediction output panel |

---

## Tech Stack

- **Frontend**: Streamlit + custom CSS (Plus Jakarta Sans, Playfair Display)
- **Tabular model**: PyTorch 2.x — Multitask MLP (18 → 128 → 64 → [4-class + 3-reg])
- **Facial model**: TensorFlow / Keras — Grayscale CNN trained on FER-2013
- **Audio model**: scikit-learn — Ensemble on 280-dim MFCC + spectral features
- **Computer Vision**: OpenCV Haar cascades for real-time face detection

---

## Team

Built with ❤️ for **Hack4Health**
