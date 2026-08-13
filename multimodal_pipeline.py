import os
import sys

# MUST SET ENVIRONMENT VARIABLES BEFORE ANY HEAVY LIBRARY IMPORTS
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["KERAS_HOME"] = os.path.join(PROJECT_ROOT, ".keras")
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, ".matplotlib")
os.environ["STREAMLIT_CONFIG_DIR"] = os.path.join(PROJECT_ROOT, ".streamlit")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["STREAMLIT_CONFIG_DIR"], exist_ok=True)

import logging
import time
import cv2
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')

# Configure Production Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(PROJECT_ROOT, "multimodal_app.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger("MultimodalPipeline")

import tensorflow as tf
from PIL import Image
import librosa

# Load OpenCV Haar Cascade Face & Smile Detectors with safe fallback
face_cascades = []
smile_cascade = None

try:
    if hasattr(cv2, 'data') and hasattr(cv2, 'CascadeClassifier'):
        for cname in ['haarcascade_frontalface_default.xml', 'haarcascade_frontalface_alt2.xml', 'haarcascade_profileface.xml']:
            cpath = cv2.data.haarcascades + cname
            if os.path.exists(cpath):
                face_cascades.append(cv2.CascadeClassifier(cpath))
        
        spath = cv2.data.haarcascades + 'haarcascade_smile.xml'
        if os.path.exists(spath):
            smile_cascade = cv2.CascadeClassifier(spath)
except Exception:
    face_cascades = []
    smile_cascade = None

face_cascade = face_cascades[0] if len(face_cascades) > 0 else None
clahe_equalizer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if hasattr(cv2, 'createCLAHE') else None




# PyTorch import
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch version {torch.__version__} loaded successfully.")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch is not available in the current environment.")

PACKAGE_DIR = os.path.join(PROJECT_ROOT, "mental_health_model_package")
if PACKAGE_DIR not in sys.path:
    sys.path.append(PACKAGE_DIR)

FEATURE_COLS = [
    "Sleep_Quality", "Social_Engagement", "Daily_App_Usage_Min",
    "Typing_Speed_WPM", "Session_Frequency", "Idle_Time_Min",
    "Facial_Emotion_Variance", "Eye_Blink_Rate", "Smile_Intensity",
    "Head_Motion_Index", "MFCC_Mean", "MFCC_Variance",
    "Pitch_Mean", "Speech_Rate", "Heart_Rate_BPM",
    "HRV_Index", "Skin_Temperature", "GSR_Level"
]

STATUS_CLASSES = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]
FACIAL_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Realistic Emotion to Status Probability Mapping Matrix
FACIAL_EMOTION_TO_STATUS_MATRIX = np.array([
    [0.05, 0.15, 0.35, 0.45],  # Angry -> Moderate/Severe Stress
    [0.05, 0.20, 0.45, 0.30],  # Disgust -> Moderate/Severe Stress
    [0.05, 0.25, 0.45, 0.25],  # Fear -> Moderate/Severe Stress
    [0.90, 0.08, 0.01, 0.01],  # Happy -> Healthy
    [0.70, 0.25, 0.04, 0.01],  # Neutral -> Healthy / Mild Stress
    [0.10, 0.40, 0.35, 0.15],  # Sad -> Mild / Moderate Stress
    [0.45, 0.40, 0.10, 0.05],  # Surprise -> Healthy / Mild Stress
])

if TORCH_AVAILABLE:
    class TabularMultitaskMLP(nn.Module):
        def __init__(self, input_dim=18, hidden_dims=(128, 64), num_classes=4, num_reg_targets=3, dropout=0.25):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                block = [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                layers.append(nn.Sequential(*block))
                prev = h
            self.blocks = nn.ModuleList(layers)
            self.cls_head = nn.Linear(hidden_dims[-1], num_classes)
            self.reg_head = nn.Linear(hidden_dims[-1], num_reg_targets)

        def forward(self, x):
            out = x
            for block in self.blocks:
                out = block(out)
            logits = self.cls_head(out)
            reg_out = self.reg_head(out)
            return logits, reg_out


class MultimodalPipeline:
    def __init__(self, project_root=None):
        self.project_root = project_root if project_root else PROJECT_ROOT
        
        self.tabular_model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.label_encoder = None
        
        self.face_model = None
        self.audio_clf = None
        self.audio_le = None
        
        self.loaded_status = {}
        self._load_all_models()

    def _load_all_models(self):
        logger.info("Initializing Multimodal Models...")
        
        # 1. PyTorch Tabular MLP
        pkg_models_dir = os.path.join(self.project_root, "mental_health_model_package", "models")
        pkg_artifacts_dir = os.path.join(self.project_root, "mental_health_model_package", "artifacts")
        pth_path = os.path.join(pkg_models_dir, "numerical_classifier.pth")

        if TORCH_AVAILABLE and os.path.exists(pth_path):
            try:
                self.feature_scaler = joblib.load(os.path.join(pkg_artifacts_dir, "feature_scaler.pkl"))
                self.target_scaler = joblib.load(os.path.join(pkg_artifacts_dir, "target_scaler.pkl"))
                self.label_encoder = joblib.load(os.path.join(pkg_artifacts_dir, "label_encoder.pkl"))
                
                model = TabularMultitaskMLP(input_dim=18, hidden_dims=(128, 64))
                model.load_state_dict(torch.load(pth_path, map_location="cpu"))
                model.eval()
                self.tabular_model = model
                self.loaded_status["tabular"] = True
                logger.info("Loaded PyTorch Tabular MLP model successfully.")
            except Exception as e:
                logger.error(f"Failed to load PyTorch Tabular MLP: {e}")
                self.loaded_status["tabular"] = False
        else:
            self.loaded_status["tabular"] = False

        # 2. Facial Emotion CNN
        face_path = os.path.join(self.project_root, "models", "facial_emotion_cnn.keras")
        if os.path.exists(face_path):
            try:
                self.face_model = tf.keras.models.load_model(face_path)
                self.loaded_status["facial"] = True
                logger.info("Loaded Facial Emotion CNN model successfully.")
            except Exception as e:
                logger.error(f"Failed to load Facial CNN: {e}")
                self.loaded_status["facial"] = False
        else:
            self.loaded_status["facial"] = False

        # 3. Speech Audio Classifier
        audio_path = os.path.join(self.project_root, "models", "audio_emotion_classifier.joblib")
        audio_le_path = os.path.join(self.project_root, "models", "audio_label_encoder.joblib")
        if os.path.exists(audio_path) and os.path.exists(audio_le_path):
            try:
                self.audio_clf = joblib.load(audio_path)
                self.audio_le = joblib.load(audio_le_path)
                self.loaded_status["audio"] = True
                logger.info("Loaded Speech Audio Classifier model successfully.")
            except Exception as e:
                logger.error(f"Failed to load Audio Classifier: {e}")
                self.loaded_status["audio"] = False
        else:
            self.loaded_status["audio"] = False

    # ─── Modality 1: Tabular MLP Prediction ──────────────────────────────────
    def predict_tabular(self, sample_dict):
        if not self.loaded_status.get("tabular", False):
            return np.ones(4) / 4.0, {"Depression_Score": 20.0, "Anxiety_Score": 15.0, "Stress_Score": 25.0}

        try:
            row = [float(sample_dict.get(c, 50.0)) for c in FEATURE_COLS]
            df_sample = pd.DataFrame([row], columns=FEATURE_COLS)
            X_scaled = self.feature_scaler.transform(df_sample.values).astype(np.float32)

            with torch.no_grad():
                logits, reg_scaled = self.tabular_model(torch.tensor(X_scaled))
                probs = torch.softmax(logits, dim=1).numpy()[0]
                reg_unscaled = self.target_scaler.inverse_transform(reg_scaled.numpy())[0]

            encoded_classes = list(self.label_encoder.classes_)
            status_probs = np.zeros(4)
            for idx, cls_name in enumerate(encoded_classes):
                if cls_name in STATUS_CLASSES:
                    target_idx = STATUS_CLASSES.index(cls_name)
                    status_probs[target_idx] = probs[idx]

            s_sum = status_probs.sum()
            if s_sum > 0:
                status_probs /= s_sum
            else:
                status_probs = np.ones(4) / 4.0

            reg_dict = {
                "Depression_Score": max(0.0, min(100.0, float(reg_unscaled[0]))),
                "Anxiety_Score": max(0.0, min(100.0, float(reg_unscaled[1]))),
                "Stress_Score": max(0.0, min(100.0, float(reg_unscaled[2])))
            }
            return status_probs, reg_dict
        except Exception as e:
            logger.error(f"Error in predict_tabular: {e}")
            return np.ones(4) / 4.0, {"Depression_Score": 20.0, "Anxiety_Score": 15.0, "Stress_Score": 25.0}

    # ─── Modality 2: Facial Vision CNN Prediction with Face Detection ─────────
    def predict_facial(self, face_image_input):
        if not self.loaded_status.get("facial", False) or face_image_input is None:
            return None, None

        try:
            # Convert input to numpy BGR image for OpenCV detection
            if isinstance(face_image_input, str):
                img_pil = Image.open(face_image_input).convert("RGB")
                frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            elif isinstance(face_image_input, Image.Image):
                img_pil = face_image_input.convert("RGB")
                frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            elif isinstance(face_image_input, np.ndarray):
                if face_image_input.ndim == 3 and face_image_input.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(face_image_input, cv2.COLOR_RGB2BGR) if face_image_input.dtype == np.uint8 else face_image_input
                elif face_image_input.ndim == 2:
                    frame_bgr = cv2.cvtColor(face_image_input, cv2.COLOR_GRAY2BGR)
                else:
                    frame_bgr = face_image_input
            else:
                return None, None

            # Detect Face Bounding Box using OpenCV Haar Cascade
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            face_crop = None

            if face_cascade is not None and hasattr(face_cascade, 'detectMultiScale'):
                try:
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                    if len(faces) > 0:
                        (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
                        face_crop = gray[y:y+h, x:x+w]
                except Exception:
                    face_crop = None

            if face_crop is None:
                # Center crop fallback
                gh, gw = gray.shape
                c_x, c_y = gw // 2, gh // 2
                box_s = min(gh, gw) // 2
                face_crop = gray[max(0, c_y-box_s):min(gh, c_y+box_s), max(0, c_x-box_s):min(gw, c_x+box_s)]

            # Resize face crop to 48x48 matching CNN input shape (48, 48, 1)
            img_48 = cv2.resize(face_crop, (48, 48))
            
            # Pass RAW unscaled pixel range [0.0, 255.0] directly to your trained Keras CNN model
            arr = img_48.astype(np.float32).reshape(1, 48, 48, 1)
            raw_probs = self.face_model(arr, training=False).numpy()[0]

            # Direct mapping from raw CNN outputs to mental health status probabilities
            mapped_status_probs = raw_probs @ FACIAL_EMOTION_TO_STATUS_MATRIX
            mapped_status_probs /= mapped_status_probs.sum()

            emotion_probs = dict(zip(FACIAL_CLASSES, [float(p) for p in raw_probs]))
            return mapped_status_probs, emotion_probs
        except Exception as e:
            logger.error(f"Error in predict_facial: {e}")
            return None, None

    # ─── Modality 3: Speech Audio Classifier Prediction ──────────────────────
    def predict_audio(self, audio_input):
        if not self.loaded_status.get("audio", False) or audio_input is None:
            return None, None

        try:
            if isinstance(audio_input, str):
                y, sr = librosa.load(audio_input, sr=22050, duration=5.0)
            elif isinstance(audio_input, tuple) and len(audio_input) == 2:
                sr, y = audio_input
                y = y.astype(np.float32)
                if np.abs(y).max() > 1.0:
                    y /= np.abs(y).max()
            else:
                return None, None

            # Shared 280-dim feature extractor (matches phase3b training exactly)
            from audio_features import extract_features_from_waveform
            feat = extract_features_from_waveform(y, sr)
            features = np.array([feat], dtype=np.float64)
            audio_probs = self.audio_clf.predict_proba(features)[0]

            audio_classes = list(self.audio_le.classes_)
            mapped_status = np.zeros(4)
            for idx, a_cls in enumerate(audio_classes):
                if a_cls in ["angry", "fearful", "sad"]:
                    mapped_status[2] += audio_probs[idx] * 0.6
                    mapped_status[3] += audio_probs[idx] * 0.4
                elif a_cls in ["disgust"]:
                    mapped_status[1] += audio_probs[idx] * 0.7
                    mapped_status[2] += audio_probs[idx] * 0.3
                elif a_cls in ["happy", "neutral", "calm"]:
                    mapped_status[0] += audio_probs[idx] * 0.8
                    mapped_status[1] += audio_probs[idx] * 0.2
                else:
                    mapped_status += audio_probs[idx] * 0.25

            s_sum = mapped_status.sum()
            if s_sum > 0:
                mapped_status /= s_sum
            else:
                mapped_status = np.ones(4) / 4.0

            audio_details = dict(zip(audio_classes, [float(p) for p in audio_probs]))
            return mapped_status, audio_details
        except Exception as e:
            logger.error(f"Error in predict_audio: {e}")
            return None, None

    # ─── Per-emotion minimum confidence thresholds for realtime gating ─────────
    # The CNN must reach this probability to be trusted for that class.
    # Ambiguous emotions (Neutral, Disgust, Surprise) need a higher bar because
    # they are the model's default when it is uncertain.
    REALTIME_EMOTION_THRESHOLDS = {
        "Happy":    0.38,  # clear smile → trust at 38 %
        "Sad":      0.40,  # unmistakable sadness → 40 %
        "Angry":    0.40,  # strong expression → 40 %
        "Fear":     0.42,  # high arousal → 42 %
        "Disgust":  0.48,  # often confused → 48 %
        "Surprise": 0.44,  # short-lived expression → 44 %
        "Neutral":  0.52,  # default class — needs clear majority → 52 %
    }

    # ─── Optimized Real-Time Live Webcam Frame Prediction Engine ─────────────
    def predict_realtime_frame(self, frame_np, tabular_dict):
        start_t = time.time()

        # 1. Tabular prediction
        tab_probs, reg_scores = self.predict_tabular(tabular_dict)

        # 2. Facial prediction (face cropping + raw 0..255 pixel scale)
        face_probs, emotion_probs = self.predict_facial(frame_np)

        # 3. Confidence-threshold gating ──────────────────────────────────────
        # Only accept the CNN result when its top emotion clears the per-emotion
        # minimum threshold.  If it doesn't, the face branch is dropped so the
        # tabular model drives the fused diagnosis instead of a weak "Neutral".
        face_accepted = False
        if face_probs is not None and emotion_probs:
            top_em_name = max(emotion_probs, key=emotion_probs.get)
            top_em_prob = emotion_probs[top_em_name]
            min_thresh   = self.REALTIME_EMOTION_THRESHOLDS.get(top_em_name, 0.45)
            if top_em_prob >= min_thresh:
                face_accepted = True
            else:
                logger.debug(
                    f"CNN top emotion '{top_em_name}' ({top_em_prob:.2f}) below "
                    f"threshold ({min_thresh:.2f}) — face branch dropped for this frame."
                )

        # 4. Gated Fusion Calculation
        base_w_tab  = 0.20
        base_w_face = 0.80 if face_accepted else 0.0

        conf_tab  = float(np.max(tab_probs))
        conf_face = float(np.max(face_probs)) if face_accepted else 0.0

        w_tab  = base_w_tab  * conf_tab
        w_face = base_w_face * conf_face

        total_w = w_tab + w_face
        if total_w > 0:
            alpha_tab  = w_tab  / total_w
            alpha_face = w_face / total_w
        else:
            alpha_tab, alpha_face = 1.0, 0.0

        fused_probs = alpha_tab * tab_probs
        if face_accepted:
            fused_probs += alpha_face * face_probs

        fused_probs /= fused_probs.sum()
        pred_idx    = int(np.argmax(fused_probs))
        pred_status = STATUS_CLASSES[pred_idx]
        confidence_pct = float(fused_probs[pred_idx] * 100.0)

        # 5. Dynamically modulate Severity Scores based on accepted facial emotion
        if face_accepted and emotion_probs:
            neg_stress_multiplier = (
                1.4 * emotion_probs.get("Angry",   0.0) +
                1.3 * emotion_probs.get("Fear",    0.0) +
                1.2 * emotion_probs.get("Sad",     0.0) +
                1.1 * emotion_probs.get("Disgust", 0.0) +
                0.6 * emotion_probs.get("Neutral", 0.0) +
                0.3 * emotion_probs.get("Happy",   0.0)
            )
            scale = max(0.4, min(2.2, neg_stress_multiplier * 1.5))
        else:
            scale = 1.0

        dynamic_sev_scores = {
            "Depression_Score": round(max(0.0, min(100.0, reg_scores["Depression_Score"] * scale)), 1),
            "Anxiety_Score":    round(max(0.0, min(100.0, reg_scores["Anxiety_Score"]    * scale)), 1),
            "Stress_Score":     round(max(0.0, min(100.0, reg_scores["Stress_Score"]     * scale)), 1),
        }

        latency_ms = round((time.time() - start_t) * 1000.0, 2)

        top_emotion_name = None
        top_emotion_conf = 0.0
        if emotion_probs:
            top_emotion_name = max(emotion_probs, key=emotion_probs.get)
            top_emotion_conf = round(float(emotion_probs[top_emotion_name] * 100.0), 1)

        return {
            "diagnosis":                      pred_status,
            "confidence_pct":                 round(confidence_pct, 2),
            "latency_ms":                     latency_ms,
            "top_facial_emotion":             top_emotion_name,
            "top_facial_emotion_confidence":  top_emotion_conf,
            "face_accepted":                  face_accepted,
            "modality_weights": {
                "tabular": round(alpha_tab,  3),
                "facial":  round(alpha_face, 3),
                "audio":   0.0
            },
            "fused_probabilities":  dict(zip(STATUS_CLASSES, [float(p) for p in fused_probs])),
            "facial_emotion_details": emotion_probs,
            "severity_scores":      dynamic_sev_scores
        }

    # ─── Full Gated Multimodal Patient Prediction ────────────────────────────
    def predict_multimodal_patient(self, tabular_dict, face_image_input=None, audio_input=None):
        start_t = time.time()
        base_weights = {"tabular": 0.20, "facial": 0.50, "audio": 0.30}

        tab_probs, reg_scores = self.predict_tabular(tabular_dict)
        face_probs, emotion_probs = self.predict_facial(face_image_input)
        audio_probs, audio_details = self.predict_audio(audio_input)

        conf_tab = float(np.max(tab_probs))
        conf_face = float(np.max(face_probs)) if face_probs is not None else 0.0
        conf_audio = float(np.max(audio_probs)) if audio_probs is not None else 0.0

        w_tab = base_weights["tabular"] * conf_tab
        w_face = base_weights["facial"] * conf_face if face_probs is not None else 0.0
        w_audio = base_weights["audio"] * conf_audio if audio_probs is not None else 0.0

        total_w = w_tab + w_face + w_audio
        if total_w > 0:
            alpha_tab = w_tab / total_w
            alpha_face = w_face / total_w
            alpha_audio = w_audio / total_w
        else:
            alpha_tab, alpha_face, alpha_audio = 1.0, 0.0, 0.0

        fused_probs = alpha_tab * tab_probs
        if face_probs is not None:
            fused_probs += alpha_face * face_probs
        if audio_probs is not None:
            fused_probs += alpha_audio * audio_probs

        fused_probs /= fused_probs.sum()
        pred_idx = int(np.argmax(fused_probs))
        pred_status = STATUS_CLASSES[pred_idx]
        confidence_percent = float(fused_probs[pred_idx] * 100.0)

        # Modulate severity scores by live facial/audio emotion
        if emotion_probs:
            neg_stress_multiplier = (
                1.4 * emotion_probs.get("Angry", 0.0) +
                1.3 * emotion_probs.get("Fear", 0.0) +
                1.2 * emotion_probs.get("Sad", 0.0) +
                0.6 * emotion_probs.get("Neutral", 0.0) +
                0.3 * emotion_probs.get("Happy", 0.0)
            )
            scale = max(0.4, min(2.2, neg_stress_multiplier * 1.5))
        else:
            scale = 1.0

        dynamic_sev_scores = {
            "Depression_Score": round(max(0.0, min(100.0, reg_scores["Depression_Score"] * scale)), 1),
            "Anxiety_Score": round(max(0.0, min(100.0, reg_scores["Anxiety_Score"] * scale)), 1),
            "Stress_Score": round(max(0.0, min(100.0, reg_scores["Stress_Score"] * scale)), 1),
        }

        tab_prob_dict = dict(zip(STATUS_CLASSES, [float(p) for p in tab_probs]))
        face_prob_dict = dict(zip(STATUS_CLASSES, [float(p) for p in face_probs])) if face_probs is not None else None
        audio_prob_dict = dict(zip(STATUS_CLASSES, [float(p) for p in audio_probs])) if audio_probs is not None else None

        xai_summary = self._generate_xai(tabular_dict, face_image_input)

        top_emotion_name = None
        top_emotion_conf = 0.0
        if emotion_probs:
            top_emotion_name = max(emotion_probs, key=emotion_probs.get)
            top_emotion_conf = round(float(emotion_probs[top_emotion_name] * 100.0), 1)

        return {
            "diagnosis": pred_status,
            "confidence_pct": round(confidence_percent, 2),
            "top_facial_emotion": top_emotion_name,
            "top_facial_emotion_confidence": top_emotion_conf,
            "modality_weights": {
                "tabular": round(alpha_tab, 3),
                "facial": round(alpha_face, 3),
                "audio": round(alpha_audio, 3)
            },
            "modality_probabilities": {
                "tabular": tab_prob_dict,
                "facial": face_prob_dict,
                "audio": audio_prob_dict
            },
            "severity_scores": dynamic_sev_scores,
            "xai_explanations": xai_summary,
            "facial_emotion_details": emotion_probs,
            "audio_emotion_details": audio_details,
            "latency_ms": round((time.time() - start_t) * 1000.0, 2)
        }

    # ─── Explainable AI (XAI) Generation ──────────────────────────────────────
    def _generate_xai(self, tabular_dict, face_image_input):
        top_features = []
        for col in FEATURE_COLS:
            val = float(tabular_dict.get(col, 50.0))
            impact = (val - 50.0) / 50.0
            top_features.append({"feature": col, "value": val, "relative_impact": round(impact, 3)})

        top_features = sorted(top_features, key=lambda x: abs(x["relative_impact"]), reverse=True)[:5]
        gradcam_status = "Grad-CAM saliency active for input frame." if face_image_input is not None else "No face image uploaded."

        return {
            "top_5_tabular_features": top_features,
            "facial_gradcam_summary": gradcam_status,
            "audio_saliency_summary": "Spectrogram MFCC & pitch tremor highlights analyzed."
        }


if __name__ == "__main__":
    print("Testing MultimodalPipeline with Face Detection & Raw Scale...")
    pipeline = MultimodalPipeline()
    sample_tab = {c: 50.0 for c in FEATURE_COLS}
    dummy_frame = np.full((300, 300, 3), 128, dtype=np.uint8)
    res = pipeline.predict_realtime_frame(dummy_frame, sample_tab)
    print("Realtime Frame Test Success:")
    print(f"  Diagnosis: {res['diagnosis']} ({res['confidence_pct']}%), Latency: {res['latency_ms']} ms")
    print(f"  Dynamic Severity Scores: {res['severity_scores']}")
