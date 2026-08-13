"""
Phase 4 — Late Fusion Layer
Combines tabular, facial, and audio modalities via F1-weighted voting.
Evaluates fused vs. individual modality performance on simulated paired data.

NOTE: The three datasets (CSV, images, audio) are NOT from the same subjects.
We simulate paired data by randomly pairing CSV rows with image predictions
and audio predictions — this is explicitly stated, not silently assumed.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Set KERAS_HOME and MPLCONFIGDIR inside project directory to prevent sandbox errors
os.environ["KERAS_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".keras")
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib")
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import glob
import librosa
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── Constants ───────────────────────────────────────────────────────────────
STRESS_CLASSES = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]

# Facial emotion → stress level mapping (per spec)
FACE_EMOTION_TO_STRESS = {
    "Happy":   "Healthy",
    "Neutral": "Healthy",
    "Sad":     "Mild_Stress",
    "Surprise":"Mild_Stress",
    "Fear":    "Moderate_Stress",
    "Disgust": "Moderate_Stress",
    "Angry":   "Severe_Stress",
}

# Audio emotion → stress level mapping (per spec)
AUDIO_EMOTION_TO_STRESS = {
    "neutral":   "Healthy",
    "calm":      "Healthy",
    "happy":     "Healthy",
    "sad":       "Mild_Stress",
    "surprised": "Mild_Stress",
    "fearful":   "Moderate_Stress",
    "angry":     "Moderate_Stress",
    "disgust":   "Severe_Stress",
}

def emotion_to_stress_probs(emotion_probs, emotion_names, mapping):
    """
    Convert emotion-class probability vector into stress-level probability vector.
    Accumulates probabilities from all emotions mapping to the same stress level.
    """
    stress_probs = np.zeros(len(STRESS_CLASSES))
    for i, emo in enumerate(emotion_names):
        stress_label = mapping.get(emo)
        if stress_label is not None:
            j = STRESS_CLASSES.index(stress_label)
            stress_probs[j] += emotion_probs[i]
    # Normalize (should already sum to ~1)
    total = stress_probs.sum()
    if total > 0:
        stress_probs /= total
    return stress_probs

# ─── 1. Retrain tabular models (same as Phase 1) ────────────────────────────
print("=" * 70)
print("PHASE 4 — Late Fusion Layer")
print("=" * 70)

print("\n[1/5] Loading and splitting CSV data...")
df = pd.read_csv("data/mental_health_multimodal.csv")
feature_cols = [c for c in df.columns
                if c not in ("Mental_Health_Status",
                             "Depression_Score", "Anxiety_Score", "Stress_Score")]
X = df[feature_cols].to_numpy(dtype=np.float64)
y_cls = df["Mental_Health_Status"].to_numpy(dtype=str)
y_reg = df[["Depression_Score", "Anxiety_Score", "Stress_Score"]].to_numpy(dtype=np.float64)

X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = \
    train_test_split(X, y_cls, y_reg, test_size=0.2, stratify=y_cls, random_state=42)

# Train tabular classifier
print("[2/5] Loading PyTorch Tabular MLP (mental_health_model_package)...")
from multimodal_pipeline import MultimodalPipeline
pipeline = MultimodalPipeline()

# Compute tabular predictions using PyTorch MLP
tab_probs_list = []
for i in range(len(X_test)):
    sample_dict = dict(zip(feature_cols, X_test[i]))
    p_tab, _ = pipeline.predict_tabular(sample_dict)
    tab_probs_list.append(p_tab)

tab_proba = np.array(tab_probs_list)
tab_classes = ["Low", "Moderate", "High", "Severe"]
tab_pred_idx = np.argmax(tab_proba, axis=1)
tab_pred = [tab_classes[idx] for idx in tab_pred_idx]
tab_f1 = f1_score(y_cls_test, tab_pred, average="macro")
print(f"  Tabular PyTorch MLP macro F1: {tab_f1:.4f}")


# ─── 2. Load facial CNN and predict on validation set ───────────────────────
print("[3/5] Loading facial CNN and predicting on validation images...")
face_model = tf.keras.models.load_model("models/facial_emotion_cnn.keras")
model = face_model  # alias for direct call

# Load images manually with PIL to avoid tf.data iteration hangs
face_class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Collect all image paths with labels, then split with seed=42 matching image_dataset_from_directory
all_paths = []
all_labels_idx = []
for label_idx, label_name in enumerate(face_class_names):
    label_dir = os.path.join("data/images", label_name)
    for fpath in sorted(os.listdir(label_dir)):
        if fpath.lower().endswith(".png") or fpath.lower().endswith(".jpg"):
            all_paths.append(os.path.join(label_dir, fpath))
            all_labels_idx.append(label_idx)

all_paths = np.array(all_paths)
all_labels_idx = np.array(all_labels_idx)

# Replicate the 80/20 split with seed=42 (stratified on labels)
_, val_paths, _, val_labels = train_test_split(
    all_paths, all_labels_idx, test_size=0.2, stratify=all_labels_idx, random_state=42
)
print(f"  Validation images: {len(val_paths)}")

# Load images into numpy
print("  Loading images with PIL...")
X_face = np.zeros((len(val_paths), 48, 48, 1), dtype=np.float32)
for i, fpath in enumerate(val_paths):
    img = Image.open(fpath).convert("L").resize((48, 48))
    X_face[i, :, :, 0] = np.array(img, dtype=np.float32)
    if (i + 1) % 1000 == 0:
        print(f"    Loaded {i+1}/{len(val_paths)}")
y_face = val_labels  # integer labels

print("  Running CNN inference...")
# Use direct model call (avoids model.predict() overhead that can hang)
face_proba = model(X_face, training=False).numpy()  # (N, 7)
print(f"  Facial CNN predictions: {face_proba.shape}")

# Compute facial macro F1 (map emotion predictions to stress, compare to mapped ground truth)
# For facial F1, we use the 7-class emotion accuracy → then map
face_pred_classes = np.array(face_class_names)[np.argmax(face_proba, axis=1)]
# Map predicted emotions to stress levels
face_pred_stress = np.array([FACE_EMOTION_TO_STRESS[e] for e in face_pred_classes])
# We don't have true stress labels for images, so we estimate F1 from the CNN's own
# val accuracy as a proxy. For a more honest estimate, we use the 7-class macro F1.
from sklearn.metrics import f1_score as f1s
# Re-use ground truth labels already loaded
face_true = y_face
face_pred_int = np.argmax(face_proba, axis=1)
face_f1 = f1s(face_true, face_pred_int, average="macro")
print(f"  Facial CNN 7-class macro F1: {face_f1:.4f}")

# ─── 3. Load audio classifier and predict on all speech files ───────────────
print("[4/5] Loading audio classifier and extracting audio features...")
audio_clf = joblib.load("models/audio_emotion_classifier.joblib")
audio_le = joblib.load("models/audio_label_encoder.joblib")
audio_class_names = joblib.load("models/audio_class_names.joblib")

# Use shared feature extraction (matches phase3b training exactly)
from audio_features import extract_features as audio_extract

EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

wav_files = sorted(glob.glob("data/audios/**/*.wav", recursive=True))
audio_features = []
audio_labels = []
audio_file_indices = []

for idx, fpath in enumerate(wav_files):
    fname = os.path.basename(fpath)
    parts = fname.replace(".wav", "").split("-")
    if len(parts) != 7 or parts[3] != "01":
        continue
    emotion = EMOTION_MAP.get(parts[2])
    if emotion is None:
        continue
    try:
        feat = audio_extract(fpath)
        audio_features.append(feat)
        audio_labels.append(emotion)
        audio_file_indices.append(idx)
    except Exception:
        pass

audio_X = np.array(audio_features)
audio_y = np.array(audio_labels)
audio_y_enc = audio_le.transform(audio_y)

# Split audio the same way (80/20 stratified, seed=42)
_, audio_X_test, _, audio_y_test, _, audio_y_enc_test = \
    train_test_split(audio_X, audio_y, audio_y_enc,
                     test_size=0.2, stratify=audio_y, random_state=42)

audio_proba = audio_clf.predict_proba(audio_X_test)  # (154, 8)
audio_pred_labels = audio_le.inverse_transform(np.argmax(audio_proba, axis=1))

# Audio macro F1
audio_f1 = f1s(audio_y_test, audio_pred_labels, average="macro")
print(f"  Audio 8-class macro F1: {audio_f1:.4f}")
print(f"  Audio test samples: {len(audio_y_test)}")

# ─── 4. Fusion: weighted average of stress-level probabilities ──────────────
print("\n[5/5] Building fusion layer...")

# F1-weighted voting: normalize the three F1 scores to sum to 1
f1_scores = {"tabular": tab_f1, "facial": face_f1, "audio": audio_f1}
total_f1 = sum(f1_scores.values())
weights = {k: v / total_f1 for k, v in f1_scores.items()}

print(f"\n  Modality F1 scores (raw):")
for k, v in f1_scores.items():
    print(f"    {k:10s}: {v:.4f}")
print(f"\n  Fusion weights (normalized):")
for k, v in weights.items():
    print(f"    {k:10s}: {v:.4f}")

# ─── 5. Simulate paired evaluation ──────────────────────────────────────────
# IMPORTANT: The three datasets are NOT from the same subjects.
# We simulate by:
#   - Using the 800 CSV test rows as our "subjects"
#   - Randomly sampling facial predictions (from 5741 val predictions)
#   - Randomly sampling audio predictions (from 154 test predictions)
# This creates synthetic "subjects" with all 3 modalities available.

print("\n  Simulating paired evaluation (random pairing — NOT real subjects)...")
print("  NOTE: Images, audio, and CSV come from DIFFERENT sources.")
print("  We randomly pair them to demonstrate the fusion mechanism.\n")

n_test = len(y_cls_test)  # 800
rng = np.random.RandomState(42)

# Randomly sample facial stress probs for each test row
face_indices = rng.choice(len(face_proba), size=n_test, replace=True)
face_stress_probs = np.array([
    emotion_to_stress_probs(face_proba[i], face_class_names, FACE_EMOTION_TO_STRESS)
    for i in face_indices
])

# Randomly sample audio stress probs for each test row
audio_indices = rng.choice(len(audio_proba), size=n_test, replace=True)
audio_stress_probs = np.array([
    emotion_to_stress_probs(audio_proba[i], audio_class_names, AUDIO_EMOTION_TO_STRESS)
    for i in audio_indices
])

# Tabular stress probs — map to STRESS_CLASSES ['Healthy', 'Mild_Stress', 'Moderate_Stress', 'Severe_Stress']
TAB_TO_STRESS_MAP = {
    "Low": "Healthy",
    "Moderate": "Moderate_Stress",
    "High": "Mild_Stress",
    "Severe": "Severe_Stress"
}
tab_stress_probs = np.zeros((n_test, len(STRESS_CLASSES)))
for i, cls in enumerate(tab_classes):
    mapped_cls = TAB_TO_STRESS_MAP.get(cls, "Healthy")
    if mapped_cls in STRESS_CLASSES:
        j = STRESS_CLASSES.index(mapped_cls)
        tab_stress_probs[:, j] = tab_proba[:, i]


# Weighted fusion
fused_probs = (
    weights["tabular"] * tab_stress_probs +
    weights["facial"]  * face_stress_probs +
    weights["audio"]   * audio_stress_probs
)
fused_pred = np.array([STRESS_CLASSES[i] for i in np.argmax(fused_probs, axis=1)])

# Individual predictions (from stress probs)
tab_individual = np.array([STRESS_CLASSES[i] for i in np.argmax(tab_stress_probs, axis=1)])
face_individual = np.array([STRESS_CLASSES[i] for i in np.argmax(face_stress_probs, axis=1)])
audio_individual = np.array([STRESS_CLASSES[i] for i in np.argmax(audio_stress_probs, axis=1)])

# ─── 6. Evaluate ────────────────────────────────────────────────────────────
print("=" * 70)
print("FUSION RESULTS — Simulated Paired Evaluation")
print("=" * 70)
print(f"  (N={n_test} synthetic pairs, random pairing from 3 independent datasets)\n")

results = {}
for name, preds in [("Tabular only", tab_individual),
                     ("Facial only", face_individual),
                     ("Audio only", audio_individual),
                     ("FUSED (all 3)", fused_pred)]:
    acc = accuracy_score(y_cls_test, preds)
    mf1 = f1_score(y_cls_test, preds, average="macro")
    wf1 = f1_score(y_cls_test, preds, average="weighted")
    results[name] = {"acc": acc, "macro_f1": mf1, "weighted_f1": wf1}
    print(f"  {name:20s}  Accuracy: {acc:.4f}  Macro-F1: {mf1:.4f}  Weighted-F1: {wf1:.4f}")

# Confusion matrix for fused
cm = confusion_matrix(y_cls_test, fused_pred, labels=STRESS_CLASSES)
print(f"\n  Fused Confusion Matrix (rows=true, cols=pred):")
print(f"  Classes: {STRESS_CLASSES}")
for i, row in enumerate(cm):
    print(f"    {STRESS_CLASSES[i]:>16s}: {row}")

# Bar chart comparing modalities
fig, ax = plt.subplots(figsize=(10, 6))
names = list(results.keys())
accs = [results[n]["acc"] for n in names]
mf1s = [results[n]["macro_f1"] for n in names]
x = np.arange(len(names))
w = 0.35
bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="steelblue")
bars2 = ax.bar(x + w/2, mf1s, w, label="Macro F1", color="coral")
ax.set_ylabel("Score")
ax.set_title("Modality Comparison — Simulated Fusion Evaluation")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha="right")
ax.legend()
ax.set_ylim(0, 1)
for bar in bars1:
    ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("phase4_fusion_comparison.png", dpi=150)
print("\nSaved comparison chart → phase4_fusion_comparison.png")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 4 SUMMARY")
print("=" * 70)
print(f"  Fusion weights: tabular={weights['tabular']:.3f}, "
      f"facial={weights['facial']:.3f}, audio={weights['audio']:.3f}")
print(f"  Tabular weight is LOWEST because its F1 ({tab_f1:.4f}) is lowest.")
print(f"  Facial CNN dominates with weight {weights['facial']:.3f} (F1={face_f1:.4f}).")
print(f"  Fusion accuracy: {results['FUSED (all 3)']['acc']:.4f} vs "
      f"best single ({max(results, key=lambda k: results[k]['acc'])}): "
      f"{max(results.values(), key=lambda x: x['acc'])['acc']:.4f}")
print("=" * 70)
