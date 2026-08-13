"""
Phase 3 — Audio Emotion Classifier
Extracts MFCC + pitch + ZCR features from RAVDESS-style .wav files and
trains an XGBoost classifier for 8-way emotion recognition.
"""

import os
import glob
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─── RAVDESS filename parsing ────────────────────────────────────────────────
# Filename format: XX-YY-ZZ-AA-BB-CC-DD.wav
# Position 3 (index 2) = emotion label:
#   01=neutral, 02=calm, 03=happy, 04=sad, 05=angry,
#   06=fearful, 07=disgust, 08=surprised
# We filter out vocal_channel != 01 (position 4, index 3) if song files present

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

def parse_filename(filepath):
    """Parse RAVDESS filename and return (emotion_label, vocal_channel)."""
    fname = os.path.basename(filepath)
    parts = fname.replace(".wav", "").split("-")
    if len(parts) != 7:
        return None, None
    emotion_code = parts[2]
    vocal_channel = parts[3]
    emotion = EMOTION_MAP.get(emotion_code, None)
    return emotion, vocal_channel

# ─── Collect wav files and parse labels ──────────────────────────────────────
print("Scanning for .wav files in data/audios/ ...")
wav_files = sorted(glob.glob("data/audios/**/*.wav", recursive=True))
print(f"Found {len(wav_files)} total .wav files")

features_list = []
labels = []
skipped = 0

for i, fpath in enumerate(wav_files):
    emotion, vocal_channel = parse_filename(fpath)

    # Filter: only speech (vocal_channel == "01"), skip song ("02" if present)
    if vocal_channel != "01":
        skipped += 1
        continue

    if emotion is None:
        skipped += 1
        continue

    # Load audio and extract features
    try:
        y, sr = librosa.load(fpath, sr=None, duration=5.0)

        # 40 MFCCs — mean and variance across frames
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfccs, axis=1)  # shape (40,)
        mfcc_var = np.var(mfccs, axis=1)    # shape (40,)

        # Pitch via YIN
        f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"),
                         fmax=librosa.note_to_hz("C7"), sr=sr)
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) > 0:
            pitch_mean = np.mean(f0_valid)
            pitch_std = np.std(f0_valid)
            pitch_range = np.max(f0_valid) - np.min(f0_valid)
        else:
            pitch_mean = 0.0
            pitch_std = 0.0
            pitch_range = 0.0

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_var = np.var(zcr)

        # Combine feature vector
        feat = np.concatenate([
            mfcc_mean,        # 40
            mfcc_var,         # 40
            [pitch_mean, pitch_std, pitch_range, zcr_mean, zcr_var],  # 5
        ])  # total = 85

        features_list.append(feat)
        labels.append(emotion)

    except Exception as e:
        skipped += 1
        if i < 5:
            print(f"  Warning: could not process {fpath}: {e}")

    if (i + 1) % 200 == 0:
        print(f"  Processed {i + 1}/{len(wav_files)} files...")

print(f"\nProcessed: {len(features_list)} files, skipped: {skipped}")

X = np.array(features_list)
y = np.array(labels)
print(f"Feature matrix shape: {X.shape}")
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# Encode string labels to integers for XGBoost
le = LabelEncoder()
class_names = sorted(np.unique(y))
le.fit(class_names)
y_enc = le.transform(y)

# ─── Train / test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# ─── Train XGBoost classifier ───────────────────────────────────────────────
print("\nTraining XGBoost audio classifier...")
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
        use_label_encoder=False,
    )),
])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# ─── Evaluate ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EVALUATION — Audio Emotion Classifier")
print("=" * 70)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print(f"\n  Accuracy:     {acc:.4f}")
print(f"  Macro F1:     {f1_macro:.4f}")
print(f"  Weighted F1:  {f1_weighted:.4f}")

# Decode integer predictions back to labels for evaluation
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels, target_names=class_names))

cm = confusion_matrix(y_test_labels, y_pred_labels, labels=class_names)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Audio Emotion Classifier — Confusion Matrix")
plt.tight_layout()
plt.savefig("phase3_confusion_matrix.png", dpi=150)
print("Saved confusion matrix plot → phase3_confusion_matrix.png")

# ─── Save model ──────────────────────────────────────────────────────────────
import joblib
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/audio_emotion_classifier.joblib")
print("Saved model → models/audio_emotion_classifier.joblib")

# Also save the label encoder and class names
joblib.dump(le, "models/audio_label_encoder.joblib")
joblib.dump(class_names, "models/audio_class_names.joblib")

print("\n" + "=" * 70)
print(f"FINAL — Audio classifier accuracy: {acc:.4f}, macro F1: {f1_macro:.4f}")
print("=" * 70)
