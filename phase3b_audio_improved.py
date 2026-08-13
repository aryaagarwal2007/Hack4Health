"""
Phase 3b — Audio Emotion Classifier (Improved Features)
Adds delta/delta-delta MFCCs, spectral centroid/rolloff/bandwidth/flatness,
RMS energy, and chroma on top of the base MFCC+pitch+ZCR features.
"""

import os
import glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from audio_features import extract_features

EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}


# ─── Collect and extract ─────────────────────────────────────────────────────
print("Scanning for .wav files ...")
wav_files = sorted(glob.glob("data/audios/**/*.wav", recursive=True))
print(f"Found {len(wav_files)} files")

features_list, labels = [], []
skipped = 0
for i, fpath in enumerate(wav_files):
    parts = os.path.basename(fpath).replace(".wav", "").split("-")
    if len(parts) != 7 or parts[3] != "01":
        skipped += 1
        continue
    emotion = EMOTION_MAP.get(parts[2])
    if emotion is None:
        skipped += 1
        continue
    try:
        features_list.append(extract_features(fpath))
        labels.append(emotion)
    except Exception:
        skipped += 1
    if (i + 1) % 200 == 0:
        print(f"  Processed {i+1}/{len(wav_files)} ...")

print(f"\nProcessed: {len(features_list)}, skipped: {skipped}")
X = np.array(features_list)
y = np.array(labels)
print(f"Feature matrix: {X.shape}")

le = LabelEncoder()
class_names = sorted(np.unique(y))
le.fit(class_names)
y_enc = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# ─── Train tuned XGBoost ─────────────────────────────────────────────────────
print("\nTraining tuned XGBoost audio classifier ...")
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        random_state=42, eval_metric="mlogloss",
    )),
])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# ─── Evaluate ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EVALUATION — Audio Classifier (Improved Features)")
print("=" * 70)
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")
print(f"\n  Accuracy:     {acc:.4f}")
print(f"  Macro F1:     {f1_macro:.4f}")
print(f"  Weighted F1:  {f1_weighted:.4f}")

y_pred_lbl = le.inverse_transform(y_pred)
y_test_lbl = le.inverse_transform(y_test)
print("\nClassification Report:")
print(classification_report(y_test_lbl, y_pred_lbl, target_names=class_names))

cm = confusion_matrix(y_test_lbl, y_pred_lbl, labels=class_names)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Audio Classifier (Improved) — Confusion Matrix")
plt.tight_layout()
plt.savefig("phase3b_confusion_matrix.png", dpi=150)
print("Saved → phase3b_confusion_matrix.png")

joblib.dump(clf, "models/audio_emotion_classifier.joblib")
joblib.dump(le, "models/audio_label_encoder.joblib")
joblib.dump(class_names, "models/audio_class_names.joblib")
print("Saved model → models/audio_emotion_classifier.joblib (overwrote)")

print("\n" + "=" * 70)
print(f"FINAL — Audio (improved) accuracy: {acc:.4f}, macro F1: {f1_macro:.4f}")
print("=" * 70)
