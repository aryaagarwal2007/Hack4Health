"""
Fair comparison: evaluate the CNN trained on CLEANED data against the
ORIGINAL validation set (the exact 5741 files used before cleaning).
Replicates keras' seed=42 stratified split on the original 28,709-file list.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ─── Reconstruct the ORIGINAL 28,709-file list (sorted, as keras sees it) ───
paths = []
for p in sorted(glob.glob("data/images/*/*")):
    if os.path.isfile(p):
        paths.append(p)
# Add quarantined files back at their ORIGINAL locations
for p in sorted(glob.glob("data/images_quarantine/*/*")):
    if os.path.isfile(p):
        cls = os.path.basename(os.path.dirname(p))
        paths.append(os.path.join("data/images", cls, os.path.basename(p)))
paths = sorted(paths)
labels = np.array([CLASS_NAMES.index(p.split(os.sep)[-2]) for p in paths])
print(f"Reconstructed original dataset: {len(paths)} files")

# Verify keras split replication against the CURRENT on-disk state
idx = np.arange(len(paths))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)

# Sanity check: files at original locations in the val split should mostly
# still exist on disk (quarantined ones were train-side, so val untouched)
orig_val_paths = [paths[i] for i in va_idx]
missing = [p for p in orig_val_paths if not os.path.exists(p)]
print(f"Original val split: {len(orig_val_paths)} files, missing from disk: {len(missing)}")
if len(missing) > len(orig_val_paths) * 0.01:
    print("WARNING: split replication looks wrong (too many missing val files).")

# ─── Load the original val images ────────────────────────────────────────────
valid = [(p, labels[i]) for i, p in zip(va_idx, orig_val_paths) if os.path.exists(p)]
print(f"Loading {len(valid)} validation images ...")
X = np.empty((len(valid), 48, 48, 1), dtype=np.float32)
y = np.zeros(len(valid), dtype=int)
for k, (p, lab) in enumerate(valid):
    img = Image.open(p).convert("L").resize((48, 48))
    X[k, :, :, 0] = np.array(img, dtype=np.float32)
    y[k] = lab

# ─── Head-to-head: OLD (pre-cleaning) vs NEW (cleaned) on identical set ─────
def evaluate(model_path, X, y):
    m = tf.keras.models.load_model(model_path)
    probs = []
    for i in range(0, len(X), 2048):
        out = m(X[i:i + 2048], training=False)
        probs.append(out.numpy() if hasattr(out, "numpy") else np.array(out))
    return np.concatenate(probs).argmax(axis=1)

print(f"\nEvaluating OLD model (pre-cleaning) ...")
pred_old = evaluate("/tmp/facial_cnn_pre_cleaning.keras", X, y)
print(f"Evaluating NEW model (trained on cleaned data) ...")
pred_new = evaluate("models/facial_emotion_cnn.keras", X, y)

acc_old = (pred_old == y).mean()
acc_new = (pred_new == y).mean()
print(f"\n=== HEAD-TO-HEAD on identical {len(y)} val images ===")
print(f"OLD (pre-cleaning) accuracy: {acc_old:.4f}")
print(f"NEW (cleaned data) accuracy: {acc_new:.4f}")
print(f"Delta: {acc_new - acc_old:+.4f}")
print("\nPer-class F1 (old -> new):")
from sklearn.metrics import f1_score
for ci, cn in enumerate(CLASS_NAMES):
    f_old = f1_score(y == ci, pred_old == ci)
    f_new = f1_score(y == ci, pred_new == ci)
    print(f"  {cn:>9}: {f_old:.2f} -> {f_new:.2f} ({f_new - f_old:+.2f})")
print("\nNEW model full report:")
print(classification_report(y, pred_new, target_names=CLASS_NAMES, digits=2))
