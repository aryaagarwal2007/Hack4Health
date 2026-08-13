"""
Dataset Cleaning — Confidence-based label-noise detection.
Uses the trained facial CNN to find training images whose folder label is
very likely wrong (model disagrees at >=85% confidence) and MOVES them to
data/images_quarantine/<Class>/ — reversible, nothing is deleted.

Validation split membership is computed with the same seed=42 split so the
quarantine only touches training-side files.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import shutil
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONF_THRESHOLD = 0.85  # only quarantine when model is very confident

# ─── Load model ──────────────────────────────────────────────────────────────
print("Loading facial CNN ...")
model = tf.keras.models.load_model("models/facial_emotion_cnn.keras")

# ─── Get training-split file paths (same split keras uses, seed=42) ─────────
print("Computing training split (seed=42) ...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/images", validation_split=0.2, subset="training", seed=42,
    color_mode="grayscale", image_size=(48, 48), batch_size=256,
    label_mode="int",
)
class_names = train_ds.class_names
paths = list(train_ds.file_paths)
labels = np.array([train_ds.class_names.index(p.split(os.sep)[-2]) for p in paths])
print(f"Training split: {len(paths)} images, classes: {class_names}")

# ─── Load images with PIL (matches phase4 approach) ─────────────────────────
print("Loading images ...")
X = np.empty((len(paths), 48, 48, 1), dtype=np.float32)
for i, p in enumerate(paths):
    img = Image.open(p).convert("L").resize((48, 48))
    X[i, :, :, 0] = np.array(img, dtype=np.float32)
    if (i + 1) % 5000 == 0:
        print(f"  Loaded {i + 1}/{len(paths)}")

# ─── Inference in batches (direct call — avoids predict() hangs) ────────────
print("Running CNN inference ...")
probs = []
BATCH = 2048
for i in range(0, len(X), BATCH):
    out = model(X[i:i + BATCH], training=False)
    probs.append(out.numpy() if hasattr(out, "numpy") else np.array(out))
probs = np.concatenate(probs, axis=0)

pred = probs.argmax(axis=1)
conf = probs.max(axis=1)

# ─── Flag confident disagreements ────────────────────────────────────────────
flagged = (pred != labels) & (conf >= CONF_THRESHOLD)
print(f"\nFlagged as likely mislabeled: {flagged.sum()} / {len(paths)} "
      f"({100 * flagged.sum() / len(paths):.1f}%)")

print("\nPer-class breakdown (flagged / total):")
for ci, cn in enumerate(class_names):
    mask = labels == ci
    n_flag = int((flagged & mask).sum())
    print(f"  {cn:>9}: {n_flag:>5} / {int(mask.sum()):>6}"
          f"  ({100 * n_flag / max(mask.sum(), 1):.1f}%)")

# Where do flagged images get re-assigned?
print("\nTop mislabel directions (true -> model-predicted):")
from collections import Counter
directions = Counter((class_names[labels[i]], class_names[pred[i]])
                     for i in np.where(flagged)[0])
for (t, p), n in directions.most_common(10):
    print(f"  {t:>9} -> {p:<9}: {n}")

# ─── Sample montage of flagged images for manual review ─────────────────────
idx = np.where(flagged)[0][:24]
fig, axes = plt.subplots(4, 6, figsize=(14, 10))
for ax, i in zip(axes.ravel(), idx):
    ax.imshow(X[i, :, :, 0], cmap="gray")
    ax.set_title(f"{class_names[labels[i]]}→{class_names[pred[i]]}\n"
                 f"(conf {conf[i]:.2f})", fontsize=8)
    ax.axis("off")
for ax in axes.ravel()[len(idx):]:
    ax.axis("off")
fig.suptitle("Sample of quarantined images: folder label → model prediction", y=0.99)
plt.tight_layout()
plt.savefig("cleaning_sample_montage.png", dpi=110)
plt.close()
print("\nSaved sample montage → cleaning_sample_montage.png")

# ─── Quarantine (move, not delete) ───────────────────────────────────────────
qroot = "data/images_quarantine"
os.makedirs(qroot, exist_ok=True)
moved = 0
for i in np.where(flagged)[0]:
    cls = class_names[labels[i]]
    dst_dir = os.path.join(qroot, cls)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(paths[i]))
    if not os.path.exists(dst):
        shutil.move(paths[i], dst)
        moved += 1
print(f"\nMoved {moved} images → {qroot}/<class>/")
print("Remaining training images:", len(paths) - moved)
print("DONE")
