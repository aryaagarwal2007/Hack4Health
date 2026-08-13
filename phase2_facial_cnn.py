"""
Phase 2 — Facial Emotion CNN
Small 3-block CNN trained on 48x48 grayscale face images (FER2013-style).
7 classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Load dataset ────────────────────────────────────────────────────────────
print("Loading image dataset from data/images/ ...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/images",
    validation_split=0.2,
    subset="training",
    seed=42,
    color_mode="grayscale",
    image_size=(48, 48),
    batch_size=64,
    label_mode="categorical",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/images",
    validation_split=0.2,
    subset="validation",
    seed=42,
    color_mode="grayscale",
    image_size=(48, 48),
    batch_size=64,
    label_mode="categorical",
)

class_names = train_ds.class_names
print(f"Classes: {class_names}")
print(f"Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
print(f"Val batches:   {tf.data.experimental.cardinality(val_ds).numpy()}")

# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ─── Build model ─────────────────────────────────────────────────────────────
model = tf.keras.Sequential([
    # Block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                           input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Block 3
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Head
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(7, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ─── Train ────────────────────────────────────────────────────────────────────
print("\nTraining for 12 epochs...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    verbose=1,
)

# ─── Evaluate ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EVALUATION — Facial Emotion CNN")
print("=" * 70)

val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"\n  Validation Loss:     {val_loss:.4f}")
print(f"  Validation Accuracy: {val_acc:.4f}")

# Collect predictions for confusion matrix
print("\nGenerating confusion matrix...")
y_true = []
y_pred = []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    label_arr = np.array(labels)
    pred_arr = np.array(preds)
    y_true.extend(np.argmax(label_arr, axis=1))
    y_pred.extend(np.argmax(pred_arr, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Facial Emotion CNN — Confusion Matrix")
plt.tight_layout()
plt.savefig("phase2_confusion_matrix.png", dpi=150)
print("Saved confusion matrix plot → phase2_confusion_matrix.png")

# ─── Save model ──────────────────────────────────────────────────────────────
model.save("models/facial_emotion_cnn.keras")
print("Saved model → models/facial_emotion_cnn.keras")

print("\n" + "=" * 70)
print(f"FINAL — Facial CNN val accuracy: {val_acc:.4f}")
print("=" * 70)
