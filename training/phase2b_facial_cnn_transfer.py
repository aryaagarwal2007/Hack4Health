"""
Phase 2b — Facial Emotion CNN (Improved Custom Architecture)
Original 3-block CNN enhanced with BatchNormalization, data augmentation,
and longer training. Target: 55-60% val accuracy.
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

IMG_SIZE = (48, 48)
BATCH = 64

# ─── Load dataset ────────────────────────────────────────────────────────────
print("Loading image dataset (48x48 grayscale) ...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/images", validation_split=0.2, subset="training", seed=42,
    color_mode="grayscale", image_size=IMG_SIZE, batch_size=BATCH,
    label_mode="categorical",
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/images", validation_split=0.2, subset="validation", seed=42,
    color_mode="grayscale", image_size=IMG_SIZE, batch_size=BATCH,
    label_mode="categorical",
)
class_names = train_ds.class_names
print(f"Classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(2000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ─── Build improved CNN with BatchNorm + augmentation ───────────────────────
print("Building improved CNN (BatchNorm + augmentation) ...")

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="augmentation")

inputs = tf.keras.layers.Input(shape=(48, 48, 1))
x = data_augmentation(inputs)

# Block 1
x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

# Block 2
x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

# Block 3
x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

# Head
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(7, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy", metrics=["accuracy"],
)
model.summary(print_fn=lambda s: print("  " + s))

# ─── Train with cosine decay LR schedule + checkpointing ────────────────────
EPOCHS = 30
STEPS_PER_EPOCH = 359
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=EPOCHS * STEPS_PER_EPOCH
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr_schedule),
    loss="categorical_crossentropy", metrics=["accuracy"],
)

# Save the best-val model and stop if it plateaus (avoids overfitting regressions)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "models/facial_emotion_cnn.keras",
    monitor="val_accuracy", save_best_only=True, verbose=1,
)
early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=6, restore_best_weights=True,
)

print(f"\nTraining for up to {EPOCHS} epochs with cosine LR decay ...")
history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS,
    callbacks=[checkpoint_cb, early_stop_cb],
)

# ─── Evaluate ────────────────────────────────────────────────────────────────
# Reload the best checkpoint so evaluation matches the deployed model
model = tf.keras.models.load_model("models/facial_emotion_cnn.keras")
print("\n" + "=" * 70)
print("EVALUATION — Facial CNN (Improved, best checkpoint)")
print("=" * 70)
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"\n  Validation Loss:     {val_loss:.4f}")
print(f"  Validation Accuracy: {val_acc:.4f}")

y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(np.array(labels), axis=1))
    y_pred.extend(np.argmax(np.array(preds), axis=1))
y_true, y_pred = np.array(y_true), np.array(y_pred)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Facial CNN (Improved) — Confusion Matrix")
plt.tight_layout()
plt.savefig("phase2b_confusion_matrix.png", dpi=150)
print("Saved → phase2b_confusion_matrix.png")
print("Best model already saved by checkpoint → models/facial_emotion_cnn.keras")

print("\n" + "=" * 70)
print(f"FINAL — Facial CNN (improved) val accuracy: {val_acc:.4f}")
print("=" * 70)
