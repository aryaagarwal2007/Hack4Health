"""
Phase 2c — Enhanced Facial CNN with Boosted "Sad" Sensitivity
Architecture: VGG-style 4-block deep CNN (32 -> 64 -> 128 -> 256) with
BatchNormalization, Spatial Dropout, Class Weighting for 'Sad', and L2 regularization.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE = (48, 48)
BATCH = 64

# ─── 1. Load Datasets ────────────────────────────────────────────────────────
print("Loading image dataset from data/images/ ...")
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
print(f"Classes ({len(class_names)}): {class_names}")
sad_index = class_names.index("Sad") if "Sad" in class_names else 5
print(f"Sad class index: {sad_index}")

# Calculate Class Weights to boost 'Sad' & underrepresented classes
all_labels = []
for _, labels in train_ds:
    all_labels.extend(np.argmax(labels.numpy(), axis=1))
all_labels = np.array(all_labels)

class_weights = compute_class_weight("balanced", classes=np.unique(all_labels), y=all_labels)
class_weight_dict = dict(enumerate(class_weights))
# Boost Sad class weight by an additional 1.35x multiplier
class_weight_dict[sad_index] = class_weight_dict[sad_index] * 1.35
print(f"Computed Class Weights: {class_weight_dict}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(3000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ─── 2. Build Deep VGG-Style CNN Architecture ───────────────────────────────
print("\nBuilding Deep VGG-style CNN with Sad-boosted sensitivity ...")

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.12),
    tf.keras.layers.RandomZoom(0.12),
    tf.keras.layers.RandomTranslation(0.08, 0.08),
], name="data_augmentation")

inputs = tf.keras.layers.Input(shape=(48, 48, 1))
x = data_augmentation(inputs)

# Block 1 (32 filters)
x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Dropout(0.2)(x)

# Block 2 (64 filters)
x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Dropout(0.25)(x)

# Block 3 (128 filters)
x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Dropout(0.3)(x)

# Block 4 (256 filters)
x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Dropout(0.35)(x)

# Fully Connected Head
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, kernel_initializer="he_normal")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dropout(0.45)(x)

x = tf.keras.layers.Dense(128, kernel_initializer="he_normal")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dropout(0.35)(x)

outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# ─── 3. Train Model ──────────────────────────────────────────────────────────
EPOCHS = 35
STEPS_PER_EPOCH = len(all_labels) // BATCH

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=EPOCHS * STEPS_PER_EPOCH,
    alpha=1e-5
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary(print_fn=lambda s: print("  " + s))

checkpoint_path = "models/facial_emotion_cnn.keras"
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1
)
early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=8, restore_best_weights=True
)

print(f"\nTraining for up to {EPOCHS} epochs with Sad-boosted class weights...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[checkpoint_cb, early_stop_cb],
)

# ─── 4. Evaluate ─────────────────────────────────────────────────────────────
model = tf.keras.models.load_model(checkpoint_path)
print("\n" + "=" * 70)
print("EVALUATION — Sad-Enhanced Facial CNN")
print("=" * 70)

val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"\n  Validation Loss:     {val_loss:.4f}")
print(f"  Validation Accuracy: {val_acc:.4f}")

y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nClassification Report:")
report_str = classification_report(y_true, y_pred, target_names=class_names)
print(report_str)

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Sad-Enhanced Facial CNN — Confusion Matrix")
plt.tight_layout()
plt.savefig("phase2c_sad_enhanced_confusion_matrix.png", dpi=150)
print("Saved confusion matrix → phase2c_sad_enhanced_confusion_matrix.png")
