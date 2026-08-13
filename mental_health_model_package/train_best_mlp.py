import json
import os
import random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Config & Paths
SEED = 42
DATA_PATH = r"E:\H4H\Data\mental_health_multimodal.csv"
MODELS_DIR = Path(r"E:\H4H\models")
ARTIFACTS_DIR = Path(r"E:\H4H\artifacts")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Mental_Health_Status"
REG_TARGETS = ["Depression_Score", "Anxiety_Score", "Stress_Score"]

def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Executing on device:", device)

# 1. Load Data
df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c not in REG_TARGETS + [TARGET_COL]]

# 2. Encode Labels & Save Encoder
label_encoder = LabelEncoder()
y_cls_all = label_encoder.fit_transform(df[TARGET_COL])
class_names = label_encoder.classes_.tolist()
severe_idx = class_names.index("Severe_Stress")
joblib.dump(label_encoder, ARTIFACTS_DIR / "label_encoder.pkl")

# 3. Stratified 70/15/15 Data Splitting
X_all = df[feature_cols].astype(np.float32).values
y_reg_all = df[REG_TARGETS].astype(np.float32).values

idx = np.arange(len(df))
train_idx, temp_idx, y_train_strat, y_temp_strat = train_test_split(
    idx,
    y_cls_all,
    test_size=0.30,
    random_state=SEED,
    stratify=y_cls_all,
)
val_idx, test_idx, _, _ = train_test_split(
    temp_idx,
    y_temp_strat,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp_strat,
)

# 4. Preprocessing (Fit Scalers ONLY on Training Split)
x_scaler = StandardScaler()
y_reg_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_all[train_idx]).astype(np.float32)
X_val = x_scaler.transform(X_all[val_idx]).astype(np.float32)
X_test = x_scaler.transform(X_all[test_idx]).astype(np.float32)

y_reg_train = y_reg_scaler.fit_transform(y_reg_all[train_idx]).astype(np.float32)
y_reg_val = y_reg_scaler.transform(y_reg_all[val_idx]).astype(np.float32)
y_reg_test = y_reg_scaler.transform(y_reg_all[test_idx]).astype(np.float32)

y_cls_train = y_cls_all[train_idx].astype(np.int64)
y_cls_val = y_cls_all[val_idx].astype(np.int64)
y_cls_test = y_cls_all[test_idx].astype(np.int64)

joblib.dump(x_scaler, ARTIFACTS_DIR / "feature_scaler.pkl")
joblib.dump(y_reg_scaler, ARTIFACTS_DIR / "target_scaler.pkl")

# 5. Model Architecture
class TabularMultitaskMLP(nn.Module):
    def __init__(
        self,
        input_dim=18,
        hidden_dims=(128, 64),
        num_classes=4,
        num_reg_targets=3,
        dropout=0.25,
        activation="relu",
        use_batchnorm=True,
    ):
        super().__init__()
        act = nn.GELU if activation.lower() == "gelu" else nn.ReLU
        layers = []
        prev = input_dim
        for h in hidden_dims:
            block = [nn.Linear(prev, h)]
            if use_batchnorm:
                block.append(nn.BatchNorm1d(h))
            block += [act(), nn.Dropout(dropout)]
            layers.append(nn.Sequential(*block))
            prev = h
        self.blocks = nn.ModuleList(layers)
        self.embedding_dim = hidden_dims[-1]
        self.cls_head = nn.Linear(self.embedding_dim, num_classes)
        self.reg_head = nn.Linear(self.embedding_dim, num_reg_targets)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        logits = self.cls_head(out)
        reg_outputs = self.reg_head(out)
        return logits, reg_outputs, out

def class_weights_from_y(y, power=1.0):
    counts = np.bincount(y, minlength=len(class_names)).astype(np.float32)
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    weights = weights ** power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

from torch.utils.data import DataLoader, TensorDataset

ds_tr = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_cls_train, dtype=torch.long),
    torch.tensor(y_reg_train, dtype=torch.float32),
)
loader_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)

model = TabularMultitaskMLP(
    input_dim=len(feature_cols),
    hidden_dims=(128, 64),
    num_classes=len(class_names),
    num_reg_targets=3,
    dropout=0.25,
    activation="relu",
).to(device)

alpha = class_weights_from_y(y_cls_train, 1.0).to(device)
cls_loss_fn = nn.CrossEntropyLoss(weight=alpha)
reg_loss_fn = nn.HuberLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
)

# 6. Train Model
best_val_macro = -np.inf
best_epoch = -1
best_state = None
stale = 0

print("Training Optimal 2-Layer Multitask ReLU MLP...")
for epoch in range(1, 121):
    model.train()
    for xb, yb, rb in loader_tr:
        xb, yb, rb = xb.to(device), yb.to(device), rb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, reg_out, _ = model(xb)
        loss = 1.0 * cls_loss_fn(logits, yb) + 0.25 * reg_loss_fn(reg_out, rb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        val_logits, _, _ = model(torch.tensor(X_val, dtype=torch.float32).to(device))
        val_preds = val_logits.argmax(dim=1).cpu().numpy()
        val_macro = f1_score(y_cls_val, val_preds, average="macro", zero_division=0)
        
    scheduler.step(val_macro)
        
    if val_macro > best_val_macro:
        best_val_macro = val_macro
        best_epoch = epoch
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        stale = 0
    else:
        stale += 1
        
    if stale >= 15:
        break

model.load_state_dict(best_state)
torch.save(model.state_dict(), MODELS_DIR / "numerical_classifier.pth")

# 7. Final Test Set Evaluation
model.eval()
with torch.no_grad():
    test_logits, test_reg_scaled, _ = model(torch.tensor(X_test, dtype=torch.float32).to(device))
    test_preds = test_logits.argmax(dim=1).cpu().numpy()
    test_reg_pred = test_reg_scaled.cpu().numpy()

prec, rec, f1_cls, _ = precision_recall_fscore_support(y_cls_test, test_preds, labels=np.arange(len(class_names)), zero_division=0)

test_acc = accuracy_score(y_cls_test, test_preds)
test_macro_f1 = f1_score(y_cls_test, test_preds, average="macro", zero_division=0)
test_weighted_f1 = f1_score(y_cls_test, test_preds, average="weighted", zero_division=0)
test_bal_acc = balanced_accuracy_score(y_cls_test, test_preds)

print("\n==================================================")
print("FINAL UNTOUCHED TEST EVALUATION")
print("==================================================")
print(f"Test Accuracy:          {test_acc:.4f}")
print(f"Test Macro F1:         {test_macro_f1:.4f}")
print(f"Test Weighted F1:      {test_weighted_f1:.4f}")
print(f"Test Balanced Accuracy: {test_bal_acc:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_cls_test, test_preds, target_names=class_names, zero_division=0))

y_reg_true_orig = y_reg_scaler.inverse_transform(y_reg_test)
y_reg_pred_orig = y_reg_scaler.inverse_transform(test_reg_pred)

print("=== Multitask Regression Metrics ===")
for i, target in enumerate(REG_TARGETS):
    mae = mean_absolute_error(y_reg_true_orig[:, i], y_reg_pred_orig[:, i])
    rmse = float(np.sqrt(mean_squared_error(y_reg_true_orig[:, i], y_reg_pred_orig[:, i])))
    r2 = r2_score(y_reg_true_orig[:, i], y_reg_pred_orig[:, i])
    print(f"{target:18s} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# 8. Save Configuration & Experiment Table
config = {
    "model_name": "Optimal 2-Layer Multitask ReLU MLP",
    "architecture": "(128, 64)",
    "input_dim": 18,
    "num_classes": 4,
    "num_reg_targets": 3,
    "activation": "relu",
    "dropout": 0.25,
    "use_batchnorm": True,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "scheduler": "ReduceLROnPlateau",
    "best_epoch": best_epoch,
    "val_macro_f1": float(best_val_macro),
    "test_macro_f1": float(test_macro_f1),
    "test_accuracy": float(test_acc),
}

with open(ARTIFACTS_DIR / "model_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\nSuccessfully saved updated model and artifacts to models/ and artifacts/.")
