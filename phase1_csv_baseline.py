"""
Phase 1 — CSV Baseline: Confirm the known ceiling for tabular-only models.
Trains a RandomForestClassifier for Mental_Health_Status and a
MultiOutputRegressor(XGBRegressor) for the three continuous scores.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
)
import warnings
warnings.filterwarnings("ignore")

# ─── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/mental_health_multimodal.csv")
print(f"Dataset shape: {df.shape}")

# Separate features and targets
feature_cols = [c for c in df.columns
                if c not in ("Mental_Health_Status",
                             "Depression_Score", "Anxiety_Score", "Stress_Score")]
X = df[feature_cols].to_numpy(dtype=np.float64)
y_cls = df["Mental_Health_Status"].to_numpy(dtype=str)
y_reg = df[["Depression_Score", "Anxiety_Score", "Stress_Score"]].to_numpy(dtype=np.float64)

# ─── Stratified 80/20 split ──────────────────────────────────────────────────
X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = \
    train_test_split(X, y_cls, y_reg, test_size=0.2,
                     stratify=y_cls, random_state=42)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Class distribution (train): {dict(zip(*np.unique(y_cls_train, return_counts=True)))}")
print(f"Class distribution (test):  {dict(zip(*np.unique(y_cls_test, return_counts=True)))}")

# ─── Classification ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("CLASSIFICATION — RandomForestClassifier (class_weight='balanced')")
print("="*70)

clf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_cls_train)
y_cls_pred = clf.predict(X_test)
y_cls_proba = clf.predict_proba(X_test)

classes = clf.classes_
acc = accuracy_score(y_cls_test, y_cls_pred)
f1_macro = f1_score(y_cls_test, y_cls_pred, average="macro")
f1_weighted = f1_score(y_cls_test, y_cls_pred, average="weighted")

# ROC-AUC (one-vs-rest, needs label binarization)
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_cls_test, classes=classes)
roc_macro = roc_auc_score(y_test_bin, y_cls_proba, average="macro", multi_class="ovr")

print(f"\n  Accuracy:        {acc:.4f}")
print(f"  Macro F1:        {f1_macro:.4f}")
print(f"  Weighted F1:     {f1_weighted:.4f}")
print(f"  Macro ROC-AUC:   {roc_macro:.4f}")
print(f"\n  Confusion Matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_cls_test, y_cls_pred, labels=classes)
print(f"  Classes: {list(classes)}")
for i, row in enumerate(cm):
    print(f"    {classes[i]:>16s}: {row}")

# ─── Regression ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("REGRESSION — MultiOutputRegressor(XGBRegressor)")
print("="*70)

reg = MultiOutputRegressor(
    XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                 random_state=42, n_jobs=-1)
)
reg.fit(X_train, y_reg_train)
y_reg_pred = reg.predict(X_test)

target_names = ["Depression_Score", "Anxiety_Score", "Stress_Score"]
print()
for i, name in enumerate(target_names):
    y_true_i = y_reg_test[:, i]
    y_pred_i = y_reg_pred[:, i]
    mae = mean_absolute_error(y_true_i, y_pred_i)
    mse = mean_squared_error(y_true_i, y_pred_i)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_i, y_pred_i)
    ev = explained_variance_score(y_true_i, y_pred_i)
    print(f"  {name}:")
    print(f"    MAE:               {mae:.4f}")
    print(f"    MSE:               {mse:.4f}")
    print(f"    RMSE:              {rmse:.4f}")
    print(f"    R²:                {r2:.4f}")
    print(f"    Explained Var:     {ev:.4f}")
    print()

# ─── Summary ──────────────────────────────────────────────────────────────────
print("="*70)
print("SUMMARY — Tabular-only ceiling (expected: ~39% acc, ~0.19 F1, ~0.49 AUC)")
print("="*70)
print(f"  Accuracy:     {acc:.4f}")
print(f"  Macro F1:     {f1_macro:.4f}")
print(f"  ROC-AUC:      {roc_macro:.4f}")
print(f"  R² (Dep):     {r2_score(y_reg_test[:,0], y_reg_pred[:,0]):.4f}")
print(f"  R² (Anx):     {r2_score(y_reg_test[:,1], y_reg_pred[:,1]):.4f}")
print(f"  R² (Str):     {r2_score(y_reg_test[:,2], y_reg_pred[:,2]):.4f}")
print()
print("These numbers confirm the known data ceiling — tabular features alone")
print("carry near-zero predictive signal for the target labels.")
