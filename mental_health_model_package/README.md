# Multimodal Mental Health System - Tabular MLP Package

This package contains the trained 2-Layer Multitask ReLU PyTorch MLP model and inference code.

## Package Contents:
- `models/numerical_classifier.pth`: Serialized PyTorch model weights.
- `artifacts/feature_scaler.pkl`: StandardScaler for 18 input features.
- `artifacts/target_scaler.pkl`: StandardScaler for regression targets.
- `artifacts/label_encoder.pkl`: LabelEncoder for 4 mental health categories.
- `artifacts/model_config.json`: Architecture specs and performance metrics.
- `inference.py`: Python script to load the model and run predictions on new data.
- `train_best_mlp.py`: Complete reproducible training script.

## Quickstart Inference:
To make predictions on new data:
```bash
python inference.py
```
