import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 1. Define Model Architecture (Must match saved weights)
class TabularMultitaskMLP(nn.Module):
    def __init__(self, input_dim=18, hidden_dims=(128, 64), num_classes=4, num_reg_targets=3, dropout=0.25):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            block = [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            layers.append(nn.Sequential(*block))
            prev = h
        self.blocks = nn.ModuleList(layers)
        self.cls_head = nn.Linear(hidden_dims[-1], num_classes)
        self.reg_head = nn.Linear(hidden_dims[-1], num_reg_targets)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        logits = self.cls_head(out)
        reg_out = self.reg_head(out)
        return logits, reg_out

def load_trained_model(weights_path="models/numerical_classifier.pth", artifacts_dir="artifacts"):
    # Load scalers and label encoder
    feature_scaler = joblib.load(f"{artifacts_dir}/feature_scaler.pkl")
    target_scaler = joblib.load(f"{artifacts_dir}/target_scaler.pkl")
    label_encoder = joblib.load(f"{artifacts_dir}/label_encoder.pkl")
    
    # Initialize and load model
    model = TabularMultitaskMLP(input_dim=18, hidden_dims=(128, 64))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    
    return model, feature_scaler, target_scaler, label_encoder

def predict_sample(sample_dict, model, feature_scaler, target_scaler, label_encoder):
    # Expect 18 feature values
    df_sample = pd.DataFrame([sample_dict])
    X_scaled = feature_scaler.transform(df_sample.values).astype(np.float32)
    
    with torch.no_grad():
        logits, reg_scaled = model(torch.tensor(X_scaled))
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_class_idx = probs.argmax()
        pred_class_label = label_encoder.inverse_transform([pred_class_idx])[0]
        
        reg_unscaled = target_scaler.inverse_transform(reg_scaled.numpy())[0]
        
    return {
        "Predicted_Status": pred_class_label,
        "Class_Probabilities": dict(zip(label_encoder.classes_, probs)),
        "Depression_Score": float(reg_unscaled[0]),
        "Anxiety_Score": float(reg_unscaled[1]),
        "Stress_Score": float(reg_unscaled[2]),
    }

if __name__ == "__main__":
    print("Loading model and artifacts...")
    model, f_scaler, t_scaler, l_encoder = load_trained_model("models/numerical_classifier.pth", "artifacts")
    print("Model ready for inference!")
