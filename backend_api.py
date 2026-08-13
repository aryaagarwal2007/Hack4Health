"""
FastAPI backend — serves the MultimodalPipeline over HTTP for the
MindSense AI Next.js frontend (~/mindsense-ai).

Endpoints:
  GET  /api/health            → model load status
  POST /api/predict/face      → {image_base64, features?} → realtime fused assessment
  POST /api/predict/tabular   → {features}                → tabular MLP assessment
  POST /api/predict/audio     → multipart .wav upload     → speech emotion assessment
  POST /api/predict/multimodal→ multipart features+image+audio → full patient fusion

Run:  python backend_api.py   (serves http://localhost:8000)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import base64
import io
import json
import tempfile
import time

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from multimodal_pipeline import MultimodalPipeline, FEATURE_COLS, STATUS_CLASSES

app = FastAPI(title="MindSense AI Backend", version="1.0")

# CORS is open for local dev; the frontend also proxies via Next.js rewrites.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: MultimodalPipeline | None = None


@app.on_event("startup")
def load_pipeline():
    global pipeline
    t0 = time.time()
    pipeline = MultimodalPipeline()
    print(f"[backend] Pipeline ready in {time.time() - t0:.1f}s "
          f"(models: {pipeline.loaded_status})")


# ─── Request schemas ──────────────────────────────────────────────────────────
class FaceRequest(BaseModel):
    image_base64: str          # raw base64 or data URL
    features: dict | None = None   # optional 18 tabular features


class TabularRequest(BaseModel):
    features: dict


def _decode_image(b64: str) -> np.ndarray:
    if "," in b64[:128]:       # strip data-URL prefix (e.g. "data:image/jpeg;base64,")
        b64 = b64.split(",", 1)[1]
    try:
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image payload")
    return np.array(img)


def _require_pipeline() -> MultimodalPipeline:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models still loading")
    return pipeline


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    if pipeline is None:
        return {"status": "loading", "models": {}}
    return {
        "status": "ready",
        "models": pipeline.loaded_status,
        "status_classes": STATUS_CLASSES,
        "feature_cols": FEATURE_COLS,
    }


@app.post("/api/predict/face")
def predict_face(req: FaceRequest):
    """Real-time frame assessment: face crop → CNN emotion → gated fusion."""
    pipe = _require_pipeline()
    frame = _decode_image(req.image_base64)
    res = pipe.predict_realtime_frame(frame, req.features or {})
    res["face_detected"] = res.get("facial_emotion_details") is not None
    return res


@app.post("/api/predict/tabular")
def predict_tabular(req: TabularRequest):
    """Tabular-only branch (PyTorch multitask MLP)."""
    pipe = _require_pipeline()
    probs, scores = pipe.predict_tabular(req.features)
    return {
        "status_probabilities": dict(zip(STATUS_CLASSES, [float(p) for p in probs])),
        "severity_scores": scores,
    }


@app.post("/api/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    """Speech emotion branch from an uploaded .wav clip."""
    pipe = _require_pipeline()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        probs, details = pipe.predict_audio(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    if probs is None:
        raise HTTPException(status_code=422, detail="Audio analysis failed")
    return {
        "status_probabilities": dict(zip(STATUS_CLASSES, [float(p) for p in probs])),
        "audio_emotion_details": details,
    }


@app.post("/api/predict/multimodal")
async def predict_multimodal(
    features: str = Form("{}"),
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
):
    """Full gated multimodal fusion: tabular (+ optional face image / audio)."""
    pipe = _require_pipeline()
    try:
        feats = json.loads(features)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="features must be JSON")

    img_path = aud_path = None
    try:
        if image is not None:
            suffix = os.path.splitext(image.filename or "img.jpg")[1] or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(await image.read())
                img_path = tmp.name
        if audio is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(await audio.read())
                aud_path = tmp.name
        return pipe.predict_multimodal_patient(feats, img_path, aud_path)
    finally:
        for p in (img_path, aud_path):
            if p and os.path.exists(p):
                os.unlink(p)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
