"""
api.py
======
FastAPI backend for voice gender prediction.

Run locally:
    uvicorn api:app --reload
"""

import io
import joblib
import soundfile as sf
from fastapi import FastAPI, File, UploadFile

from feature_utils import extract_features  # shared, unmodified

app = FastAPI(title="Voice Gender Classification API")

# Simple, direct path — no model_artifacts/, no env vars
MODEL_PATH = "voice_gender_model.joblib"

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
scaler = artifact["scaler"]
labels = artifact["labels"]
SAMPLE_RATE = artifact["sample_rate"]
N_MFCC = artifact["n_mfcc"]
DURATION = artifact["duration"]


@app.get("/")
def home():
    return {"message": "Voice Gender Classification API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    data, sr = sf.read(io.BytesIO(audio_bytes))

    # Same feature pipeline used in training — guarantees train/inference parity
    feats = extract_features((sr, data), sr=SAMPLE_RATE, n_mfcc=N_MFCC, duration=DURATION)
    feats_scaled = scaler.transform(feats.reshape(1, -1))

    probs = model.predict_proba(feats_scaled)[0]
    pred_idx = int(probs.argmax())

    return {"prediction": labels[pred_idx]}
