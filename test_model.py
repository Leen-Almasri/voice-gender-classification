"""
test_model.py
=============
Quick manual sanity check: load the saved artifact and predict on one
known file from each class, using the SAME extract_features pipeline
as training and the Gradio app.
"""

import joblib
from feature_utils import extract_features

artifact = joblib.load("voice_gender_model.joblib")
model, scaler, labels = artifact["model"], artifact["scaler"], artifact["labels"]

test_cases = [
    ("data/Actor_02/03-01-01-01-01-01-02.wav", "female"),
    ("data/Actor_01/03-01-01-01-01-01-01.wav", "male"),
]

for path, expected in test_cases:
    feats = extract_features(path, sr=artifact["sample_rate"], n_mfcc=artifact["n_mfcc"], duration=artifact["duration"])
    probs = model.predict_proba(scaler.transform(feats.reshape(1, -1)))[0]
    pred = labels[int(probs.argmax())]
    status = "OK" if pred == expected else "MISMATCH"
    print(f"[{status}] {path} -> expected={expected} predicted={pred} probs={dict(zip(labels, probs.round(3)))}")
