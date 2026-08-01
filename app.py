import pickle
import numpy as np
import librosa
from flask import Flask, request, jsonify

app = Flask(__name__)

# تحميل الموديل
model = pickle.load(open("model.pkl", "rb"))

# استخراج الخصائص من الصوت
def extract_features(file):
    y, sr = librosa.load(file, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc.reshape(1, -1)

@app.route("/")
def home():
    return "Voice Gender API Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"})

    file = request.files["file"]

    features = extract_features(file)
    prediction = model.predict(features)[0]

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)