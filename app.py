import pickle
import numpy as np
import librosa
import tempfile
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))


# Extract audio features (MFCC)
def extract_features(file):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file.save(temp.name)

            # Load audio with fixed sample rate
            y, sr = librosa.load(
                temp.name,
                sr=22050,       # Force consistent sample rate
                duration=3,     # Limit duration
                offset=0.5      # Skip initial silence
            )

        # Check if audio is empty
        if len(y) == 0:
            return None

        # Extract MFCC features
        mfcc = np.mean(
            librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=40
            ).T,
            axis=0
        )

        return mfcc.reshape(1, -1)

    except Exception as e:
        print("Feature extraction error:", e)
        return None


@app.route("/")
def home():
    return "Voice Gender API Running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    # Check if file exists in request
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    # Extract features from audio
    features = extract_features(file)

    if features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    try:
        # Make prediction
        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": str(prediction)
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(debug=True)
