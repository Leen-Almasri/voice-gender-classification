import pickle
import numpy as np
import librosa
import tempfile
import soundfile as sf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))


# Extract audio features using soundfile (no ffmpeg needed)
def extract_features(file):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file.save(temp.name)

            # Read audio file
            y, sr = sf.read(temp.name)

        # Convert stereo to mono if needed
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        # Resample to 22050 Hz if needed
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050

        # Keep only first 3 seconds
        y = y[:sr * 3]

        if len(y) == 0:
            return None

        # Extract MFCC features
        mfcc = np.mean(
            librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,
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
    # Check if file is included
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    # Extract features
    features = extract_features(file)

    if features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    try:
        # Predict gender
        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": str(prediction)
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(debug=True)
