import pickle
import numpy as np
import librosa
import tempfile
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))



def extract_features(file):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file.save(temp.name)

            
            y, sr = librosa.load(temp.name, sr=22050)

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
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    features = extract_features(file)

    if features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    try:
        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": str(prediction)
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(debug=True)
