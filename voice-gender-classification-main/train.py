import os
import numpy as np
import librosa
import pickle
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "data"

X = []
y = []


def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)


for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            try:
                
                parts = file.split("-")
                actor_id = int(parts[-1].split(".")[0])

                
                if actor_id % 2 == 0:
                    label = "female"
                else:
                    label = "male"

                
                features = extract_features(file_path)

                X.append(features)
                y.append(label)

            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

print("✅ Feature extraction complete")
print(f"Samples: {len(X)}")


model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)


with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("🎉 Model saved as model.pkl")