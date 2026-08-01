from fastapi import FastAPI, File, UploadFile
import numpy as np
import librosa
import pickle

app = FastAPI()


model = pickle.load(open("model.pkl", "rb"))

def extract_features(file_bytes):
    import io
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    return mfccs_mean.reshape(1, -1)

@app.get("/")
def home():
    return {"message": "Voice Gender Classification API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    
    features = extract_features(audio_bytes)
    prediction = model.predict(features)[0]

    gender = "male" if prediction == 1 else "female"

    return {"gender": gender}