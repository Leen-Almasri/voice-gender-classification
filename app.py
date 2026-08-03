import pickle
import numpy as np
import librosa
import soundfile as sf
import gradio as gr

model = pickle.load(open("model.pkl", "rb"))

def predict(audio):
    y, sr = audio

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)

    y = y[:22050 * 3]

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=22050, n_mfcc=40).T, axis=0)
    mfcc = mfcc.reshape(1, -1)

    prediction = model.predict(mfcc)[0]
    return str(prediction)

gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="numpy"),
    outputs="text"
).launch()
