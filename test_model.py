import pickle
import librosa
import numpy as np

# تحميل المودل
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# جربي ملف صوت من الداتا
file_path = "data/Actor_01/03-01-01-01-01-01-01.wav"

# قراءة الصوت
audio, sr = librosa.load(file_path, duration=3)

# استخراج الميزات
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
features = np.mean(mfcc.T, axis=0).reshape(1, -1)

# التنبؤ
prediction = model.predict(features)

print("Prediction:", prediction[0])