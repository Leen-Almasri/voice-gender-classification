"""
streamlit_app.py
================
Streamlit frontend for voice gender prediction.
Sends the audio file to the FastAPI backend (api.py) via requests.

Run locally (in a second terminal, after starting the API):
    streamlit run streamlit_app.py
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Voice Gender Classifier", page_icon="🎙️")

st.title("🎙️ Voice Gender Classifier")
st.write("Upload or record a short voice clip (.wav) and click **Predict**.")

audio_bytes = None
filename = "audio.wav"

tab1, tab2 = st.tabs(["📁 Upload file", "🎤 Record audio"])

with tab1:
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        filename = uploaded_file.name
        st.audio(audio_bytes, format="audio/wav")

with tab2:
    recorded = st.audio_input("Record your voice")
    if recorded is not None:
        audio_bytes = recorded.read()
        filename = "recording.wav"
        st.audio(audio_bytes, format="audio/wav")

st.divider()

if st.button("Predict", type="primary", disabled=audio_bytes is None):
    with st.spinner("Predicting..."):
        try:
            files = {"file": (filename, audio_bytes, "audio/wav")}
            response = requests.post(API_URL, files=files, timeout=30)

            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]

                if prediction == "male":
                    st.success("### Prediction:  **Male**")
                else:
                    st.success("### Prediction: **Female**")
            else:
                st.error(f"API error ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the API. Make sure it's running:\n\n"
                "`uvicorn api:app --reload`"
            )

if audio_bytes is None:
    st.info("Upload or record an audio clip to enable the Predict button.")
