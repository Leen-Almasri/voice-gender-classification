"""
feature_utils.py
=================
SINGLE SOURCE OF TRUTH for audio preprocessing + feature extraction.

CRITICAL DESIGN PRINCIPLE:
    train.py and app.py BOTH import `extract_features` from this file.
    They must NEVER each define their own copy. This is the #1 root cause
    of "model predicts only one class after deployment" bugs: the training
    pipeline and the inference pipeline slowly drift apart (different
    n_mfcc, different sample rate, different pooling, missing scaler...)
    until the feature vector the model sees at inference time no longer
    resembles anything it was trained on. A RandomForest fed
    out-of-distribution features will often just fall back to predicting
    whatever class was most common in the training leaves closest to the
    (essentially random) input -> collapse to a single class.

CONFIG is centralized here too, so both scripts are guaranteed to agree.
"""

import numpy as np
import librosa
import logging

logger = logging.getLogger("voice_gender")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# GLOBAL CONFIG — change values here ONLY. Both train.py and app.py read this.
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050        # librosa default; MUST match everywhere
DURATION = 3.0             # seconds — fixed audio length fed to the model
N_MFCC = 40                # number of MFCC coefficients
FIXED_LEN = int(SAMPLE_RATE * DURATION)

# feature vector layout (for documentation / debugging):
#   40 MFCC mean + 40 MFCC std
#   40 delta-MFCC mean + 40 delta-MFCC std
#   12 chroma mean
#   1  zero-crossing-rate mean
#   1  spectral centroid mean
#   1  RMS energy mean
# total = 40+40+40+40+12+1+1+1 = 216
FEATURE_DIM = N_MFCC * 4 + 12 + 3


def load_audio(path_or_array, sr=SAMPLE_RATE):
    """
    Load audio from a file path OR accept an already-loaded (data, samplerate)
    tuple (this is what gr.Audio(type='numpy') gives us in app.py).

    ALWAYS returns: mono float32 numpy array, resampled to `sr`.
    This function is the single place where sampling-rate / mono-conversion
    logic lives — again, to prevent train/inference drift.
    """
    if isinstance(path_or_array, str):
        # librosa.load already does mono conversion + resampling to `sr`
        y, _ = librosa.load(path_or_array, sr=sr, mono=True)
        return y.astype(np.float32)

    if isinstance(path_or_array, tuple):
        native_sr, data = path_or_array
        data = np.asarray(data)

        # Gradio mic/upload input is frequently int16 PCM -> normalize to [-1, 1]
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        else:
            data = data.astype(np.float32)

        # Stereo -> mono (Gradio gives shape (n_samples, n_channels) sometimes,
        # or (n_channels, n_samples) — handle both defensively)
        if data.ndim == 2:
            data = np.mean(data, axis=-1) if data.shape[-1] <= 8 else np.mean(data, axis=0)

        # Resample if the recorded/uploaded sample rate differs from our target
        if native_sr != sr:
            data = librosa.resample(data, orig_sr=native_sr, target_sr=sr)

        return data.astype(np.float32)

    raise TypeError(f"Unsupported audio input type: {type(path_or_array)}")


def fix_length(y, target_len=FIXED_LEN):
    """
    Force every clip to EXACTLY `target_len` samples.
    Variable-length audio -> variable-length MFCC frame counts -> if you then
    pool differently (or not at all) between train and inference, the
    feature vectors are not comparable. Fixing length up front removes this
    entire class of bug.
    """
    if len(y) == 0:
        # Silent/corrupt input -> return zeros rather than crashing
        return np.zeros(target_len, dtype=np.float32)
    if len(y) > target_len:
        # Center-crop rather than always taking the start, which biases
        # toward leading silence in many recordings
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]
    elif len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    return y


def extract_features(audio_input, sr=SAMPLE_RATE, n_mfcc=N_MFCC, duration=DURATION):
    """
    THE canonical feature extraction pipeline. Identical call, identical
    output shape, whether invoked from train.py or app.py.

    Parameters
    ----------
    audio_input : str (file path) or (sample_rate, np.ndarray) tuple

    Returns
    -------
    np.ndarray of shape (FEATURE_DIM,) dtype float32
    """
    y = load_audio(audio_input, sr=sr)
    y = fix_length(y, target_len=int(sr * duration))

    # Guard against a clip that is all zeros (silence) — MFCC on pure silence
    # can produce NaNs/Infs in some librosa versions, which silently poison
    # the RandomForest split logic and is a classic hidden cause of
    # single-class collapse.
    if np.allclose(y, 0.0):
        logger.warning("Input audio is silent after preprocessing — features will be near-zero.")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)

    feats = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta_mfcc.mean(axis=1), delta_mfcc.std(axis=1),
        chroma.mean(axis=1),
        [zcr.mean()],
        [centroid.mean()],
        [rms.mean()],
    ]).astype(np.float32)

    # Replace any NaN/Inf that slipped through (e.g. from silent/degenerate audio)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    assert feats.shape[0] == FEATURE_DIM, (
        f"Feature vector has wrong length: got {feats.shape[0]}, expected {FEATURE_DIM}. "
        f"This means train.py and app.py have drifted out of sync — "
        f"they must both import extract_features from feature_utils.py."
    )
    return feats
