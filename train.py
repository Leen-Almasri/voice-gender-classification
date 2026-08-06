"""
train.py
========
Trains a RandomForestClassifier for male/female voice classification.

Supports TWO dataset layouts, auto-detected:

(A) RAVDESS-style (what the user's project actually uses):
    data/
        Actor_01/03-01-06-02-02-02-17.wav   (last number before .wav = actor id)
        Actor_02/...
        ...
    Label rule (RAVDESS convention): odd actor id -> male, even -> female.

(B) Simple folder-per-class:
    dataset/
        male/*.wav
        female/*.wav

Run:
    python train.py --data_dir ./data
"""

import os
import re
import argparse
import logging
import numpy as np
import joblib
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from feature_utils import extract_features, SAMPLE_RATE, DURATION, N_MFCC, FEATURE_DIM

logger = logging.getLogger("voice_gender")

LABELS = ["female", "male"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL = {i: l for i, l in enumerate(LABELS)}
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

RAVDESS_RE = re.compile(r"(\d+)\.(wav|mp3|flac)$", re.IGNORECASE)


def build_dataset_ravdess(data_dir):
    """
    RAVDESS convention: filename ...-<actor_id>.wav, odd id = male, even = female.
    `group` = actor_id, so we can later split by SPEAKER, not by sample —
    this is what the original project's train.py never did, which is why its
    reported "training accuracy" of 1.0 was meaningless (it evaluated on the
    exact rows it trained on, and even a correct held-out split that mixes
    each actor's clips across train/test would let the model memorize voice
    identity rather than learn gender).
    """
    X, y, groups = [], [], []
    skipped = 0
    for root, _dirs, files in os.walk(data_dir):
        for fname in files:
            if not fname.lower().endswith(AUDIO_EXTS):
                continue
            m = RAVDESS_RE.search(fname)
            if not m:
                skipped += 1
                continue
            actor_id = int(m.group(1))
            label = "female" if actor_id % 2 == 0 else "male"
            fpath = os.path.join(root, fname)
            try:
                feats = extract_features(fpath)
                X.append(feats)
                y.append(LABEL2IDX[label])
                groups.append(actor_id)
            except Exception as e:
                skipped += 1
                logger.warning(f"Skipping {fpath}: {e}")
    if skipped:
        logger.warning(f"Skipped {skipped} files that didn't match the expected pattern or failed to load.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(groups)


def build_dataset_foldered(data_dir):
    """Simple dataset/male/*.wav, dataset/female/*.wav layout. groups=None (per-file split)."""
    X, y = [], []
    skipped = 0
    for label in LABELS:
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(AUDIO_EXTS):
                continue
            fpath = os.path.join(class_dir, fname)
            try:
                feats = extract_features(fpath)
                X.append(feats)
                y.append(LABEL2IDX[label])
            except Exception as e:
                skipped += 1
                logger.warning(f"Skipping {fpath}: {e}")
    if skipped:
        logger.warning(f"Skipped {skipped} unreadable files.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), None


def build_dataset(data_dir):
    if os.path.isdir(os.path.join(data_dir, "male")) or os.path.isdir(os.path.join(data_dir, "female")):
        logger.info("Detected folder-per-class layout (male/, female/).")
        return build_dataset_foldered(data_dir)
    logger.info("Detected RAVDESS-style layout (Actor_XX/ folders, filename-encoded actor id).")
    return build_dataset_ravdess(data_dir)


def sanity_check_dataset(y):
    counts = Counter(y)
    if len(counts) < 2:
        raise ValueError(f"Dataset only contains {len(counts)} class(es): {counts}.")
    ratio = min(counts.values()) / max(counts.values())
    if ratio < 0.3:
        logger.warning(f"Dataset is imbalanced (ratio={ratio:.2f}, counts={counts}).")


def bias_collapse_test(model, scaler, X_test, y_test, header="Held-out test"):
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)

    pred_counts = Counter(preds)
    logger.info(f"[{header}] Prediction distribution: "
                f"{ {IDX2LABEL[k]: v for k, v in pred_counts.items()} }")

    print(f"\nSample predictions ({header}):")
    for i in range(min(8, len(y_test))):
        true_label, pred_label = IDX2LABEL[y_test[i]], IDX2LABEL[preds[i]]
        conf = probs[i][preds[i]]
        marker = "OK" if true_label == pred_label else "X "
        print(f"  [{marker}] true={true_label:<7} pred={pred_label:<7} conf={conf:.2f}")

    if len(pred_counts) < 2:
        raise RuntimeError(
            f"MODEL COLLAPSE DETECTED on {header}: only one class predicted. "
            f"Do not deploy this model."
        )
    print(f"✅ Bias-collapse check passed for {header}.")

    acc = accuracy_score(y_test, preds)
    print(f"{header} accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=LABELS))
    print("Confusion matrix ['female','male']:\n", confusion_matrix(y_test, preds))
    return acc


def main(args):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger.info(f"Config: sr={SAMPLE_RATE}, duration={DURATION}s, n_mfcc={N_MFCC}, feature_dim={FEATURE_DIM}")

    X, y, groups = build_dataset(args.data_dir)
    sanity_check_dataset(y)
    logger.info(f"Dataset: X={X.shape}, class distribution={ {IDX2LABEL[k]: v for k, v in Counter(y).items()} }")

    if groups is not None:
        n_speakers = len(set(groups))
        logger.info(f"Detected {n_speakers} unique speakers/groups. Using SPEAKER-GROUPED split "
                     f"so test speakers are never seen in training — this is the metric that "
                     f"actually predicts real-world (unseen-voice) performance.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))
    else:
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
        )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    logger.info(f"Train dist: {Counter(y_train)} | Test dist: {Counter(y_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # Held-out (unseen-speaker) evaluation — the real deployment-relevant number
    bias_collapse_test(model, scaler, X_test, y_test, header="Held-out unseen-speaker test")

    # Extra robustness check: k-fold cross-validation grouped by speaker
    if groups is not None and len(set(groups)) >= 5:
        gkf = GroupKFold(n_splits=5)
        cv_scores = cross_val_score(
            RandomForestClassifier(n_estimators=300, max_depth=18, class_weight="balanced",
                                    random_state=42, n_jobs=-1),
            StandardScaler().fit_transform(X), y, groups=groups, cv=gkf
        )
        logger.info(f"5-fold speaker-grouped CV accuracy: mean={cv_scores.mean():.3f} "
                    f"std={cv_scores.std():.3f} folds={np.round(cv_scores, 3).tolist()}")

    artifact = {
        "model": model,
        "scaler": scaler,
        "labels": LABELS,
        "sample_rate": SAMPLE_RATE,
        "duration": DURATION,
        "n_mfcc": N_MFCC,
        "feature_dim": FEATURE_DIM,
    }
    out_path = "voice_gender_model.joblib"  # simple, direct path — matches api.py
    joblib.dump(artifact, out_path)
    logger.info(f"Saved model artifact to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()
    main(args)
