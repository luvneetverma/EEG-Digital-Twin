"""
audio_data.py — Audio Mental-Health Pipeline (companion to eeg_data.py)

Implements a synthetic audio-feature dataset for depression detection.
Mirrors the EEG pipeline structure so it slots into the same
"Cloud-Based Digital Twin" framework.

Feature vector per sample (acoustic biomarkers commonly cited in the
voice-based depression literature):

    [mfcc_mean (13)] + [mfcc_std (13)] + [pitch_mean, pitch_std,
     jitter, shimmer, energy_mean, energy_std, zcr_mean,
     speaking_rate, spectral_centroid, spectral_rolloff,
     hnr (harmonics-to-noise ratio)] = 36 features total

MDD (label=1):  lower mean pitch, higher jitter/shimmer, lower energy,
                slower speaking rate, lower HNR (hoarser voice).
Healthy (label=0): opposite pattern.

These directional shifts are well-documented in clinical voice research
(Cummins et al., 2015; Mundt et al., 2007).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_MFCC = 13

MFCC_MEAN_NAMES = [f"mfcc_mean_{i}"  for i in range(1, N_MFCC + 1)]
MFCC_STD_NAMES  = [f"mfcc_std_{i}"   for i in range(1, N_MFCC + 1)]

PROSODIC_NAMES = [
    "pitch_mean", "pitch_std",
    "jitter", "shimmer",
    "energy_mean", "energy_std",
    "zcr_mean", "speaking_rate",
    "spectral_centroid", "spectral_rolloff",
    "hnr",
]

FEATURE_NAMES = MFCC_MEAN_NAMES + MFCC_STD_NAMES + PROSODIC_NAMES


# ── Synthetic feature generator ────────────────────────────────────────────────

def _sample_healthy():
    row = {}
    # MFCC: roughly centred, modest variance
    for i, n in enumerate(MFCC_MEAN_NAMES):
        row[n] = np.random.normal(loc=0.0 + 0.2 * (i % 3), scale=2.0)
    for n in MFCC_STD_NAMES:
        row[n] = max(0.05, np.random.normal(1.6, 0.4))

    # Prosodic — healthy voice
    row["pitch_mean"]        = np.random.normal(180.0, 25.0)   # Hz
    row["pitch_std"]         = max(2.0, np.random.normal(28.0, 6.0))
    row["jitter"]            = max(0.001, np.random.normal(0.012, 0.004))   # 1.2% typical
    row["shimmer"]           = max(0.01,  np.random.normal(0.045, 0.012))   # 4.5% typical
    row["energy_mean"]       = np.random.normal(0.075, 0.018)
    row["energy_std"]        = max(0.005, np.random.normal(0.030, 0.008))
    row["zcr_mean"]          = max(0.005, np.random.normal(0.080, 0.020))
    row["speaking_rate"]     = np.random.normal(4.4, 0.6)      # syllables / sec
    row["spectral_centroid"] = np.random.normal(2200, 350)     # Hz
    row["spectral_rolloff"]  = np.random.normal(4200, 600)     # Hz
    row["hnr"]               = np.random.normal(18.0, 3.0)     # dB — higher = cleaner voice
    return row


def _sample_mdd():
    row = {}
    # MFCC: shifted mean, slightly compressed dynamics
    for i, n in enumerate(MFCC_MEAN_NAMES):
        row[n] = np.random.normal(loc=-0.6 + 0.2 * (i % 3), scale=2.2)
    for n in MFCC_STD_NAMES:
        row[n] = max(0.05, np.random.normal(1.2, 0.35))   # less variation = monotone

    # Prosodic — depressed voice (clinically observed shifts)
    row["pitch_mean"]        = np.random.normal(155.0, 22.0)   # ↓ pitch
    row["pitch_std"]         = max(2.0, np.random.normal(18.0, 5.0))   # ↓ variation
    row["jitter"]            = max(0.001, np.random.normal(0.024, 0.007))  # ↑ jitter
    row["shimmer"]           = max(0.01,  np.random.normal(0.078, 0.018))  # ↑ shimmer
    row["energy_mean"]       = np.random.normal(0.045, 0.014)             # ↓ energy
    row["energy_std"]        = max(0.005, np.random.normal(0.018, 0.006))  # ↓ dynamics
    row["zcr_mean"]          = max(0.005, np.random.normal(0.060, 0.018))
    row["speaking_rate"]     = np.random.normal(3.2, 0.5)                  # ↓ slower speech
    row["spectral_centroid"] = np.random.normal(1850, 320)
    row["spectral_rolloff"]  = np.random.normal(3500, 550)
    row["hnr"]               = np.random.normal(12.5, 3.0)                 # ↓ hoarser
    return row


def generate_audio_features(n_samples=500):
    """
    Generate a synthetic audio-feature dataset.
    Returns a DataFrame with FEATURE_NAMES + 'label' (0=Healthy, 1=MDD).
    """
    np.random.seed(RANDOM_SEED)
    n_mdd     = n_samples // 2
    n_healthy = n_samples - n_mdd
    rows      = []

    for _ in range(n_healthy):
        r = _sample_healthy(); r["label"] = 0; rows.append(r)
    for _ in range(n_mdd):
        r = _sample_mdd();    r["label"] = 1; rows.append(r)

    df = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df[FEATURE_NAMES + ["label"]]


def load_dataset(n_samples=500, test_size=0.2):
    """
    Returns: X_train, X_test, y_train, y_test, meta
    `meta` includes per-feature mean/std for z-score normalisation
    (used by the AudioDigitalTwin at inference time).
    """
    df = generate_audio_features(n_samples=n_samples)
    X  = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y  = df["label"].to_numpy(dtype=np.int64)

    mu    = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_norm = (X - mu) / sigma

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    meta = {
        "feature_names": FEATURE_NAMES,
        "n_features":    len(FEATURE_NAMES),
        "n_samples":     len(df),
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "mu":            mu,
        "sigma":         sigma,
    }
    return X_train, X_test, y_train, y_test, meta


if __name__ == "__main__":
    df = generate_audio_features(20)
    print(df.head())
    print(f"\nFeatures: {len(FEATURE_NAMES)}  Samples: {len(df)}")
