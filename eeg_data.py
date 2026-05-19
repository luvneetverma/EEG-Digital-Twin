"""
eeg_data.py — IEEE Paper Section III.A–D implementation
Synthetic EEG dataset generator using the paper's exact feature vector:
    xi = [PSDi, Conni, Demoi]   (Eq. 2)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# EEG frequency bands — Section III.B
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 50.0),
}

# Coherence channel pairs — Section III.C
CHANNEL_PAIRS = [
    ("Fp1","Fp2"),("F3","F4"),("C3","C4"),("P3","P4"),("O1","O2"),
    ("Fp1","F3"),("Fp2","F4"),("F3","C3"),("F4","C4"),("C3","P3"),
]
COHERENCE_NAMES = [f"coh_{a}_{b}" for a, b in CHANNEL_PAIRS]

# All 18 feature names (must stay consistent across all files)
FEATURE_NAMES = (
    ["psd_delta","psd_theta","psd_alpha","psd_beta","psd_gamma",
     "alpha_asymmetry"]
    + COHERENCE_NAMES
    + ["age","gender"]
)


def generate_eeg_features(n_samples=500):
    """
    Synthetic EEG feature dataset.
    MDD (label=1): elevated delta/theta, reduced alpha/beta/gamma, lower coherence.
    Healthy (label=0): opposite pattern.
    Returns pd.DataFrame with FEATURE_NAMES columns + 'label' column.
    """
    np.random.seed(RANDOM_SEED)
    n_mdd     = n_samples // 2
    n_healthy = n_samples - n_mdd
    records   = []

    for lbl, n in [(0, n_healthy), (1, n_mdd)]:
        for _ in range(n):
            row = {}
            if lbl == 0:   # Healthy
                row["psd_delta"]       = max(0.05, np.random.normal(1.0, 0.35))
                row["psd_theta"]       = max(0.05, np.random.normal(0.8, 0.30))
                row["psd_alpha"]       = max(0.05, np.random.normal(2.0, 0.40))
                row["psd_beta"]        = max(0.05, np.random.normal(1.5, 0.35))
                row["psd_gamma"]       = max(0.05, np.random.normal(0.5, 0.20))
                row["alpha_asymmetry"] = np.random.normal(0.05, 0.15)
                coh_mu = 0.65
            else:           # MDD
                row["psd_delta"]       = max(0.05, np.random.normal(2.2, 0.45))
                row["psd_theta"]       = max(0.05, np.random.normal(1.8, 0.40))
                row["psd_alpha"]       = max(0.05, np.random.normal(1.0, 0.35))
                row["psd_beta"]        = max(0.05, np.random.normal(0.7, 0.30))
                row["psd_gamma"]       = max(0.05, np.random.normal(0.3, 0.18))
                row["alpha_asymmetry"] = np.random.normal(0.30, 0.20)
                coh_mu = 0.42

            for i, name in enumerate(COHERENCE_NAMES):
                mu = coh_mu if i < 5 else (coh_mu + 0.08)
                row[name] = float(np.clip(np.random.normal(mu, 0.12), 0.0, 1.0))

            row["age"]    = float(np.random.randint(20, 71))
            row["gender"] = float(np.random.randint(0, 2))
            row["label"]  = lbl
            records.append(row)

    df = pd.DataFrame(records).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


def load_dataset(n_samples=500, test_size=0.20):
    """
    Preprocessing pipeline per paper Section III.D:
      1. Missing value imputation (mean)
      2. Remove non-informative variables (subject ID)
      3. Z-score normalisation  x' = (x - μ) / σ   (Eq. 6)
      4. Label encoding (already 0/1 binary)
      5. Stratified 80:20 train-test split

    Returns X_train, X_test, y_train, y_test, meta
    meta contains: feature_names, mu, sigma (numpy arrays), n_features, n_samples
    """
    df = generate_eeg_features(n_samples)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    X = df[FEATURE_NAMES].values.astype(np.float64)
    y = df["label"].values.astype(int)

    # Z-score normalisation (Eq. 6)
    mu    = X.mean(axis=0)          # shape (n_features,)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_norm = (X - mu) / sigma

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )

    meta = {
        "feature_names": FEATURE_NAMES,
        "mu":            mu,          # numpy array — consistent everywhere
        "sigma":         sigma,       # numpy array
        "n_features":    len(FEATURE_NAMES),
        "n_samples":     n_samples,
        "n_train":       len(X_train),
        "n_test":        len(X_test),
    }
    return X_train, X_test, y_train, y_test, meta


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, meta = load_dataset()
    print(f"Samples: {meta['n_samples']}  |  Features: {meta['n_features']}")
    print(f"Train: {meta['n_train']}  |  Test: {meta['n_test']}")
    print(f"Features: {meta['feature_names']}")
