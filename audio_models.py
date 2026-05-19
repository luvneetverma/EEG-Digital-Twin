"""
audio_models.py — Trains acoustic-biomarker classifiers for the
Audio Digital Twin (companion to train_models.py).

Trains:
  • SVM (RBF) — primary, mirrors EEG paper Section III.G
  • LSTM (if TensorFlow is available) — sequence-style classifier on the
    36-d feature vector reshaped as a short pseudo-sequence
  • Random Forest — tabular baseline

Picks the highest-F1 model and saves it as the audio digital twin.
"""

import os
import importlib.util
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "saved_models")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
MODEL_PATH  = os.path.join(MODELS_DIR, "audio_twin_model.pkl")
RANDOM_SEED = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
TF_AVAILABLE = False
if importlib.util.find_spec("tensorflow") is not None:
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
        from tensorflow.keras.callbacks import EarlyStopping
        TF_AVAILABLE = True
    except Exception:
        pass


def _evaluate(name, y_test, y_pred):
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
    }


def _print_section(title):
    print("\n" + "=" * 56)
    print(f"  {title}")
    print("=" * 56)


class _LSTMWrapper:
    """Thin wrapper so a Keras model behaves like sklearn (predict / predict_proba)."""

    def __init__(self, keras_model):
        self.model = keras_model

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32).reshape(-1, X.shape[-1])
        seq = X.reshape(X.shape[0], 4, X.shape[1] // 4) if X.shape[1] % 4 == 0 else X.reshape(X.shape[0], 1, X.shape[1])
        proba = self.model.predict(seq, verbose=0).flatten()
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32).reshape(-1, X.shape[-1])
        seq = X.reshape(X.shape[0], 4, X.shape[1] // 4) if X.shape[1] % 4 == 0 else X.reshape(X.shape[0], 1, X.shape[1])
        p1 = self.model.predict(seq, verbose=0).flatten()
        return np.stack([1 - p1, p1], axis=1)


def train_and_evaluate():
    """Train SVM, RF and (optional) LSTM. Save the highest-F1 as the audio twin."""
    import sys
    sys.path.insert(0, BASE_DIR)
    from audio_data import load_dataset, FEATURE_NAMES

    _print_section("Loading audio feature dataset")
    X_train, X_test, y_train, y_test, meta = load_dataset(500)
    n_feat = meta["n_features"]
    print(f"  Features: {n_feat}   Train: {meta['n_train']}   Test: {meta['n_test']}")

    results, models, preds = [], {}, {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── SVM ─────────────────────────────────────────────────────────────────
    _print_section("Training SVM (RBF kernel) — primary acoustic model")
    svm = SVC(kernel="rbf", C=1.5, gamma="scale", probability=True, random_state=RANDOM_SEED)
    svm.fit(X_train, y_train)
    p_svm = svm.predict(X_test)
    results.append(_evaluate("SVM", y_test, p_svm))
    models["SVM"] = svm
    preds["SVM"]  = p_svm

    # ── Random Forest ───────────────────────────────────────────────────────
    _print_section("Training Random Forest")
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    p_rf = rf.predict(X_test)
    results.append(_evaluate("Random Forest", y_test, p_rf))
    models["Random Forest"] = rf
    preds["Random Forest"]  = p_rf

    # ── LSTM (if TensorFlow available) ──────────────────────────────────────
    if TF_AVAILABLE:
        _print_section("Training LSTM (sequence over feature blocks)")
        # Reshape 36-d vector into 4 timesteps × 9 channels
        steps = 4 if n_feat % 4 == 0 else 1
        per   = n_feat // steps
        Xtr = X_train.reshape(-1, steps, per).astype(np.float32)
        Xte = X_test.reshape(-1,  steps, per).astype(np.float32)

        keras_model = Sequential([
            LSTM(64, input_shape=(steps, per), return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1,  activation="sigmoid"),
        ])
        keras_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        keras_model.fit(
            Xtr, y_train, epochs=40, batch_size=16,
            validation_split=0.15, verbose=0,
            callbacks=[EarlyStopping(patience=6, restore_best_weights=True)],
        )
        wrapper = _LSTMWrapper(keras_model)
        p_lstm  = wrapper.predict(X_test)
        results.append(_evaluate("LSTM", y_test, p_lstm))
        models["LSTM"] = wrapper
        preds["LSTM"]  = p_lstm
    else:
        print("\n[audio_models] TensorFlow not installed — skipping LSTM.")

    # ── Pick best by F1-Score ───────────────────────────────────────────────
    best = max(results, key=lambda r: r["F1-Score"])
    best_name = best["Model"]
    best_model = models[best_name]
    print("\n" + "=" * 56)
    print(f"  Best audio model: {best_name}  (F1 = {best['F1-Score']:.3f})")
    print("=" * 56)

    # ── Confusion matrices ──────────────────────────────────────────────────
    for ax, name in zip(axes, ["SVM", "Random Forest", "LSTM"]):
        if name not in preds:
            ax.axis("off")
            ax.set_title(f"{name}\n(skipped)")
            continue
        cm = confusion_matrix(y_test, preds[name])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Healthy", "MDD"], yticklabels=["Healthy", "MDD"],
                    ax=ax, cbar=False)
        ax.set_title(f"{name} — Audio")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "audio_confusion_matrices.png"), dpi=130)
    plt.close(fig)

    # ── Persist everything ──────────────────────────────────────────────────
    payload = {
        "model":      best_model,
        "model_name": best_name,
        "results":    results,
        "meta": {
            "feature_names": FEATURE_NAMES,
            "n_features":    n_feat,
            "n_samples":     meta["n_samples"],
            "mu":            meta["mu"],
            "sigma":         meta["sigma"],
        },
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\n  Saved audio twin → {MODEL_PATH}")

    print("\n  Summary:")
    for r in results:
        marker = " ← BEST" if r["Model"] == best_name else ""
        print(f"    {r['Model']:<14s}  Acc={r['Accuracy']:.3f}  Prec={r['Precision']:.3f}  "
              f"Rec={r['Recall']:.3f}  F1={r['F1-Score']:.3f}{marker}")
    return results, best_name


if __name__ == "__main__":
    train_and_evaluate()
