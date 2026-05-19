"""
digital_twin.py — IEEE Paper Section III.J
Digital Twin Model: ŷt = f(xt)  (Equation 10)
Maintains a continuously updated virtual representation of an
individual's mental health state.
"""

import os
import time
import joblib
import numpy as np
from collections import deque

# Absolute path — works from any directory
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "saved_models", "digital_twin_model.pkl")
HISTORY_LEN = 20


class EEGDigitalTwin:
    """
    Implements the Digital Twin function f(·) trained on EEG features.
    Continuously updates its virtual state as new EEG data arrives.
    """

    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "Run python train_models.py first."
            )
        payload           = joblib.load(model_path)
        self.model        = payload["model"]
        self.meta         = payload["meta"]
        self.history      = deque(maxlen=HISTORY_LEN)
        self.state        = None
        self.last_updated = None

        # Ensure mu and sigma are numpy arrays regardless of how they were saved
        mu    = self.meta["mu"]
        sigma = self.meta["sigma"]
        self.meta["mu"]    = np.asarray(mu,    dtype=np.float64).flatten()
        self.meta["sigma"] = np.asarray(sigma, dtype=np.float64).flatten()
        # Guard against zero sigma
        self.meta["sigma"][self.meta["sigma"] == 0] = 1.0

        print(f"[DigitalTwin] Loaded — {self.meta['n_features']} features, "
              f"{self.meta['n_samples']} training samples.")

    # ── Core inference (Eq. 10) ───────────────────────────────────────────────

    def predict(self, x):
        """Return integer label: 0 = Healthy, 1 = MDD."""
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        label = int(self.model.predict(x)[0])
        self.history.append({"label": label, "timestamp": time.time()})
        return label

    def predict_proba(self, x):
        """Return MDD probability in [0, 1]."""
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        return round(float(self.model.predict_proba(x)[0][1]), 4)

    # ── Rolling risk (% of recent predictions that are MDD) ──────────────────

    def _rolling_risk(self):
        """
        Returns rolling MDD risk percentage from recent history.
        Called by both update_state() and api.py /twin/history endpoint.
        """
        if not self.history:
            return 0.0
        return round(
            sum(h["label"] for h in self.history) / len(self.history) * 100, 1
        )

    # ── Digital twin state update ─────────────────────────────────────────────

    def update_state(self, raw_features):
        """
        Accepts a dict of raw EEG feature values (un-normalised).
        Normalises using stored μ/σ, runs inference, updates history.

        Returns full status dict compatible with IEEE Table II.
        """
        feature_names = self.meta["feature_names"]
        mu            = self.meta["mu"]
        sigma         = self.meta["sigma"]

        x_raw    = np.array([raw_features[f] for f in feature_names], dtype=np.float64)
        x_normed = ((x_raw - mu) / sigma).reshape(1, -1)

        t0    = time.perf_counter()
        label = self.predict(x_normed)
        proba = self.predict_proba(x_normed)
        ms    = round((time.perf_counter() - t0) * 1000, 2)

        self.state        = label
        self.last_updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        return {
            "timestamp":       self.last_updated,
            "prediction":      "MDD" if label == 1 else "Healthy",
            "mdd_probability": proba,
            "confidence":      round(max(proba, 1.0 - proba), 4),
            "inference_ms":    ms,
            "rolling_risk":    self._rolling_risk(),
            "alert":           proba > 0.70,
        }

    def get_status(self):
        """Return current twin state without running new inference."""
        if self.state is None:
            return {"status": "No data received yet. Send EEG features to /twin/update."}
        return {
            "current_state":    "MDD" if self.state == 1 else "Healthy",
            "rolling_risk_pct": self._rolling_risk(),
            "last_updated":     self.last_updated,
            "history_length":   len(self.history),
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, BASE_DIR)
    from eeg_data import generate_eeg_features
    twin = EEGDigitalTwin()
    row  = generate_eeg_features(1).drop(columns=["label"]).iloc[0].to_dict()
    s    = twin.update_state(row)
    print("\nDigital Twin Status:")
    for k, v in s.items():
        print(f"  {k:<22}: {v}")
