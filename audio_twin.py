"""
audio_twin.py — Audio Digital Twin

Mirror of digital_twin.py for the acoustic-biomarker model.
ŷt = g(zt)  where zt = audio feature vector.
"""

import os
import time
import joblib
import numpy as np
from collections import deque

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "saved_models", "audio_twin_model.pkl")
HISTORY_LEN = 20


class AudioDigitalTwin:
    """Continuously updated virtual representation of voice-based mental state."""

    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Audio model not found at '{model_path}'. "
                "Run python audio_models.py first."
            )
        payload          = joblib.load(model_path)
        self.model       = payload["model"]
        self.model_name  = payload.get("model_name", "SVM")
        self.meta        = payload["meta"]
        self.history     = deque(maxlen=HISTORY_LEN)
        self.state       = None
        self.last_updated = None

        mu    = self.meta["mu"]
        sigma = self.meta["sigma"]
        self.meta["mu"]    = np.asarray(mu,    dtype=np.float64).flatten()
        self.meta["sigma"] = np.asarray(sigma, dtype=np.float64).flatten()
        self.meta["sigma"][self.meta["sigma"] == 0] = 1.0

        print(f"[AudioTwin] Loaded {self.model_name} — {self.meta['n_features']} features.")

    # ── Internal ────────────────────────────────────────────────────────────

    def _vectorise(self, x):
        """Accept a dict (keyed by feature_names) or a list/array."""
        if isinstance(x, dict):
            x = [x[n] for n in self.meta["feature_names"]]
        return np.asarray(x, dtype=np.float64).reshape(1, -1)

    def _normalise(self, x):
        return (x - self.meta["mu"]) / self.meta["sigma"]

    # ── Public API ──────────────────────────────────────────────────────────

    def predict(self, x):
        x = self._normalise(self._vectorise(x))
        label = int(self.model.predict(x)[0])
        self.history.append({"label": label, "timestamp": time.time()})
        return label

    def predict_proba(self, x):
        x = self._normalise(self._vectorise(x))
        if hasattr(self.model, "predict_proba"):
            p1 = float(self.model.predict_proba(x)[0][1])
        else:  # fallback for models without proba
            p1 = float(self.model.predict(x)[0])
        return p1

    def update_state(self, x):
        """Run prediction and update twin state — returns full status dict."""
        t0    = time.perf_counter()
        prob  = self.predict_proba(x)
        label = 1 if prob >= 0.5 else 0
        self.history.append({"label": label, "prob": prob, "timestamp": time.time()})
        self.state = {
            "prediction":      "MDD" if label == 1 else "Healthy",
            "label":           label,
            "mdd_probability": round(prob, 4),
            "confidence":      round(max(prob, 1 - prob), 4),
            "alert":           bool(prob >= 0.7),
            "model_name":      self.model_name,
            "inference_ms":    round((time.perf_counter() - t0) * 1000, 2),
            "modality":        "audio",
        }
        self.last_updated = time.time()
        return self.state

    def get_status(self):
        return self.state or {
            "prediction":  None,
            "label":       None,
            "history_len": len(self.history),
            "model_name":  self.model_name,
            "modality":    "audio",
        }


if __name__ == "__main__":
    from audio_data import generate_audio_features, FEATURE_NAMES
    twin = AudioDigitalTwin()
    sample = generate_audio_features(1).iloc[0].to_dict()
    sample.pop("label", None)
    print(twin.update_state(sample))
