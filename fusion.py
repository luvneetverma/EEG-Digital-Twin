"""
fusion.py — Late-Fusion of EEG + Audio Digital Twins

Combines the two modality probabilities into a single mental-health
verdict using a weighted average:

    p_final = w_eeg * p_eeg + w_audio * p_audio
    label   = 1 if p_final >= 0.5 else 0

Default weighting: 0.6 EEG, 0.4 audio (EEG carries more clinical
weight per the literature; audio is a strong corroborating signal).
"""

import time
from typing import Optional


DEFAULT_WEIGHTS = {"eeg": 0.6, "audio": 0.4}
ALERT_THRESHOLD = 0.7


def fuse(eeg_result: Optional[dict],
         audio_result: Optional[dict],
         w_eeg: float = DEFAULT_WEIGHTS["eeg"],
         w_audio: float = DEFAULT_WEIGHTS["audio"]) -> dict:
    """
    Late-fusion of two twin results.

    Each *_result is the dict returned by twin.update_state() — must
    contain key 'mdd_probability'. Either may be None (single-modality
    fall-back).
    """
    t0 = time.perf_counter()

    p_eeg   = eeg_result.get("mdd_probability")   if eeg_result   else None
    p_audio = audio_result.get("mdd_probability") if audio_result else None

    if p_eeg is None and p_audio is None:
        return {
            "error": "Both modalities missing — nothing to fuse.",
            "prediction": None, "mdd_probability": None,
        }

    # Re-normalise weights when one modality is missing
    if p_eeg is None:
        w_eeg, w_audio = 0.0, 1.0
    elif p_audio is None:
        w_eeg, w_audio = 1.0, 0.0
    else:
        s = w_eeg + w_audio
        w_eeg, w_audio = w_eeg / s, w_audio / s

    p_final = (w_eeg   * (p_eeg   or 0.0)) + \
              (w_audio * (p_audio or 0.0))
    label = 1 if p_final >= 0.5 else 0
    confidence = max(p_final, 1 - p_final)

    # Severity score (0–100) — weighted prob × 100
    severity = round(p_final * 100, 1)

    # Risk band (used for the dashboard pill)
    if severity < 30:
        band = "Low"
    elif severity < 60:
        band = "Moderate"
    elif severity < 80:
        band = "High"
    else:
        band = "Critical"

    return {
        "prediction":       "MDD" if label == 1 else "Healthy",
        "label":            label,
        "mdd_probability":  round(p_final, 4),
        "confidence":       round(confidence, 4),
        "severity_score":   severity,
        "risk_band":        band,
        "alert":            bool(p_final >= ALERT_THRESHOLD),
        "weights":          {"eeg": round(w_eeg, 2), "audio": round(w_audio, 2)},
        "components": {
            "eeg":   {
                "probability": round(p_eeg,   4) if p_eeg   is not None else None,
                "label":       eeg_result.get("prediction")   if eeg_result   else None,
            },
            "audio": {
                "probability": round(p_audio, 4) if p_audio is not None else None,
                "label":       audio_result.get("prediction") if audio_result else None,
            },
        },
        "fusion_method":  "weighted-late-fusion",
        "fusion_ms":      round((time.perf_counter() - t0) * 1000, 3),
    }


if __name__ == "__main__":
    eeg   = {"prediction": "MDD",     "mdd_probability": 0.81}
    audio = {"prediction": "Healthy", "mdd_probability": 0.32}
    print(fuse(eeg, audio))
