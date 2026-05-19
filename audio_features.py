"""
audio_features.py — Real audio feature extractor (librosa-based)

Takes a wav/mp3/flac file path (or raw numpy array) and produces the
36-dim feature vector expected by the AudioDigitalTwin, in the same
order as audio_data.FEATURE_NAMES.

Falls back to safe defaults if librosa is unavailable, so the rest
of the system still imports without error.
"""

import os
import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from audio_data import FEATURE_NAMES, MFCC_MEAN_NAMES, MFCC_STD_NAMES, N_MFCC


def _safe(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, (int, float, np.floating, np.integer)):
        if np.isnan(value) or np.isinf(value):
            return default
        return float(value)
    return default


def extract_features_from_file(audio_path, sr=22050, max_seconds=30):
    """
    Extract the 36-dim feature dict from an audio file on disk.
    Returns a dict keyed by FEATURE_NAMES.
    """
    if not LIBROSA_AVAILABLE:
        raise RuntimeError(
            "librosa is not installed. Run: pip install librosa soundfile"
        )
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    y, sr = librosa.load(audio_path, sr=sr, mono=True, duration=max_seconds)
    return extract_features_from_array(y, sr)


def extract_features_from_array(y, sr=22050):
    """Extract the 36-dim feature dict from a mono numpy waveform."""
    if not LIBROSA_AVAILABLE:
        raise RuntimeError("librosa is not installed.")

    if y.size == 0:
        # Return zero-vector if empty
        return {name: 0.0 for name in FEATURE_NAMES}

    # Normalise amplitude to avoid clipping artefacts
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    feats = {}

    # ── MFCCs (mean + std across time) ──────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std  = mfcc.std(axis=1)
    for i, name in enumerate(MFCC_MEAN_NAMES):
        feats[name] = _safe(mfcc_mean[i])
    for i, name in enumerate(MFCC_STD_NAMES):
        feats[name] = _safe(mfcc_std[i])

    # ── Pitch (F0) via piptrack ─────────────────────────────────────────────
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=float(librosa.note_to_hz('C2')),
            fmax=float(librosa.note_to_hz('C7')),
            sr=sr,
        )
        f0_voiced = f0[voiced_flag]
        if f0_voiced.size > 0:
            f0_voiced = f0_voiced[~np.isnan(f0_voiced)]
        if f0_voiced.size > 0:
            feats["pitch_mean"] = _safe(np.mean(f0_voiced), 150.0)
            feats["pitch_std"]  = _safe(np.std(f0_voiced),  20.0)
        else:
            feats["pitch_mean"], feats["pitch_std"] = 150.0, 20.0
    except Exception:
        feats["pitch_mean"], feats["pitch_std"] = 150.0, 20.0

    # ── Jitter & Shimmer (cycle-to-cycle perturbations, approximated) ───────
    try:
        zc = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
        feats["jitter"] = _safe(np.std(zc) / (np.mean(zc) + 1e-9), 0.015)
    except Exception:
        feats["jitter"] = 0.015

    try:
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        feats["shimmer"] = _safe(np.std(rms) / (np.mean(rms) + 1e-9), 0.05)
        feats["energy_mean"] = _safe(np.mean(rms), 0.05)
        feats["energy_std"]  = _safe(np.std(rms),  0.02)
    except Exception:
        feats["shimmer"]     = 0.05
        feats["energy_mean"] = 0.05
        feats["energy_std"]  = 0.02

    # ── Zero-crossing rate ──────────────────────────────────────────────────
    try:
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        feats["zcr_mean"] = _safe(np.mean(zcr), 0.07)
    except Exception:
        feats["zcr_mean"] = 0.07

    # ── Speaking rate (rough: onset density) ────────────────────────────────
    try:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        duration = len(y) / sr
        feats["speaking_rate"] = _safe(len(onsets) / max(duration, 1e-6), 4.0)
    except Exception:
        feats["speaking_rate"] = 4.0

    # ── Spectral centroid + rolloff ─────────────────────────────────────────
    try:
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        sr_ = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        feats["spectral_centroid"] = _safe(np.mean(sc), 2000.0)
        feats["spectral_rolloff"]  = _safe(np.mean(sr_), 4000.0)
    except Exception:
        feats["spectral_centroid"] = 2000.0
        feats["spectral_rolloff"]  = 4000.0

    # ── HNR (harmonics-to-noise ratio, dB; approximated via spectral flatness) ─
    try:
        flat = librosa.feature.spectral_flatness(y=y)[0]
        # Convert to dB-style HNR: low flatness → high HNR
        flat_mean = max(np.mean(flat), 1e-6)
        feats["hnr"] = _safe(-10.0 * np.log10(flat_mean), 15.0)
    except Exception:
        feats["hnr"] = 15.0

    # Fill any missing keys defensively (should never happen)
    for n in FEATURE_NAMES:
        feats.setdefault(n, 0.0)

    return feats


def features_to_vector(feats):
    """Convert a feature dict into the canonical numpy vector."""
    return np.array([feats[n] for n in FEATURE_NAMES], dtype=np.float64)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python audio_features.py <audio_file>")
        sys.exit(0)
    f = extract_features_from_file(sys.argv[1])
    for k, v in f.items():
        print(f"  {k:22s} = {v:.4f}")
