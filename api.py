"""
api.py — IEEE Paper Section III.K + Audio Extension (multimodal late-fusion)
REST API exposing the EEG + Audio Digital Twins as a cloud service.

Pipelines:
    EEG   Input -> Preprocessing -> EEG Digital Twin   -> Output  (Eq. 11)
    Audio Input -> MFCC/Prosody  -> Audio Digital Twin -> Output
                                       |
                            Late-Fusion (60% EEG / 40% Audio)
                                       |
                              Final Verdict + PDF

Endpoints
---------
GET  /                     -> Web dashboard
GET  /health               -> Health check (now reports both modalities)
GET  /docs                 -> API reference

EEG modality
GET  /twin/status          -> EEG twin state
POST /twin/predict         -> Predict from EEG features
POST /twin/update          -> Update EEG twin + full status
GET  /twin/demo            -> Demo EEG inference
GET  /twin/history         -> Rolling risk + recent labels

Audio modality
POST /audio/predict        -> Predict from posted audio FEATURE dict
POST /audio/upload         -> Upload .wav/.mp3 -> extract -> predict
GET  /audio/demo           -> Demo audio inference (synthetic)
GET  /audio/status         -> Audio twin state

Fusion + reports
POST /fused/predict        -> Combine EEG + audio probabilities
POST /report/generate      -> 7-section EEG-only PDF (legacy)
POST /report/full          -> Patient-flow PDF (demographics + clinical + EEG + audio + fusion)
POST /report/preview       -> JSON dry-run for the dashboard
"""

import os
import sys
import time
import tempfile
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

# Ensure imports work regardless of working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

try:
    from flask_cors import CORS
    _cors_available = True
except ImportError:
    _cors_available = False

app = Flask(__name__, static_folder=BASE_DIR)
if _cors_available:
    CORS(app)   # Allow all origins — needed for internet deployment

# ── Auto-train if model missing, then load digital twin ──────────────────────
print("[API] Initialising EEG Digital Twin …")

def _ensure_model():
    model_path = os.path.join(BASE_DIR, "saved_models", "digital_twin_model.pkl")
    if not os.path.exists(model_path):
        print("[API] Model not found — training now (first-time setup) …")
        from train_models import train_and_evaluate
        train_and_evaluate()
        print("[API] Training complete.")

_ensure_model()

try:
    from digital_twin import EEGDigitalTwin
    twin        = EEGDigitalTwin()
    MODEL_READY = True
    print("[API] Digital twin ready.")
except Exception as e:
    print(f"[API] ERROR loading twin: {e}")
    MODEL_READY = False
    twin        = None

from eeg_data import generate_eeg_features   # for /twin/demo

# ── Audio twin: auto-train + load ────────────────────────────────────────────
def _ensure_audio_model():
    audio_path = os.path.join(BASE_DIR, "saved_models", "audio_twin_model.pkl")
    if not os.path.exists(audio_path):
        print("[API] Audio model not found - training now (first-time setup)...")
        from audio_models import train_and_evaluate as _train_audio
        _train_audio()

_ensure_audio_model()

try:
    from audio_twin import AudioDigitalTwin
    audio_twin = AudioDigitalTwin()
    AUDIO_READY = True
    print(f"[API] Audio digital twin ready ({audio_twin.model_name}).")
except Exception as e:
    print(f"[API] WARNING audio twin unavailable: {e}")
    audio_twin = None
    AUDIO_READY = False

from audio_data import generate_audio_features, FEATURE_NAMES as AUDIO_FEATURE_NAMES

try:
    from audio_features import extract_features_from_file, LIBROSA_AVAILABLE
except Exception as e:
    LIBROSA_AVAILABLE = False
    extract_features_from_file = None
    print(f"[API] librosa unavailable - /audio/upload disabled: {e}")

from fusion import fuse, DEFAULT_WEIGHTS

# Optional PDF report generator
try:
    from report_generator import generate_clinical_report
    _PDF_READY = True
except ImportError:
    _PDF_READY = False
    print("[API] reportlab not installed - /report/generate unavailable. pip install reportlab")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_model():
    if not MODEL_READY:
        return jsonify({
            "error": "Model not ready. Check server logs.",
            "hint":  "Run python train_models.py manually if this persists."
        }), 503
    return None


def _make_demo_features():
    df = generate_eeg_features(n_samples=1)
    return df.drop(columns=["label"]).iloc[0].to_dict()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def dashboard():
    """Serve the web dashboard."""
    return send_from_directory(BASE_DIR, "dashboard.html")


@app.route("/health", methods=["GET"])
def health():
    """Table II: API response stability check."""
    return jsonify({
        "status":            "ok",
        "model_ready":       MODEL_READY,
        "eeg_ready":         MODEL_READY,
        "audio_ready":       AUDIO_READY,
        "librosa_available": bool(LIBROSA_AVAILABLE),
        "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "framework":         "Cloud-Based EEG + Audio Digital Twin",
    })


@app.route("/docs", methods=["GET"])
def docs():
    return jsonify({
        "title":     "EEG Digital Twin REST API",
        "paper":     "Cloud-Based EEG Digital Twin Framework for Real-Time Mental Monitoring",
        "endpoints": {
            "GET  /":               "Web dashboard (open in browser)",
            "GET  /health":         "Health check",
            "GET  /twin/status":    "Current digital twin state",
            "POST /twin/predict":   "Predict mental health from EEG features",
            "POST /twin/update":    "Update twin + get full status dict",
            "GET  /twin/history":   "Rolling risk + recent labels",
            "GET  /twin/demo":      "One demo inference with synthetic data",
        },
        "predict_body_schema": {
            "psd_delta":        "float  — Delta band power (0.5–4 Hz)",
            "psd_theta":        "float  — Theta band power (4–8 Hz)",
            "psd_alpha":        "float  — Alpha band power (8–13 Hz)",
            "psd_beta":         "float  — Beta band power (13–30 Hz)",
            "psd_gamma":        "float  — Gamma band power (30+ Hz)",
            "alpha_asymmetry":  "float  — Right-left frontal alpha asymmetry",
            "coh_Fp1_Fp2":      "float  — Coherence [0,1]",
            "... (10 coherences total)": "see /twin/demo for full schema",
            "age":              "float",
            "gender":           "float  — 0 or 1",
        }
    })


@app.route("/twin/status", methods=["GET"])
def twin_status():
    err = _require_model()
    if err:
        return err
    return jsonify(twin.get_status())


@app.route("/twin/predict", methods=["POST"])
def twin_predict():
    """
    Run inference on posted EEG features.
    Target latency ≤ 120 ms (Table II).
    """
    err = _require_model()
    if err:
        return err

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required. See /docs for schema."}), 400

    try:
        t0     = time.perf_counter()
        status = twin.update_state(data)
        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        return jsonify({
            "prediction":      status["prediction"],
            "mdd_probability": status["mdd_probability"],
            "confidence":      status["confidence"],
            "inference_ms":    total_ms,
            "alert":           status["alert"],
        })
    except KeyError as e:
        return jsonify({"error": f"Missing feature in request body: {e}",
                        "hint": "See /docs for the full feature schema."}), 422
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/twin/update", methods=["POST"])
def twin_update():
    """Update digital twin state — returns full status dict."""
    err = _require_model()
    if err:
        return err

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required."}), 400

    try:
        return jsonify(twin.update_state(data))
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 422
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/twin/demo", methods=["GET"])
def twin_demo():
    """Demo inference with auto-generated synthetic EEG data."""
    err = _require_model()
    if err:
        return err
    features = _make_demo_features()
    status   = twin.update_state(features)
    return jsonify({
        "message":  "Demo inference using synthetic EEG data",
        "features": {k: round(float(v), 4) for k, v in features.items()},
        "result":   status,
    })


@app.route("/twin/history", methods=["GET"])
def twin_history():
    """Rolling MDD risk and recent prediction labels."""
    err = _require_model()
    if err:
        return err
    return jsonify({
        "rolling_risk_pct": twin._rolling_risk(),          # fixed: method now exists
        "history_length":   len(twin.history),
        "recent_labels":    [h["label"] for h in list(twin.history)[-10:]],
    })


@app.route("/report/generate", methods=["POST"])
def report_generate():
    """
    Generate a structured clinical PDF report from real-world patient EEG data.

    Expected JSON body:
    {
        "patient_info": {
            "patient_name": "John Doe",
            "patient_id":   "PT-001",
            "doctor_name":  "Dr. Smith",
            "hospital":     "City Hospital",
            "report_date":  "2026-04-15",     (optional)
            "clinical_notes": "..."           (optional)
        },
        "eeg_features": {
            "psd_delta": 1.2, "psd_theta": 0.9, ... (all 18 features)
        }
    }

    Returns: PDF file (application/pdf)
    """
    if not _PDF_READY:
        return jsonify({
            "error": "reportlab not installed.",
            "fix":   "pip install reportlab"
        }), 503

    err = _require_model()
    if err:
        return err

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required."}), 400

    patient_info  = data.get("patient_info", {})
    eeg_features  = data.get("eeg_features")
    if not eeg_features:
        return jsonify({"error": "eeg_features field required."}), 400

    try:
        # Run digital twin inference on the provided features
        twin_result = twin.update_state(eeg_features)

        # Enrich result with rolling risk
        twin_result["rolling_risk"] = twin_result.get("rolling_risk", twin._rolling_risk())

        # Generate PDF
        pdf_bytes = generate_clinical_report(patient_info, eeg_features, twin_result)

        from flask import Response
        patient_name = patient_info.get("patient_name", "patient").replace(" ", "_")
        filename = f"EEG_Report_{patient_name}_{time.strftime('%Y%m%d_%H%M%S')}.pdf"

        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(pdf_bytes)),
                "X-Prediction": twin_result.get("prediction", "-"),
                "X-MDD-Probability": str(twin_result.get("mdd_probability", 0)),
            }
        )

    except KeyError as e:
        return jsonify({"error": f"Missing EEG feature: {e}",
                        "hint": "See /docs for feature schema."}), 422
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# -----------------------------------------------------------------------------
# DASHBOARD JSON PREVIEW
# -----------------------------------------------------------------------------

@app.route("/report/preview", methods=["POST"])
def report_preview():
    """Run inference only (no PDF). Used by the dashboard for quick feedback."""
    err = _require_model()
    if err: return err
    data = request.get_json(silent=True) or {}
    eeg_features = data.get("eeg_features")
    if not eeg_features:
        return jsonify({"error": "eeg_features required."}), 400
    try:
        res = twin.update_state(eeg_features)
        res["rolling_risk"] = res.get("rolling_risk", twin._rolling_risk())
        return jsonify(res)
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 422
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# -----------------------------------------------------------------------------
# AUDIO DIGITAL TWIN ENDPOINTS
# -----------------------------------------------------------------------------

@app.route("/audio/status", methods=["GET"])
def audio_status():
    if not AUDIO_READY:
        return jsonify({"error": "Audio twin not loaded.",
                        "fix": "pip install librosa soundfile && python audio_models.py"}), 503
    return jsonify({
        "ready":       True,
        "model_name":  audio_twin.model_name,
        "n_features":  audio_twin.meta["n_features"],
        "state":       audio_twin.get_status(),
    })


@app.route("/audio/predict", methods=["POST"])
def audio_predict():
    if not AUDIO_READY:
        return jsonify({"error": "Audio twin not loaded."}), 503
    data = request.get_json(silent=True) or {}
    feats = data.get("audio_features")
    if not feats:
        return jsonify({"error": "audio_features field required."}), 400
    try:
        return jsonify(audio_twin.update_state(feats))
    except KeyError as e:
        return jsonify({"error": f"Missing audio feature: {e}"}), 422
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/audio/upload", methods=["POST"])
def audio_upload():
    """Upload an audio file (wav/mp3/webm). Extracts features and runs twin."""
    if not AUDIO_READY:
        return jsonify({"error": "Audio twin not loaded."}), 503
    try:
        from audio_features import extract_features_from_file, LIBROSA_AVAILABLE
    except Exception as e:
        return jsonify({"error": f"audio_features import failed: {e}"}), 503
    if not LIBROSA_AVAILABLE:
        return jsonify({"error": "librosa not installed on server.",
                        "fix": "pip install librosa soundfile"}), 503

    f = request.files.get("audio_file")
    if f is None:
        return jsonify({"error": "audio_file (multipart) required."}), 400

    import tempfile
    suffix = os.path.splitext(f.filename or "rec.webm")[1] or ".webm"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.save(tmp.name); tmp.close()
    try:
        feats = extract_features_from_file(tmp.name)
        res   = audio_twin.update_state(feats)
        return jsonify({"features": feats, "result": res})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    finally:
        try: os.unlink(tmp.name)
        except Exception: pass


@app.route("/audio/demo", methods=["GET"])
def audio_demo():
    if not AUDIO_READY:
        return jsonify({"error": "Audio twin not loaded."}), 503
    sample = generate_audio_features(1).iloc[0].to_dict()
    sample.pop("label", None)
    return jsonify({"features": sample, "result": audio_twin.update_state(sample)})


# -----------------------------------------------------------------------------
# FUSED (MULTIMODAL) PREDICTION
# -----------------------------------------------------------------------------

@app.route("/fused/predict", methods=["POST"])
def fused_predict():
    err = _require_model()
    if err: return err
    data = request.get_json(silent=True) or {}
    eeg_features   = data.get("eeg_features")
    audio_features = data.get("audio_features")
    weights        = data.get("weights") or DEFAULT_WEIGHTS
    if not eeg_features:
        return jsonify({"error": "eeg_features required."}), 400
    try:
        eeg_res   = twin.update_state(eeg_features)
        audio_res = (audio_twin.update_state(audio_features)
                     if (audio_features and AUDIO_READY) else None)
        fused_res = fuse(eeg_res, audio_res,
                         w_eeg=float(weights.get("eeg",   DEFAULT_WEIGHTS["eeg"])),
                         w_audio=float(weights.get("audio", DEFAULT_WEIGHTS["audio"])))
        return jsonify({"eeg": eeg_res, "audio": audio_res, "fused": fused_res})
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 422
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# -----------------------------------------------------------------------------
# PATIENT-FLOW FULL REPORT (EEG + Audio + Fusion + PDF)
# -----------------------------------------------------------------------------

@app.route("/report/full", methods=["POST"])
def report_full():
    """
    End-to-end patient-flow report.

    Two request shapes supported:
      (A) application/json  - body includes eeg_features and (optional) audio_features
                              or use_demo_audio: true to use synthetic demo acoustic features
      (B) multipart/form-data - audio_file (wav/webm/mp3) + payload (JSON string)

    Returns JSON with fused verdict plus base64-encoded PDF (pdf_base64 and
    report_pdf_base64 aliases) and audio_source provenance field.
    """
    if not _PDF_READY:
        return jsonify({"error": "reportlab not installed.",
                        "fix": "pip install reportlab"}), 503

    err = _require_model()
    if err: return err

    # ---- Parse input (multipart OR JSON) ----
    uploaded_audio_path = None
    if request.content_type and request.content_type.startswith("multipart/"):
        import json as _json, tempfile
        raw_payload = request.form.get("payload", "{}")
        try:
            data = _json.loads(raw_payload)
        except Exception:
            return jsonify({"error": "Invalid 'payload' JSON in multipart form."}), 400
        f = request.files.get("audio_file")
        if f is not None:
            suffix = os.path.splitext(f.filename or "rec.webm")[1] or ".webm"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            f.save(tmp.name); tmp.close()
            uploaded_audio_path = tmp.name
    else:
        data = request.get_json(silent=True) or {}

    patient_info     = data.get("patient_info", {})
    clinical_history = data.get("clinical_history", {})
    eeg_features     = data.get("eeg_features")
    audio_features   = data.get("audio_features")
    use_demo_audio   = bool(data.get("use_demo_audio"))
    weights          = data.get("weights") or DEFAULT_WEIGHTS

    if not eeg_features:
        if uploaded_audio_path:
            try: os.unlink(uploaded_audio_path)
            except Exception: pass
        return jsonify({"error": "eeg_features required."}), 400

    # ---- Resolve audio features from the three possible sources ----
    audio_source = "none"
    try:
        if uploaded_audio_path and AUDIO_READY:
            try:
                from audio_features import extract_features_from_file, LIBROSA_AVAILABLE
                if not LIBROSA_AVAILABLE:
                    raise RuntimeError("librosa not installed on server.")
                audio_features = extract_features_from_file(uploaded_audio_path)
                audio_source   = "uploaded_file"
            except Exception as _aerr:
                try:
                    sample = generate_audio_features(1).iloc[0].to_dict()
                    sample.pop("label", None)
                    audio_features = sample
                    audio_source   = f"demo_fallback ({type(_aerr).__name__}: {_aerr})"
                except Exception:
                    audio_features = None
        elif use_demo_audio and AUDIO_READY:
            sample = generate_audio_features(1).iloc[0].to_dict()
            sample.pop("label", None)
            audio_features = sample
            audio_source   = "demo_synthetic"
        elif audio_features and AUDIO_READY:
            audio_source   = "client_supplied"
    finally:
        if uploaded_audio_path:
            try: os.unlink(uploaded_audio_path)
            except Exception: pass

    # ---- Run both twins + fusion ----
    try:
        eeg_res   = twin.update_state(eeg_features)
        eeg_res["rolling_risk"] = eeg_res.get("rolling_risk", twin._rolling_risk())
        audio_res = (audio_twin.update_state(audio_features)
                     if (audio_features and AUDIO_READY) else None)
        fused_res = fuse(eeg_res, audio_res,
                         w_eeg=float(weights.get("eeg",   DEFAULT_WEIGHTS["eeg"])),
                         w_audio=float(weights.get("audio", DEFAULT_WEIGHTS["audio"])))

        pname = (patient_info.get("patient_name")
                 or patient_info.get("name") or "patient")
        pdf_bytes = generate_clinical_report(patient_info, eeg_features, eeg_res)

        import base64
        pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
        return jsonify({
            "patient_info":       patient_info,
            "clinical_history":   clinical_history,
            "eeg":                eeg_res,
            "audio":              audio_res,
            "audio_source":       audio_source,
            "fusion":             fused_res,
            "fused":              fused_res,
            "pdf_base64":         pdf_b64,
            "report_pdf_base64":  pdf_b64,
            "pdf_filename":       f"MentalHealth_Report_{str(pname).replace(' ','_')}_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
            "generated_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 422
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 58)
    print("  Multimodal Digital Twin Cloud API  (EEG + Audio)")
    print("  IEEE: Cloud-Based Digital Twin Framework + Audio Extension")
    print("=" * 58)
    print(f"  Dashboard   : http://127.0.0.1:{port}/")
    print(f"  Health      : http://127.0.0.1:{port}/health")
    print(f"  EEG demo    : http://127.0.0.1:{port}/twin/demo")
    print(f"  Audio demo  : http://127.0.0.1:{port}/audio/demo")
    print(f"  Docs        : http://127.0.0.1:{port}/docs")
    print("=" * 58 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
