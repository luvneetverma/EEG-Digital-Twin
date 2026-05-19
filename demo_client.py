"""
demo_client.py
--------------
Simulates real-time EEG monitoring by sending multiple
synthetic EEG samples to the running REST API.

Run AFTER starting the API:
    Terminal 1: python api.py
    Terminal 2: python demo_client.py
"""

import os
import sys
import time
import json
import requests
import numpy as np

# Allow running from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Change BASE_URL if your deployed Render URL is set via env var RENDER_URL
BASE_URL = os.environ.get("API_URL", "http://127.0.0.1:5000")


def print_section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def check_health():
    print_section("1. Health Check")
    r = requests.get(f"{BASE_URL}/health")
    print(json.dumps(r.json(), indent=2))


def run_demo_inference():
    print_section("2. Demo Inference (synthetic EEG data)")
    r = requests.get(f"{BASE_URL}/twin/demo")
    data = r.json()
    print(f"  Prediction   : {data['result']['prediction']}")
    print(f"  MDD Prob     : {data['result']['mdd_probability']}")
    print(f"  Confidence   : {data['result']['confidence']}")
    print(f"  Inference ms : {data['result']['inference_ms']} ms")
    print(f"  Alert        : {data['result']['alert']}")


def simulate_realtime_monitoring(n_samples: int = 10):
    """
    Simulate a streaming real-time monitoring session.
    Sends n_samples EEG feature vectors, 0.5 s apart.
    """
    print_section(f"3. Real-Time Monitoring Simulation ({n_samples} samples)")

    from eeg_data import generate_eeg_features
    df = generate_eeg_features(n_samples=n_samples)
    df_features = df.drop(columns=["label"])
    true_labels = df["label"].values

    latencies = []

    for i, (_, row) in enumerate(df_features.iterrows()):
        payload = row.to_dict()
        t0 = time.perf_counter()
        r  = requests.post(f"{BASE_URL}/twin/update", json=payload)
        latency = (time.perf_counter() - t0) * 1000

        if r.status_code == 200:
            result = r.json()
            latencies.append(latency)
            marker = "⚠️ " if result["alert"] else "✅ "
            print(f"  Sample {i+1:02d} | True: {'MDD' if true_labels[i]==1 else 'Healthy':8s}"
                  f" | Pred: {result['prediction']:8s}"
                  f" | MDD prob: {result['mdd_probability']:.3f}"
                  f" | {latency:.1f} ms  {marker}")
        else:
            print(f"  Sample {i+1:02d} | ERROR: {r.text}")

        time.sleep(0.2)   # simulate streaming interval

    print(f"\n  Avg latency : {np.mean(latencies):.1f} ms")
    print(f"  Max latency : {np.max(latencies):.1f} ms")
    print(f"  (Paper target: ≤ 120 ms)")


def show_twin_history():
    print_section("4. Digital Twin Rolling History")
    r = requests.get(f"{BASE_URL}/twin/history")
    data = r.json()
    print(f"  Rolling MDD risk : {data['rolling_risk_pct']}%")
    print(f"  Predictions kept : {data['history_length']}")
    print(f"  Last 10 labels   : {data['recent_labels']}")


def show_twin_status():
    print_section("5. Current Digital Twin State")
    r = requests.get(f"{BASE_URL}/twin/status")
    print(json.dumps(r.json(), indent=2))


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  EEG Digital Twin — Demo Client")
    print("="*55)

    try:
        check_health()
        run_demo_inference()
        simulate_realtime_monitoring(n_samples=12)
        show_twin_history()
        show_twin_status()
        print("\n✅  Demo complete.\n")
    except requests.exceptions.ConnectionError:
        print("\n❌  Cannot connect to API.")
        print("   Start the server first: python api.py\n")
