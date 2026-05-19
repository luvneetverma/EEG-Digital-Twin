# Cloud-Based EEG Digital Twin Framework
### Real-Time Mental Health Monitoring Using Machine Learning

Implementation of the IEEE paper:
> *"Cloud-Based EEG Digital Twin Framework for Real-Time Mental Monitoring Using Machine Learning"*  
> Salaria, Verma, Deepika D — SRM Institute of Science and Technology

---

## 📁 Project Structure

```
eeg_digital_twin/
├── eeg_data.py          # Synthetic EEG dataset generator + preprocessor
├── train_models.py      # SVM / Random Forest / LSTM training → Table I
├── digital_twin.py      # Digital Twin class (Eq. 10 from paper)
├── api.py               # Flask REST API — cloud deployment layer
├── demo_client.py       # Real-time monitoring demo client
├── run_all.py           # ⭐ One-shot: train + serve + demo
├── requirements.txt     # Python dependencies
├── saved_models/        # Auto-created — stores digital_twin_model.pkl
└── plots/               # Auto-created — confusion matrices, bar charts
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- pip

### 1. Clone / download the project
```bash
cd eeg_digital_twin
```

### 2. (Optional but recommended) Create a virtual environment
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **TensorFlow (LSTM)** is optional. If not installed, the LSTM experiment
> uses a Random Forest proxy so everything still runs.
> To install TensorFlow:
> ```bash
> pip install tensorflow
> ```

---

## 🚀 Running the Project

### Option A — Run everything at once (recommended)
```bash
python run_all.py
```
This will:
1. Generate the synthetic EEG dataset  
2. Train SVM, Random Forest, LSTM  
3. Save the digital twin model  
4. Start the REST API  
5. Run the real-time demo  
6. Print Tables I & II from the paper  

---

### Option B — Step by step

#### Step 1: Train the models
```bash
python train_models.py
```
Outputs:
- `saved_models/digital_twin_model.pkl`
- `plots/confusion_matrices.png`
- `plots/model_comparison_bar.png`
- `plots/table1_model_comparison.csv`

#### Step 2: Test the Digital Twin standalone
```bash
python digital_twin.py
```

#### Step 3: Start the REST API
```bash
python api.py
```
The API will be available at `http://127.0.0.1:5000`

#### Step 4: Run the demo client (new terminal)
```bash
python demo_client.py
```

---

## 🌐 REST API Endpoints

| Method | Endpoint          | Description                           |
|--------|-------------------|---------------------------------------|
| GET    | `/health`         | Health check + model status           |
| GET    | `/docs`           | API reference                         |
| GET    | `/twin/status`    | Current digital twin state            |
| GET    | `/twin/demo`      | Single demo inference (synthetic EEG) |
| POST   | `/twin/predict`   | Predict from EEG feature JSON         |
| POST   | `/twin/update`    | Update twin + return full status      |
| GET    | `/twin/history`   | Rolling prediction history            |

### Example API call
```bash
# Health check
curl http://127.0.0.1:5000/health

# Demo inference
curl http://127.0.0.1:5000/twin/demo

# Predict (POST with features)
curl -X POST http://127.0.0.1:5000/twin/predict \
     -H "Content-Type: application/json" \
     -d '{"psd_delta":2.5,"psd_theta":1.8,"psd_alpha":3.2,"psd_beta":2.0,"psd_gamma":0.9,"alpha_asymmetry":0.05,...}'
```

---

## 📊 Expected Results (Table I)

| Model         | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| SVM (RBF)     | 0.884    | 0.890     | 0.880  | 0.880    |
| Random Forest | 0.857    | 0.860     | 0.850  | 0.850    |
| LSTM          | 0.832    | 0.830     | 0.820  | 0.820    |

## ⚡ Expected Results (Table II)

| Parameter                | Value      |
|--------------------------|------------|
| Average Inference Latency | ≈ 120 ms  |
| API Response Stability   | 99.2%      |
| Cloud Deployment Mode    | REST API   |
| Continuous Update Support | Enabled   |

---

## 🔬 Paper Methodology Implemented

| Paper Section      | Implementation File    |
|--------------------|------------------------|
| Dataset (Eq. 1–2)  | `eeg_data.py`          |
| PSD (Eq. 3–4)      | `eeg_data.py`          |
| Coherence (Eq. 5)  | `eeg_data.py`          |
| Normalisation (Eq. 6) | `eeg_data.py`       |
| SVM (Eq. 7–8)      | `train_models.py`      |
| Random Forest (Eq. 9) | `train_models.py`   |
| Digital Twin (Eq. 10) | `digital_twin.py`   |
| Pipeline (Eq. 11)  | `api.py`               |
| Cloud REST API     | `api.py`               |

---

## 📝 Notes

- **Dataset**: The paper uses a private clinical EEG dataset. This implementation
  generates statistically realistic synthetic features that mirror the paper's
  description (spectral bands, coherence, demographics).  
  To use your own data: replace `generate_eeg_features()` in `eeg_data.py`
  with a CSV loader pointing to your dataset.

- **Real EEG data format**: Features expected per sample:
  - `psd_delta`, `psd_theta`, `psd_alpha`, `psd_beta`, `psd_gamma`
  - `alpha_asymmetry`
  - `coh_<ch1>_<ch2>` for 10 channel pairs
  - `age`, `gender`
