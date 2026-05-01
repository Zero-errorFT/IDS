Looking at your project code, here's a cool name suggestion and the updated README:

---

**Name suggestion: SENTINEL**
*(System for Early Network Threat Identification, Neutralisation, and Event Logging)*

Or shorter variants: **NetSentinel**, **CANARY** *(CAN Anomaly Recognition & Response sYstem)* — CANARY works brilliantly because it's both a network security term and literally a canary-in-the-coal-mine metaphor for early threat detection.

**I'd go with: CANARY** — it covers both CAN bus and general network traffic, it's memorable, and the metaphor is perfect for an IDS.

---

Here's the updated README:

---

# 🐦 CANARY
### CAN Anomaly Recognition & Response sYstem

> *"Like a canary in a coal mine — the first to detect danger."*

A real-time, hybrid machine learning Intrusion Detection System (IDS) that monitors live network traffic across enterprise networks, IoT deployments, and in-vehicle CAN bus systems. CANARY detects cyberattacks — DoS flooding, message spoofing, replay attacks, and fuzzy injection — in under 5 milliseconds per window, with a live monitoring dashboard that any operator can read at a glance.

---

## 🎯 What Problem Does This Solve?

Modern networks — whether inside a car, across a factory floor, or in a cloud data centre — transmit thousands of messages per second with no built-in authentication at the packet level. Attackers exploit this through:

| Attack | Description |
|---|---|
| **DoS Flooding** | Overwhelm a network node with repeated messages |
| **Spoofing** | Inject fake messages impersonating a trusted source |
| **Replay** | Re-transmit captured legitimate messages to trigger unintended actions |
| **Fuzzy / Injection** | Send randomised payloads to probe for vulnerabilities |

Traditional signature-based IDS miss novel attacks. CANARY uses machine learning trained on statistical traffic behaviour — so it detects attacks it has never explicitly seen before.

---

## 🧠 How It Works

```
Live Traffic (CAN Bus / Network / HuggingFace Stream)
                        ↓
          Preprocessing & 1-Second Windowing
                        ↓
         Feature Engineering (~5,900 features)
         [Frequency · Mean Payload · Entropy]
                        ↓
    ┌───────────────────────────────────────┐
    │  Random Forest (Supervised)           │  ← Detects known attacks
    │  Isolation Forest (Unsupervised)      │  ← Detects zero-day anomalies
    └───────────────────────────────────────┘
                        ↓
           Hybrid Fusion + Threshold Engine
           [NONE · LOW · MEDIUM · HIGH]
                        ↓
         Real-Time Dashboard + Alerts + LLM
```

---

## ✨ Key Features

- **Dual-Model Detection** — Random Forest for known attack classification; Isolation Forest for zero-day anomaly detection
- **Hybrid Severity Fusion** — Both models agree → HIGH; one flags → MEDIUM/LOW
- **Dynamic Threshold Engine** — Thresholds auto-adjust by time of day (night traffic vs. rush hour); manual override via dashboard slider
- **Protocol-Agnostic** — Works on CAN bus traffic, general TCP/IP flows, and live HuggingFace network datasets
- **Live Network Topology Map** — Interactive SVG node map; nodes pulse red during attacks; add/edit/delete devices
- **LLM Payload Inspector** — Optional Llama 3 (via Ollama) for deep hex payload analysis and natural-language explanations
- **< 5ms Inference Latency** — Suitable for edge gateway deployment
- **~96% Classification Accuracy** on held-out test data

---

## 📊 Dashboard Preview

Seven live tabs:

| Tab | What You See |
|---|---|
| **Overview** | Live prediction status, hybrid verdict, accuracy, system metrics, Chart.js timelines |
| **Network Map** | SVG topology with animated attack propagation |
| **Analytics** | Attack distribution, accuracy over time, entropy history, severity timeline |
| **Event Log** | Scrollable table of every detection decision |
| **LLM Inspector** | Llama 3 payload analysis results |
| **Settings** | Threshold sensitivity slider + profile info |
| **Roadmap** | Project milestones and future work |

---

## 🤖 Models

### Random Forest (Supervised)
- 5-class classification: `attack-free`, `DoS-attack`, `fuzzy-attack`, `spoofing-attack`, `replay-attack`
- 100 estimators, stratified 75/25 train/test split
- ~96% accuracy on held-out test set
- Feature importances available for interpretability

### Isolation Forest (Unsupervised)
- Trained on normal traffic only
- Detects previously unseen / zero-day attack patterns
- contamination = 0.1
- ~80% binary accuracy (normal vs. anomaly)

### Hybrid Fusion Logic
```
RF attack + IF anomaly  →  HIGH severity
RF attack only          →  MEDIUM severity
IF anomaly only         →  LOW severity
Both clear              →  NONE
```

---

## ⚡ Performance

| Metric | Value |
|---|---|
| Inference latency | < 5 ms per window |
| Throughput | 400+ windows/second |
| RF classification accuracy | ~96% |
| Feature dimensions | ~5,900 per window |
| Supported attack types | 4 + zero-day |

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/your-username/canary-nids.git
cd canary-nids
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Data and Train Models

```bash
python setup.py
```

This generates synthetic traffic, engineers features, trains both models, and saves all artefacts. Takes about 30–60 seconds.

### 4. Start the Dashboard

```bash
python app.py
```

Open **http://localhost:8081** in your browser. Press **START** in the dashboard.

---

## 📡 Data Sources

| Source | How to Use |
|---|---|
| **Shuffled Simulation** | Select in dashboard — auto-generated by `setup.py` |
| **Full Feature Set** | `features.csv` — all 600 windows in order |
| **Live CAN Feed** | Run `python live_feed.py` in a terminal, then select "Live CAN Feed" |
| **HuggingFace Live** | Select "HuggingFace Live" — streams `pyToshka/network-intrusion-detection` in real time (requires `pip install datasets`) |
| **Your own CSV logs** | Format: `Timestamp, CAN_ID, Payload` — drop into project folder and update `dsel` in the dashboard |

---

## 🔧 Optional: LLM Payload Inspector

CANARY can send suspicious payloads to a local Llama 3 model for natural-language analysis:

```bash
# Install Ollama: https://ollama.com
ollama serve
ollama pull llama3

# Then restart app.py — LLM inspector activates automatically
```

The LLM inspector tab will show verdicts like: *"Payload FFFFFFFFFFFFFFFF with CAN ID 0x050 — HIGH confidence DoS; all bytes at maximum value is a classic flooding pattern."*

---

## 📁 Project Structure

```
canary-nids/
├── app.py                    # Flask backend + Socket.IO streaming
├── setup.py                  # Data generation + model training (run this first)
├── feature_engineering.py    # Time-window feature extraction
├── threshold_engine.py       # Dynamic rule-based alerting
├── llm_inspector.py          # Ollama / Llama 3 payload analysis
├── train_model.py            # Standalone model training script
├── evaluate_model.py         # Held-out evaluation + confusion matrix export
├── real_time_classifier.py   # CAN bus hardware replay + live classifier
├── live_feed.py              # HuggingFace live data feed
├── data_loader.py            # CSV dataset loader for real CAN logs
├── templates/
│   └── index.html            # Dashboard (single-file SPA)
├── features.csv              # Generated feature matrix
├── shuffled_features.csv     # Shuffled version for simulation
├── live_can_feed.csv         # Live feed buffer (generated by live_feed.py)
├── random_forest_model.joblib
├── isolation_forest_model.joblib
├── label_encoder.joblib
└── requirements.txt
```

---

## 🛡️ Use Cases

- **Automotive cybersecurity** — CAN bus IDS for connected vehicles
- **IoT network monitoring** — detect anomalous device behaviour
- **Enterprise network security** — low-latency edge detection
- **Security research** — benchmark ML models on network intrusion datasets
- **Education** — demonstrates hybrid ML, real-time streaming, and dashboard design

---

## ⚠️ Limitations

- Accuracy depends on training data diversity — synthetic data may not capture all real-world noise
- Detection only, not prevention — CANARY raises alerts but does not block traffic
- Requires retraining when deployed on new network types or vehicle models
- LLM inspector requires local GPU or patient CPU inference

---

## 🔮 Roadmap

- [x] Random Forest multi-class classifier
- [x] Isolation Forest zero-day detection
- [x] Dynamic threshold engine with time-of-day profiles
- [x] Interactive network topology map
- [x] HuggingFace live streaming
- [ ] LLM payload inspector (in progress — requires Ollama)
- [ ] LSTM / Transformer sequence-aware detection
- [ ] Online learning for concept drift adaptation
- [ ] Federated learning across distributed network nodes
- [ ] ECU-level embedded deployment (STM32 / Raspberry Pi)

---

CANARY — watching your network so you don't have to.*