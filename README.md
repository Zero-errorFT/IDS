🐦 CANARY

CAN Anomaly Recognition & Response sYstem

Real-Time Hybrid Machine Learning Intrusion Detection System

“Like a canary in a coal mine — the first signal when something goes wrong.”

⸻

⚡ Overview

CANARY is a real-time Intrusion Detection System (IDS) that monitors network and CAN bus traffic using a hybrid machine learning pipeline. It detects cyberattacks such as DoS flooding, spoofing, replay attacks, and anomalous injections in near real time and visualizes them through a live web dashboard.

The system combines:

* Supervised learning (Random Forest)
* Unsupervised anomaly detection (Isolation Forest)
* Rule-based threshold engine
* Optional LLM-powered forensic analysis

⸻

🎯 Key Features

* 🧠 Hybrid ML Detection System (Random Forest + Isolation Forest)
* ⚡ Real-Time Inference Pipeline with sliding window processing
* 📊 Live Dashboard (Flask + Socket.IO + Chart.js)
* 🌐 Network Topology Visualization (SVG-based interactive map)
* 🚨 Dynamic Threat Severity Engine (NONE / LOW / MEDIUM / HIGH)
* 🧾 Event Logging with CSV Export
* 🤖 Optional LLM Payload Forensics (Llama 3 via Ollama)
* 🔄 Multi-source streaming: CSV replay / live feed / HuggingFace dataset

⸻

🧠 How It Works

Live CAN / Network Traffic
            ↓
   1-Second Windowing Layer
            ↓
 Feature Engineering Engine
 (Statistical + Entropy + Frequency Features)
            ↓
┌──────────────────────────────┐
│ Random Forest Classifier     │ → Known attacks
│ Isolation Forest             │ → Anomaly detection
└──────────────────────────────┘
            ↓
   Hybrid Fusion Engine
            ↓
 Threshold + Rule-Based Layer
            ↓
  Real-Time Dashboard + Alerts

⸻

🚨 Threats Detected

* DoS Flooding (message burst attacks)
* Spoofing (fake identity injection)
* Replay Attacks (reused valid packets)
* Fuzzy / Random Injection (malformed payloads)
* Zero-day anomalies (behavior deviation)

⸻

🧠 Machine Learning Models

🌲 Random Forest (Supervised)

* Multi-class classification
* Attack categories + normal traffic
* ~96% accuracy on held-out dataset
* Interpretable feature importance

🌲 Isolation Forest (Unsupervised)

* Trained only on normal traffic
* Detects unknown / zero-day anomalies
* Outputs anomaly score for fusion system

⚖️ Fusion Strategy

* Both agree → HIGH severity
* RF only → MEDIUM
* IF only → LOW
* None → SAFE

⸻

⚙️ System Architecture

Data Ingestion Layer
   ├── CSV Simulation
   ├── Live CAN Feed
   └── HuggingFace Stream
            ↓
Feature Engineering Layer
   - Frequency analysis
   - Payload statistics
   - Shannon entropy
            ↓
ML Inference Layer
   - Random Forest
   - Isolation Forest
            ↓
Decision Engine
   - Hybrid fusion logic
   - Dynamic thresholding
            ↓
Visualization Layer
   - Flask backend
   - Socket.IO streaming
   - Chart.js dashboard
   - Network topology map

⸻

📊 Dashboard

The live dashboard includes:

* 📈 Real-time attack timeline
* 🧭 Interactive network topology map
* 📊 Attack distribution analytics
* 📜 Live event log
* ⚙️ Sensitivity controls
* 🤖 LLM forensic inspector (optional)

⸻

⚡ Performance

* ⏱️ Inference latency: near real-time (streaming pipeline optimized)
* 📦 Feature space: ~5,000+ engineered features per window
* 📡 Supports multiple live data sources
* 🧠 Dual-model ensemble for robustness

⸻

🤖 LLM Forensics (Optional)

Integrated with Llama 3 (via Ollama) for deep packet inspection:

* Payload decoding & analysis
* Attack classification explanation
* MITRE ATT&CK mapping
* Risk scoring (0–10)
* Recommended mitigation steps

⸻

🛠️ Tech Stack

* Backend: Flask, Flask-SocketIO
* ML: Scikit-learn (Random Forest, Isolation Forest), NumPy, Pandas
* Visualization: Chart.js, SVG, HTML/CSS/JS
* Streaming: WebSockets, threading
* LLM (optional): Ollama + Llama 3
* Data: CSV simulation, CAN logs, HuggingFace datasets

⸻

🚀 Quick Start

1. Clone Repo

git clone https://github.com/Zero-errorFT/IDS.git
cd IDS

2. Install Dependencies

pip install -r requirements.txt

3. Train Models + Setup Data

python setup.py

4. Run Dashboard

python app.py

Open:

http://localhost:8081

⸻

📁 Project Structure

├── app.py                  # Flask backend + WebSocket server
├── setup.py                # Data generation + model training
├── feature_engineering.py  # Feature extraction pipeline
├── train_model.py          # Model training
├── real_time_classifier.py # Live inference engine
├── threshold_engine.py     # Rule-based detection layer
├── llm_inspector.py       # LLM forensic module
├── live_feed.py           # Streaming data generator
├── templates/             # Dashboard frontend
├── *.joblib               # Trained ML models
├── *.csv                  # Dataset + simulation data

⸻

🔮 Future Improvements

* Transformer-based sequence detection (LSTM/Attention)
* Online learning for evolving attack patterns
* Federated learning across distributed nodes
* Embedded deployment (Raspberry Pi / ECU systems)
* Automated response system (active defense mode)

⸻

🛡️ Use Cases

* Automotive cybersecurity (CAN bus IDS)
* IoT anomaly detection systems
* Enterprise network monitoring
* Cybersecurity research & education
* Edge AI security systems

⸻

👨‍💻 Author Notes

CANARY is designed as a full-stack ML security system, demonstrating:

* End-to-end ML engineering
* Real-time streaming architecture
* Hybrid model design
* Practical cybersecurity application
* Production-style dashboard engineering

⸻

🐦 Final Thought

CANARY is not just a classifier — it is a real-time behavioral security layer that observes, learns, and reacts to network anomalies as they happen.
