"""
app.py  -  NIDS Backend with HuggingFace Streaming Support
"""
import time, os, threading, random
import pandas as pd
import joblib
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from threading import Lock, Event
from sklearn.metrics import accuracy_score
from datetime import datetime
import psutil

from threshold_engine import threshold_engine
from llm_inspector import llm_inspector

app = Flask(__name__, template_folder="templates")

NODE_MAP = [
    {"id":"node_router",   "label":"Core Router",   "can_range":(0x000,0x0FF),"type":"network"},
    {"id":"node_firewall", "label":"Firewall",       "can_range":(0x100,0x1FF),"type":"security"},
    {"id":"node_server",   "label":"Main Server",    "can_range":(0x200,0x2FF),"type":"server"},
    {"id":"node_gateway",  "label":"Gateway / Hub",  "can_range":(0x300,0x3FF),"type":"gateway"},
    {"id":"node_iot_a",    "label":"IoT Device A",   "can_range":(0x400,0x4FF),"type":"iot"},
    {"id":"node_iot_b",    "label":"IoT Device B",   "can_range":(0x500,0x5FF),"type":"iot"},
    {"id":"node_endpoint", "label":"Endpoint / PC",  "can_range":(0x600,0x6FF),"type":"client"},
    {"id":"node_external", "label":"External / OTA", "can_range":(0x700,0x7FF),"type":"external"},
]

def can_id_to_node(can_id_str):
    try:
        s = str(can_id_str).strip()
        if not s:
            return "node_external"
        cid = int(s, 16) if s.lower().startswith("0x") else int(s)
        if cid < 0:
            return "node_external"
    except (ValueError, TypeError) as e:
        print(f"[WARN] Invalid CAN ID '{can_id_str}': {e}")
        return "node_external"
    for n in NODE_MAP:
        lo, hi = n["can_range"]
        if lo <= cid <= hi:
            return n["id"]
    return "node_external"

app.config["SECRET_KEY"] = "nids-secret"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")
thread = None
thread_lock = Lock()
stop_event = Event()
pause_event = Event()   # when set, streaming loops wait (paused state)

# ── Streaming state ──────────────────────────────────────────
_hf_buffer = []      # rows fetched from HuggingFace but not yet emitted
_hf_lock   = Lock()
_hf_thread = None
_hf_active = False

CSV_STREAM_DELAY = float(os.getenv("CSV_STREAM_DELAY", "0.4"))
LIVE_STREAM_DELAY = float(os.getenv("LIVE_STREAM_DELAY", "0.35"))
HF_STREAM_DELAY = float(os.getenv("HF_STREAM_DELAY", "0.75"))
HF_CONNECT_POLL_DELAY = float(os.getenv("HF_CONNECT_POLL_DELAY", "0.35"))
EMIT_MIN_INTERVAL = float(os.getenv("EMIT_MIN_INTERVAL", "0.35"))
LLM_MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "5.0"))

_prediction_engine = None  # singleton — loaded once, reused every simulation run

def get_prediction_engine():
    global _prediction_engine
    if _prediction_engine is None:
        _prediction_engine = PredictionEngine()
    return _prediction_engine

class PredictionEngine:
    def __init__(self):
        self.rf_model = None
        self.if_model = None
        self.encoder  = None
        self.feature_columns = None
        try:
            self.rf_model        = joblib.load("random_forest_model.joblib")
            self.encoder         = joblib.load("label_encoder.joblib")
            self.feature_columns = list(self.rf_model.feature_names_in_)
            print("OK  Random Forest loaded")
        except FileNotFoundError as e:
            print(f"MISSING RF model: {e}\n   --> Run: python setup.py")
        try:
            self.if_model = joblib.load("isolation_forest_model.joblib")
            print("OK  Isolation Forest loaded")
        except FileNotFoundError:
            print("Isolation Forest not found — anomaly layer disabled")

    def predict(self, fv):
        """Returns a plain-Python dict safe for JSON serialisation."""
        result = {
            "label":      "model-not-loaded",
            "if_anomaly": False,   # plain Python bool — never numpy.bool_
            "if_score":   0.0,
            "fused_sev":  "none",
            "rf_conf":    0.0,
        }
        if self.rf_model is None:
            return result

        # Align columns to exactly what the model was trained on
        fv_aligned = fv.reindex(columns=self.feature_columns, fill_value=0.0)

        # ── Random Forest ──────────────────────────────────────
        rf_enc   = self.rf_model.predict(fv_aligned)[0]
        rf_proba = self.rf_model.predict_proba(fv_aligned)[0]
        rf_label = self.encoder.inverse_transform([rf_enc])[0]
        rf_conf  = float(rf_proba.max())            # float() strips numpy scalar

        result["label"]   = str(rf_label)
        result["rf_conf"] = rf_conf
        is_rf_attack = "attack" in rf_label and "free" not in rf_label

        # ── Isolation Forest ────────────────────────────────────
        is_if_anomaly = False
        if self.if_model is not None:
            if_pred       = self.if_model.predict(fv_aligned)[0]   # numpy int
            if_score      = float(self.if_model.decision_function(fv_aligned)[0])
            is_if_anomaly = bool(if_pred == -1)     # cast to plain Python bool
            result["if_anomaly"] = is_if_anomaly
            result["if_score"]   = if_score

        # ── Hybrid fusion severity ──────────────────────────────
        if is_rf_attack and is_if_anomaly:
            result["fused_sev"] = "high"
        elif is_rf_attack:
            result["fused_sev"] = "medium"
        elif is_if_anomaly:
            result["fused_sev"] = "low"
        else:
            result["fused_sev"] = "none"

        return result

def _freq_cols(row): return [c for c in row.index if str(c).endswith("_freq")]
def _ent_cols(row):  return [c for c in row.index if str(c).endswith("_entropy")]
def _msg_rate(row):
    fc=_freq_cols(row); return float(row[fc].sum()) if fc else 0.0
def _entropy(row):
    ec=_ent_cols(row); return float(row[ec].mean()) if ec else 0.0
def _repeat(row):
    fc=_freq_cols(row)
    if not fc: return 0.0
    v=row[fc].values.astype(float); t=v.sum()
    return float(v.max()/t) if t>0 else 0.0
def _dominant_can(row):
    fc=_freq_cols(row)
    if fc: return str(max(fc,key=lambda c:row[c])).replace("_freq","")
    return "unknown"

def _synthetic_hf_payload(attack_type):
    if attack_type == "DoS-attack":
        patterns = ["FFFFFFFF00000000", "0000000000000000", "FFFF0000FFFF0000"]
    elif attack_type == "fuzzy-attack":
        patterns = [
            "".join(f"{random.randint(0, 255):02X}" for _ in range(8))
            for _ in range(3)
        ]
    elif attack_type == "spoofing-attack":
        patterns = ["A1B2C3D4E5F60718", "A1B2C3D400000000", "1122A1B2C3D45566"]
    elif attack_type == "replay-attack":
        patterns = ["4455667744556677", "1020304010203040", "A1B2C3D4A1B2C3D4"]
    else:
        patterns = ["1122334455667788", "2143658721436587", "1021324354657687"]
    return random.choice(patterns)

def _synthetic_hf_can_id(attack_type):
    if attack_type == "DoS-attack":
        return hex(random.choice([0x040, 0x050, 0x060]))
    if attack_type == "fuzzy-attack":
        return hex(random.randint(0x120, 0x1EF))
    if attack_type == "spoofing-attack":
        return hex(random.choice([0x220, 0x250, 0x2A0]))
    if attack_type == "replay-attack":
        return hex(random.choice([0x710, 0x720, 0x750]))
    return hex(random.choice([0x320, 0x355, 0x410, 0x4A0, 0x620, 0x680]))

# ── HuggingFace background fetcher ──────────────────────────
def _hf_fetcher(dataset_id="pyToshka/network-intrusion-detection"):
    """Runs in background thread: streams rows from HuggingFace into _hf_buffer."""
    global _hf_active
    try:
        from datasets import load_dataset
        print(f"[HF] Starting stream: {dataset_id}")
        ds = load_dataset(dataset_id, split="train", streaming=True)
        label_map = {
            "normal":"attack-free","dos":"DoS-attack","probe":"fuzzy-attack",
            "r2l":"spoofing-attack","u2r":"replay-attack","neptune":"DoS-attack",
            "smurf":"DoS-attack","back":"DoS-attack","teardrop":"DoS-attack",
            "ipsweep":"fuzzy-attack","satan":"fuzzy-attack",
        }
        for item in ds:
            if not _hf_active:
                break
            raw_label = str(item.get("label", item.get("class", item.get("target", "normal")))).lower()
            attack_type = "attack-free"
            for k, v in label_map.items():
                if k in raw_label:
                    attack_type = v; break

            payload = _synthetic_hf_payload(attack_type)
            can_id = _synthetic_hf_can_id(attack_type)

            row = {
                "Timestamp":   time.time(),
                "CAN_ID":      can_id,
                "Payload":     payload[:16],
                "attack_type": attack_type,
                "car_model":   "huggingface_live",
            }
            with _hf_lock:
                if len(_hf_buffer) < 500:
                    _hf_buffer.append(row)
            time.sleep(0.1)  # ~10 rows/sec → realistic live feel

    except Exception as e:
        print(f"[HF] Streamer error: {e}")
    finally:
        _hf_active = False
        print("[HF] Streamer stopped")

def _start_hf_streamer():
    global _hf_thread, _hf_active, _hf_buffer
    _hf_active = True
    _hf_buffer = []
    _hf_thread = threading.Thread(target=_hf_fetcher, daemon=True)
    _hf_thread.start()

def _stop_hf_streamer():
    global _hf_active
    _hf_active = False

def _sleep_with_pause(delay):
    if delay <= 0:
        return
    end_time = time.time() + delay
    while not stop_event.is_set() and time.time() < end_time:
        while pause_event.is_set() and not stop_event.is_set():
            socketio.sleep(0.05)
        remaining = end_time - time.time()
        if remaining <= 0:
            break
        socketio.sleep(min(0.05, remaining))

# ── Feature engineering for raw CAN rows ────────────────────
def _raw_to_feature_vector(rows, engine):
    """Convert a list of raw CAN dicts into a feature vector for prediction."""
    import numpy as np
    if not rows:
        return None, "unknown", "attack-free"

    # Dominant label + CAN ID
    labels = [r.get("attack_type","attack-free") for r in rows]
    from collections import Counter
    true_label = Counter(labels).most_common(1)[0][0]

    can_ids = []
    for r in rows:
        try:
            s = str(r.get("CAN_ID", "0")).strip()
            can_ids.append(int(s, 16) if s.startswith("0x") else int(s))
        except:
            can_ids.append(0)

    dom_can_id = Counter(can_ids).most_common(1)[0][0]
    node_id = can_id_to_node(str(dom_can_id))

    if engine.feature_columns is None:
        return None, node_id, true_label

    # Build a zero feature vector
    fv = pd.DataFrame(0.0, index=[0], columns=engine.feature_columns)

    # Fill frequency features
    # FIX (minor): removed unused `total = len(can_ids)` that was here before
    can_counts = Counter(can_ids)
    for cid, cnt in can_counts.items():
        col = f"{cid}_freq"
        if col in fv.columns:
            fv[col] = cnt

    # Fill payload mean/entropy features
    for r in rows:
        try:
            s = str(r.get("CAN_ID","0")).strip()
            cid = int(s, 16) if s.startswith("0x") else int(s)
        except:
            continue
        payload = r.get("Payload","")
        try:
            bvals = [int(payload[i:i+2], 16) for i in range(0, min(len(payload),16), 2)]
        except:
            continue
        if bvals:
            mean_col = f"{cid}_mean"
            if mean_col in fv.columns:
                fv[mean_col] = np.mean(bvals)
            ent_col = f"{cid}_entropy"
            if ent_col in fv.columns:
                probs = np.bincount(bvals, minlength=256).astype(float)
                probs = probs[probs > 0]; p = probs / probs.sum()
                fv[ent_col] = float(-np.sum(p * np.log2(p)))

    return fv, node_id, true_label

# ── Main streaming functions ─────────────────────────────────
_fv_cache: dict = {}   # dataset_path → list of pre-aligned feature vectors
_last_llm_inspect_time = 0.0

def _run_llm_inspection_async(payload_hex, node_id, prediction, msg_rate, ent_val, rep_ratio, pred_result):
    try:
        llm_result = llm_inspector.inspect(
            payload_hex   = payload_hex,
            can_id        = str(node_id),
            context       = prediction,
            msg_rate      = msg_rate,
            entropy       = ent_val,
            repeat_ratio  = rep_ratio,
            rf_conf       = pred_result["rf_conf"],
            if_anomaly    = bool(pred_result["if_anomaly"]),
            fused_sev     = pred_result["fused_sev"],
        )
        socketio.emit("llm_update", {
            "llm": llm_result,
            "timestamp": time.strftime("%H:%M:%S"),
            "node_id": str(node_id),
            "prediction": prediction,
        })
    except Exception as e:
        print(f"[LLM] Async inspection failed: {e}")

def stream_csv(model, dataset):
    """Stream from a pre-existing CSV file (shuffled_features.csv etc.)"""
    engine = get_prediction_engine()

    if not os.path.exists(dataset):
        socketio.emit("simulation_status",{"status":"error","message":f"Dataset not found: {dataset}. Run setup.py first."})
        return
    if engine.feature_columns is None:
        socketio.emit("simulation_status",{"status":"error","message":"Models not found. Run: python setup.py"})
        return

    df = pd.read_csv(dataset)
    missing = [c for c in engine.feature_columns if c not in df.columns]
    if missing:
        socketio.emit("simulation_status",{"status":"error","message":f"Dataset missing {len(missing)} feature columns. Re-run setup.py"})
        return

    # FIX 1: Pull attack_type and time_window from the DataFrame directly before
    # iterating rows. Using row.get() on a pandas Series is unreliable — a column
    # name can shadow an index label and silently return the wrong value.
    if dataset not in _fv_cache:
        print(f"[CSV] Building feature cache for {dataset}...")
        subset = df.head(200).reset_index(drop=True)
        attack_types = subset["attack_type"].astype(str) if "attack_type" in subset.columns else pd.Series(["unknown"] * len(subset))
        time_windows = subset["time_window"].astype(str) if "time_window" in subset.columns else subset.index.astype(str)
        _fv_cache[dataset] = [
            (
                row.to_frame().T[engine.feature_columns],
                attack_types.iloc[i],
                time_windows.iloc[i],
            )
            for i, (_, row) in enumerate(subset.iterrows())
        ]
        print(f"[CSV] Cached {len(_fv_cache[dataset])} rows.")
    else:
        print(f"[CSV] Using cached feature vectors for {dataset}")

    print(f"[CSV] Streaming {len(_fv_cache[dataset])} rows from {dataset} in loop mode...")
    y_true, y_pred = [], []

    while not stop_event.is_set():
        for fv, true_label, window_id in _fv_cache[dataset]:
            if stop_event.is_set():
                break
            while pause_event.is_set() and not stop_event.is_set():
                socketio.sleep(0.3)
            _emit_prediction(engine, fv, true_label, window_id, y_true, y_pred)
            _sleep_with_pause(CSV_STREAM_DELAY)

def stream_live_can(model, dataset):
    """Stream from live_can_feed.csv as raw CAN rows (feature-engineered on-the-fly)."""
    engine = get_prediction_engine()

    if not os.path.exists(dataset):
        socketio.emit("simulation_status",{"status":"error","message":f"File not found: {dataset}. Run live_feed.py first."})
        return
    if engine.feature_columns is None:
        socketio.emit("simulation_status",{"status":"error","message":"Models not found. Run: python setup.py"})
        return

    y_true, y_pred = [], []
    cycle = 0
    window_counter = 0

    while not stop_event.is_set():
        if not os.path.exists(dataset):
            socketio.emit("simulation_status",{"status":"error","message":f"File not found: {dataset}. Run live_feed.py first."})
            return

        df = pd.read_csv(dataset)
        total_rows = len(df)
        if total_rows == 0:
            socketio.emit("simulation_status",{"status":"waiting","message":"live_can_feed.csv is empty. Waiting for data..."})
            _sleep_with_pause(0.5)
            continue

        window_size = 25 if total_rows >= 250 else 10
        print(f"[LIVE_CSV] Cycle {cycle}: streaming {total_rows} raw CAN rows with window={window_size}")

        for i in range(0, total_rows, window_size):
            if stop_event.is_set():
                break
            while pause_event.is_set() and not stop_event.is_set():
                socketio.sleep(0.3)
            batch = df.iloc[i:i+window_size].to_dict("records")
            if len(batch) < 2:
                continue
            fv, node_id, true_label = _raw_to_feature_vector(batch, engine)
            if fv is None:
                continue
            _emit_prediction(engine, fv, true_label, str(window_counter), y_true, y_pred, node_id)
            window_counter += 1
            _sleep_with_pause(LIVE_STREAM_DELAY)

        cycle += 1

def stream_hf_live(model):
    """Stream from HuggingFace in real-time."""
    engine = get_prediction_engine()
    if engine.feature_columns is None:
        socketio.emit("simulation_status",{"status":"error","message":"Models not found. Run: python setup.py"})
        return

    _start_hf_streamer()
    print("[HF] Emitting live stream...")
    y_true, y_pred = [], []
    window_idx = 0
    WINDOW = 5

    socketio.emit("simulation_status",{"status":"hf_connecting","message":"Connecting to HuggingFace..."})
    # Wait briefly for first rows
    waited = 0
    while len(_hf_buffer) < WINDOW and waited < 15:
        socketio.sleep(HF_CONNECT_POLL_DELAY); waited += HF_CONNECT_POLL_DELAY

    if len(_hf_buffer) == 0:
        socketio.emit("simulation_status",{"status":"error","message":"HuggingFace unreachable. Check internet or run live_feed.py first."})
        _stop_hf_streamer()
        return

    socketio.emit("simulation_status",{"status":"hf_live","message":"HuggingFace stream connected"})

    while not stop_event.is_set():
        # FIX 2: Added missing pause support — this loop never checked pause_event
        while pause_event.is_set() and not stop_event.is_set():
            socketio.sleep(0.3)

        with _hf_lock:
            if len(_hf_buffer) < WINDOW:
                batch = list(_hf_buffer)
                _hf_buffer.clear()
            else:
                batch = _hf_buffer[:WINDOW]
                del _hf_buffer[:WINDOW]

        if not batch:
            socketio.sleep(HF_CONNECT_POLL_DELAY); continue

        fv, node_id, true_label = _raw_to_feature_vector(batch, engine)
        if fv is not None:
            _emit_prediction(engine, fv, true_label, str(window_idx), y_true, y_pred, node_id)
            window_idx += 1
        _sleep_with_pause(HF_STREAM_DELAY)

    _stop_hf_streamer()
    socketio.emit("simulation_status",{"status":"finished"})

_last_emit_time = 0.0

def _emit_prediction(engine, fv, true_label, window_id, y_true, y_pred, node_id=None):
    global _last_emit_time, _last_llm_inspect_time
    now = time.time()
    # Still compute stats every window; only throttle the socketio push
    should_emit = (now - _last_emit_time) >= EMIT_MIN_INTERVAL
    pred_result = engine.predict(fv)
    prediction  = pred_result["label"]          # plain str

    row       = fv.iloc[0]
    msg_rate  = _msg_rate(row)
    ent_val   = _entropy(row)
    rep_ratio = _repeat(row)
    if node_id is None:
        node_id = can_id_to_node(_dominant_can(row))

    tr = threshold_engine.check(
        msg_rate=msg_rate, entropy_score=ent_val,
        repeat_ratio=rep_ratio, ml_prediction=prediction)

    # Bump severity if IF also flags anomaly
    sev = tr["severity"]
    if pred_result["if_anomaly"] and sev in ("none", "low"):
        sev = "medium"

    llm_result = None
    llm_allowed = (now - _last_llm_inspect_time) >= LLM_MIN_INTERVAL
    if tr["alert"] and sev in ("medium", "high") and llm_allowed and llm_inspector.is_available():
        # Build real payload hex from the dominant CAN ID's mean bytes
        try:
            dom_col = _dominant_can(row)
            mean_col = f"{dom_col}_mean"
            mean_val = int(float(row.get(mean_col, 0))) & 0xFF
            phex = format(mean_val, '02X') * 8          # 8-byte payload from mean
        except Exception:
            phex = "FFFFFFFF00000000" if "DoS" in prediction else "A1B2C3D400000000"
        socketio.start_background_task(
            _run_llm_inspection_async,
            phex, node_id, prediction, msg_rate, ent_val, rep_ratio, pred_result.copy()
        )
        _last_llm_inspect_time = now

    y_true.append(true_label); y_pred.append(prediction)
    acc    = accuracy_score(y_true, y_pred)
    is_atk = "attack" in prediction and "free" not in prediction

    if not should_emit:
        return  # skip this emit, accumulate data but don't flood browser

    _last_emit_time = now
    socketio.emit("live_update", {
        "prediction":    prediction,
        "true_label":    true_label,
        "is_correct":    bool(prediction == true_label),  # plain bool
        "window_id":     window_id,
        "timestamp":     time.strftime("%H:%M:%S"),
        "accuracy":      round(float(acc), 4),
        "cpu_percent":   float(psutil.cpu_percent(interval=None)),
        "memory_percent":float(psutil.virtual_memory().percent),
        "disk_percent":  float(psutil.disk_usage("/").percent),
        "threshold": {
            "alert":      bool(tr["alert"]),
            "severity":   sev,
            "reasons":    tr["reasons"],
            "profile":    tr["profile"],
            "multiplier": float(tr["multiplier"]),
            "thresholds": {k: float(v) for k, v in tr["thresholds_used"].items()},
        },
        # Hybrid dual-model verdict — all values explicitly cast to Python natives
        "hybrid": {
            "rf_label":   prediction,
            "rf_conf":    round(pred_result["rf_conf"], 3),
            "if_anomaly": bool(pred_result["if_anomaly"]),
            "if_score":   round(float(pred_result["if_score"]), 3),
            "fused_sev":  pred_result["fused_sev"],
        },
        "node":     {"id": node_id, "is_attack": bool(is_atk), "severity": sev},
        "llm":      llm_result,
        "msg_rate": round(float(msg_rate), 1),
        "entropy":  round(float(ent_val), 3),
    })

def stream_data(model, dataset):
    """Router: picks the right streaming strategy based on dataset."""
    if dataset == "hf_live":
        stream_hf_live(model)
    # FIX 3: Was "live_feed.py" (a script name) — changed to the actual CSV filename
    elif dataset == "live_can_feed.csv":
        if os.path.exists(dataset):
            df = pd.read_csv(dataset, nrows=2)
            if "Payload" in df.columns and "CAN_ID" in df.columns:
                stream_live_can(model, dataset)
            else:
                stream_csv(model, dataset)
        else:
            socketio.emit("simulation_status",{"status":"error","message":"live_can_feed.csv not found. Run live_feed.py"})
    else:
        stream_csv(model, dataset)

# ── Flask routes ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/reset_cache", methods=["POST"])
def reset_cache():
    global _fv_cache, _prediction_engine
    _fv_cache.clear()
    _prediction_engine = None   # force model reload too
    return jsonify({"status": "ok", "message": "Feature cache and model cleared"})

@app.route("/api/nodes")
def get_nodes(): return jsonify(NODE_MAP)

@app.route("/api/profile")
def get_profile(): return jsonify(threshold_engine.get_current_profile())

@app.route("/api/threshold", methods=["POST"])
def set_threshold():
    mult = request.get_json().get("multiplier", None)
    threshold_engine.set_manual_multiplier(mult)
    return jsonify({"status":"ok","multiplier":mult,"profile":threshold_engine.get_current_profile()})

@app.route("/api/llm_status")
def llm_status(): return jsonify({"available":llm_inspector.is_available()})

@app.route("/api/hf_status")
def hf_status():
    return jsonify({
        "buffer_size": len(_hf_buffer),
        "active": _hf_active,
        "datasets_installed": _check_datasets_lib(),
    })

def _check_datasets_lib():
    try:
        import datasets; return True
    except ImportError:
        return False

@app.route("/api/dataset_status")
def dataset_status():
    out = []
    configs = [
        ("shuffled_features.csv","Shuffled Simulation","csv"),
        ("live_can_feed.csv",    "Live CAN Feed","raw"),
        ("features.csv",         "Full Feature Set","csv"),
        ("hf_live",              "HuggingFace Live Stream","live"),
    ]
    for fname,label,dtype in configs:
        if dtype == "live":
            out.append({
                "file":fname,"label":label,"exists":_check_datasets_lib(),
                "size_kb":0,"dtype":dtype,
                "updated":"Real-time" if _check_datasets_lib() else "Install: pip install datasets",
                "source": "HuggingFace Streaming"
            })
        elif os.path.exists(fname):
            st = os.stat(fname)
            out.append({"file":fname,"label":label,"exists":True,"dtype":dtype,
                "size_kb":st.st_size//1024,
                "updated":datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "source": "Local CSV"})
        else:
            out.append({"file":fname,"label":label,"exists":False,"size_kb":0,"dtype":dtype,"source":"—"})
    return jsonify(out)

# ── SocketIO events ──────────────────────────────────────────
@socketio.on("start_simulation")
def handle_start(data):
    global thread, _last_emit_time, _last_llm_inspect_time
    with thread_lock:
        if thread is None or not thread.is_alive():
            stop_event.clear()
            pause_event.clear()
            _last_emit_time = 0.0
            _last_llm_inspect_time = 0.0
            thread = socketio.start_background_task(stream_data, data["model"], data["dataset"])

@socketio.on("pause_simulation")
def handle_pause():
    pause_event.set()

@socketio.on("resume_simulation")
def handle_resume():
    pause_event.clear()

@socketio.on("stop_simulation")
def handle_stop():
    stop_event.set()
    pause_event.clear()
    _stop_hf_streamer()

@socketio.on("set_threshold")
def handle_threshold(data):
    threshold_engine.set_manual_multiplier(data.get("multiplier",None))
    socketio.emit("threshold_update", threshold_engine.get_current_profile())

if __name__ == "__main__":
    print("Starting NIDS server...")
    print(f"  LLM: {'active' if llm_inspector.is_available() else 'offline'}")
    print(f"  HuggingFace datasets: {'installed' if _check_datasets_lib() else 'not installed (pip install datasets)'}")
    print(f"  Threshold: {threshold_engine.get_current_profile()['label']}")
    if not os.path.exists("random_forest_model.joblib"):
        print("  WARNING: No trained model found!")
        print("  --> Run: python setup.py")
    socketio.run(app, host="0.0.0.0", port=8081, debug=True, allow_unsafe_werkzeug=True)
