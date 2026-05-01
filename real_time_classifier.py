"""
real_time_classifier.py
-----------------------
Two responsibilities:
  1. replay_can_log()    — replays a CAN CSV onto a virtual bus (hardware sim)
  2. RealTimeClassifier  — reads from a virtual bus and classifies each frame

NOTE: 'virtual' bustype requires python-can with socketcan/kvaser/etc.
For simulation without hardware, use run_simulation.py instead.
"""

import can
import pandas as pd
import joblib
import numpy as np
import time
from collections import defaultdict, deque
from feature_engineering import create_time_window_features


# ─────────────────────────────────────────────────────────────
# PART 1: CAN Log Replay  (previously misnamed as the classifier)
# ─────────────────────────────────────────────────────────────
def replay_can_log(channel: str = 'vcan0',
                   log_file_path: str = 'test_simulation.csv',
                   realtime: bool = True):
    """
    Reads a CSV log file and replays the CAN messages onto a virtual bus.

    Args:
        channel       : virtual CAN channel name (e.g. 'vcan0')
        log_file_path : path to a CSV with columns Timestamp, CAN_ID, Payload
        realtime      : if True, replay at original speed; False = as fast as possible
    """
    print(f"[REPLAY] Replaying {log_file_path} on channel '{channel}'...")

    try:
        df = pd.read_csv(log_file_path)
        df = df.rename(columns={
            'timestamp': 'Timestamp',
            'arbitration_id': 'CAN_ID',
            'data_field': 'Payload'
        })
    except FileNotFoundError:
        print(f"[REPLAY] Error: Log file not found at {log_file_path}")
        return
    except Exception as e:
        print(f"[REPLAY] Error reading CSV: {e}")
        return

    try:
        bus = can.interface.Bus(channel=channel, bustype='virtual')
    except Exception as e:
        print(f"[REPLAY] Cannot open virtual bus '{channel}': {e}")
        print("  Hint: on Linux run  sudo modprobe vcan && sudo ip link add dev vcan0 type vcan && sudo ip link set up vcan0")
        return

    start_time  = time.time()
    log_t0      = df['Timestamp'].iloc[0]
    replayed    = 0
    errors      = 0

    for _, row in df.iterrows():
        if realtime:
            elapsed     = time.time() - start_time
            log_elapsed = row['Timestamp'] - log_t0
            if log_elapsed > elapsed:
                time.sleep(log_elapsed - elapsed)
        try:
            msg = can.Message(
                arbitration_id=int(str(row['CAN_ID']).strip(), 16),
                data=bytes.fromhex(str(row['Payload']).strip()),
                is_extended_id=False
            )
            bus.send(msg)
            replayed += 1
        except Exception:
            errors += 1

    bus.shutdown()
    print(f"[REPLAY] Complete — {replayed} messages sent, {errors} skipped.")


# ─────────────────────────────────────────────────────────────
# PART 2: Real-Time Classifier  (the actual classifier)
# ─────────────────────────────────────────────────────────────
class RealTimeClassifier:
    """
    Listens on a CAN bus channel, buffers frames into 1-second windows,
    engineers features on-the-fly, and classifies with Random Forest.
    Uses Isolation Forest as a second-opinion anomaly detector.
    """

    def __init__(self,
                 rf_path:  str = 'random_forest_model.joblib',
                 if_path:  str = 'isolation_forest_model.joblib',
                 enc_path: str = 'label_encoder.joblib',
                 window_size: float = 1.0,
                 channel: str = 'vcan0'):

        self.window_size = window_size
        self.channel     = channel
        self._window_buf: list = []   # raw rows in current window
        self._window_start: float = time.time()

        # Load models
        try:
            self.rf_model  = joblib.load(rf_path)
            self.encoder   = joblib.load(enc_path)
            self.feat_cols = list(self.rf_model.feature_names_in_)
            print(f"[CLF] Random Forest loaded ({len(self.feat_cols)} features)")
        except FileNotFoundError as e:
            raise RuntimeError(f"Model not found: {e}. Run setup.py first.")

        try:
            self.if_model = joblib.load(if_path)
            print("[CLF] Isolation Forest loaded")
        except FileNotFoundError:
            self.if_model = None
            print("[CLF] WARNING: Isolation Forest not found — anomaly layer disabled")

    # ── Public ──────────────────────────────────────────────────
    def run(self, max_windows: int = 0):
        """
        Start reading from the CAN bus and classifying.
        max_windows=0 means run indefinitely (Ctrl-C to stop).
        """
        try:
            bus = can.interface.Bus(channel=self.channel, bustype='virtual')
        except Exception as e:
            print(f"[CLF] Cannot open bus: {e}")
            return

        print(f"[CLF] Listening on '{self.channel}' — window={self.window_size}s")
        window_count = 0
        try:
            while True:
                msg = bus.recv(timeout=0.1)
                if msg:
                    self._buffer_message(msg)
                if time.time() - self._window_start >= self.window_size:
                    result = self._classify_window()
                    if result:
                        self._print_result(result, window_count)
                        window_count += 1
                    self._window_buf    = []
                    self._window_start  = time.time()
                    if max_windows and window_count >= max_windows:
                        break
        except KeyboardInterrupt:
            print("\n[CLF] Stopped.")
        finally:
            bus.shutdown()

    # ── Internal ────────────────────────────────────────────────
    def _buffer_message(self, msg: can.Message):
        self._window_buf.append({
            'Timestamp': msg.timestamp,
            'CAN_ID':    msg.arbitration_id,
            'Payload':   msg.data.hex().upper(),
            'attack_type': 'unknown',
            'car_model':   'live',
        })

    def _classify_window(self) -> dict | None:
        if not self._window_buf:
            return None

        df = pd.DataFrame(self._window_buf)
        feat_df = create_time_window_features(df, window_size=self.window_size)

        if feat_df.empty:
            return None

        drop_cols = [c for c in ['time_window', 'car_model', 'attack_type'] if c in feat_df.columns]
        X = feat_df.drop(columns=drop_cols)

        # Align feature columns to trained model
        X = X.reindex(columns=self.feat_cols, fill_value=0.0)

        # ── Layer 1: Random Forest ──
        rf_pred_enc = self.rf_model.predict(X)[0]
        rf_pred     = self.encoder.inverse_transform([rf_pred_enc])[0]
        rf_proba    = self.rf_model.predict_proba(X)[0]
        rf_conf     = float(rf_proba.max())

        # ── Layer 2: Isolation Forest (anomaly score) ──
        if_verdict  = 'N/A'
        if_score    = 0.0
        if self.if_model:
            if_raw      = self.if_model.decision_function(X)[0]
            if_label    = self.if_model.predict(X)[0]   # 1=normal, -1=anomaly
            if_score    = float(if_raw)
            if_verdict  = 'ANOMALY' if if_label == -1 else 'NORMAL'

        # ── Fusion: both flag attack → HIGH confidence ──
        is_attack = 'attack' in rf_pred and 'free' not in rf_pred
        if is_attack and if_verdict == 'ANOMALY':
            fused = 'HIGH — confirmed by both models'
        elif is_attack:
            fused = 'MEDIUM — RF only'
        elif if_verdict == 'ANOMALY':
            fused = 'LOW — anomaly only (IF), RF says normal'
        else:
            fused = 'CLEAR'

        return {
            'rf_prediction': rf_pred,
            'rf_confidence': rf_conf,
            'if_verdict':    if_verdict,
            'if_score':      if_score,
            'fused_verdict': fused,
            'msg_count':     len(self._window_buf),
        }

    @staticmethod
    def _print_result(r: dict, idx: int):
        is_atk = 'attack' in r['rf_prediction'] and 'free' not in r['rf_prediction']
        col    = '\033[91m' if is_atk else '\033[92m'
        rst    = '\033[0m'
        print(f"[W{idx:04d}] RF: {col}{r['rf_prediction'].upper()}{rst} "
              f"({r['rf_confidence']*100:.1f}%)  "
              f"IF: {r['if_verdict']}  "
              f"Fusion: {r['fused_verdict']}  "
              f"msgs={r['msg_count']}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'replay':
        log = sys.argv[2] if len(sys.argv) > 2 else 'test_simulation.csv'
        replay_can_log(log_file_path=log)
    else:
        clf = RealTimeClassifier()
        clf.run()