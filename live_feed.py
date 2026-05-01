import os
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime

# ── CONFIGURATION ──────────────────────────────────────
HF_DATASET_ID   = "pyToshka/network-intrusion-detection"
HF_SPLIT        = "train"

OUTPUT_FILE     = "live_can_feed.csv"
ROWS_PER_FETCH  = 200
REFRESH_SECONDS = 2
MAX_ROWS        = 1000  # Rolling buffer

FORCE_ATTACK_MODE = True

# ── LABEL MAPPING ─────────────────────────────────────
LABEL_MAP = {
    "normal":   "attack-free",
    "dos":      "DoS-attack",
    "fuzzy":    "fuzzy-attack",
    "spoofing": "spoofing-attack",
    "replay":   "replay-attack",
}

def map_label(raw_label):
    raw = str(raw_label).lower()
    for key, val in LABEL_MAP.items():
        if key in raw:
            return val
    if raw == "0": return "attack-free"
    if raw == "1": return "DoS-attack"
    return "attack-free"

def label_to_can_id(label):
    return hex({
        "attack-free":    random.randint(0x300, 0x3FF),
        "DoS-attack":     random.randint(0x000, 0x0FF),
        "fuzzy-attack":   random.randint(0x100, 0x1FF),
        "spoofing-attack":random.randint(0x200, 0x2FF),
        "replay-attack":  random.randint(0x700, 0x7FF),
    }.get(label, 0x300))

# ── HUGGINGFACE FETCH ─────────────────────────────────
def fetch_from_hf():
    """
    FIX: The datasets library caches an internal HTTP client that becomes
    permanently broken after a DNS/network error ('client has been closed').
    Clearing the module-level fsspec/requests cache forces a clean reconnect.
    """
    try:
        # Force a fresh import context so any cached broken clients are dropped
        import importlib
        import datasets as ds_module
        importlib.reload(ds_module)   # <- resets internal HTTP session state

        from datasets import load_dataset
        ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT, streaming=True)
        shuffled_ds = ds.shuffle(seed=random.randint(0, 10000), buffer_size=100)
        batch = list(shuffled_ds.take(ROWS_PER_FETCH))
        df = pd.DataFrame(batch)
        return df

    except Exception as e:
        print(f"[HF] ❌ Error: {e}")
        # Also try clearing fsspec cache if available — handles the
        # 'Cannot send a request, as the client has been closed' state
        try:
            import fsspec
            fsspec.filesystem("https").clear_instance_cache()
        except Exception:
            pass
        return pd.DataFrame()

# ── PROCESSING & CONVERSION ───────────────────────────
def convert_to_can(df):
    label_col = next(
        (c for c in df.columns if any(x in c.lower() for x in ['label', 'class', 'type'])),
        None
    )
    rows = []
    base_ts = time.time()
    for i, row in df.iterrows():
        if FORCE_ATTACK_MODE and i % 5 == 0:
            attack = random.choice(["DoS-attack", "fuzzy-attack"])
        else:
            attack = map_label(row[label_col]) if label_col else "attack-free"
        rows.append({
            "Timestamp":   round(base_ts + i * 0.1, 4),
            "CAN_ID":      label_to_can_id(attack),
            "Payload":     "A1B2C3D4E5F60718",
            "attack_type": attack,
            "car_model":   "hf_live_node"
        })
    return pd.DataFrame(rows)

# ── MAIN LOOP ─────────────────────────────────────────
def run_feed():
    print("🌐 NIDS LIVE FEED STARTING...")
    print(f"📡 SOURCE: HuggingFace ({HF_DATASET_ID})")
    print(f"📁 SINK: {OUTPUT_FILE} (Rolling Buffer: {MAX_ROWS} rows)\n")

    consecutive_failures = 0

    while True:
        print(f"[FEED] Refresh @ {datetime.now().strftime('%H:%M:%S')}")

        hf_df = fetch_from_hf()

        if hf_df.empty:
            consecutive_failures += 1
            wait = min(2 * consecutive_failures, 30)  # back off up to 30s
            print(f"[WARN] No data fetched (failure #{consecutive_failures}). Retrying in {wait}s...")
            time.sleep(wait)
            continue

        consecutive_failures = 0  # reset on success
        new_data = convert_to_can(hf_df)

        if os.path.exists(OUTPUT_FILE):
            existing_df = pd.read_csv(OUTPUT_FILE)
            combined_df = pd.concat([existing_df, new_data]).tail(MAX_ROWS)
            combined_df.to_csv(OUTPUT_FILE, index=False)
        else:
            new_data.to_csv(OUTPUT_FILE, index=False)

        counts = new_data['attack_type'].value_counts().to_dict()
        print(f"[FEED] ✅ Appended {len(new_data)} rows. Distribution: {counts}")

        time.sleep(REFRESH_SECONDS)

if __name__ == "__main__":
    try:
        run_feed()
    except KeyboardInterrupt:
        print("\n🛑 Feed stopped by user.")