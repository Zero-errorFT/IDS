"""
setup.py — Run this ONCE to fix everything
==========================================
Fixes vs original:
  1. Entropy function renamed from calc_entropy → _setup_entropy to avoid
     collision with feature_engineering.py's calculate_entropy
  2. features_df.to_csv(..., index=True) → index=False (prevents 'Unnamed: 0'
     column appearing in CSV and breaking app.py column alignment)
  3. shuffled.to_csv also uses index=False consistently
  4. calc_entropy had dead code after 'return' — removed
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import random
from scipy.stats import entropy as scipy_entropy

print("=" * 55)
print("  NIDS SETUP — Generating data and training models")
print("=" * 55)

# ─────────────────────────────────────────────────────────────
# STEP 1: Generate synthetic CAN/network traffic
# ─────────────────────────────────────────────────────────────
print("\n[1/4] Generating synthetic network traffic...")

random.seed(42)
np.random.seed(42)

ATTACK_TYPES = ['attack-free', 'DoS-attack', 'fuzzy-attack', 'spoofing-attack', 'replay-attack']
CAN_IDS = list(range(0x100, 0x800, 8))

def make_payload(attack_type):
    if attack_type == 'DoS-attack':
        return bytes([0xFF] * 8)
    elif attack_type == 'fuzzy-attack':
        return bytes([random.randint(0, 255) for _ in range(8)])
    elif attack_type == 'spoofing-attack':
        return bytes([random.randint(100, 200) for _ in range(8)])
    elif attack_type == 'replay-attack':
        return bytes([0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x18])
    else:
        return bytes([random.randint(20, 80) for _ in range(8)])

rows = []
t = 0.0
for window_idx in range(600):
    weights     = [0.60, 0.10, 0.10, 0.10, 0.10]
    attack_type = random.choices(ATTACK_TYPES, weights=weights)[0]

    if attack_type == 'DoS-attack':
        n_msgs = random.randint(200, 400)
    elif attack_type == 'fuzzy-attack':
        n_msgs = random.randint(60, 120)
    else:
        n_msgs = random.randint(30, 80)

    for i in range(n_msgs):
        if attack_type == 'DoS-attack':
            can_id = 0x000
        elif attack_type == 'fuzzy-attack':
            can_id = random.randint(0x000, 0x7FF)
        else:
            can_id = random.choice(CAN_IDS)

        payload = make_payload(attack_type)
        rows.append({
            'Timestamp':   round(t, 5),
            'CAN_ID':      can_id,
            'Payload':     payload.hex().upper(),
            'attack_type': attack_type,
            'car_model':   'synthetic',
        })
        t += 1.0 / n_msgs

    t = float(window_idx + 1)

raw_df = pd.DataFrame(rows)
print(f"   Generated {len(raw_df):,} raw messages across {window_idx+1} time windows")
print(f"   Attack distribution:\n{raw_df['attack_type'].value_counts().to_string()}")


# ─────────────────────────────────────────────────────────────
# STEP 2: Feature engineering
# FIX: renamed to _setup_entropy to avoid naming clash with
#      feature_engineering.py's calculate_entropy
# ─────────────────────────────────────────────────────────────
print("\n[2/4] Engineering features...")

def hex_to_bytes(hex_str):
    try:
        clean = str(hex_str).replace(' ', '')
        return [int(clean[i:i+2], 16) for i in range(0, min(len(clean), 16), 2)]
    except:
        return []


def _setup_entropy(series):
    """
    FIX: renamed from calc_entropy.
    Also removed dead code that appeared after the return statement in original.
    """
    if series is None or (hasattr(series, 'empty') and series.empty):
        return 0
    values = series.dropna() if hasattr(series, 'dropna') else series
    if len(values) == 0:
        return 0
    probs = pd.Series(values).value_counts(normalize=True)
    return float(-(probs * np.log2(probs)).sum())


df = raw_df.copy()
df['time_window'] = df['Timestamp'].astype(int)

# Frequency features per CAN ID per window
freq_df = df.pivot_table(
    index='time_window', columns='CAN_ID',
    values='Timestamp', aggfunc='count'
).fillna(0)
freq_df.columns = [f'{int(c)}_freq' for c in freq_df.columns]

# Payload byte features
df['bytes'] = df['Payload'].apply(hex_to_bytes)
df_exploded = df[['time_window', 'CAN_ID', 'bytes']].explode('bytes')
df_exploded = df_exploded.dropna(subset=['bytes'])
df_exploded['bytes'] = pd.to_numeric(df_exploded['bytes'], errors='coerce').dropna()
df_exploded = df_exploded.dropna(subset=['bytes'])

grouped = df_exploded.groupby(['time_window', 'CAN_ID'])['bytes']
mean_df = grouped.mean().unstack(fill_value=0)
entr_df = grouped.apply(_setup_entropy).unstack(fill_value=0)   # FIX: use renamed function

mean_df.columns = [f'{int(c)}_mean' for c in mean_df.columns]
entr_df.columns = [f'{int(c)}_entropy' for c in entr_df.columns]

labels_df   = df.groupby('time_window')[['car_model', 'attack_type']].first()
features_df = freq_df.join(mean_df).join(entr_df).join(labels_df).fillna(0)
print(f"   Feature matrix: {features_df.shape[0]} windows × {features_df.shape[1]} columns")


# ─────────────────────────────────────────────────────────────
# STEP 3: Save datasets
# FIX: index=False prevents 'Unnamed: 0' column in CSV
# ─────────────────────────────────────────────────────────────
print("\n[3/4] Saving datasets...")

features_df_with_tw = features_df.reset_index()   # makes time_window a column
features_df_with_tw.to_csv('features.csv', index=False)   # FIX: was index=True
print("   Saved: features.csv (index=False, no Unnamed column)")

shuffled = features_df_with_tw.sample(frac=1, random_state=42).reset_index(drop=True)
shuffled.to_csv('shuffled_features.csv', index=False)
print("   Saved: shuffled_features.csv")


# ─────────────────────────────────────────────────────────────
# STEP 4: Train models
# ─────────────────────────────────────────────────────────────
print("\n[4/4] Training ML models...")

cols_to_drop = [c for c in ['time_window', 'car_model', 'attack_type'] if c in features_df_with_tw.columns]
X      = features_df_with_tw.drop(columns=cols_to_drop)
y_text = features_df_with_tw['attack_type']

encoder = LabelEncoder()
y       = encoder.fit_transform(y_text)
joblib.dump(encoder, 'label_encoder.joblib')
print(f"   Classes: {list(encoder.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
score = rf.score(X_test, y_test)
joblib.dump(rf, 'random_forest_model.joblib')
print(f"   Random Forest accuracy (held-out 25%): {score:.1%}")
print(f"   Saved: random_forest_model.joblib")

# Isolation Forest — trained on NORMAL traffic only
normal_X = X[y_text == 'attack-free']
iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
iso.fit(normal_X)
joblib.dump(iso, 'isolation_forest_model.joblib')
print(f"   Saved: isolation_forest_model.joblib")

# Move index.html to templates/ if not already there
os.makedirs("templates", exist_ok=True)
if os.path.exists("index.html") and not os.path.exists("templates/index.html"):
    import shutil
    shutil.copy("index.html", "templates/index.html")
    print("   Copied index.html → templates/index.html")


# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SETUP COMPLETE")
print("=" * 55)
files = ['features.csv', 'shuffled_features.csv',
         'random_forest_model.joblib', 'label_encoder.joblib',
         'isolation_forest_model.joblib']
for f in files:
    size   = os.path.getsize(f) // 1024 if os.path.exists(f) else 0
    status = "OK" if os.path.exists(f) else "MISSING"
    print(f"    {status}  {f}  ({size} KB)")

print(f"\n  Next steps:")
print(f"    1. python app.py")
print(f"    2. Open http://localhost:8081")
print(f"    3. Press Start in the dashboard")