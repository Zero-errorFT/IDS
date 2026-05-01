"""
evaluate_model.py
-----------------
Evaluates trained models on a HELD-OUT test set to prevent data leakage.

Fix applied:
  - Previous version evaluated on the full features.csv (same data used for training)
    → inflated accuracy due to data leakage.
  - This version re-splits features.csv with the same seed as train_model.py
    and evaluates ONLY on the held-out 30% test portion.
  - A separate shuffled_features.csv (if present) can also be used as
    an independent out-of-distribution evaluation set.
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')   # headless — no display needed
import matplotlib.pyplot as plt
import os


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop unnamed index column that setup.py may have added
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    return df


def prepare_X_y(df: pd.DataFrame, encoder=None):
    """Drop meta-columns; encode attack_type labels."""
    drop_cols = [c for c in ['time_window', 'car_model', 'attack_type'] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y_text = df.get('attack_type', pd.Series(['unknown'] * len(df)))
    if encoder:
        y = encoder.transform(y_text)
    else:
        y = y_text
    return X, y, y_text


def save_confusion_matrix(cm, labels, title, path):
    """Save a confusion matrix heatmap to disk."""
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(title, fontsize=12, pad=14)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Confusion matrix saved → {path}")


# ─────────────────────────────────────────────────────────────
# Random Forest evaluation
# ─────────────────────────────────────────────────────────────
def evaluate_random_forest(model_path, feature_path, encoder_path,
                            test_size=0.30, random_state=42):
    print("\n" + "="*60)
    print("  Random Forest — Held-Out Test Evaluation")
    print("="*60)

    model   = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    df      = load_features(feature_path)

    X, y, y_text = prepare_X_y(df, encoder)

    # ── KEY FIX: same split as train_model.py (same seed + stratify) ──
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y
    )

    print(f"  Evaluating on {len(X_test)} held-out samples "
          f"({test_size*100:.0f}% of {len(df)} total)\n")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy (held-out): {acc:.4f}  ({acc*100:.2f}%)\n")
    print(classification_report(
        y_test, y_pred,
        target_names=encoder.classes_,
        zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, encoder.classes_,
                          f'RF Confusion Matrix (held-out {test_size*100:.0f}%)',
                          'eval_rf_confusion.png')

    # Feature importance (top 15)
    importances = model.feature_importances_
    feat_names  = model.feature_names_in_
    top_idx     = np.argsort(importances)[::-1][:15]
    print("\n  Top 15 Feature Importances:")
    for i, idx in enumerate(top_idx, 1):
        print(f"    {i:2d}. {feat_names[idx]:<30s}  {importances[idx]:.4f}")

    return acc


# ─────────────────────────────────────────────────────────────
# Isolation Forest evaluation
# ─────────────────────────────────────────────────────────────
def evaluate_isolation_forest(model_path, feature_path,
                               test_size=0.30, random_state=42):
    print("\n" + "="*60)
    print("  Isolation Forest — Held-Out Anomaly Detection")
    print("="*60)

    model = joblib.load(model_path)
    df    = load_features(feature_path)

    drop_cols = [c for c in ['time_window', 'car_model', 'attack_type'] if c in df.columns]
    X         = df.drop(columns=drop_cols)

    # Binary ground-truth: 1=normal, -1=anomaly
    y_true = df['attack_type'].apply(
        lambda x: 1 if x == 'attack-free' else -1
    )

    # Same split for comparability
    _, X_test, _, y_true_test = train_test_split(
        X, y_true, test_size=test_size, random_state=random_state,
        stratify=y_true
    )

    print(f"  Evaluating on {len(X_test)} held-out samples\n")

    y_pred = model.predict(X_test)   # 1 or -1

    acc = accuracy_score(y_true_test, y_pred)
    print(f"  Accuracy (held-out): {acc:.4f}  ({acc*100:.2f}%)\n")
    print(classification_report(
        y_true_test, y_pred,
        target_names=['Anomaly (-1)', 'Normal (+1)'],
        zero_division=0
    ))

    cm = confusion_matrix(y_true_test, y_pred, labels=[-1, 1])
    save_confusion_matrix(cm, ['Anomaly', 'Normal'],
                          f'IF Confusion Matrix (held-out {test_size*100:.0f}%)',
                          'eval_if_confusion.png')
    return acc


# ─────────────────────────────────────────────────────────────
# Optional: out-of-distribution test on shuffled_features.csv
# ─────────────────────────────────────────────────────────────
def evaluate_ood(model_path, ood_path, encoder_path):
    """
    If shuffled_features.csv was generated from a DIFFERENT data split
    than features.csv, this gives a genuine out-of-distribution score.
    (Requires shuffled_features to not overlap with training data.)
    """
    if not os.path.exists(ood_path):
        print(f"\n  OOD eval skipped — '{ood_path}' not found.")
        return

    print("\n" + "="*60)
    print("  RF — Out-of-Distribution Evaluation (shuffled_features.csv)")
    print("="*60)

    model   = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    df      = load_features(ood_path)

    if 'attack_type' not in df.columns:
        print("  No attack_type column — skipping OOD eval.")
        return

    X, y, _ = prepare_X_y(df, encoder)
    # Align columns
    X = X.reindex(columns=model.feature_names_in_, fill_value=0.0)

    y_pred = model.predict(X)
    acc    = accuracy_score(y, y_pred)
    print(f"  OOD Accuracy: {acc:.4f}  ({acc*100:.2f}%)\n")
    print(classification_report(y, y_pred,
                                target_names=encoder.classes_,
                                zero_division=0))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    feature_file  = 'features.csv'
    shuffled_file = 'shuffled_features.csv'

    required = {
        'features.csv':                 feature_file,
        'random_forest_model.joblib':   'random_forest_model.joblib',
        'isolation_forest_model.joblib':'isolation_forest_model.joblib',
        'label_encoder.joblib':         'label_encoder.joblib',
    }

    missing = [k for k, v in required.items() if not os.path.exists(v)]
    if missing:
        print(f"ERROR: Missing files: {missing}")
        print("Run: python setup.py")
        raise SystemExit(1)

    rf_acc = evaluate_random_forest(
        'random_forest_model.joblib', feature_file, 'label_encoder.joblib'
    )
    if_acc = evaluate_isolation_forest(
        'isolation_forest_model.joblib', feature_file
    )
    evaluate_ood(
        'random_forest_model.joblib', shuffled_file, 'label_encoder.joblib'
    )

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  RF  accuracy (held-out 30%): {rf_acc:.4f}")
    print(f"  IF  accuracy (held-out 30%): {if_acc:.4f}")
    print(f"  Confusion matrices saved: eval_rf_confusion.png, eval_if_confusion.png")
    print()