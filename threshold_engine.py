"""
threshold_engine.py
-------------------
Dynamic threshold engine for CAN/network traffic.
Thresholds automatically scale with time-of-day traffic patterns.
Works for any network: automotive, enterprise, home IoT.

Usage:
    from threshold_engine import ThresholdEngine
    engine = ThresholdEngine()
    is_anomaly = engine.check(msg_rate, entropy_score, prediction)
"""

import time
from datetime import datetime


# ─────────────────────────────────────────────
# TIME-OF-DAY TRAFFIC PROFILES
# Each entry: (start_hour, end_hour, label, traffic_multiplier)
# Multiplier scales the base threshold up or down.
# 1.0 = normal, 1.5 = busy (threshold higher = less sensitive),
# 0.6 = quiet (threshold lower = more sensitive)
# ─────────────────────────────────────────────
TRAFFIC_PROFILES = [
    (0,  6,  "Night / Low Traffic",       0.5),   # 12am–6am  → very quiet → tight threshold
    (6,  9,  "Morning Rush",              1.3),   # 6am–9am   → busy
    (9,  12, "Mid Morning",               1.0),   # normal
    (12, 14, "Lunch Peak",                1.2),   # slightly elevated
    (14, 18, "Afternoon / Work Hours",    1.0),   # normal
    (18, 21, "Evening Peak",              1.4),   # commute / home activity
    (21, 24, "Late Evening Wind-Down",    0.7),   # quieting down
]

# ─────────────────────────────────────────────
# BASE THRESHOLDS (tune these to your network)
# ─────────────────────────────────────────────
BASE_THRESHOLDS = {
    "msg_rate":      50.0,   # messages per second
    "entropy":        6.5,   # max Shannon entropy (bits) before suspicious
    "repeat_ratio":   0.85,  # max ratio of repeated CAN IDs (DoS indicator)
}

# Manual override — set via dashboard slider (None = use auto)
_manual_multiplier = None


class ThresholdEngine:
    def __init__(self):
        self._history = []          # stores last N decisions for drift detection
        self._window_count = 0

    # ── Public API ─────────────────────────────

    def set_manual_multiplier(self, value: float):
        """
        Override automatic time-based multiplier with a manual value.
        Call with None to re-enable automatic mode.
        value: float between 0.3 (very sensitive) and 2.0 (very loose)
        """
        global _manual_multiplier
        _manual_multiplier = float(value) if value is not None else None

    def get_current_profile(self) -> dict:
        """Returns current traffic profile info for the dashboard."""
        hour = datetime.now().hour
        label, multiplier = self._get_profile(hour)
        if _manual_multiplier is not None:
            multiplier = _manual_multiplier
            label = "Manual Override"
        return {
            "hour": hour,
            "label": label,
            "multiplier": round(multiplier, 2),
            "thresholds": {
                k: round(v * multiplier, 2)
                for k, v in BASE_THRESHOLDS.items()
            }
        }

    def check(self, msg_rate: float, entropy_score: float,
              repeat_ratio: float = 0.0, ml_prediction: str = "") -> dict:
        """
        Main check — returns a verdict dict.

        Parameters:
            msg_rate      : messages per second in this window
            entropy_score : Shannon entropy of payloads
            repeat_ratio  : fraction of duplicate CAN IDs (0–1)
            ml_prediction : label from your Random Forest

        Returns dict:
            {
              "alert": bool,
              "severity": "none" | "low" | "medium" | "high",
              "reasons": [list of triggered rules],
              "thresholds_used": {...},
              "profile": "Morning Rush" etc.
            }
        """
        profile = self.get_current_profile()
        t = profile["thresholds"]
        reasons = []

        # Rule 1: message rate
        if msg_rate > t["msg_rate"] * 1.5:
            reasons.append(f"CRITICAL msg_rate {msg_rate:.1f} >> threshold {t['msg_rate']:.1f}")
        elif msg_rate > t["msg_rate"]:
            reasons.append(f"High msg_rate {msg_rate:.1f} > threshold {t['msg_rate']:.1f}")

        # Rule 2: entropy (high entropy = randomised/spoofed payload)
        if entropy_score > t["entropy"]:
            reasons.append(f"High payload entropy {entropy_score:.2f} > {t['entropy']:.2f}")

        # Rule 3: repeat ratio (very high = DoS flood of same ID)
        if repeat_ratio > t["repeat_ratio"]:
            reasons.append(f"Repeat CAN ID ratio {repeat_ratio:.2f} > {t['repeat_ratio']:.2f}")

        # Rule 4: ML model flagged it
        ml_attack = ml_prediction and "attack" in ml_prediction and "free" not in ml_prediction
        if ml_attack:
            reasons.append(f"ML model flagged: {ml_prediction}")

        # Severity
        alert = len(reasons) > 0
        if len(reasons) >= 3 or (ml_attack and len(reasons) >= 2):
            severity = "high"
        elif len(reasons) == 2:
            severity = "medium"
        elif len(reasons) == 1:
            severity = "low"
        else:
            severity = "none"

        self._window_count += 1
        self._history.append(alert)
        if len(self._history) > 100:
            self._history.pop(0)

        return {
            "alert": alert,
            "severity": severity,
            "reasons": reasons,
            "thresholds_used": t,
            "profile": profile["label"],
            "multiplier": profile["multiplier"],
            "window_count": self._window_count,
        }

    def get_alert_rate(self) -> float:
        """Returns fraction of recent windows that triggered an alert."""
        if not self._history:
            return 0.0
        return round(sum(self._history) / len(self._history), 3)

    # ── Internal ───────────────────────────────

    def _get_profile(self, hour: int):
        for start, end, label, mult in TRAFFIC_PROFILES:
            if start <= hour < end:
                return label, mult
        return "Default", 1.0


# ─────────────────────────────────────────────
# Singleton for import by app.py
# ─────────────────────────────────────────────
threshold_engine = ThresholdEngine()


if __name__ == "__main__":
    # Quick self-test
    engine = ThresholdEngine()
    print("Current profile:", engine.get_current_profile())

    result = engine.check(
        msg_rate=120.0,
        entropy_score=7.1,
        repeat_ratio=0.9,
        ml_prediction="DoS-attack"
    )
    print("\nCheck result:", result)

    result2 = engine.check(
        msg_rate=30.0,
        entropy_score=4.2,
        repeat_ratio=0.3,
        ml_prediction="attack-free"
    )
    print("\nNormal check:", result2)