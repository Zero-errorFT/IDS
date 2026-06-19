"""
llm_inspector.py
----------------
Sends suspicious CAN/network payloads to a LOCAL Llama 3 model
running via Ollama for DETAILED forensic analysis.

Changes from v1:
  - System prompt asks for deep forensic reasoning, not just a verdict
  - JSON schema expanded: summary, byte_analysis, indicators, context_alignment,
    recommended_action, ioc_signatures, mitre_attack_ref, risk_score
  - num_predict raised to 900 tokens (allows proper paragraph-level reasoning)
  - Payload bytes decoded to decimal + pattern description before sending
  - Cache key includes msg_rate bucket so different traffic levels get fresh analysis
  - inspect() now accepts repeat_ratio, rf_conf, if_anomaly, fused_sev for richer context

Fix: self.available is no longer permanently latched to False after a connection error.
     is_available() re-probes Ollama if it has been down for >30s, so a brief blip
     doesn't disable the inspector for the rest of the session.
"""

import requests
import json
import time

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "llama3"

# ── System prompt — detailed forensic analyst persona ─────────────────────────
SYSTEM_PROMPT = """You are a senior automotive and network cybersecurity analyst specialising in CAN bus intrusion forensics.

Your task is to perform a DETAILED forensic analysis of a suspicious network/CAN frame and produce a structured report.

You MUST respond with ONLY a valid JSON object — no preamble, no markdown fences, no extra text.you need to wor like a seasoned analyst who has seen it all.

The JSON must exactly follow this schema:
{
  "verdict": "ATTACK" | "NORMAL" | "SUSPICIOUS",
  "attack_type": "DoS" | "Spoofing" | "Replay" | "Fuzzy Injection" | "Payload Injection" | "None" | "Unknown",
  "confidence": <integer 0-100>,
  "risk_score": <integer 0-10>,
  "summary": "<2-3 sentences: what this traffic looks like and why it is or is not suspicious>",
  "byte_analysis": "<Analyse the specific hex bytes: patterns, repeated values, known attack signatures, entropy observations, what the values could mean in a CAN context>",
  "indicators": ["<specific observable indicator 1>", "<indicator 2>", "<indicator 3>"],
  "context_alignment": "<Does the ML model verdict match your independent byte-level analysis? Explain agreement or disagreement in 1-2 sentences>",
  "recommended_action": "<Concrete action for the security team: block / monitor / escalate / ignore — and why>",
  "ioc_signatures": ["<Indicator of Compromise pattern 1>", "<IOC 2 if applicable>"],
  "mitre_attack_ref": "<Closest MITRE ATT&CK for ICS technique ID and name, e.g. T0814 Denial of Service, or N/A>"
}"""

# ── User prompt — rich context passed to the model ────────────────────────────
USER_TEMPLATE = """Perform a full forensic analysis of this CAN bus / network frame:

═══ FRAME METADATA ═══
  CAN ID / Node:        {can_id}
  Raw Payload (hex):    {payload_hex}
  Payload length:       {payload_len} bytes
  Decoded bytes (dec):  {payload_dec}
  Byte pattern notes:   {byte_pattern}

═══ TRAFFIC STATISTICS (this time window) ═══
  Message rate:         {msg_rate:.1f} msg/s  {rate_flag}
  Shannon entropy:      {entropy:.3f} bits    {entropy_flag}
  Repeat ratio:         {repeat_ratio:.2f}    {repeat_flag}

═══ ML MODEL CONTEXT ═══
  Random Forest verdict:     {rf_label}
  RF confidence:             {rf_conf:.1f}%
  Isolation Forest anomaly:  {if_anomaly}
  Hybrid fused severity:     {fused_sev}

═══ ADDITIONAL CONTEXT ═══
  {extra_context}

Produce your detailed forensic JSON report now."""


def _decode_payload(hex_str: str) -> tuple:
    """Turn hex payload into decimal bytes + pattern description."""
    try:
        clean = hex_str.replace(" ", "")[:32]
        raw   = bytes.fromhex(clean)
        dec_str = " ".join(str(b) for b in raw)

        patterns = []
        if len(set(raw)) == 1:
            patterns.append(f"all bytes identical (0x{raw[0]:02X} repeated x{len(raw)})")
        if all(b == 0xFF for b in raw):
            patterns.append("MAX VALUE FLOOD — classic DoS signature")
        elif all(b == 0x00 for b in raw):
            patterns.append("all-zero payload — possible null injection")
        elif raw == bytes(sorted(raw)):
            patterns.append("monotonically increasing byte sequence")
        elif raw == bytes(sorted(raw, reverse=True)):
            patterns.append("monotonically decreasing byte sequence")
        else:
            unique_ratio = len(set(raw)) / max(len(raw), 1)
            if unique_ratio > 0.875:
                patterns.append(f"high byte diversity ({unique_ratio:.0%} unique) — possible fuzzy/random injection")
            elif unique_ratio < 0.25:
                patterns.append(f"low byte diversity ({unique_ratio:.0%} unique) — possible replay or templated payload")
        if raw[:2] == bytes([0xA1, 0xB2]):
            patterns.append("header matches known replay attack template")

        return dec_str, ("; ".join(patterns) if patterns else "no obvious byte-level pattern")
    except Exception:
        return "parse error", "unknown"


def _rate_flag(r: float) -> str:
    if r > 300: return "⚠ CRITICAL — far exceeds normal CAN bus rate (~100 msg/s)"
    if r > 150: return "⚠ HIGH — elevated, possible DoS flood"
    if r > 80:  return "△ ELEVATED"
    return "✓ within normal range"

def _entropy_flag(e: float) -> str:
    if e < 0.5:  return "⚠ VERY LOW — near-constant payload, DoS/Replay signature"
    if e > 6.5:  return "⚠ HIGH — high randomness, possible fuzzy injection"
    if e > 5.5:  return "△ MODERATELY HIGH"
    return "✓ normal"

def _repeat_flag(r: float) -> str:
    if r > 0.9: return "⚠ CRITICAL — single CAN ID flooding"
    if r > 0.7: return "⚠ HIGH — dominant CAN ID, suspicious"
    return "✓ normal distribution"


class LLMInspector:
    # How long to wait before re-probing Ollama after a connection failure
    _RETRY_INTERVAL = 30  # seconds

    def __init__(self, model: str = MODEL_NAME, timeout: int = 45):
        self.model     = model
        self.timeout   = timeout
        self._cache: dict = {}
        self.available = self._check_ollama()
        self._unavailable_since: float = 0.0  # timestamp when Ollama last went down

    # ── Public API ──────────────────────────────────────────────────────────────

    def inspect(self,
                payload_hex:  str,
                can_id:       str   = "unknown",
                context:      str   = "",
                msg_rate:     float = 0.0,
                entropy:      float = 0.0,
                repeat_ratio: float = 0.0,
                rf_conf:      float = 0.0,
                if_anomaly:   bool  = False,
                fused_sev:    str   = "none") -> dict:
        """
        Full forensic inspection.
        Returns a rich dict with verdict, byte_analysis, indicators, MITRE ref, etc.
        Falls back gracefully if Ollama is not running, and auto-recovers if it
        comes back up after being down (re-probes every _RETRY_INTERVAL seconds).
        """
        # FIX: re-probe Ollama if it was previously down but enough time has passed
        if not self.available:
            if time.time() - self._unavailable_since >= self._RETRY_INTERVAL:
                self.available = self._check_ollama()
            if not self.available:
                return self._fallback("Ollama not running — start with: ollama serve")

        # Cache key includes rate bucket so same payload at different traffic levels
        # gets a fresh analysis
        rate_bucket = int(msg_rate // 50) * 50
        cache_key   = f"{can_id}:{payload_hex[:16]}:r{rate_bucket}:if{if_anomaly}"
        if cache_key in self._cache:
            cached = self._cache[cache_key].copy()
            cached["from_cache"] = True
            return cached

        payload_dec, byte_pattern = _decode_payload(payload_hex)

        prompt = USER_TEMPLATE.format(
            can_id        = can_id,
            payload_hex   = payload_hex[:32].upper(),
            payload_len   = len(payload_hex) // 2,
            payload_dec   = payload_dec,
            byte_pattern  = byte_pattern,
            msg_rate      = msg_rate,
            rate_flag     = _rate_flag(msg_rate),
            entropy       = entropy,
            entropy_flag  = _entropy_flag(entropy),
            repeat_ratio  = repeat_ratio,
            repeat_flag   = _repeat_flag(repeat_ratio),
            rf_label      = context or "Not provided",
            rf_conf       = (rf_conf * 100) if rf_conf <= 1.0 else rf_conf,
            if_anomaly    = "YES — anomaly detected" if if_anomaly else "No",
            fused_sev     = fused_sev.upper(),
            extra_context = (
                "Both Random Forest and Isolation Forest agree — treat as high-confidence incident."
                if if_anomaly and "attack" in (context or "").lower()
                else "Single-model flag — medium confidence, investigate further."
            ),
        )

        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model":  self.model,
                    "system": SYSTEM_PROMPT,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature":    0.15,
                        "num_predict":    900,
                        "top_p":          0.9,
                        "repeat_penalty": 1.1,
                    }
                },
                timeout=self.timeout
            )
            resp.raise_for_status()
            raw_text = resp.json().get("response", "")
            result   = self._parse_response(raw_text)
            result.update({
                "from_cache":   False,
                "payload_hex":  payload_hex[:32].upper(),
                "payload_dec":  payload_dec,
                "byte_pattern": byte_pattern,
                "msg_rate":     msg_rate,
                "entropy":      round(entropy, 3),
                "timestamp":    time.strftime("%H:%M:%S"),
            })
            self._cache[cache_key] = result
            return result

        except requests.exceptions.ConnectionError:
            # FIX: record when it went down so is_available() can retry later,
            # rather than permanently latching available=False for the session
            self.available = False
            self._unavailable_since = time.time()
            return self._fallback("Cannot connect to Ollama. Run: ollama serve")
        except requests.exceptions.Timeout:
            return self._fallback(
                f"Ollama timed out after {self.timeout}s — model may be loading, retry shortly"
            )
        except Exception as e:
            return self._fallback(f"LLM error: {str(e)[:120]}")

    def is_available(self) -> bool:
        if not self.available and time.time() - self._unavailable_since >= self._RETRY_INTERVAL:
            self.available = self._check_ollama()
            if self.available:
                self._unavailable_since = 0.0
        return self.available

    def clear_cache(self):
        self._cache.clear()

    # ── Internal ────────────────────────────────────────────────────────────────

    def _check_ollama(self) -> bool:
        try:
            requests.get("http://localhost:11434/", timeout=3)
            print("✅ Ollama is running — LLM inspector active.")
            return True
        except Exception:
            self._unavailable_since = time.time()
            print("⚠️  Ollama not detected. LLM inspector disabled.")
            print("    To enable: run  ollama serve  then restart app.py")
            return False

    def _parse_response(self, raw: str) -> dict:
        """Robustly extract JSON — handles markdown fences and truncated output."""
        raw = raw.strip()
        for fence in ["```json", "```JSON", "```"]:
            raw = raw.replace(fence, "")

        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(raw[start:end])
                return {
                    "verdict":            str(data.get("verdict",      "UNKNOWN")).upper(),
                    "attack_type":        str(data.get("attack_type",  "Unknown")),
                    "confidence":         int(data.get("confidence",   0)),
                    "risk_score":         int(data.get("risk_score",   0)),
                    "summary":            str(data.get("summary",      "No summary provided.")),
                    "byte_analysis":      str(data.get("byte_analysis","No byte analysis provided.")),
                    "indicators":         list(data.get("indicators",  [])),
                    "context_alignment":  str(data.get("context_alignment", "")),
                    "recommended_action": str(data.get("recommended_action","No recommendation.")),
                    "ioc_signatures":     list(data.get("ioc_signatures",   [])),
                    "mitre_attack_ref":   str(data.get("mitre_attack_ref",  "N/A")),
                    "raw":                raw[start:end],
                }
            except json.JSONDecodeError as e:
                return self._partial_parse(raw, str(e))

        return self._fallback(f"No JSON found in LLM output: {raw[:150]}")

    def _partial_parse(self, raw: str, err: str) -> dict:
        """Best-effort extraction when JSON is malformed (e.g. truncated)."""
        result = self._fallback(f"JSON parse error ({err}) — partial extraction")
        for field in ["verdict", "attack_type", "summary", "recommended_action"]:
            marker = f'"{field}"'
            idx = raw.find(marker)
            if idx != -1:
                val_start = raw.find('"', idx + len(marker) + 2)
                val_end   = raw.find('"', val_start + 1)
                if val_start != -1 and val_end > val_start:
                    result[field] = raw[val_start+1:val_end]
        result["raw"] = raw[:600]
        return result

    def _fallback(self, message: str) -> dict:
        return {
            "verdict":            "UNKNOWN",
            "attack_type":        "Unknown",
            "confidence":         0,
            "risk_score":         0,
            "summary":            message,
            "byte_analysis":      "",
            "indicators":         [],
            "context_alignment":  "",
            "recommended_action": "Check Ollama status.",
            "ioc_signatures":     [],
            "mitre_attack_ref":   "N/A",
            "from_cache":         False,
            "raw":                "",
        }


# ── Singleton for import by app.py ────────────────────────────────────────────
llm_inspector = LLMInspector()


if __name__ == "__main__":
    inspector = LLMInspector()
    if inspector.is_available():
        print("\n🔍 Test 1: DoS flood")
        r1 = inspector.inspect(
            payload_hex   = "FFFFFFFFFFFFFFFF",
            can_id        = "0x000",
            context       = "DoS-attack",
            msg_rate      = 310.0,
            entropy       = 0.05,
            repeat_ratio  = 0.97,
            rf_conf       = 0.98,
            if_anomaly    = True,
            fused_sev     = "high",
        )
        print(json.dumps(r1, indent=2))

        print("\n✅ Test 2: Normal traffic")
        r2 = inspector.inspect(
            payload_hex   = "2A4F1C3E7B9D0521",
            can_id        = "0x3A0",
            context       = "attack-free",
            msg_rate      = 42.0,
            entropy       = 5.8,
            repeat_ratio  = 0.2,
            rf_conf       = 0.91,
            if_anomaly    = False,
            fused_sev     = "none",
        )
        print(json.dumps(r2, indent=2))
    else:
        print("Start Ollama first: ollama serve")
