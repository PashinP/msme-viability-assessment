"""
Gemini Chat Agent — Conversational feature extraction
=======================================================
Multi-turn conversation with a business owner to naturally
extract the 11 features needed for loan viability prediction.

Uses Google Gemini API for the conversational layer.
Our validation logic ensures extracted features are within valid ranges.
"""
import os
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (works regardless of cwd)
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Try to import google.generativeai
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

from api.prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

# Valid ranges for each feature (for validation)
FEATURE_RANGES = {
    "Term":               (1, 480),
    "NoEmp":              (0, 500),
    "NewExist":           (1, 2),
    "CreateJob":          (0, 200),
    "RetainedJob":        (0, 200),
    "DisbursementGross":  (100, 10_000_000),
    "UrbanRural":         (0, 2),
    "RevLineCr":          (0, 1),
    "LowDoc":             (0, 1),
    "SBA_Appv":           (50, 10_000_000),
    "GrAppv":             (100, 10_000_000),
}

REQUIRED_FEATURES = list(FEATURE_RANGES.keys())


def validate_features(features: dict) -> tuple[dict, list[str]]:
    """
    Validate and clamp extracted features to valid ranges.
    Returns (cleaned_features, warnings).
    """
    cleaned = {}
    warnings = []

    for feat, (lo, hi) in FEATURE_RANGES.items():
        val = features.get(feat)
        if val is None:
            warnings.append(f"Missing feature: {feat}")
            continue

        try:
            val = float(val)
        except (ValueError, TypeError):
            warnings.append(f"Invalid value for {feat}: {features[feat]}")
            continue

        # Integer features
        if feat in ("NoEmp", "NewExist", "CreateJob", "RetainedJob",
                     "UrbanRural", "RevLineCr", "LowDoc"):
            val = int(round(val))

        if val < lo:
            warnings.append(f"{feat} ({val}) below minimum ({lo}), clamped")
            val = lo
        if val > hi:
            warnings.append(f"{feat} ({val}) above maximum ({hi}), clamped")
            val = hi

        cleaned[feat] = val

    return cleaned, warnings


def extract_json_from_text(text: str) -> dict | None:
    """Extract JSON block from Gemini's response."""
    # Look for ```json ... ```
    pattern = r"```json\s*(\{[\s\S]*?\})\s*```"
    match = re.search(pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: look for any JSON object
    pattern2 = r"\{[^{}]*\"Term\"[^{}]*\}"
    match2 = re.search(pattern2, text)
    if match2:
        try:
            return json.loads(match2.group())
        except json.JSONDecodeError:
            pass

    return None


class ChatAgent:
    """
    Manages multi-turn conversation with Gemini for feature extraction.
    """

    def __init__(self):
        if not HAS_GEMINI:
            raise RuntimeError("google-generativeai package not installed. "
                               "Run: pip install google-generativeai")
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set in .env file")

        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT,
        )

    def chat(self, messages: list[dict]) -> dict:
        """
        Process a conversation and return the agent's response.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}

        Returns:
            {
                "response": "agent's message",
                "features_extracted": {...} or None,
                "validation_warnings": [...],
                "extraction_complete": bool
            }
        """
        # Build Gemini conversation history
        gemini_history = []

        # Add few-shot examples
        for ex in FEW_SHOT_EXAMPLES:
            gemini_history.append({"role": "user", "parts": [ex["user"]]})
            gemini_history.append({"role": "model", "parts": [ex["assistant"]]})

        # Add actual conversation history (everything except the last message)
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        # Start chat with history and send the last message
        chat = self.model.start_chat(history=gemini_history)
        last_msg = messages[-1]["content"]

        # Retry with exponential backoff for rate limits (Gemini free tier: 5 RPM)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = chat.send_message(last_msg)
                response_text = response.text
                break
            except Exception as e:
                error_str = str(e).lower()
                if ("429" in error_str or "resource" in error_str or
                     "rate" in error_str or "quota" in error_str):
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                        print(f"[ChatAgent] Rate limited, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(wait)
                        continue
                    else:
                        return {
                            "response": "⏳ The AI service is temporarily busy. Please wait a moment and try again.",
                            "features_extracted": None,
                            "validation_warnings": [],
                            "extraction_complete": False,
                        }
                else:
                    return {
                        "response": f"⚠️ Something went wrong: {str(e)[:200]}. Please try again.",
                        "features_extracted": None,
                        "validation_warnings": [],
                        "extraction_complete": False,
                    }

        # Check if features were extracted
        features_raw = extract_json_from_text(response_text)
        features_cleaned = None
        warnings = []
        complete = False

        if features_raw:
            # Remove non-feature keys
            features_only = {k: v for k, v in features_raw.items()
                             if k in REQUIRED_FEATURES}
            features_cleaned, warnings = validate_features(features_only)
            complete = len(features_cleaned) == len(REQUIRED_FEATURES)

        return {
            "response": response_text,
            "features_extracted": features_cleaned if complete else None,
            "validation_warnings": warnings,
            "extraction_complete": complete,
        }
