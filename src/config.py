"""Central configuration loaded from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
POLICIES_DIR = DATA_DIR / "policies"
CHROMA_DIR = ROOT_DIR / os.getenv("CHROMA_PERSIST_DIR", "chroma_db")

# LLM
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = "claude-haiku-4-5-20251001"  # Fast and cheap; swap for sonnet for higher quality

# MLflow — use SQLite backend (MLflow 3.x dropped the file store)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
MLFLOW_EXPERIMENT_NAME = "financial-risk-detection"

# Risk thresholds
LOW_RISK_THRESHOLD = float(os.getenv("LOW_RISK_THRESHOLD", 0.3))
MEDIUM_RISK_THRESHOLD = float(os.getenv("MEDIUM_RISK_THRESHOLD", 0.6))
HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", 0.85))

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))


def risk_tier(score: float) -> str:
    if score >= HIGH_RISK_THRESHOLD:
        return "CRITICAL"
    elif score >= MEDIUM_RISK_THRESHOLD:
        return "HIGH"
    elif score >= LOW_RISK_THRESHOLD:
        return "MEDIUM"
    return "LOW"


TIER_ACTIONS = {
    "CRITICAL": "Immediately escalate to the Financial Crime team. Freeze transaction pending review.",
    "HIGH": "Flag for same-day review by a risk analyst. Do not release funds until cleared.",
    "MEDIUM": "Add to the daily review queue. Monitor account for additional suspicious activity.",
    "LOW": "Log for audit trail. No immediate action required.",
}
