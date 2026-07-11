"""
SHAP-based explainability for the Isolation Forest model.

Generates:
  - Global feature importance (bar chart + beeswarm)
  - Per-prediction waterfall chart
  - Human-readable top-driver text for the LLM prompt
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for server use)
from pathlib import Path
from src.config import ROOT_DIR

PLOTS_DIR = ROOT_DIR / "app" / "static"


def build_explainer(model, X_transformed: np.ndarray):
    """Build a SHAP TreeExplainer for the Isolation Forest."""
    return shap.TreeExplainer(model)


def get_shap_values(explainer, X_transformed: np.ndarray) -> np.ndarray:
    return explainer.shap_values(X_transformed)


def top_drivers_text(shap_vals: np.ndarray, feature_names: list[str], n: int = 3) -> str:
    """
    Return a human-readable string describing the top N SHAP drivers.
    This is injected into the LLM prompt.

    Example output:
        "1. velocity_1h (+0.18): unusually high transaction frequency in the last hour
         2. high_risk_country (+0.14): transaction originated from a high-risk jurisdiction
         3. amount_vs_avg_ratio (+0.12): amount is 8x the account's 30-day average"
    """
    feature_impact = list(zip(feature_names, shap_vals))
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
    top = feature_impact[:n]

    descriptions = {
        "amount": "transaction amount is unusually large",
        "log_amount": "transaction amount is unusually large (log-scaled)",
        "velocity_1h": "unusually high transaction frequency in the last hour",
        "velocity_24h": "unusually high transaction volume in the last 24 hours",
        "high_risk_country": "transaction linked to a high-risk jurisdiction",
        "amount_vs_avg_ratio": "amount significantly exceeds the account's recent average",
        "hour": "transaction occurred at an unusual time of day",
        "is_night": "transaction occurred during the high-risk overnight window (11pm–5am)",
        "merchant_risk_tier": "merchant is in a high-risk category",
        "days_since_account_open": "account is relatively new",
        "is_weekend": "transaction occurred on a weekend",
    }

    lines = []
    for i, (feat, val) in enumerate(top, 1):
        direction = "+" if val > 0 else "-"
        desc = descriptions.get(feat, f"{feat} contributed to the anomaly score")
        lines.append(f"{i}. {feat} ({direction}{abs(val):.3f}): {desc}")

    return "\n".join(lines)


def save_global_importance_plot(shap_values, feature_names: list[str]) -> Path:
    """Save a global SHAP bar chart (average |SHAP| per feature)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / "shap_global_importance.png"
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def save_beeswarm_plot(shap_values, X_df: pd.DataFrame) -> Path:
    """Save a SHAP beeswarm plot showing direction of feature effects."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / "shap_beeswarm.png"
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_df, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def save_waterfall_plot(explainer, shap_values, X_df: pd.DataFrame, idx: int) -> Path:
    """Save a SHAP waterfall plot for a single prediction."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"shap_waterfall_{idx}.png"
    plt.figure(figsize=(8, 5))
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_df.iloc[idx].values,
        feature_names=list(X_df.columns),
    )
    shap.waterfall_plot(explanation, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path
