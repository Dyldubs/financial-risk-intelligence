"""
Unit tests for src/models/explainer.py

Tests the top_drivers_text function which converts SHAP values
into human-readable strings for the LLM prompt.
"""

import numpy as np
import pytest

from src.models.explainer import top_drivers_text


FEATURE_NAMES = [
    "amount", "hour", "merchant_risk_tier",
    "velocity_1h", "velocity_24h", "high_risk_country",
    "amount_vs_avg_ratio", "days_since_account_open", "is_weekend",
]


class TestTopDriversText:
    def test_returns_n_lines_by_default(self):
        shap_vals = np.array([0.3, 0.1, 0.05, 0.2, 0.08, 0.15, 0.07, 0.01, 0.02])
        result = top_drivers_text(shap_vals, FEATURE_NAMES, n=3)
        assert len(result.strip().split("\n")) == 3

    def test_returns_one_line_when_n_equals_1(self):
        shap_vals = np.ones(len(FEATURE_NAMES))
        result = top_drivers_text(shap_vals, FEATURE_NAMES, n=1)
        assert len(result.strip().split("\n")) == 1

    def test_highest_absolute_shap_appears_first(self):
        # velocity_1h (index 3) has the highest value
        shap_vals = np.array([0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01])
        result = top_drivers_text(shap_vals, FEATURE_NAMES, n=3)
        first_line = result.strip().split("\n")[0]
        assert "velocity_1h" in first_line

    def test_negative_absolute_value_also_ranks(self):
        # amount_vs_avg_ratio (index 6) has large negative SHAP
        shap_vals = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, -0.99, 0.01, 0.01])
        result = top_drivers_text(shap_vals, FEATURE_NAMES, n=1)
        assert "amount_vs_avg_ratio" in result

    def test_positive_shap_shows_plus_sign(self):
        shap_vals = np.array([0.5] + [0.0] * 8)
        result = top_drivers_text(shap_vals, FEATURE_NAMES, n=1)
        assert "(+" in result

    def test_negative_shap_shows_minus_sign(self):
        shap_vals = np.array([-0.5] + [0.0] * 8)
        result = top_drivers_text(shap_vals, FEATURE_NAMES, n=1)
        assert "(-" in result

    def test_known_features_get_human_descriptions(self):
        shap_vals = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0])
        result = top_drivers_text(shap_vals, FEATURE_NAMES, n=1)
        # high_risk_country should have a description
        assert "high-risk" in result.lower() or "jurisdiction" in result.lower()

    def test_unknown_feature_gets_fallback_description(self):
        names = ["unknown_xyz_feature"] + FEATURE_NAMES[1:]
        shap_vals = np.array([0.99] + [0.0] * 8)
        result = top_drivers_text(shap_vals, names, n=1)
        assert "unknown_xyz_feature" in result
        assert "contributed to the anomaly score" in result

    def test_line_numbers_start_at_one(self):
        shap_vals = np.array([0.3, 0.2, 0.1] + [0.0] * 6)
        result = top_drivers_text(shap_vals, FEATURE_NAMES, n=3)
        lines = result.strip().split("\n")
        assert lines[0].startswith("1.")
        assert lines[1].startswith("2.")
        assert lines[2].startswith("3.")

    def test_returns_string(self):
        shap_vals = np.zeros(len(FEATURE_NAMES))
        result = top_drivers_text(shap_vals, FEATURE_NAMES)
        assert isinstance(result, str)
