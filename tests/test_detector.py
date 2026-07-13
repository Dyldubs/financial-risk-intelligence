"""
Unit tests for src/models/detector.py

Tests score normalisation, transaction risk scoring, and the
risk_tier / TIER_ACTIONS config values.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.detector import _normalise_scores, score
from src.config import risk_tier, TIER_ACTIONS


# ---------------------------------------------------------------------------
# _normalise_scores
# ---------------------------------------------------------------------------

class TestNormaliseScores:
    def test_output_minimum_is_zero(self):
        raw = np.array([-0.5, -0.2, 0.0, 0.1, 0.3])
        result = _normalise_scores(raw)
        assert result.min() == pytest.approx(0.0)

    def test_output_maximum_is_one(self):
        raw = np.array([-0.5, -0.2, 0.0, 0.1, 0.3])
        result = _normalise_scores(raw)
        assert result.max() == pytest.approx(1.0)

    def test_values_bounded_between_zero_and_one(self):
        raw = np.random.randn(200)
        result = _normalise_scores(raw)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_lower_raw_score_means_higher_risk(self):
        # IsolationForest: lower (more negative) = more anomalous = higher risk
        raw = np.array([-0.5, -0.3, -0.1])
        result = _normalise_scores(raw)
        assert result[0] > result[1] > result[2]

    def test_all_same_values_returns_zeros(self):
        raw = np.array([0.2, 0.2, 0.2])
        result = _normalise_scores(raw)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_output_shape_matches_input(self):
        raw = np.random.randn(50)
        result = _normalise_scores(raw)
        assert result.shape == raw.shape

    def test_single_element_array(self):
        raw = np.array([0.1])
        result = _normalise_scores(raw)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

class TestScore:
    def test_adds_risk_score_column(self, sample_df, trained_model, fitted_pipeline):
        result = score(sample_df, trained_model, fitted_pipeline)
        assert "risk_score" in result.columns

    def test_adds_risk_tier_column(self, sample_df, trained_model, fitted_pipeline):
        result = score(sample_df, trained_model, fitted_pipeline)
        assert "risk_tier" in result.columns

    def test_adds_recommended_action_column(self, sample_df, trained_model, fitted_pipeline):
        result = score(sample_df, trained_model, fitted_pipeline)
        assert "recommended_action" in result.columns

    def test_risk_scores_in_zero_one_range(self, sample_df, trained_model, fitted_pipeline):
        result = score(sample_df, trained_model, fitted_pipeline)
        assert (result["risk_score"] >= 0.0).all()
        assert (result["risk_score"] <= 1.0).all()

    def test_risk_tiers_are_valid(self, sample_df, trained_model, fitted_pipeline):
        result = score(sample_df, trained_model, fitted_pipeline)
        valid = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert set(result["risk_tier"].unique()).issubset(valid)

    def test_recommended_actions_are_non_empty(self, sample_df, trained_model, fitted_pipeline):
        result = score(sample_df, trained_model, fitted_pipeline)
        assert result["recommended_action"].str.len().gt(0).all()

    def test_output_row_count_matches_input(self, sample_df, trained_model, fitted_pipeline):
        result = score(sample_df, trained_model, fitted_pipeline)
        assert len(result) == len(sample_df)

    def test_original_columns_preserved(self, sample_df, trained_model, fitted_pipeline):
        result = score(sample_df, trained_model, fitted_pipeline)
        for col in sample_df.columns:
            assert col in result.columns


# ---------------------------------------------------------------------------
# risk_tier() and TIER_ACTIONS
# ---------------------------------------------------------------------------

class TestRiskTier:
    def test_low_tier_below_threshold(self):
        assert risk_tier(0.1) == "LOW"
        assert risk_tier(0.0) == "LOW"

    def test_medium_tier_in_band(self):
        assert risk_tier(0.45) == "MEDIUM"

    def test_high_tier_in_band(self):
        assert risk_tier(0.75) == "HIGH"

    def test_critical_tier_above_threshold(self):
        assert risk_tier(0.9) == "CRITICAL"
        assert risk_tier(1.0) == "CRITICAL"

    def test_boundary_at_low_medium(self):
        # At exactly the low threshold, should be MEDIUM (>= comparison)
        assert risk_tier(0.3) == "MEDIUM"

    def test_boundary_at_medium_high(self):
        assert risk_tier(0.6) == "HIGH"

    def test_boundary_at_high_critical(self):
        assert risk_tier(0.85) == "CRITICAL"


class TestTierActions:
    def test_all_four_tiers_present(self):
        for tier in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            assert tier in TIER_ACTIONS

    def test_all_actions_are_non_empty_strings(self):
        for tier, action in TIER_ACTIONS.items():
            assert isinstance(action, str)
            assert len(action.strip()) > 0

    def test_critical_action_implies_urgency(self):
        # CRITICAL action should communicate urgency
        action = TIER_ACTIONS["CRITICAL"].lower()
        assert any(word in action for word in ("escalate", "freeze", "immediately", "urgent"))
