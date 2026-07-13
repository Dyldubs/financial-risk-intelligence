"""
Unit tests for src/llm/summariser.py

Tests prompt construction, the fallback summary (no API key),
and that generate_risk_summary() correctly routes based on API key presence.
"""

import pytest
from unittest.mock import patch

from src.llm.summariser import build_risk_prompt, _fallback_summary, generate_risk_summary


SAMPLE_TXN = {
    "transaction_id": "TXN-TEST-001",
    "amount": 9800.0,
    "risk_score": 0.87,
    "risk_tier": "CRITICAL",
    "hour": 2,
    "high_risk_country": 1,
    "velocity_1h": 5,
    "velocity_24h": 18,
    "amount_vs_avg_ratio": 8.5,
    "recommended_action": "Immediately escalate to the Financial Crime team.",
}
SAMPLE_DRIVERS = (
    "1. velocity_1h (+0.28): unusually high transaction frequency in the last hour\n"
    "2. high_risk_country (+0.21): transaction linked to a high-risk jurisdiction\n"
    "3. amount_vs_avg_ratio (+0.18): amount significantly exceeds the account's recent average"
)
SAMPLE_POLICY = "AML/CTF Act: transactions above AUD 10,000 require reporting to AUSTRAC."


# ---------------------------------------------------------------------------
# build_risk_prompt
# ---------------------------------------------------------------------------

class TestBuildRiskPrompt:
    def test_contains_transaction_id(self):
        prompt = build_risk_prompt(SAMPLE_TXN, SAMPLE_DRIVERS, SAMPLE_POLICY)
        assert "TXN-TEST-001" in prompt

    def test_contains_formatted_amount(self):
        prompt = build_risk_prompt(SAMPLE_TXN, SAMPLE_DRIVERS, SAMPLE_POLICY)
        assert "9,800.00" in prompt

    def test_contains_risk_score(self):
        prompt = build_risk_prompt(SAMPLE_TXN, SAMPLE_DRIVERS, SAMPLE_POLICY)
        assert "0.87" in prompt

    def test_contains_risk_tier(self):
        prompt = build_risk_prompt(SAMPLE_TXN, SAMPLE_DRIVERS, SAMPLE_POLICY)
        assert "CRITICAL" in prompt

    def test_contains_shap_drivers(self):
        prompt = build_risk_prompt(SAMPLE_TXN, SAMPLE_DRIVERS, SAMPLE_POLICY)
        assert SAMPLE_DRIVERS in prompt

    def test_contains_policy_context(self):
        prompt = build_risk_prompt(SAMPLE_TXN, SAMPLE_DRIVERS, SAMPLE_POLICY)
        assert SAMPLE_POLICY in prompt

    def test_returns_string(self):
        result = build_risk_prompt(SAMPLE_TXN, SAMPLE_DRIVERS, SAMPLE_POLICY)
        assert isinstance(result, str)

    def test_handles_missing_optional_fields(self):
        minimal_txn = {"amount": 100.0}
        # Should not raise KeyError on missing optional fields
        prompt = build_risk_prompt(minimal_txn, "", "")
        assert isinstance(prompt, str)


# ---------------------------------------------------------------------------
# _fallback_summary
# ---------------------------------------------------------------------------

class TestFallbackSummary:
    def test_returns_non_empty_string(self):
        result = _fallback_summary(SAMPLE_TXN, SAMPLE_DRIVERS)
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_contains_risk_tier(self):
        result = _fallback_summary(SAMPLE_TXN, SAMPLE_DRIVERS)
        assert "CRITICAL" in result

    def test_contains_risk_score(self):
        result = _fallback_summary(SAMPLE_TXN, SAMPLE_DRIVERS)
        assert "0.87" in result

    def test_contains_transaction_id(self):
        result = _fallback_summary(SAMPLE_TXN, SAMPLE_DRIVERS)
        assert "TXN-TEST-001" in result

    def test_indicates_llm_unavailable(self):
        result = _fallback_summary(SAMPLE_TXN, SAMPLE_DRIVERS)
        assert "ANTHROPIC_API_KEY" in result or "LLM summary unavailable" in result

    def test_contains_recommended_action(self):
        result = _fallback_summary(SAMPLE_TXN, SAMPLE_DRIVERS)
        assert "Financial Crime" in result or "escalate" in result.lower()


# ---------------------------------------------------------------------------
# generate_risk_summary
# ---------------------------------------------------------------------------

class TestGenerateRiskSummary:
    def test_uses_fallback_when_api_key_empty(self):
        with patch("src.llm.summariser.ANTHROPIC_API_KEY", ""):
            result = generate_risk_summary(SAMPLE_TXN, SAMPLE_DRIVERS, SAMPLE_POLICY)
        assert "LLM summary unavailable" in result

    def test_returns_string_without_api_key(self):
        with patch("src.llm.summariser.ANTHROPIC_API_KEY", ""):
            result = generate_risk_summary(SAMPLE_TXN, SAMPLE_DRIVERS, SAMPLE_POLICY)
        assert isinstance(result, str)

    def test_does_not_call_llm_without_api_key(self):
        with patch("src.llm.summariser.ANTHROPIC_API_KEY", ""), \
             patch("src.llm.summariser.ChatAnthropic") as mock_llm:
            generate_risk_summary(SAMPLE_TXN, SAMPLE_DRIVERS, SAMPLE_POLICY)
            mock_llm.assert_not_called()
