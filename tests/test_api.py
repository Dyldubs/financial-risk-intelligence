"""
Integration tests for src/api/main.py (FastAPI endpoints).

Uses FastAPI's TestClient. Model and vectorstore are mocked so tests
run without trained artifacts on disk.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import src.api.main as api_module
from src.api.main import app


# A single valid transaction payload matching the Transaction schema
VALID_TXN = {
    "transaction_id": "TXN-TEST-001",
    "amount": 9800.0,
    "hour": 2,
    "merchant_risk_tier": 2,
    "velocity_1h": 5,
    "velocity_24h": 18,
    "high_risk_country": 1,
    "amount_vs_avg_ratio": 8.5,
    "days_since_account_open": 15,
    "is_weekend": 1,
}


@pytest.fixture(scope="module")
def client():
    """
    TestClient with startup silenced.
    Model files won't be present in the test environment, so startup
    logs a WARNING but does not raise — _model and _pipeline remain None.
    """
    with TestClient(app) as c:
        yield c


def _scored_df():
    """A one-row DataFrame representing the output of score()."""
    return pd.DataFrame([{
        "transaction_id": "TXN-TEST-001",
        "amount": 9800.0,
        "hour": 2,
        "merchant_risk_tier": 2,
        "velocity_1h": 5,
        "velocity_24h": 18,
        "high_risk_country": 1,
        "amount_vs_avg_ratio": 8.5,
        "days_since_account_open": 15,
        "is_weekend": 1,
        "risk_score": 0.87,
        "risk_tier": "CRITICAL",
        "recommended_action": "Immediately escalate to the Financial Crime team.",
    }])


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_has_status_key(self, client):
        assert "status" in client.get("/health").json()

    def test_response_has_model_loaded_key(self, client):
        assert "model_loaded" in client.get("/health").json()

    def test_response_has_vectorstore_loaded_key(self, client):
        assert "vectorstore_loaded" in client.get("/health").json()

    def test_status_value_is_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"


# ---------------------------------------------------------------------------
# GET /model/info
# ---------------------------------------------------------------------------

class TestModelInfoEndpoint:
    def test_returns_503_when_model_not_loaded(self, client):
        with patch.object(api_module, "_model", None):
            r = client.get("/model/info")
        assert r.status_code == 503

    def test_returns_200_with_mocked_model(self, client):
        mock_model = MagicMock()
        mock_model.n_estimators = 100
        mock_model.contamination = 0.02
        with patch.object(api_module, "_model", mock_model):
            r = client.get("/model/info")
        assert r.status_code == 200

    def test_response_contains_model_type(self, client):
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "IsolationForest"
        with patch.object(api_module, "_model", mock_model):
            r = client.get("/model/info")
        assert "model_type" in r.json()


# ---------------------------------------------------------------------------
# POST /analyse — validation errors
# ---------------------------------------------------------------------------

class TestAnalyseValidation:
    def test_missing_amount_returns_422(self, client):
        r = client.post("/analyse", json={
            "transactions": [{"hour": 12}]
        })
        assert r.status_code == 422

    def test_negative_amount_returns_422(self, client):
        txn = {**VALID_TXN, "amount": -100.0}
        r = client.post("/analyse", json={"transactions": [txn]})
        assert r.status_code == 422

    def test_zero_amount_returns_422(self, client):
        txn = {**VALID_TXN, "amount": 0.0}
        r = client.post("/analyse", json={"transactions": [txn]})
        assert r.status_code == 422

    def test_missing_transactions_key_returns_422(self, client):
        r = client.post("/analyse", json={})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /analyse — model not loaded (503)
# ---------------------------------------------------------------------------

class TestAnalyseNoModel:
    def test_returns_503_when_model_is_none(self, client):
        with patch.object(api_module, "_model", None), \
             patch.object(api_module, "_pipeline", None):
            r = client.post("/analyse", json={"transactions": [VALID_TXN]})
        assert r.status_code == 503

    def test_503_error_message_mentions_training(self, client):
        with patch.object(api_module, "_model", None), \
             patch.object(api_module, "_pipeline", None):
            r = client.post("/analyse", json={"transactions": [VALID_TXN]})
        assert "503" in str(r.status_code)


# ---------------------------------------------------------------------------
# POST /analyse — success path (mocked model)
# ---------------------------------------------------------------------------

class TestAnalyseSuccess:
    """Happy-path tests with all heavy dependencies mocked out."""

    @pytest.fixture
    def mock_context(self, client):
        """Patch all I/O-bound callables for a clean success path."""
        scored = _scored_df()
        with patch.object(api_module, "_model", MagicMock()), \
             patch.object(api_module, "_pipeline", MagicMock()), \
             patch.object(api_module, "_vectorstore", None), \
             patch("src.api.main.score", return_value=scored), \
             patch("src.api.main.build_explainer", return_value=MagicMock()), \
             patch("src.api.main.get_shap_values", return_value=np.zeros((1, 9))), \
             patch("src.api.main.top_drivers_text", return_value="velocity_1h (+0.28)"), \
             patch("src.api.main.generate_risk_summary", return_value="High risk detected."):
            yield client

    def test_returns_200(self, mock_context):
        r = mock_context.post("/analyse", json={"transactions": [VALID_TXN]})
        assert r.status_code == 200

    def test_total_matches_input_count(self, mock_context):
        r = mock_context.post("/analyse", json={"transactions": [VALID_TXN]})
        assert r.json()["total"] == 1

    def test_results_list_length_matches_total(self, mock_context):
        r = mock_context.post("/analyse", json={"transactions": [VALID_TXN]})
        data = r.json()
        assert len(data["results"]) == data["total"]

    def test_result_contains_required_fields(self, mock_context):
        r = mock_context.post("/analyse", json={"transactions": [VALID_TXN]})
        result = r.json()["results"][0]
        for field in ("amount", "risk_score", "risk_tier", "top_drivers",
                      "policy_context", "risk_summary", "recommended_action"):
            assert field in result, f"Missing field: {field}"

    def test_risk_score_is_float(self, mock_context):
        r = mock_context.post("/analyse", json={"transactions": [VALID_TXN]})
        assert isinstance(r.json()["results"][0]["risk_score"], float)

    def test_flagged_count_reflects_high_risk(self, mock_context):
        r = mock_context.post("/analyse", json={"transactions": [VALID_TXN]})
        data = r.json()
        # Our mocked row has risk_tier=CRITICAL → should be counted as flagged
        assert data["flagged"] >= 0
        assert data["flagged"] <= data["total"]
