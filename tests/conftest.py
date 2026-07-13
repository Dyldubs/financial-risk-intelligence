"""
Shared pytest fixtures used across the test suite.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from src.data.features import build_feature_pipeline, get_feature_names, prepare_training_data


@pytest.fixture
def sample_df():
    """Small synthetic transaction DataFrame with known values."""
    return pd.DataFrame([
        {
            "amount": 50.0, "hour": 14, "merchant_risk_tier": 0,
            "velocity_1h": 1, "velocity_24h": 5, "high_risk_country": 0,
            "amount_vs_avg_ratio": 1.0, "days_since_account_open": 500,
            "is_weekend": 0, "is_fraud": 0,
        },
        {
            "amount": 100.0, "hour": 10, "merchant_risk_tier": 0,
            "velocity_1h": 1, "velocity_24h": 4, "high_risk_country": 0,
            "amount_vs_avg_ratio": 0.9, "days_since_account_open": 730,
            "is_weekend": 1, "is_fraud": 0,
        },
        {
            "amount": 9800.0, "hour": 2, "merchant_risk_tier": 2,
            "velocity_1h": 5, "velocity_24h": 18, "high_risk_country": 1,
            "amount_vs_avg_ratio": 8.5, "days_since_account_open": 15,
            "is_weekend": 1, "is_fraud": 1,
        },
        {
            "amount": 200.0, "hour": 16, "merchant_risk_tier": 0,
            "velocity_1h": 1, "velocity_24h": 6, "high_risk_country": 0,
            "amount_vs_avg_ratio": 1.2, "days_since_account_open": 400,
            "is_weekend": 0, "is_fraud": 0,
        },
        {
            "amount": 75.0, "hour": 9, "merchant_risk_tier": 1,
            "velocity_1h": 2, "velocity_24h": 7, "high_risk_country": 0,
            "amount_vs_avg_ratio": 0.7, "days_since_account_open": 600,
            "is_weekend": 0, "is_fraud": 0,
        },
    ])


@pytest.fixture
def feature_names(sample_df):
    return get_feature_names(sample_df)


@pytest.fixture
def fitted_pipeline(sample_df, feature_names):
    """A feature pipeline fitted on sample_df."""
    pipeline = build_feature_pipeline(feature_names)
    pipeline.fit(sample_df[feature_names])
    return pipeline


@pytest.fixture
def trained_model(sample_df, fitted_pipeline, feature_names):
    """A tiny IsolationForest trained on sample_df (no MLflow, fast)."""
    X_t = fitted_pipeline.transform(sample_df[feature_names])
    model = IsolationForest(n_estimators=10, random_state=42)
    model.fit(X_t)
    return model
