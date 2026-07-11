"""
Feature engineering pipeline.

Works with both the real Kaggle dataset and the synthetic dataset.
The pipeline is a scikit-learn Pipeline so it can be saved and reloaded
consistently between training and inference.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


# --- Feature sets ---

# Features available in the synthetic dataset
SYNTHETIC_NUMERIC_FEATURES = [
    "amount",
    "hour",
    "merchant_risk_tier",
    "velocity_1h",
    "velocity_24h",
    "high_risk_country",
    "amount_vs_avg_ratio",
    "days_since_account_open",
    "is_weekend",
]

# Features from the real Kaggle dataset (PCA features V1..V28 + amount)
KAGGLE_NUMERIC_FEATURES = [f"V{i}" for i in range(1, 29)] + ["amount"]


class LogAmountTransformer(BaseEstimator, TransformerMixin):
    """Log-transform 'amount' to reduce the effect of extreme values."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "amount" in X.columns:
            X["log_amount"] = np.log1p(X["amount"])
        return X


class IsNightTransformer(BaseEstimator, TransformerMixin):
    """Flag transactions between 11pm and 5am as high-risk time window."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "hour" in X.columns:
            X["is_night"] = ((X["hour"] >= 23) | (X["hour"] <= 5)).astype(int)
        return X


def build_feature_pipeline(feature_names: list[str]) -> Pipeline:
    """
    Build a reproducible scikit-learn preprocessing pipeline.

    Args:
        feature_names: List of column names to include as numeric features.

    Returns:
        A fitted-ready Pipeline that scales numeric features.
    """
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_names),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("log_amount", LogAmountTransformer()),
        ("is_night", IsNightTransformer()),
        ("preprocessor", preprocessor),
    ])


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Detect which feature set to use based on available columns."""
    if "V1" in df.columns:
        return KAGGLE_NUMERIC_FEATURES
    return SYNTHETIC_NUMERIC_FEATURES


def prepare_training_data(df: pd.DataFrame):
    """
    Split into features (X) and labels (y), excluding non-feature columns.

    Returns:
        X (DataFrame of features), y (Series of labels), feature_names (list)
    """
    feature_names = get_feature_names(df)
    # Only keep columns that actually exist in df
    feature_names = [f for f in feature_names if f in df.columns]
    X = df[feature_names].copy()
    y = df["is_fraud"].astype(int) if "is_fraud" in df.columns else None
    return X, y, feature_names
