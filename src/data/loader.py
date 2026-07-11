"""
Data loading utilities.

Supports two modes:
  1. Real data: Kaggle Credit Card Fraud Detection dataset (creditcard.csv)
     Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
     Place at: data/raw/creditcard.csv

  2. Synthetic data: generated automatically if no CSV is found.
     Useful for development and demo without Kaggle account.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import RAW_DATA_DIR


def load_transactions(path: Path | None = None) -> pd.DataFrame:
    """Load transaction data. Falls back to synthetic data if file not found."""
    csv_path = path or RAW_DATA_DIR / "creditcard.csv"

    if csv_path.exists():
        print(f"Loading real data from {csv_path}")
        df = pd.read_csv(csv_path)
        df = _normalise_real_data(df)
    else:
        print("No creditcard.csv found — generating synthetic dataset.")
        df = _generate_synthetic_data(n_samples=10_000, fraud_rate=0.02)

    print(f"Loaded {len(df):,} transactions | fraud rate: {df['is_fraud'].mean():.2%}")
    return df


def _normalise_real_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns for the Kaggle credit card fraud dataset."""
    df = df.rename(columns={"Time": "time", "Amount": "amount", "Class": "is_fraud"})
    # V1..V28 are PCA-transformed features — keep them as-is
    return df.reset_index(drop=True)


def _generate_synthetic_data(n_samples: int = 10_000, fraud_rate: float = 0.02) -> pd.DataFrame:
    """
    Generate a synthetic transaction dataset with realistic structure.
    Fraudulent transactions have higher amounts, unusual hours, and elevated velocity.
    """
    rng = np.random.default_rng(42)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    def make_legit(n):
        return {
            "amount": rng.lognormal(mean=3.5, sigma=1.2, size=n),
            "hour": rng.choice(range(24), size=n, p=_hour_dist()),
            "merchant_risk_tier": rng.choice([0, 1, 2], size=n, p=[0.7, 0.25, 0.05]),
            "velocity_1h": rng.poisson(lam=1.5, size=n),
            "velocity_24h": rng.poisson(lam=8, size=n),
            "high_risk_country": rng.choice([0, 1], size=n, p=[0.97, 0.03]),
            "amount_vs_avg_ratio": rng.lognormal(mean=0, sigma=0.4, size=n),
            "days_since_account_open": rng.integers(30, 2000, size=n),
            "is_weekend": rng.choice([0, 1], size=n, p=[0.71, 0.29]),
            "is_fraud": np.zeros(n, dtype=int),
        }

    def make_fraud(n):
        return {
            "amount": rng.lognormal(mean=5.5, sigma=1.5, size=n),
            "hour": rng.choice(range(24), size=n, p=_fraud_hour_dist()),
            "merchant_risk_tier": rng.choice([0, 1, 2], size=n, p=[0.2, 0.4, 0.4]),
            "velocity_1h": rng.poisson(lam=5, size=n),
            "velocity_24h": rng.poisson(lam=18, size=n),
            "high_risk_country": rng.choice([0, 1], size=n, p=[0.6, 0.4]),
            "amount_vs_avg_ratio": rng.lognormal(mean=1.5, sigma=0.8, size=n),
            "days_since_account_open": rng.integers(1, 180, size=n),
            "is_weekend": rng.choice([0, 1], size=n, p=[0.5, 0.5]),
            "is_fraud": np.ones(n, dtype=int),
        }

    legit = pd.DataFrame(make_legit(n_legit))
    fraud = pd.DataFrame(make_fraud(n_fraud))
    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    df["transaction_id"] = [f"TXN-{i:07d}" for i in range(len(df))]
    return df


def _hour_dist():
    """Realistic legitimate transaction hour distribution (busy 9am-9pm)."""
    weights = np.array([0.5, 0.3, 0.2, 0.2, 0.2, 0.4, 0.8, 1.5, 2.5,
                        4.0, 5.5, 6.0, 6.5, 6.0, 5.5, 5.0, 5.5, 6.0,
                        6.5, 6.0, 5.0, 4.0, 3.0, 1.5])
    return weights / weights.sum()


def _fraud_hour_dist():
    """Fraud skews heavily to late night / early morning."""
    weights = np.array([4.0, 5.0, 5.5, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5,
                        2.0, 2.0, 2.0, 2.5, 3.0, 4.0])
    return weights / weights.sum()
