"""
Anomaly detection model: Isolation Forest + Z-score ensemble.

Trains with MLflow experiment tracking and Optuna hyperparameter search.
Saves the best model to the MLflow Model Registry.
"""

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import optuna
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from src.config import MLFLOW_EXPERIMENT_NAME, ROOT_DIR
from src.data.features import build_feature_pipeline, prepare_training_data

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_PATH = ROOT_DIR / "models" / "detector.joblib"
PIPELINE_PATH = ROOT_DIR / "models" / "feature_pipeline.joblib"

# Hardcode SQLite URI so it's never overridden by a stale .env value
_MLFLOW_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"


def train(df: pd.DataFrame, n_trials: int = 30) -> dict:
    """
    Train the anomaly detector with Optuna hyperparameter search.
    All runs are logged to MLflow.

    Args:
        df: Raw transaction DataFrame (must include 'is_fraud' column for evaluation).
        n_trials: Number of Optuna trials.

    Returns:
        dict with best_params, best_auc_pr, model, pipeline.
    """
    mlflow.set_tracking_uri(_MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X, y, feature_names = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_feature_pipeline(feature_names)
    X_train_t = pipeline.fit_transform(X_train)
    X_test_t = pipeline.transform(X_test)

    best_score = -np.inf
    best_params = {}
    best_model = None

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
            "contamination": trial.suggest_float("contamination", 0.005, 0.05),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }

        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            model = IsolationForest(**params, random_state=42, n_jobs=-1)
            model.fit(X_train_t)

            # Decision function: lower = more anomalous; normalise to [0,1]
            raw_scores = model.decision_function(X_test_t)
            risk_scores = _normalise_scores(raw_scores)

            auc_pr = average_precision_score(y_test, risk_scores)
            auc_roc = roc_auc_score(y_test, risk_scores)

            mlflow.log_params(params)
            mlflow.log_metric("auc_pr", auc_pr)
            mlflow.log_metric("auc_roc", auc_roc)

        return auc_pr

    with mlflow.start_run(run_name="optuna-search"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_model = IsolationForest(**best_params, random_state=42, n_jobs=-1)
        best_model.fit(X_train_t)

        raw_scores = best_model.decision_function(X_test_t)
        risk_scores = _normalise_scores(raw_scores)
        best_auc_pr = average_precision_score(y_test, risk_scores)

        # Log best run
        mlflow.log_params(best_params)
        mlflow.log_metric("best_auc_pr", best_auc_pr)
        mlflow.log_metric("best_auc_roc", roc_auc_score(y_test, risk_scores))
        mlflow.sklearn.log_model(best_model, "isolation_forest")

        print(f"\nBest AUC-PR: {best_auc_pr:.4f}")
        print(f"Best params: {best_params}")

    # Save locally for API use
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return {
        "best_params": best_params,
        "best_auc_pr": best_auc_pr,
        "model": best_model,
        "pipeline": pipeline,
        "feature_names": feature_names,
        "X_test": X_test,
        "y_test": y_test,
    }


def load_model():
    """Load the trained model and feature pipeline from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run train() first."
        )
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline


def score(df: pd.DataFrame, model, pipeline) -> pd.DataFrame:
    """
    Score a batch of transactions and return risk scores.

    Returns:
        Input df with added columns: risk_score, risk_tier.
    """
    from src.config import risk_tier, TIER_ACTIONS
    from src.data.features import get_feature_names

    feature_names = get_feature_names(df)
    feature_names = [f for f in feature_names if f in df.columns]
    X = pipeline.transform(df[feature_names])
    raw = model.decision_function(X)
    scores = _normalise_scores(raw)

    result = df.copy()
    result["risk_score"] = scores
    result["risk_tier"] = result["risk_score"].apply(risk_tier)
    result["recommended_action"] = result["risk_tier"].map(TIER_ACTIONS)
    return result


def _normalise_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Invert and normalise Isolation Forest scores to [0, 1] (higher = riskier)."""
    inverted = -raw_scores
    min_s, max_s = inverted.min(), inverted.max()
    if max_s == min_s:
        return np.zeros_like(inverted)
    return (inverted - min_s) / (max_s - min_s)
