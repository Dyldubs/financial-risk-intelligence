"""
Supervised XGBoost classifier for fraud detection.

Complements the Isolation Forest (unsupervised) with a supervised approach
that uses is_fraud labels. Use this when labelled training data is available.

Key design choices:
  - scale_pos_weight handles the ~49:1 class imbalance
  - eval_metric='aucpr' optimises for precision-recall, not accuracy
  - Optuna tunes 7 hyperparameters with MLflow tracking
  - Shares the same feature pipeline as the Isolation Forest
"""

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
import optuna
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import MLFLOW_EXPERIMENT_NAME, ROOT_DIR
from src.data.features import build_feature_pipeline, prepare_training_data

optuna.logging.set_verbosity(optuna.logging.WARNING)

XGB_MODEL_PATH     = ROOT_DIR / "models" / "xgb_classifier.joblib"
XGB_PIPELINE_PATH  = ROOT_DIR / "models" / "xgb_feature_pipeline.joblib"

# Bypass stale .env values (same pattern as detector.py)
_MLFLOW_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"


def train_xgb(df: pd.DataFrame, n_trials: int = 30) -> dict:
    """
    Train a supervised XGBoost fraud classifier with Optuna hyperparameter search.

    Args:
        df:       Raw transaction DataFrame — must include an 'is_fraud' column.
        n_trials: Number of Optuna trials (default: 30).

    Returns:
        dict with best_params, best_auc_pr, best_auc_roc, model, pipeline,
        feature_names, X_test, y_test.
    """
    mlflow.set_tracking_uri(_MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X, y, feature_names = prepare_training_data(df)
    if y is None:
        raise ValueError(
            "XGBoost requires fraud labels. Ensure 'is_fraud' column is present."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit the shared feature pipeline on training data only
    pipeline = build_feature_pipeline(feature_names)
    X_train_t = pipeline.fit_transform(X_train)
    X_test_t  = pipeline.transform(X_test)

    # Class imbalance ratio — guides scale_pos_weight search range
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    imbalance_ratio = n_neg / n_pos
    print(f"  Training set: {n_neg:,} legitimate | {n_pos:,} fraud "
          f"(imbalance ratio {imbalance_ratio:.0f}:1)")

    def objective(trial):
        params = {
            "n_estimators"    : trial.suggest_int  ("n_estimators",     100, 500),
            "max_depth"       : trial.suggest_int  ("max_depth",          3,   8),
            "learning_rate"   : trial.suggest_float("learning_rate",   0.01, 0.3,  log=True),
            "min_child_weight": trial.suggest_int  ("min_child_weight",   1,  10),
            "subsample"       : trial.suggest_float("subsample",         0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree",  0.5, 1.0),
            "gamma"           : trial.suggest_float("gamma",             0.0, 1.0),
            # Search within ±50% of the natural imbalance ratio
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight",
                imbalance_ratio * 0.5,
                imbalance_ratio * 2.0,
            ),
        }

        with mlflow.start_run(run_name=f"xgb-trial-{trial.number}", nested=True):
            model = XGBClassifier(
                **params,
                random_state=42,
                n_jobs=1,           # -1 causes segfault on macOS/Python 3.14 via OpenMP
                tree_method="hist", # explicit; avoids fallback that triggers extra threads
                eval_metric="aucpr",
                verbosity=0,
            )
            model.fit(X_train_t, y_train)
            proba   = model.predict_proba(X_test_t)[:, 1]
            auc_pr  = average_precision_score(y_test, proba)
            auc_roc = roc_auc_score(y_test, proba)
            mlflow.log_params(params)
            mlflow.log_metric("auc_pr",  auc_pr)
            mlflow.log_metric("auc_roc", auc_roc)

        return auc_pr

    with mlflow.start_run(run_name="xgb-optuna-search"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_model  = XGBClassifier(
            **best_params,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            eval_metric="aucpr",
            verbosity=0,
        )
        best_model.fit(X_train_t, y_train)

        proba        = best_model.predict_proba(X_test_t)[:, 1]
        best_auc_pr  = average_precision_score(y_test, proba)
        best_auc_roc = roc_auc_score(y_test, proba)

        mlflow.log_params(best_params)
        mlflow.log_metric("best_auc_pr",  best_auc_pr)
        mlflow.log_metric("best_auc_roc", best_auc_roc)
        mlflow.xgboost.log_model(best_model, "xgb_classifier")

        print(f"\nXGBoost — Best AUC-PR : {best_auc_pr:.4f}")
        print(f"XGBoost — Best AUC-ROC: {best_auc_roc:.4f}")
        print(f"Best params: {best_params}")

    XGB_MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, XGB_MODEL_PATH)
    joblib.dump(pipeline,   XGB_PIPELINE_PATH)
    print(f"XGBoost model saved to {XGB_MODEL_PATH}")

    return {
        "best_params" : best_params,
        "best_auc_pr" : best_auc_pr,
        "best_auc_roc": best_auc_roc,
        "model"       : best_model,
        "pipeline"    : pipeline,
        "feature_names": feature_names,
        "X_test"      : X_test,
        "y_test"      : y_test,
    }


def load_xgb_model():
    """Load the trained XGBoost model and pipeline from disk."""
    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No XGBoost model found at {XGB_MODEL_PATH}. "
            "Run: python scripts/train.py --model xgboost"
        )
    model    = joblib.load(XGB_MODEL_PATH)
    pipeline = joblib.load(XGB_PIPELINE_PATH)
    return model, pipeline


def score_xgb(df: pd.DataFrame, model, pipeline) -> pd.DataFrame:
    """
    Score transactions with the XGBoost classifier.

    Returns input df with added columns: risk_score (fraud probability),
    risk_tier, recommended_action.
    """
    from src.config import risk_tier, TIER_ACTIONS
    from src.data.features import get_feature_names

    feature_names = get_feature_names(df)
    feature_names = [f for f in feature_names if f in df.columns]
    X     = pipeline.transform(df[feature_names])
    proba = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["risk_score"]         = proba
    result["risk_tier"]          = result["risk_score"].apply(risk_tier)
    result["recommended_action"] = result["risk_tier"].map(TIER_ACTIONS)
    return result
