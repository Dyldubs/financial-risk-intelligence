"""
Training script. Trains and saves the anomaly detection / classification models.

Usage:
    python scripts/train.py                          # train all models (default)
    python scripts/train.py --model isolation_forest
    python scripts/train.py --model xgboost
    python scripts/train.py --model scorecard
    python scripts/train.py --model both            # isolation_forest + xgboost

Options:
    --model MODEL  Which model to train:
                     isolation_forest | xgboost | scorecard | both | all
                     (default: all)
    --trials N     Number of Optuna hyperparameter search trials (default: 30)
    --data PATH    Path to a custom CSV (default: uses synthetic data)
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_transactions
from src.data.features import get_feature_names


def train_isolation_forest(df, n_trials):
    from src.models.detector import train
    from src.models.explainer import (
        build_explainer, get_shap_values,
        save_global_importance_plot, save_beeswarm_plot,
    )

    print("\n── Isolation Forest (unsupervised) ──────────────────────")
    result = train(df, n_trials=n_trials)

    print(f"\n  Best AUC-PR: {result['best_auc_pr']:.4f}")

    print("  Generating SHAP plots...")
    feature_names = result["feature_names"]
    X_test   = result["X_test"]
    pipeline = result["pipeline"]
    model    = result["model"]

    X_test_t = pipeline.transform(X_test[feature_names])
    explainer = build_explainer(model, X_test_t)
    shap_vals = get_shap_values(explainer, X_test_t)

    p1 = save_global_importance_plot(shap_vals, feature_names)
    p2 = save_beeswarm_plot(shap_vals, X_test[feature_names])
    print(f"  Saved: {p1}")
    print(f"  Saved: {p2}")

    return result


def train_xgboost(df, n_trials):
    from src.models.xgb_classifier import train_xgb

    print("\n── XGBoost Classifier (supervised) ─────────────────────")

    if "is_fraud" not in df.columns:
        print("  WARNING: 'is_fraud' column not found — skipping XGBoost.")
        return None

    result = train_xgb(df, n_trials=n_trials)
    print(f"\n  Best AUC-PR : {result['best_auc_pr']:.4f}")
    print(f"  Best AUC-ROC: {result['best_auc_roc']:.4f}")
    return result


def train_scorecard(df, **kwargs):
    from src.models.scorecard import train_scorecard as _train

    print("\n── Logistic Regression Scorecard (WoE/IV) ───────────────")

    if "is_fraud" not in df.columns:
        print("  WARNING: 'is_fraud' column not found — skipping scorecard.")
        return None

    result = _train(df)
    m = result["metrics"]
    print(f"\n  AUC-PR : {m['auc_pr']:.4f}")
    print(f"  AUC-ROC: {m['auc_roc']:.4f}")
    print(f"  Gini   : {m['gini']:.4f}")
    print(f"  KS     : {m['ks_stat']:.4f}")
    print(f"  Features selected (IV ≥ 0.02): {m['n_features_selected']}")

    print("\n  Information Value summary:")
    print(result["all_iv_summary"].to_string(index=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["isolation_forest", "xgboost", "scorecard", "both", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna trials per model (default: 30)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to custom CSV (default: synthetic data)")
    args = parser.parse_args()

    print("=" * 60)
    print("Financial Risk Intelligence — Model Training")
    print("=" * 60)

    path = Path(args.data) if args.data else None
    df   = load_transactions(path)

    if_result  = None
    xgb_result = None
    sc_result  = None

    run_if  = args.model in ("isolation_forest", "both", "all")
    run_xgb = args.model in ("xgboost", "both", "all")
    run_sc  = args.model in ("scorecard", "all")

    if run_if:
        if_result = train_isolation_forest(df, args.trials)
    if run_xgb:
        xgb_result = train_xgboost(df, args.trials)
    if run_sc:
        sc_result = train_scorecard(df)

    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    if if_result:
        print(f"  Isolation Forest  — AUC-PR : {if_result['best_auc_pr']:.4f}")
    if xgb_result:
        print(f"  XGBoost           — AUC-PR : {xgb_result['best_auc_pr']:.4f}")
        print(f"  XGBoost           — AUC-ROC: {xgb_result['best_auc_roc']:.4f}")
    if sc_result:
        m = sc_result["metrics"]
        print(f"  LR Scorecard      — AUC-PR : {m['auc_pr']:.4f}")
        print(f"  LR Scorecard      — Gini   : {m['gini']:.4f}")
        print(f"  LR Scorecard      — KS     : {m['ks_stat']:.4f}")

    print("\nNext steps:")
    print("  See notebooks/02_model_comparison.ipynb for IF vs XGBoost.")
    print("  See notebooks/04_scorecard.ipynb for the scorecard deep-dive.")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
