"""
Training script. Run once to train and save the anomaly detection model.

Usage:
    python scripts/train.py

Options:
    --trials N     Number of Optuna hyperparameter search trials (default: 30)
    --data PATH    Path to a custom CSV file (default: uses synthetic data)
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data.loader import load_transactions
from src.models.detector import train
from src.models.explainer import (
    build_explainer, get_shap_values,
    save_global_importance_plot, save_beeswarm_plot,
)
from src.data.features import get_feature_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("Financial Risk Intelligence — Model Training")
    print("=" * 60)

    # Load data
    path = Path(args.data) if args.data else None
    df = load_transactions(path)

    # Train
    result = train(df, n_trials=args.trials)
    model = result["model"]
    pipeline = result["pipeline"]
    feature_names = result["feature_names"]
    X_test = result["X_test"]

    print(f"\nBest AUC-PR: {result['best_auc_pr']:.4f}")

    # Generate global SHAP plots
    print("\nGenerating SHAP plots...")
    X_test_t = pipeline.transform(X_test[feature_names])
    explainer = build_explainer(model, X_test_t)
    shap_vals = get_shap_values(explainer, X_test_t)

    importance_path = save_global_importance_plot(shap_vals, feature_names)
    beeswarm_path = save_beeswarm_plot(shap_vals, X_test[feature_names])

    print(f"  Saved: {importance_path}")
    print(f"  Saved: {beeswarm_path}")
    print("\nTraining complete. You can now run:")
    print("  streamlit run app/streamlit_app.py")
    print("  uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    main()
