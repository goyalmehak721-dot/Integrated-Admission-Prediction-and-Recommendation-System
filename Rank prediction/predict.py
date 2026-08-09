"""
Phase 3.3 -- Prediction Interface
==================================
Provides a simple function to predict JEE Advanced rank brackets.

Usage:
    python predict.py
    -> Runs sample predictions

    from predict import predict_rank
    predict_rank(180, max_marks=360)
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import warnings
import joblib

warnings.filterwarnings("ignore")

# Import model classes so pickle can deserialize them
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_training import (
    LightGBMNative, XGBoostNative, NumpyMLP, PolynomialQuantileRegressor
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_models():
    """Load all trained models and config."""
    models = joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))
    with open(os.path.join(MODELS_DIR, "config.json"), "r") as f:
        config = json.load(f)
    lookup = pd.read_csv(os.path.join(MODELS_DIR, "quantile_lookup.csv"))

    # Determine best model from results
    results_file = os.path.join(BASE_DIR, "results", "best_model.txt")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            best_model_name = f.read().strip()
    else:
        best_model_name = "lightgbm"  # default

    return models, config, lookup, best_model_name


def get_quantile_features(pct_score, lookup):
    """
    Look up historical quantile features for a given percentage score.
    Uses nearest-bin matching from the lookup table.
    """
    pct_bin = (pct_score // 1.0) * 1.0  # 1% bin
    row = lookup.iloc[(lookup["pct_bin"] - pct_bin).abs().argsort()[:1]]

    if len(row) > 0:
        return {
            "log_best_rank": row.iloc[0]["log_best_rank"],
            "log_worst_rank": row.iloc[0]["log_worst_rank"],
            "log_median_rank": row.iloc[0]["log_median_rank"],
        }
    else:
        return {
            "log_best_rank": np.log(1),
            "log_worst_rank": np.log(33701),
            "log_median_rank": np.log(10000),
        }


def model_predict(model_obj, X, model_name):
    """Get prediction from a model object (handles different model types)."""
    if model_name in ("lightgbm", "xgboost"):
        return model_obj.predict(X)
    elif model_name in ("mlp", "polynomial"):
        return model_obj.predict(X)
    else:
        # Generic fallback
        return model_obj.predict(X)


def predict_rank(marks, max_marks=360, verbose=True):
    """
    Predict JEE Advanced rank bracket for given marks.

    Args:
        marks: Raw marks obtained
        max_marks: Maximum marks for the exam (default 360 for recent years)
        verbose: If True, print formatted output

    Returns:
        dict with keys: optimistic, expected, pessimistic (rank values)
    """
    models, config, lookup, best_model_name = load_models()

    pct_score = (marks / max_marks) * 100
    q_features = get_quantile_features(pct_score, lookup)

    # Build feature vector: [pct_score, log_best_rank, log_worst_rank, log_median_rank]
    X = np.array([[
        pct_score,
        q_features["log_best_rank"],
        q_features["log_worst_rank"],
        q_features["log_median_rank"],
    ]])

    results = {}
    for q_name in ["optimistic", "expected", "pessimistic"]:
        if q_name in models and best_model_name in models[q_name]:
            model_obj = models[q_name][best_model_name]
            log_rank_pred = model_predict(model_obj, X, best_model_name)
            if isinstance(log_rank_pred, np.ndarray):
                log_rank_pred = log_rank_pred.flatten()[0]
            rank_pred = int(np.round(np.exp(log_rank_pred)))
            results[q_name] = max(1, rank_pred)
        else:
            results[q_name] = None

    if verbose:
        opt = results.get('optimistic', 'N/A')
        exp = results.get('expected', 'N/A')
        pes = results.get('pessimistic', 'N/A')

        opt_str = f"{opt:>8,}" if isinstance(opt, int) else str(opt)
        exp_str = f"{exp:>8,}" if isinstance(exp, int) else str(exp)
        pes_str = f"{pes:>8,}" if isinstance(pes, int) else str(pes)

        print()
        print("+" + "-" * 56 + "+")
        print("|" + "  JEE Advanced Rank Prediction  ".center(56) + "|")
        print("+" + "-" * 56 + "+")
        print(f"|  Marks:       {marks}/{max_marks} ({pct_score:.1f}%)".ljust(57) + "|")
        print(f"|  Model:       {best_model_name}".ljust(57) + "|")
        print("+" + "-" * 56 + "+")
        print(f"|  [Optimistic]  (tough paper):   AIR {opt_str}".ljust(57) + "|")
        print(f"|  [Expected]    (median paper):  AIR {exp_str}".ljust(57) + "|")
        print(f"|  [Pessimistic] (easy paper):    AIR {pes_str}".ljust(57) + "|")
        print("+" + "-" * 56 + "+")
        print()

    return results


def run_sample_predictions():
    """Run sample predictions to demonstrate the system."""
    print("\n" + "=" * 60)
    print("  Sample Predictions (Max Marks = 360)")
    print("=" * 60)

    test_cases = [
        # (300, 360, "Top scorer"),
        # (250, 360, "Strong performer"),
        # (200, 360, "Above average"),
        # (180, 360, "Average"),
        # (150, 360, "Below average"),
        # (130, 360, "Near cutoff"),
        # (120, 360, "Low score"),
        (100, 360, "Very low score"),
        # (144, 360, "2025 custome"),
        (341, 360, "2025 custome"),
    ]

    for marks, max_marks, description in test_cases:
        print(f"\n--- {description} ---")
        predict_rank(marks, max_marks, verbose=True)


if __name__ == "__main__":
    run_sample_predictions()
