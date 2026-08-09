"""
Phase 3.2 -- Model Evaluation & Comparison
===========================================
Leave-One-Year-Out Cross-Validation to evaluate all models.
Reports MAE, RMSE, R2 on log-transformed and original rank scale.

No sklearn dependency -- uses native LightGBM/XGBoost APIs + numpy.

Input:  features_data.csv
Output: evaluation results, comparison table, plots
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# Import custom model classes
sys.path.insert(0, os.path.dirname(__file__))
from model_training import (
    LightGBMNative, XGBoostNative, NumpyMLP, PolynomialQuantileRegressor
)

BASE_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(BASE_DIR, "features_data.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = ["pct_score", "log_best_rank", "log_worst_rank", "log_median_rank"]

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "figure.dpi": 150,
    "font.size": 10,
})


# ─── Metrics ────────────────────────────────────────────────────────────────────

def calc_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def calc_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ─── Model Training Functions ──────────────────────────────────────────────────

def train_and_predict(model_name, X_train, y_train, X_test, quantile=0.5):
    """Train a single model and return predictions."""
    if model_name == "lightgbm":
        m = LightGBMNative(quantile=quantile)
        m.fit(X_train, y_train)
        return m.predict(X_test)

    elif model_name == "xgboost":
        m = XGBoostNative(quantile=quantile)
        m.fit(X_train, y_train)
        return m.predict(X_test)

    elif model_name == "mlp":
        m = NumpyMLP(layer_sizes=[64, 32, 16], quantile=quantile,
                     lr=0.001, epochs=300, batch_size=64)
        m.fit(X_train, y_train)
        return m.predict(X_test)

    elif model_name == "polynomial":
        m = PolynomialQuantileRegressor(degree=3, quantile=quantile, max_iter=100)
        m.fit(X_train, y_train)
        return m.predict(X_test)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def leave_one_year_out_cv(df, model_names, quantile=0.5):
    """Leave-One-Year-Out CV."""
    years = sorted(df["Year"].unique())
    results = {name: {"y_true_log": [], "y_pred_log": [],
                       "y_true_rank": [], "y_pred_rank": []}
               for name in model_names}

    for hold_year in years:
        train_df = df[df["Year"] != hold_year]
        test_df = df[df["Year"] == hold_year]
        if len(test_df) < 5:
            continue

        X_train = train_df[FEATURES].values
        y_train = train_df["log_rank"].values
        X_test = test_df[FEATURES].values
        y_true_log = test_df["log_rank"].values
        y_true_rank = test_df["AIR"].values

        for name in model_names:
            try:
                y_pred_log = train_and_predict(name, X_train, y_train,
                                                X_test, quantile)
                y_pred_rank = np.exp(y_pred_log)
                results[name]["y_true_log"].extend(y_true_log.tolist())
                results[name]["y_pred_log"].extend(y_pred_log.tolist())
                results[name]["y_true_rank"].extend(y_true_rank.tolist())
                results[name]["y_pred_rank"].extend(y_pred_rank.tolist())
            except Exception as e:
                print(f"    [WARN] {name} failed on year {hold_year}: {e}")

    return results


def compute_metrics(results):
    """Compute MAE, RMSE, R2 for each model."""
    metrics = {}
    for name, data in results.items():
        if len(data["y_true_log"]) == 0:
            continue
        yt_log = np.array(data["y_true_log"])
        yp_log = np.array(data["y_pred_log"])
        yt_rank = np.array(data["y_true_rank"])
        yp_rank = np.array(data["y_pred_rank"])
        metrics[name] = {
            "MAE_log": calc_mae(yt_log, yp_log),
            "RMSE_log": calc_rmse(yt_log, yp_log),
            "R2_log": calc_r2(yt_log, yp_log),
            "MAE_rank": calc_mae(yt_rank, yp_rank),
            "RMSE_rank": calc_rmse(yt_rank, yp_rank),
            "R2_rank": calc_r2(yt_rank, yp_rank),
        }
    return metrics


def plot_model_comparison(metrics_df):
    """Bar chart comparing models."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metric_cols = ["MAE_log", "RMSE_log", "R2_log",
                   "MAE_rank", "RMSE_rank", "R2_rank"]
    titles = ["MAE (log scale)", "RMSE (log scale)", "R2 (log scale)",
              "MAE (rank scale)", "RMSE (rank scale)", "R2 (rank scale)"]
    colors = ["#f97316", "#ef4444", "#22c55e", "#f97316", "#ef4444", "#22c55e"]

    for ax, col, title, color in zip(axes.flatten(), metric_cols, titles, colors):
        vals = metrics_df[col].values
        names = metrics_df.index.tolist()
        bars = ax.barh(names, vals, color=color, alpha=0.8, edgecolor="#30363d")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.2, axis="x")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + abs(bar.get_width()) * 0.02,
                    bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=8, color="#c9d1d9")

    plt.suptitle("Model Comparison -- Leave-One-Year-Out CV (Median Quantile)",
                 fontsize=14, color="#58a6ff")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "07_model_comparison.png"),
                bbox_inches="tight")
    plt.close()
    print("  [OK] Model comparison chart")


def plot_residuals(results, best_model_name):
    """Residual analysis for the best model."""
    data = results[best_model_name]
    yt = np.array(data["y_true_log"])
    yp = np.array(data["y_pred_log"])
    residuals = yt - yp

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(yt, yp, alpha=0.3, s=8, color="#58a6ff")
    axes[0].plot([yt.min(), yt.max()], [yt.min(), yt.max()],
                 "--", color="#f97316", linewidth=1.5)
    axes[0].set_xlabel("Actual log(Rank)")
    axes[0].set_ylabel("Predicted log(Rank)")
    axes[0].set_title(f"Predicted vs Actual -- {best_model_name}")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=50, color="#22c55e", alpha=0.7,
                 edgecolor="#30363d")
    axes[1].set_xlabel("Residual (actual - predicted)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")
    axes[1].axvline(0, color="#f97316", linewidth=1.5, linestyle="--")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(yp, residuals, alpha=0.3, s=8, color="#a855f7")
    axes[2].axhline(0, color="#f97316", linewidth=1.5, linestyle="--")
    axes[2].set_xlabel("Predicted log(Rank)")
    axes[2].set_ylabel("Residual")
    axes[2].set_title("Residuals vs Predicted")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"Residual Analysis -- {best_model_name}",
                 fontsize=13, color="#58a6ff")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "08_residual_analysis.png"),
                bbox_inches="tight")
    plt.close()
    print("  [OK] Residual analysis plot")


def plot_quantile_prediction_year(df, best_model, test_year=2024):
    """Show prediction brackets for a held-out year."""
    train_df = df[df["Year"] != test_year]
    test_df = df[df["Year"] == test_year].sort_values("pct_score", ascending=False)

    if len(test_df) < 5:
        print(f"  [!] Not enough data for year {test_year}")
        return

    X_train = train_df[FEATURES].values
    y_train = train_df["log_rank"].values
    X_test = test_df[FEATURES].values

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(test_df["pct_score"], test_df["AIR"],
            "o-", color="#ffffff", markersize=3, linewidth=1.5,
            label=f"Actual {test_year}", zorder=5)

    q_config = [
        ("optimistic", 0.10, "#22c55e", "Optimistic (10th %ile - tough paper)"),
        ("expected",   0.50, "#3b82f6", "Expected (50th %ile - median)"),
        ("pessimistic", 0.90, "#ef4444", "Pessimistic (90th %ile - easy paper)"),
    ]

    pred_data = {}
    for q_name, q_val, color, label in q_config:
        y_pred_log = train_and_predict(best_model, X_train, y_train,
                                        X_test, q_val)
        y_pred_rank = np.exp(y_pred_log)
        pred_data[q_name] = y_pred_rank
        ax.plot(test_df["pct_score"].values, y_pred_rank,
                "--", color=color, linewidth=1.5, alpha=0.9, label=label)

    if "optimistic" in pred_data and "pessimistic" in pred_data:
        ax.fill_between(
            test_df["pct_score"].values,
            pred_data["optimistic"],
            pred_data["pessimistic"],
            alpha=0.12, color="#3b82f6", label="Prediction Range"
        )

    ax.set_xlabel("Percentage Score (%)")
    ax.set_ylabel("AIR (Rank)")
    ax.set_title(f"Quantile Prediction Brackets -- {test_year} (Held Out)")
    ax.set_yscale("log")
    ax.legend(fontsize=9, framealpha=0.3, edgecolor="#30363d")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "09_prediction_brackets.png"),
                bbox_inches="tight")
    plt.close()
    print(f"  [OK] Prediction brackets for {test_year}")


def plot_per_year_error(df, best_model):
    """MAE per held-out year."""
    years = sorted(df["Year"].unique())
    year_errors = []

    for hold_year in years:
        train_df = df[df["Year"] != hold_year]
        test_df = df[df["Year"] == hold_year]
        if len(test_df) < 5:
            continue

        X_train = train_df[FEATURES].values
        y_train = train_df["log_rank"].values
        X_test = test_df[FEATURES].values
        y_true = test_df["log_rank"].values

        y_pred = train_and_predict(best_model, X_train, y_train, X_test, 0.5)
        year_errors.append({
            "Year": hold_year,
            "MAE_log": calc_mae(y_true, y_pred),
            "R2_log": calc_r2(y_true, y_pred),
            "n_samples": len(test_df),
        })

    err_df = pd.DataFrame(year_errors)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(err_df)))
    bars = axes[0].bar(err_df["Year"].astype(str), err_df["MAE_log"],
                       color=colors, edgecolor="#30363d")
    axes[0].set_title(f"MAE (log) per Year -- {best_model}")
    axes[0].set_ylabel("MAE (log scale)")
    axes[0].grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, err_df["MAE_log"]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                     color="#c9d1d9")
    plt.sca(axes[0])
    plt.xticks(rotation=45)

    colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(err_df)))
    bars2 = axes[1].bar(err_df["Year"].astype(str), err_df["R2_log"],
                        color=colors2, edgecolor="#30363d")
    axes[1].set_title(f"R2 (log) per Year -- {best_model}")
    axes[1].set_ylabel("R2 Score")
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars2, err_df["R2_log"]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                     color="#c9d1d9")
    plt.sca(axes[1])
    plt.xticks(rotation=45)

    plt.suptitle("Per-Year Error Analysis (Leave-One-Year-Out)",
                 fontsize=13, color="#58a6ff")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "10_per_year_error.png"),
                bbox_inches="tight")
    plt.close()
    print("  [OK] Per-year error analysis")

    return err_df


def main():
    print("=" * 60)
    print("Phase 3.2: Model Evaluation & Comparison")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded {len(df)} records")
    print(f"Years available: {sorted(df['Year'].unique())}")

    model_names = ["lightgbm", "xgboost", "mlp", "polynomial"]

    print(f"\n{'--'*25}")
    print("Running Leave-One-Year-Out CV (median quantile)...")
    print("This trains each model 17 times (once per held-out year).")
    print(f"{'--'*25}")

    results = leave_one_year_out_cv(df, model_names, quantile=0.5)
    metrics = compute_metrics(results)

    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df.sort_values("R2_log", ascending=False)

    print(f"\n{'='*70}")
    print("MODEL COMPARISON -- Leave-One-Year-Out CV (Median Quantile)")
    print(f"{'='*70}")
    print(metrics_df.round(4).to_string())

    metrics_df.round(6).to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))

    best_model = metrics_df.index[0]
    print(f"\n* Best model (by R2 on log scale): {best_model}")
    print(f"  R2 = {metrics_df.loc[best_model, 'R2_log']:.4f}")
    print(f"  MAE (log) = {metrics_df.loc[best_model, 'MAE_log']:.4f}")
    print(f"  RMSE (log) = {metrics_df.loc[best_model, 'RMSE_log']:.4f}")
    print(f"  MAE (rank) = {metrics_df.loc[best_model, 'MAE_rank']:.1f}")

    print(f"\n{'--'*25}")
    print(f"Evaluating {best_model} across all quantiles...")
    print(f"{'--'*25}")

    quantile_results = {}
    for q_name, q_val in [("optimistic", 0.10), ("expected", 0.50),
                           ("pessimistic", 0.90)]:
        res = leave_one_year_out_cv(df, [best_model], quantile=q_val)
        met = compute_metrics(res)
        quantile_results[q_name] = met.get(best_model, {})
        print(f"  {q_name} (alpha={q_val}):")
        if best_model in met:
            for k, v in met[best_model].items():
                print(f"    {k}: {v:.4f}")

    q_df = pd.DataFrame(quantile_results).T
    q_df.round(6).to_csv(os.path.join(RESULTS_DIR, "quantile_evaluation.csv"))

    print(f"\nGenerating evaluation plots...")
    plot_model_comparison(metrics_df)
    plot_residuals(results, best_model)
    plot_quantile_prediction_year(df, best_model, test_year=2024)
    per_year_df = plot_per_year_error(df, best_model)
    per_year_df.to_csv(os.path.join(RESULTS_DIR, "per_year_errors.csv"),
                       index=False)

    with open(os.path.join(RESULTS_DIR, "best_model.txt"), "w") as f:
        f.write(best_model)

    print(f"\n{'='*50}")
    print(f"All results saved to: {RESULTS_DIR}")
    print(f"Best model: {best_model}")

    return metrics_df


if __name__ == "__main__":
    main()
