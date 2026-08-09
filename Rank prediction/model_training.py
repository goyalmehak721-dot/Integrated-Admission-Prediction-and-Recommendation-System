"""
Phase 3.1 -- Model Training (Quantile Regression)
==================================================
Trains multiple ML models with quantile regression to output rank brackets:
  - Optimistic (10th percentile -- tough year)
  - Expected   (50th percentile -- median year)
  - Pessimistic (90th percentile -- easy year)

Models: LightGBM (native), XGBoost (native), Custom MLP (numpy),
        Polynomial Quantile Regression (numpy)

All implementations use native APIs -- no sklearn dependency.

Input:  features_data.csv
Output: trained models saved to models/ directory
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
import joblib
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(BASE_DIR, "features_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

QUANTILES = {
    "optimistic": 0.10,
    "expected": 0.50,
    "pessimistic": 0.90,
}

FEATURES = ["pct_score", "log_best_rank", "log_worst_rank", "log_median_rank"]


def pinball_loss(y_true, y_pred, quantile):
    """Pinball (quantile) loss function."""
    delta = y_true - y_pred
    return np.mean(np.where(delta >= 0, quantile * delta, (quantile - 1) * delta))


# ─── LightGBM (Native API) ─────────────────────────────────────────────────────

class LightGBMNative:
    """LightGBM using native API with quantile objective."""

    def __init__(self, quantile=0.5):
        self.quantile = quantile
        self.model = None

    def fit(self, X, y):
        dtrain = lgb.Dataset(X, label=y)
        params = {
            "objective": "quantile",
            "alpha": self.quantile,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "seed": 42,
            "n_jobs": -1,
        }
        self.model = lgb.train(params, dtrain, num_boost_round=500)
        return self

    def predict(self, X):
        return self.model.predict(X)


# ─── XGBoost (Native API) ──────────────────────────────────────────────────────

class XGBoostNative:
    """XGBoost using native API with quantile error objective."""

    def __init__(self, quantile=0.5):
        self.quantile = quantile
        self.model = None

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": self.quantile,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": 0,
            "seed": 42,
        }
        self.model = xgb.train(params, dtrain, num_boost_round=500)
        return self

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)


# ─── Custom MLP with Pinball Loss (Pure NumPy) ─────────────────────────────────

class NumpyMLP:
    """
    Multi-Layer Perceptron in pure NumPy with pinball (quantile) loss.
    Architecture configurable via layer_sizes parameter.
    """
    def __init__(self, layer_sizes, quantile=0.5, lr=0.001, epochs=500,
                 batch_size=64, seed=42):
        self.layer_sizes = layer_sizes
        self.quantile = quantile
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.weights = []
        self.biases = []
        self.mean_ = None
        self.std_ = None

    def _init_weights(self, input_dim):
        dims = [input_dim] + self.layer_sizes + [1]
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            w = self.rng.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros((1, dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_deriv(self, x):
        return (x > 0).astype(float)

    def _forward(self, X):
        activations = [X]
        pre_activations = []
        a = X
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            a = self._relu(z)
            activations.append(a)
        z = a @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z)
        activations.append(z)
        return activations, pre_activations

    def _pinball_grad(self, y_true, y_pred):
        delta = y_true - y_pred
        grad = np.where(delta >= 0, -self.quantile, 1 - self.quantile)
        return grad / len(y_true)

    def fit(self, X, y):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        X_scaled = (X - self.mean_) / self.std_
        y = y.reshape(-1, 1)

        self._init_weights(X_scaled.shape[1])
        n = len(X_scaled)

        for epoch in range(self.epochs):
            idx = self.rng.permutation(n)
            X_shuf = X_scaled[idx]
            y_shuf = y[idx]

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]

                activations, pre_activations = self._forward(X_batch)
                y_pred = activations[-1]
                d_out = self._pinball_grad(y_batch, y_pred)

                grads_w = []
                grads_b = []
                delta = d_out
                for i in range(len(self.weights) - 1, -1, -1):
                    a_prev = activations[i]
                    dw = a_prev.T @ delta
                    db = delta.sum(axis=0, keepdims=True)
                    grads_w.insert(0, dw)
                    grads_b.insert(0, db)
                    if i > 0:
                        delta = (delta @ self.weights[i].T) * \
                                self._relu_deriv(pre_activations[i-1])

                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * grads_w[i]
                    self.biases[i] -= self.lr * grads_b[i]

        return self

    def predict(self, X):
        X_scaled = (X - self.mean_) / self.std_
        activations, _ = self._forward(X_scaled)
        return activations[-1].flatten()


# ─── Polynomial Quantile Regression ────────────────────────────────────────────

class PolynomialQuantileRegressor:
    """
    Polynomial regression with quantile loss optimization via IRLS.
    """
    def __init__(self, degree=4, quantile=0.5, max_iter=100):
        self.degree = degree
        self.quantile = quantile
        self.max_iter = max_iter
        self.coeffs = None
        self.mean_ = None
        self.std_ = None

    def _poly_features(self, X):
        features = [np.ones((len(X), 1))]
        for d in range(1, self.degree + 1):
            features.append(X ** d)
        n_features = X.shape[1]
        for i in range(n_features):
            for j in range(i+1, n_features):
                features.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack(features)

    def fit(self, X, y):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        X_scaled = (X - self.mean_) / self.std_
        X_poly = self._poly_features(X_scaled)
        y = y.flatten()

        self.coeffs = np.linalg.lstsq(X_poly, y, rcond=None)[0]

        for _ in range(self.max_iter):
            y_pred = X_poly @ self.coeffs
            residuals = y - y_pred
            weights = np.where(residuals >= 0, self.quantile, 1 - self.quantile)
            weights = weights / (np.abs(residuals) + 1e-6)
            W = np.diag(weights)
            try:
                self.coeffs = np.linalg.solve(
                    X_poly.T @ W @ X_poly + 1e-6 * np.eye(X_poly.shape[1]),
                    X_poly.T @ W @ y
                )
            except np.linalg.LinAlgError:
                break

        return self

    def predict(self, X):
        X_scaled = (X - self.mean_) / self.std_
        X_poly = self._poly_features(X_scaled)
        return X_poly @ self.coeffs


# ─── Main Training Pipeline ────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 3.1: Model Training (Quantile Regression)")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded {len(df)} records with {len(df.columns)} columns")

    X = df[FEATURES].values
    y = df["log_rank"].values

    print(f"Features: {FEATURES}")
    print(f"Target: log_rank (range {y.min():.2f} -- {y.max():.2f})")
    print(f"Training samples: {len(X)}")

    all_models = {}

    for q_name, q_val in QUANTILES.items():
        print(f"\n{'--'*25}")
        print(f"Training for quantile: {q_name} (alpha={q_val})")
        print(f"{'--'*25}")

        models = {}

        # 1. LightGBM (Native)
        print("  Training LightGBM...", end=" ")
        lgbm = LightGBMNative(quantile=q_val)
        lgbm.fit(X, y)
        models["lightgbm"] = lgbm
        preds = lgbm.predict(X)
        tl = pinball_loss(y, preds, q_val)
        print(f"[OK] (train pinball: {tl:.4f})")

        # 2. XGBoost (Native)
        print("  Training XGBoost...", end=" ")
        xgb_m = XGBoostNative(quantile=q_val)
        xgb_m.fit(X, y)
        models["xgboost"] = xgb_m
        preds = xgb_m.predict(X)
        tl = pinball_loss(y, preds, q_val)
        print(f"[OK] (train pinball: {tl:.4f})")

        # 3. Custom MLP
        print("  Training MLP...", end=" ")
        mlp = NumpyMLP(
            layer_sizes=[64, 32, 16],
            quantile=q_val,
            lr=0.001,
            epochs=300,
            batch_size=64,
        )
        mlp.fit(X, y)
        models["mlp"] = mlp
        preds = mlp.predict(X)
        tl = pinball_loss(y, preds, q_val)
        print(f"[OK] (train pinball: {tl:.4f})")

        # 4. Polynomial Quantile Regression
        print("  Training Polynomial QR...", end=" ")
        poly = PolynomialQuantileRegressor(degree=3, quantile=q_val, max_iter=100)
        poly.fit(X, y)
        models["polynomial"] = poly
        preds = poly.predict(X)
        tl = pinball_loss(y, preds, q_val)
        print(f"[OK] (train pinball: {tl:.4f})")

        all_models[q_name] = models

    # Save everything
    joblib.dump(all_models, os.path.join(MODELS_DIR, "all_models.pkl"))

    config = {
        "features": FEATURES,
        "quantiles": QUANTILES,
        "target": "log_rank",
    }
    with open(os.path.join(MODELS_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    feature_stats = {
        "mean": {k: float(v) for k, v in df[FEATURES].mean().to_dict().items()},
        "std": {k: float(v) for k, v in df[FEATURES].std().to_dict().items()},
        "min": {k: float(v) for k, v in df[FEATURES].min().to_dict().items()},
        "max": {k: float(v) for k, v in df[FEATURES].max().to_dict().items()},
    }
    with open(os.path.join(MODELS_DIR, "feature_stats.json"), "w") as f:
        json.dump(feature_stats, f, indent=2)

    agg_cols = ["pct_bin", "log_best_rank", "log_worst_rank", "log_median_rank",
                "best_rank", "worst_rank", "median_rank"]
    available_cols = [c for c in agg_cols if c in df.columns]
    agg_table = df[available_cols].drop_duplicates("pct_bin").sort_values("pct_bin")
    agg_table.to_csv(os.path.join(MODELS_DIR, "quantile_lookup.csv"), index=False)

    print(f"\n{'='*50}")
    print(f"All models saved to: {MODELS_DIR}")
    print(f"  - all_models.pkl")
    print(f"  - config.json")
    print(f"  - feature_stats.json")
    print(f"  - quantile_lookup.csv")

    return all_models


if __name__ == "__main__":
    main()
