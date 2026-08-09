"""
Phase 2.1 — Feature Engineering
=================================
Builds features for the ML models:
  1. Percentage Score (marks / max_marks)
  2. Log-Rank target (log(AIR))
  3. Historical Quantile Aggregation (best/worst/median rank per 1% score bin)

Input:  interpolated_data.csv
Output: features_data.csv
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(BASE_DIR, "interpolated_data.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "features_data.csv")


def compute_percentage_score(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute percentage score for consistency."""
    df = df.copy()
    df["pct_score"] = (df["Mark"] / df["MaxMarks"]) * 100
    return df


def compute_log_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log transformation to the rank target."""
    df = df.copy()
    df["log_rank"] = np.log(df["AIR"])
    return df


def compute_quantile_aggregation(df: pd.DataFrame, bin_width: float = 1.0) -> pd.DataFrame:
    """
    For each 1% bin of percentage score, compute:
      - best_rank:   min rank across all years (optimistic — tough year)
      - worst_rank:  max rank across all years (pessimistic — easy year)
      - median_rank: median rank across all years
      - log versions of each

    These features capture the historical range of outcomes for a given score level.
    """
    df = df.copy()

    # Create percentage bins (0%, 1%, 2%, ..., 100%)
    df["pct_bin"] = (df["pct_score"] // bin_width) * bin_width

    # Compute aggregations per bin
    agg = df.groupby("pct_bin").agg(
        best_rank=("AIR", "min"),     # lowest rank (best outcome)
        worst_rank=("AIR", "max"),    # highest rank (worst outcome)
        median_rank=("AIR", "median"),
        mean_rank=("AIR", "mean"),
        count=("AIR", "count"),
    ).reset_index()

    # Log-transform the aggregated ranks
    agg["log_best_rank"] = np.log(agg["best_rank"].clip(lower=1))
    agg["log_worst_rank"] = np.log(agg["worst_rank"].clip(lower=1))
    agg["log_median_rank"] = np.log(agg["median_rank"].clip(lower=1))

    # Merge back into the main DataFrame
    df = df.merge(agg, on="pct_bin", how="left", suffixes=("", "_agg"))

    print(f"\n  Quantile aggregation table (sample):")
    print(agg[agg["count"] >= 3].head(20).to_string(index=False))

    return df


def main():
    print("=" * 60)
    print("Phase 2.1: Feature Engineering")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded {len(df)} records")

    # Step 1: Percentage Score
    df = compute_percentage_score(df)
    print(f"\n1. Percentage Score computed")
    print(f"   Range: {df['pct_score'].min():.2f}% — {df['pct_score'].max():.2f}%")

    # Step 2: Log-Rank Target
    df = compute_log_rank(df)
    print(f"\n2. Log-Rank target computed")
    print(f"   log(AIR) range: {df['log_rank'].min():.2f} — {df['log_rank'].max():.2f}")
    print(f"   (corresponds to AIR {df['AIR'].min()} — {df['AIR'].max()})")

    # Step 3: Historical Quantile Aggregation
    print(f"\n3. Historical Quantile Aggregation (1% bins):")
    df = compute_quantile_aggregation(df, bin_width=1.0)

    # Summary
    print(f"\n{'='*40}")
    print(f"Final feature set:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Feature statistics:")
    print(df[["pct_score", "log_rank", "best_rank", "worst_rank",
              "median_rank"]].describe().round(2).to_string())

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved features to: {OUTPUT_FILE}")

    return df


if __name__ == "__main__":
    main()
