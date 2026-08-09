"""
Phase 1.2 — Missing Value Interpolation
========================================
Uses Monotonic Cubic Spline Interpolation (PCHIP) grouped by year to fill
missing rank-mark pairs. Validates monotonicity after interpolation.

Input:  cleaned_data.csv
Output: interpolated_data.csv
"""

import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
import os
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(BASE_DIR, "cleaned_data.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "interpolated_data.csv")


def interpolate_year(year_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Interpolate missing marks for a single year using PCHIP (monotonic cubic).
    Falls back to linear interpolation if insufficient data points.

    The relationship is: as AIR increases, Mark decreases (strictly monotonic).
    """
    df = year_df.copy().sort_values("AIR").reset_index(drop=True)

    # Separate known and missing
    known = df[df["Mark"].notna()].copy()
    missing = df[df["Mark"].isna()].copy()

    if len(missing) == 0:
        return df  # nothing to interpolate

    if len(known) < 3:
        print(f"  [WARN] Year {year}: Only {len(known)} known points -- skipping interpolation")
        return df

    # Use PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
    # This preserves monotonicity
    x_known = known["AIR"].values.astype(float)
    y_known = known["Mark"].values.astype(float)

    try:
        interpolator = PchipInterpolator(x_known, y_known)

        max_marks = df["MaxMarks"].iloc[0]
        filled_count = 0

        for idx in df.index:
            if pd.isna(df.loc[idx, "Mark"]):
                air_val = float(df.loc[idx, "AIR"])
                # Only interpolate within the range of known data (no extrapolation)
                if x_known.min() <= air_val <= x_known.max():
                    interp_mark = float(interpolator(air_val))
                    df.loc[idx, "Mark"] = round(interp_mark, 1)
                    df.loc[idx, "Percentage"] = round((interp_mark / max_marks) * 100, 2)
                    filled_count += 1

        return df

    except Exception as e:
        print(f"  [WARN] Year {year}: PCHIP failed ({e}), using linear interpolation")
        # Fallback: set marks as index, interpolate
        df = df.set_index("AIR").sort_index()
        df["Mark"] = df["Mark"].interpolate(method="index")
        max_marks = df["MaxMarks"].iloc[0]
        df["Percentage"] = (df["Mark"] / max_marks) * 100
        df = df.reset_index()
        return df


def validate_monotonicity(df: pd.DataFrame) -> dict:
    """
    Check that for each year, marks strictly decrease as rank increases.
    Returns a dict of {year: n_violations}.
    """
    violations = {}
    for year in sorted(df["Year"].unique()):
        year_df = df[(df["Year"] == year) & (df["Mark"].notna())].sort_values("AIR")
        marks = year_df["Mark"].values
        # Marks should be non-increasing
        n_violations = int(np.sum(np.diff(marks) > 0))
        if n_violations > 0:
            violations[year] = n_violations
    return violations


def main():
    print("=" * 60)
    print("Phase 1.2: Missing Value Interpolation")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded {len(df)} records from cleaned_data.csv")
    total_missing_before = int(df["Mark"].isna().sum())
    print(f"Total missing marks: {total_missing_before}")

    # Interpolate year by year
    interpolated_dfs = []
    for year in sorted(df["Year"].unique()):
        year_df = df[df["Year"] == year].copy()
        n_missing_before = int(year_df["Mark"].isna().sum())
        year_df = interpolate_year(year_df, year)
        n_missing_after = int(year_df["Mark"].isna().sum())
        n_filled = n_missing_before - n_missing_after
        print(f"  Year {year}: {n_filled} values interpolated "
              f"(remaining missing: {n_missing_after})")
        interpolated_dfs.append(year_df)

    result = pd.concat(interpolated_dfs, ignore_index=True)

    # Validate monotonicity
    print("\nMonotonicity validation:")
    violations = validate_monotonicity(result)
    if violations:
        print(f"  Violations found: {violations}")
        # Fix minor violations by ensuring non-increasing with cummin
        for year in violations:
            mask = result["Year"] == year
            year_sorted = result.loc[mask].sort_values("AIR")
            # Apply cumulative minimum to enforce monotonicity
            fixed_marks = year_sorted["Mark"].cummin()
            result.loc[year_sorted.index, "Mark"] = fixed_marks
            result.loc[year_sorted.index, "Percentage"] = (
                fixed_marks / result.loc[year_sorted.index, "MaxMarks"] * 100
            ).round(2)
        print("  Fixed violations using cumulative minimum.")
        # Re-validate
        violations_after = validate_monotonicity(result)
        if violations_after:
            print(f"  Violations after fix: {violations_after}")
        else:
            print("  All violations fixed [OK]")
    else:
        print("  All years pass monotonicity check [OK]")

    # Summary
    total_missing_after = int(result["Mark"].isna().sum())
    total_filled = total_missing_before - total_missing_after
    print(f"\nFinal dataset:")
    print(f"  Total records: {len(result)}")
    print(f"  Values interpolated: {total_filled}")
    print(f"  Remaining NaN marks: {total_missing_after}")
    print(f"  Records with valid marks: {int(result['Mark'].notna().sum())}")

    # Drop rows that are still NaN (outside interpolation range)
    result_clean = result.dropna(subset=["Mark"]).reset_index(drop=True)
    print(f"  After dropping uninterpolatable rows: {len(result_clean)}")

    result_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved interpolated data to: {OUTPUT_FILE}")

    return result_clean


if __name__ == "__main__":
    main()
