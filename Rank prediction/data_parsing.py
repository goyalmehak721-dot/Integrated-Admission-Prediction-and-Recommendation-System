"""
Phase 1.1 — Data Parsing & Reshaping
=====================================
Parses the complex wide-format JEE Advanced CSV into a clean long-format DataFrame.

Output columns: Year, MaxMarks, AIR, Mark, Percentage
Saves: cleaned_data.csv
"""

import pandas as pd
import numpy as np
import re
import os

CSV_FILE = os.path.join(
    os.path.dirname(__file__),
    "JEE Advanced Rank vs Marks 2008-2025.xlsx - JEE Advanced Rank vs Marks 2008-2025.csv"
)
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "cleaned_data.csv")


def parse_header(filepath: str):
    """
    Parse the dual-row header to extract year info and column mapping.
    Row 1: Year, 2025 Estimation (360), , 2024 (360), , ...
    Row 2: AIR, Mark, Percentage, Mark, Percentage, ...
    """
    with open(filepath, "r", encoding="utf-8") as f:
        header_line = f.readline().strip().rstrip(",")
        sub_header_line = f.readline().strip().rstrip(",")

    # Split headers
    headers = header_line.split(",")
    sub_headers = sub_header_line.split(",")

    # Parse year blocks from the header row
    # Format: "2024 (360)" or "2025 Estimation (360)"
    year_blocks = []  # list of (year, max_marks, col_start_index)
    i = 1  # skip the first "Year" column
    while i < len(headers):
        cell = headers[i].strip()
        if cell == "" or cell == "Year":
            i += 1
            continue
        # Try to parse year and max marks
        match = re.search(r"(\d{4}).*?\((\d+)\)", cell)
        if match:
            year = int(match.group(1))
            max_marks = int(match.group(2))
            year_blocks.append((year, max_marks, i))
            i += 2  # skip Mark and Percentage columns
        else:
            i += 1

    return year_blocks


def parse_data(filepath: str, year_blocks: list) -> pd.DataFrame:
    """
    Read data rows and reshape into long format.
    """
    # Read all data rows (skip the 2-row header)
    raw_df = pd.read_csv(filepath, header=None, skiprows=2, dtype=str)

    records = []
    for _, row in raw_df.iterrows():
        # AIR is in column 0
        air_str = str(row.iloc[0]).strip()
        if air_str == "" or air_str == "nan":
            continue
        try:
            air = int(air_str)
        except ValueError:
            continue

        for year, max_marks, col_start in year_blocks:
            # Skip 2025 estimation
            if year == 2025:
                continue

            mark_str = str(row.iloc[col_start]).strip() if col_start < len(row) else "-"
            pct_str = str(row.iloc[col_start + 1]).strip() if (col_start + 1) < len(row) else "-"

            # Handle missing values
            if mark_str in ["-", "", "nan", "None"]:
                mark = np.nan
            else:
                try:
                    mark = float(mark_str)
                except ValueError:
                    mark = np.nan

            if pct_str in ["-", "", "nan", "None"]:
                pct = np.nan
            else:
                # Remove % sign if present
                pct_str = pct_str.replace("%", "")
                try:
                    pct = float(pct_str)
                except ValueError:
                    pct = np.nan

            records.append({
                "Year": year,
                "MaxMarks": max_marks,
                "AIR": air,
                "Mark": mark,
                "Percentage": pct,
            })

    df = pd.DataFrame(records)
    return df


def main():
    print("=" * 60)
    print("Phase 1.1: Data Parsing & Reshaping")
    print("=" * 60)

    # 1. Parse header
    year_blocks = parse_header(CSV_FILE)
    print(f"\nDetected {len(year_blocks)} year columns:")
    for year, max_marks, col_idx in year_blocks:
        label = " (EXCLUDED — estimation)" if year == 2025 else ""
        print(f"  {year} — Max Marks: {max_marks}, Col Index: {col_idx}{label}")

    # Filter out 2025
    training_blocks = [(y, m, c) for y, m, c in year_blocks if y != 2025]
    print(f"\nTraining years: {len(training_blocks)}")

    # 2. Parse data rows into long format
    df = parse_data(CSV_FILE, year_blocks)

    # 3. Summary statistics
    print(f"\nTotal records: {len(df)}")
    print(f"Years: {sorted(df['Year'].unique())}")
    print(f"AIR range: {df['AIR'].min()} — {df['AIR'].max()}")

    # Data availability by year
    print("\nData availability per year (non-null marks):")
    for year in sorted(df["Year"].unique()):
        year_df = df[df["Year"] == year]
        total = len(year_df)
        non_null = year_df["Mark"].notna().sum()
        null_count = year_df["Mark"].isna().sum()
        air_max = year_df.loc[year_df["Mark"].notna(), "AIR"].max() if non_null > 0 else 0
        print(f"  {year}: {non_null}/{total} rows with data "
              f"(missing: {null_count}), max AIR with data: {air_max}")

    # 4. Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved cleaned long-format data to: {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")
    print(f"\nSample rows:")
    print(df.head(20).to_string(index=False))

    return df


if __name__ == "__main__":
    main()
