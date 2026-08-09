"""
JoSAA Rank-to-College Prediction Engine (IIT-Only)
====================================================
High-performance data ingestion, filtering, and probabilistic prediction
engine for IIT admissions using JEE Advanced ranks from JoSAA counseling
data (2016–2020).
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import List, Dict

# ──────────────────────────────────────────────────────────────
# Institute name normalization (resolve historical changes)
# ──────────────────────────────────────────────────────────────
INSTITUTE_NAME_NORMALIZATION = {
    "Indian School of Mines Dhanbad": "Indian Institute of Technology (ISM) Dhanbad",
}


def is_iit(name: str) -> bool:
    """Check if an institute is an IIT."""
    return "indian institute of technology" in name.lower()


class DataLoader:
    """
    Scans the data/ directory, ingests all CSVs, sanitizes and normalizes
    into a single high-performance DataFrame filtered to IITs only.
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.data_dir = data_dir
        self.df = self._load_and_sanitize()

    def _load_and_sanitize(self) -> pd.DataFrame:
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        frames = []
        for fpath in csv_files:
            try:
                chunk = pd.read_csv(fpath, dtype=str, encoding="utf-8")
                frames.append(chunk)
            except Exception as e:
                print(f"[WARN] Skipping {fpath}: {e}")

        df = pd.concat(frames, ignore_index=True)
        print(f"[INFO] Loaded {len(df)} raw rows from {len(csv_files)} files")

        # ── Strip whitespace from all string columns ──
        str_cols = ["Institute", "Academic Program Name", "Quota", "Seat Type", "Gender"]
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()

        # ── Normalize institute names ──
        df["Institute"] = df["Institute"].replace(INSTITUTE_NAME_NORMALIZATION)

        # ── Keep only IITs ──
        df = df[df["Institute"].apply(is_iit)].copy()
        print(f"[INFO] After IIT filter: {len(df)} rows")

        # ── Keep only AI (All India) quota ──
        df = df[df["Quota"] == "AI"].copy()

        # ── Drop PwD rank rows (contain 'P' suffix — separate ranking list) ──
        pwd_mask = (
            df["Opening Rank"].str.contains("P", na=False) |
            df["Closing Rank"].str.contains("P", na=False)
        )
        df = df[~pwd_mask].copy()

        # ── Drop rows with missing / non-numeric ranks ──
        df["Opening Rank"] = pd.to_numeric(df["Opening Rank"], errors="coerce")
        df["Closing Rank"] = pd.to_numeric(df["Closing Rank"], errors="coerce")
        df = df.dropna(subset=["Opening Rank", "Closing Rank"]).copy()

        # ── Cast numeric columns to int ──
        df["Opening Rank"] = df["Opening Rank"].astype(int)
        df["Closing Rank"] = df["Closing Rank"].astype(int)
        df["Year"] = df["Year"].astype(int)
        df["Round"] = df["Round"].astype(int)

        # ── Normalize empty Gender (2016–2017) → Gender-Neutral ──
        df["Gender"] = df["Gender"].replace({"": "Gender-Neutral", "nan": "Gender-Neutral"})
        df.loc[df["Gender"].isna(), "Gender"] = "Gender-Neutral"

        print(f"[INFO] After full sanitization: {len(df)} rows, {df['Institute'].nunique()} IITs")
        return df

    def get_metadata(self) -> Dict:
        """Return available filter options for the frontend."""
        categories = sorted(self.df["Seat Type"].unique().tolist())
        genders = sorted(self.df["Gender"].unique().tolist())
        years = sorted(self.df["Year"].unique().tolist())
        iits = sorted(self.df["Institute"].unique().tolist())
        return {
            "categories": categories,
            "genders": genders,
            "years": years,
            "iits": iits,
        }


class PredictionEngine:
    """
    Core query and prediction logic for IIT admissions.
    Accepts user inputs (rank, category, gender), applies categorical
    masks, then produces probabilistic admission predictions.
    """

    def __init__(self, data_loader: DataLoader = None):
        if data_loader is None:
            data_loader = DataLoader()
        self.loader = data_loader
        self.df = data_loader.df

    def predict(
        self,
        predicted_rank: int,
        category: str,
        gender: str,
    ) -> List[Dict]:
        """
        Run the full prediction pipeline.

        Parameters
        ----------
        predicted_rank : int
            The user's predicted JEE Advanced rank.
        category : str
            Seat Type (e.g., 'OPEN', 'OBC-NCL', 'SC', 'ST', 'EWS').
        gender : str
            'Gender-Neutral' or 'Female-only (including Supernumerary)'.

        Returns
        -------
        List[Dict]
            Sorted list of program predictions with risk tiers.
        """
        df = self.df.copy()

        # ── Step 1: Seat Type filter ──
        df = df[df["Seat Type"] == category]

        # ── Step 2: Gender filter ──
        # Female candidates can sit in both Gender-Neutral and Female-only seats
        # Male / Gender-Neutral candidates only see Gender-Neutral seats
        if gender == "Female-only (including Supernumerary)":
            df = df[df["Gender"].isin(["Gender-Neutral", "Female-only (including Supernumerary)"])]
        else:
            df = df[df["Gender"] == "Gender-Neutral"]

        if df.empty:
            return []

        # ── Step 3: Rank eligibility — keep rows where rank <= Closing Rank ──
        eligible = df[df["Closing Rank"] >= predicted_rank]

        # ── Step 4: Compute per-program stats from last round of each year ──
        last_round_df = df.sort_values(["Year", "Round"])
        last_round_per_year = last_round_df.groupby(
            ["Institute", "Academic Program Name", "Year"]
        ).last().reset_index()

        program_stats = last_round_per_year.groupby(
            ["Institute", "Academic Program Name"]
        ).agg(
            min_closing=("Closing Rank", "min"),
            max_closing=("Closing Rank", "max"),
            avg_closing=("Closing Rank", "mean"),
            years_available=("Year", "nunique"),
        ).reset_index()

        # ── Step 5: Compute earliest admission round from eligible rows ──
        if len(eligible) > 0:
            earliest_round = eligible.groupby(
                ["Institute", "Academic Program Name"]
            ).agg(
                earliest_round=("Round", "min"),
            ).reset_index()
        else:
            earliest_round = pd.DataFrame(
                columns=["Institute", "Academic Program Name", "earliest_round"]
            )

        # ── Step 6: Merge stats and classify risk tiers ──
        results = program_stats.merge(
            earliest_round,
            on=["Institute", "Academic Program Name"],
            how="left",
        )

        max_round = int(last_round_per_year["Round"].max()) if len(last_round_per_year) > 0 else 7

        predictions = []
        for _, row in results.iterrows():
            min_cr = int(row["min_closing"])
            max_cr = int(row["max_closing"])
            avg_cr = round(float(row["avg_closing"]))
            years_avail = int(row["years_available"])
            e_round = int(row["earliest_round"]) if pd.notna(row["earliest_round"]) else None

            # Risk tier classification
            if predicted_rank <= min_cr:
                tier = "High Chance (Safe)"
                tier_code = 1
            elif predicted_rank <= max_cr:
                tier = "Moderate Chance (Realistic)"
                tier_code = 2
            elif predicted_rank <= max_cr * 1.10:
                tier = "Low Chance (Ambitious)"
                tier_code = 3
            else:
                # Beyond 10% relaxation — skip
                continue

            # For ambitious programs without a direct eligible round, estimate last round
            if e_round is None:
                e_round = max_round

            predictions.append({
                "institute": row["Institute"],
                "program": row["Academic Program Name"],
                "risk_tier": tier,
                "risk_tier_code": tier_code,
                "min_closing_rank": min_cr,
                "max_closing_rank": max_cr,
                "avg_closing_rank": avg_cr,
                "earliest_round": e_round,
                "years_of_data": years_avail,
            })

        # ── Step 7: Sort — primary by earliest_round ASC, secondary by avg_closing ASC ──
        predictions.sort(key=lambda x: (x["earliest_round"], x["avg_closing_rank"]))

        return predictions
