from __future__ import annotations

import os
from pathlib import Path
import glob
import pandas as pd


# project root = folder where main.py lives (one level above /src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_and_merge_data():
    data_path = PROJECT_ROOT / "data" / "raw"
    files = list(data_path.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in: {data_path}\n"
            f"Put your Serie A season CSVs there (e.g., I1_2012.csv, ...)."
        )

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Keep only required columns
    needed_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}.\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[needed_cols]
    df.columns = ["date", "home_team", "away_team", "home_goals", "away_goals"]

    # Try common formats in football-data.co.uk CSVs
    dt = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
    dt2 = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["date"] = dt.fillna(dt2)

# If still NaT, fallback (rare)
    mask = df["date"].isna()
    if mask.any():
        df.loc[mask, "date"] = pd.to_datetime(df.loc[mask, "date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date")

        return df


def train_test_split(df: pd.DataFrame, split_date="2023-07-01"):
    train = df[df["date"] < split_date]
    test = df[df["date"] >= split_date]
    return train, test