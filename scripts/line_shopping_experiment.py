"""Line-shopping experiment: reprice the validation-locked bets at best market odds.

Question: the validation-locked selectors lose money at the *average* opening price
(``AvgH/AvgD/AvgA`` is what the pipeline uses as opening odds). Would the exact same
bets be profitable if executed at the *best available* pre-match price
(``MaxH/MaxD/MaxA`` from football-data.co.uk)?

Leakage/selection safety: nothing is re-selected here. The bet set is reproduced
from the saved per-match test predictions using the thresholds already locked on
validation (``final_bet_selector_*.csv``); only the execution price changes. This
isolates the bookmaker-margin effect from model skill, and reports bootstrap CIs
so a sign flip is not over-interpreted.

Run offline (does not touch any canonical artifact):

    python scripts/line_shopping_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bet_selection import _candidate_bets  # noqa: E402
from src.config import FINAL_CONFIG  # noqa: E402
from src.team_names import normalize_team_name  # noqa: E402

ARTIFACTS = Path("artifacts")
RAW_DIR = Path("data/raw")
BOOTSTRAP_ITERS = 10_000
SEED = 7


def _latest(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["run_ts_utc"] == df["run_ts_utc"].max()].copy()


def load_max_odds() -> pd.DataFrame:
    """Best pre-match 1X2 price per match from the raw football-data CSVs."""
    frames = []
    for league_dir in sorted(RAW_DIR.iterdir()):
        if not league_dir.is_dir():
            continue
        for csv_path in sorted(league_dir.glob("*.csv")):
            if csv_path.stem.endswith("_fixtures"):
                continue
            df = pd.read_csv(csv_path, encoding="latin-1", on_bad_lines="skip")
            triplet = ("MaxH", "MaxD", "MaxA") if "MaxH" in df.columns else ("BbMxH", "BbMxD", "BbMxA")
            if not set(triplet).issubset(df.columns) or "Date" not in df.columns:
                continue
            h, d, a = triplet
            out = pd.DataFrame({
                "league": league_dir.name,
                "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
                "home_team": df["HomeTeam"].map(lambda t: normalize_team_name(t, league_dir.name)),
                "away_team": df["AwayTeam"].map(lambda t: normalize_team_name(t, league_dir.name)),
                "max_odds_home": pd.to_numeric(df[h], errors="coerce"),
                "max_odds_draw": pd.to_numeric(df[d], errors="coerce"),
                "max_odds_away": pd.to_numeric(df[a], errors="coerce"),
            })
            frames.append(out.dropna(subset=["date"]))
    merged = pd.concat(frames, ignore_index=True)
    return merged.drop_duplicates(subset=["league", "date", "home_team", "away_team"], keep="last")


def rebuild_locked_bets(dump: pd.DataFrame, model: str, selector_row: pd.Series) -> pd.DataFrame:
    probs = dump[[f"{model}_p_h", f"{model}_p_d", f"{model}_p_a"]].to_numpy(dtype=float)
    odds = dump[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    y_true = dump["y_true"].to_numpy(dtype=int)
    match_info = dump[[
        "date", "league", "home_team", "away_team",
        "open_odds_home", "open_odds_draw", "open_odds_away",
        "close_odds_home", "close_odds_draw", "close_odds_away",
    ]].to_dict("records")
    return _candidate_bets(
        probs, odds, y_true, match_info,
        edge_threshold=float(selector_row["edge_threshold"]),
        min_probability=float(selector_row["min_probability"]),
        min_odds=float(selector_row["min_odds"]),
        max_odds=float(selector_row["max_odds"]),
    )


def reprice_at_max(bets: pd.DataFrame, max_odds: pd.DataFrame) -> pd.DataFrame:
    bets = bets.copy()
    bets["date"] = pd.to_datetime(bets["date"])
    bets = bets.merge(max_odds, on=["league", "date", "home_team", "away_team"], how="left")
    suffix = bets["pred_choice"].map({0: "home", 1: "draw", 2: "away"})
    max_taken = np.array([
        row[f"max_odds_{s}"] if isinstance(s, str) else np.nan
        for (_, row), s in zip(bets.iterrows(), suffix)
    ], dtype=float)
    # Fall back to the average price where the max price is missing or implausible
    # (a max below the average price for the same match is a data glitch).
    usable = np.isfinite(max_taken) & (max_taken >= bets["odds_taken"].to_numpy(dtype=float))
    bets["max_odds_taken"] = np.where(usable, max_taken, bets["odds_taken"])
    bets["max_price_found"] = usable
    bets["return_max"] = np.where(bets["won"] == 1, bets["stake"] * bets["max_odds_taken"], 0.0)
    bets["profit_max"] = bets["return_max"] - bets["stake"]
    return bets


def roi(profit: np.ndarray, stake: np.ndarray) -> float:
    inv = float(stake.sum())
    return float(profit.sum()) / inv * 100.0 if inv > 0 else 0.0


def bootstrap_roi_ci(profit: np.ndarray, stake: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(profit)
    idx = rng.integers(0, n, size=(BOOTSTRAP_ITERS, n))
    inv = stake[idx].sum(axis=1)
    rois = np.where(inv > 0, profit[idx].sum(axis=1) / inv * 100.0, 0.0)
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))


def main() -> None:
    exp = FINAL_CONFIG.experiment_name
    dump = _latest(pd.read_csv(ARTIFACTS / f"final_per_match_predictions_{exp}.csv"))
    dump["date"] = pd.to_datetime(dump["date"])
    selector = _latest(pd.read_csv(ARTIFACTS / f"final_bet_selector_{exp}.csv"))
    locked = selector[selector["selector_status"] == "validation_locked_positive_fold_roi"]
    max_odds = load_max_odds()

    rows = []
    for _, sel in locked.iterrows():
        model = sel["model"]
        bets = rebuild_locked_bets(dump, model, sel)
        expected = int(sel["test_bets"])
        if len(bets) != expected:
            print(f"WARNING {model}: rebuilt {len(bets)} bets, artifact says {expected} - skipping")
            continue
        bets = reprice_at_max(bets, max_odds)
        profit_avg = bets["profit"].to_numpy(dtype=float)
        profit_max = bets["profit_max"].to_numpy(dtype=float)
        stake = bets["stake"].to_numpy(dtype=float)
        lo_a, hi_a = bootstrap_roi_ci(profit_avg, stake)
        lo_m, hi_m = bootstrap_roi_ci(profit_max, stake)
        rows.append({
            "model": model,
            "bets": len(bets),
            "max_price_coverage_pct": round(100.0 * bets["max_price_found"].mean(), 1),
            "avg_odds_taken": round(float(bets["odds_taken"].mean()), 4),
            "avg_max_odds_taken": round(float(bets["max_odds_taken"].mean()), 4),
            "roi_avg_price": round(roi(profit_avg, stake), 4),
            "roi_avg_ci_low": round(lo_a, 2),
            "roi_avg_ci_high": round(hi_a, 2),
            "roi_max_price": round(roi(profit_max, stake), 4),
            "roi_max_ci_low": round(lo_m, 2),
            "roi_max_ci_high": round(hi_m, 2),
            "artifact_test_roi": round(float(sel["test_roi"]), 4),
        })

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))
    csv_path = ARTIFACTS / f"line_shopping_experiment_{exp}.csv"
    out.to_csv(csv_path, index=False)
    print(f"\nWrote: {csv_path}")


if __name__ == "__main__":
    main()
