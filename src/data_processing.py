from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLOSING_COLS = ("PSCH", "PSCD", "PSCA")

BOOK_TRIPLES = [
    ("B365H", "B365D", "B365A"),
    ("BWH", "BWD", "BWA"),
    ("GBH", "GBD", "GBA"),
    ("IWH", "IWD", "IWA"),
    ("LBH", "LBD", "LBA"),
    ("PSH", "PSD", "PSA"),
    ("WHH", "WHD", "WHA"),
    ("SJH", "SJD", "SJA"),
    ("VCH", "VCD", "VCA"),
    ("BSH", "BSD", "BSA"),
]

REQUIRED_BASE = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def _valid_odds_triplet(oh, od, oa):
    if not (np.isfinite(oh) and np.isfinite(od) and np.isfinite(oa)):
        return False
    if oh <= 1.0001 or od <= 1.0001 or oa <= 1.0001:
        return False
    return True


def _odds_to_fair_probs(oh, od, oa):
    inv = np.array([1.0 / oh, 1.0 / od, 1.0 / oa], dtype=float)
    s = inv.sum()
    if s <= 0 or not np.isfinite(s):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return inv / s


def _fair_probs_to_odds(p):
    p = np.asarray(p, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        return (np.nan, np.nan, np.nan)

    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()

    return (float(1.0 / p[0]), float(1.0 / p[1]), float(1.0 / p[2]))


def _pick_best_or_avg_odds_row(row, available_cols):
    # 1) Pinnacle closing
    if all(c in available_cols for c in CLOSING_COLS):
        oh = pd.to_numeric(row.get(CLOSING_COLS[0]), errors="coerce")
        od = pd.to_numeric(row.get(CLOSING_COLS[1]), errors="coerce")
        oa = pd.to_numeric(row.get(CLOSING_COLS[2]), errors="coerce")

        if _valid_odds_triplet(oh, od, oa):
            return float(oh), float(od), float(oa)

    # 2) Average across books in fair-probability space
    probs_list = []
    for h, d, a in BOOK_TRIPLES:
        if h in available_cols and d in available_cols and a in available_cols:
            oh = pd.to_numeric(row.get(h), errors="coerce")
            od = pd.to_numeric(row.get(d), errors="coerce")
            oa = pd.to_numeric(row.get(a), errors="coerce")

            if _valid_odds_triplet(oh, od, oa):
                probs_list.append(_odds_to_fair_probs(float(oh), float(od), float(oa)))

    if len(probs_list) == 0:
        return (np.nan, np.nan, np.nan)

    p_avg = np.nanmean(np.vstack(probs_list), axis=0)
    return _fair_probs_to_odds(p_avg)


def load_league_data(league_name):
    data_path = PROJECT_ROOT / "data" / "raw" / league_name
    files = list(data_path.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in: {data_path}\n"
            f"Make sure you have downloaded the data for {league_name}."
        )

    dfs = []

    for file in sorted(files):
        df = pd.read_csv(file)

        missing = [c for c in REQUIRED_BASE if c not in df.columns]
        if missing:
            raise ValueError(
                f"[{file.name}] Missing required columns {missing}.\n"
                f"Available columns: {list(df.columns)}"
            )

        want_cols = set(REQUIRED_BASE)

        for c in CLOSING_COLS:
            if c in df.columns:
                want_cols.add(c)

        for h, d, a in BOOK_TRIPLES:
            if h in df.columns and d in df.columns and a in df.columns:
                want_cols.update([h, d, a])

        df = df[list(want_cols)].copy()

        df = df.rename(columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
        })

        # parse dates
        dt1 = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
        dt2 = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        df["date"] = dt1.fillna(dt2)

        mask = df["date"].isna()
        if mask.any():
            df.loc[mask, "date"] = pd.to_datetime(
                df.loc[mask, "date"], dayfirst=True, errors="coerce"
            )

        df = df.dropna(subset=["date"])

        # keep played + future fixtures
        df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
        df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
        df["is_played"] = df["home_goals"].notna() & df["away_goals"].notna()

        available_cols = set(df.columns)

        odds = df.apply(
            lambda r: _pick_best_or_avg_odds_row(r, available_cols),
            axis=1,
            result_type="expand"
        )
        odds.columns = ["odds_home", "odds_draw", "odds_away"]

        df = pd.concat([df, odds], axis=1)

        df["odds_home"] = pd.to_numeric(df["odds_home"], errors="coerce")
        df["odds_draw"] = pd.to_numeric(df["odds_draw"], errors="coerce")
        df["odds_away"] = pd.to_numeric(df["odds_away"], errors="coerce")

        dfs.append(
            df[[
                "date",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "odds_home",
                "odds_draw",
                "odds_away",
                "is_played",
            ]]
        )

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)

    # remove exact duplicates if fixture files overlap with historical rows
    out = out.drop_duplicates(
        subset=["date", "home_team", "away_team"],
        keep="first"
    ).reset_index(drop=True)

    return out