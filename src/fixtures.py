import pandas as pd


def get_current_or_next_matchday_fixtures(df_league: pd.DataFrame, max_window_days: int = 4):
    today = pd.Timestamp.now().normalize()

    future_df = df_league[df_league["is_played"] == False].copy()
    future_df = future_df.sort_values("date")

    if future_df.empty:
        return pd.DataFrame(), None

    candidate_df = future_df[future_df["date"] >= today].copy()

    if candidate_df.empty:
        candidate_df = future_df.copy()

    if candidate_df.empty:
        return pd.DataFrame(), None

    matchday_start = candidate_df["date"].min()
    matchday_end = matchday_start + pd.Timedelta(days=max_window_days)

    fixtures = candidate_df[
        (candidate_df["date"] >= matchday_start) &
        (candidate_df["date"] <= matchday_end)
    ].copy()

    return fixtures, matchday_start
