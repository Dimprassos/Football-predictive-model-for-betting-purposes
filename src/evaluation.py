import numpy as np
import pandas as pd


def labels_from_df(df: pd.DataFrame) -> np.ndarray:
    y = []
    for _, r in df.iterrows():
        if r["home_goals"] > r["away_goals"]:
            y.append(0)
        elif r["home_goals"] == r["away_goals"]:
            y.append(1)
        else:
            y.append(2)
    return np.array(y, dtype=int)


def simulate_value_betting(probs, raw_odds, y_true, edge_threshold=0.05):
    stats = {
        "Home (1)": {"count": 0, "wins": 0, "invested": 0, "return": 0, "odds_sum": 0},
        "Draw (X)": {"count": 0, "wins": 0, "invested": 0, "return": 0, "odds_sum": 0},
        "Away (2)": {"count": 0, "wins": 0, "invested": 0, "return": 0, "odds_sum": 0},
    }

    for i in range(len(probs)):
        p_h, p_d, p_a = probs[i]
        o_h, o_d, o_a = raw_odds[i]

        if not (np.isfinite(o_h) and np.isfinite(o_d) and np.isfinite(o_a)):
            continue

        evs = [
            (p_h * o_h - 1, 0, o_h, "Home (1)"),
            (p_d * o_d - 1, 1, o_d, "Draw (X)"),
            (p_a * o_a - 1, 2, o_a, "Away (2)"),
        ]

        best_ev, choice, odds_taken, label = max(evs)

        if best_ev > edge_threshold:
            stats[label]["count"] += 1
            stats[label]["invested"] += 1.0
            stats[label]["odds_sum"] += odds_taken

            if choice == y_true[i]:
                stats[label]["wins"] += 1
                stats[label]["return"] += odds_taken

    print(f"\n{'Market Segment':<15} | {'Bets':<5} | {'Win%':<7} | {'ROI%':<8}")
    print("-" * 45)

    total_bets = 0
    total_wins = 0
    total_inv = 0
    total_ret = 0
    total_odds_sum = 0

    for label, s in stats.items():
        if s["count"] > 0:
            win_pc = (s["wins"] / s["count"]) * 100
            roi = ((s["return"] - s["invested"]) / s["invested"]) * 100
            print(f"{label:<15} | {s['count']:<5} | {win_pc:>6.1f}% | {roi:>7.2f}%")

            total_bets += s["count"]
            total_wins += s["wins"]
            total_inv += s["invested"]
            total_ret += s["return"]
            total_odds_sum += s["odds_sum"]

    final_profit = total_ret - total_inv
    final_roi = (final_profit / total_inv * 100) if total_inv > 0 else 0
    avg_odds = (total_odds_sum / total_bets) if total_bets > 0 else 0

    print("-" * 45)
    print(f"{'TOTAL':<15} | {int(total_inv):<5} | {'-':>7} | {final_roi:>7.2f}%")

    return total_bets, total_wins, final_profit, final_roi, avg_odds
