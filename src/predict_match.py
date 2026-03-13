import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

# Προσθήκη του project root στο path για να βλέπουμε τα modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_processing import load_league_data
from src.poisson_model import (
    fit_team_strengths_home_away_weighted,
    predict_lambdas_home_away,
    apply_elo_to_lambdas,
    match_outcome_probs_dc,
    top_k_scorelines_dc
)
from src.elo import expected_score, match_result, margin_multiplier
from src.calibration import temperature_scale_probs

# Ρυθμίσεις Αρχείων
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EXPERIMENT_NAME = "baseline_xgboost_v1"
PARAMS_FILE = ARTIFACTS_DIR / f"best_params_{EXPERIMENT_NAME}.json"
MODEL_FILE = ARTIFACTS_DIR / f"meta_model_{EXPERIMENT_NAME}.json"
MLP_FILE = ARTIFACTS_DIR / f"mlp_model_{EXPERIMENT_NAME}.pkl"

def load_artifacts():
    print("Loading models and parameters...")
    
    if not PARAMS_FILE.exists():
        sys.exit("Error: Parameters file not found. Run main.py first.")
    with open(PARAMS_FILE, "r") as f:
        params = json.load(f)

    if not MODEL_FILE.exists():
        sys.exit("Error: XGBoost model not found. Run main.py first.")
    meta_model = XGBClassifier()
    meta_model.load_model(str(MODEL_FILE))

    mlp_model = None
    if MLP_FILE.exists():
        with open(MLP_FILE, "rb") as f:
            mlp_model = pickle.load(f)
    
    return params, meta_model, mlp_model

def get_league_state(league_name, params):
    """
    Υπολογίζει την τρέχουσα κατάσταση (Elo, Attack, Defense) για όλη τη λίγκα
    ώστε να είμαστε έτοιμοι για πρόβλεψη.
    """
    df = load_league_data(league_name)
    df = df[df["is_played"] == True].sort_values("date")
    
    # 1. Υπολογισμός Team Strengths (Poisson)
    decay = params[league_name]["decay"]
    l_avg_h, l_avg_a, att_h, def_h, att_a, def_a = fit_team_strengths_home_away_weighted(
        df, decay=decay
    )
    
    # 2. Υπολογισμός Elo Ratings
    ratings = {}
    K = params[league_name]["K"]
    ha = params[league_name]["ha"]

    def get_init(r):
        if len(r) >= 5:
            return sum(sorted(r.values())[:3]) / 3.0
        return 1500.0

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        
        init_r = get_init(ratings)
        rh = ratings.get(h, init_r)
        ra = ratings.get(a, init_r)
        
        if h not in ratings: ratings[h] = rh
        if a not in ratings: ratings[a] = ra
        
        exp_h = expected_score(rh + ha, ra)
        sh, sa = match_result(hg, ag)
        mult = margin_multiplier(hg - ag)
        
        ratings[h] = rh + (K * mult) * (sh - exp_h)
        ratings[a] = ra + (K * mult) * (sa - (1 - exp_h))
        
    return {
        "ratings": ratings,
        "att_h": att_h, "def_h": def_h,
        "att_a": att_a, "def_a": def_a,
        "l_avg_h": l_avg_h, "l_avg_a": l_avg_a,
        "params": params[league_name]
    }

def predict_custom_match(home, away, odds_h, odds_d, odds_a, state, meta_model, mlp_model):
    p = state["params"]
    
    # Ανάκτηση δεδομένων ομάδων
    elo_h = state["ratings"].get(home, 1500.0)
    elo_a = state["ratings"].get(away, 1500.0)
    
    # Υπολογισμός Lambdas (Expected Goals)
    lam_h, lam_a = predict_lambdas_home_away(
        home, away,
        state["l_avg_h"], state["l_avg_a"],
        state["att_h"], state["def_h"],
        state["att_a"], state["def_a"]
    )
    
    # Εφαρμογή Elo διόρθωσης
    lam_h, lam_a = apply_elo_to_lambdas(lam_h, lam_a, elo_h, elo_a, beta=p["beta"])
    
    # Στατιστική Πιθανότητα (Poisson/Dixon-Coles)
    pH, pD, pA = match_outcome_probs_dc(lam_h, lam_a, rho=p["rho"])
    model_probs_raw = np.array([[pH, pD, pA]])
    
    # Calibration
    model_probs_cal = temperature_scale_probs(model_probs_raw, p["T"])[0]
    
    # Διαχείριση Αποδόσεων
    if odds_h > 1.0 and odds_d > 1.0 and odds_a > 1.0:
        # Market probabilities (inverse odds)
        mkt_probs = np.array([1/odds_h, 1/odds_d, 1/odds_a])
        mkt_probs /= mkt_probs.sum()
    else:
        # Αν ο χρήστης δεν δώσει αποδόσεις, χρησιμοποιούμε τις πιθανότητες του μοντέλου
        mkt_probs = model_probs_cal

    # Feature Engineering για το Meta-Model
    def safe_logit(val):
        val = np.clip(val, 1e-12, 1 - 1e-12)
        return np.log(val) - np.log(1 - val)

    features = [
        safe_logit(model_probs_cal[0]), safe_logit(model_probs_cal[1]), safe_logit(model_probs_cal[2]),
        safe_logit(mkt_probs[0]), safe_logit(mkt_probs[1]), safe_logit(mkt_probs[2]),
        (elo_h - elo_a) / 400.0,
        lam_h + lam_a,
        lam_h - lam_a
    ]
    
    X = np.array([features])
    
    # Τελικές Προβλέψεις
    meta_probs = meta_model.predict_proba(X)[0]
    mlp_probs = mlp_model.predict_proba(X)[0] if mlp_model else [0, 0, 0]
    
    # Top Scorelines
    top_scores = top_k_scorelines_dc(lam_h, lam_a, p["rho"], k=3)
    
    return {
        "meta": meta_probs,
        "mlp": mlp_probs,
        "base": model_probs_cal,
        "elo": (elo_h, elo_a),
        "xg": (lam_h, lam_a),
        "scores": top_scores
    }

def main():
    print("=== Interactive Match Predictor ===")
    params, meta_model, mlp_model = load_artifacts()
    
    leagues = ["england", "spain", "italy", "germany", "france"]
    
    while True:
        print("\nAvailable Leagues:")
        for i, l in enumerate(leagues):
            print(f"{i+1}. {l}")
        
        try:
            choice = int(input("\nSelect League (number) or 0 to exit: "))
            if choice == 0: break
            if choice < 1 or choice > len(leagues): continue
        except ValueError:
            continue
            
        league = leagues[choice-1]
        print(f"Loading data for {league}...")
        state = get_league_state(league, params)
        
        teams = sorted(state["ratings"].keys())
        
        while True:
            print(f"\n--- {league.upper()} Prediction ---")
            home_team = input("Home Team Name (part or full): ").strip()
            if not home_team: break
            
            # Απλή αναζήτηση ονόματος
            matches_h = [t for t in teams if home_team.lower() in t.lower()]
            if not matches_h:
                print("Team not found!")
                continue
            home_real = matches_h[0]
            print(f"Selected Home: {home_real}")
            
            away_team = input("Away Team Name (part or full): ").strip()
            matches_a = [t for t in teams if away_team.lower() in t.lower()]
            if not matches_a:
                print("Team not found!")
                continue
            away_real = matches_a[0]
            print(f"Selected Away: {away_real}")
            
            try:
                odds_str = input("Enter Odds (Home Draw Away) e.g., '1.90 3.50 4.00' (or enter to skip): ")
                if odds_str.strip():
                    oh, od, oa = map(float, odds_str.split())
                else:
                    oh, od, oa = 0, 0, 0
            except:
                print("Invalid odds format. Ignoring odds.")
                oh, od, oa = 0, 0, 0
            
            res = predict_custom_match(home_real, away_real, oh, od, oa, state, meta_model, mlp_model)
            
            print("\n" + "="*40)
            print(f"{home_real} vs {away_real}")
            print(f"Elo: {res['elo'][0]:.0f} vs {res['elo'][1]:.0f}")
            print(f"xG:  {res['xg'][0]:.2f} vs {res['xg'][1]:.2f}")
            print("-" * 40)
            print(f"{'Model':<15} | {'Home %':<8} | {'Draw %':<8} | {'Away %':<8}")
            print("-" * 40)
            print(f"{'Base (Poisson)':<15} | {res['base'][0]*100:5.1f}%   | {res['base'][1]*100:5.1f}%   | {res['base'][2]*100:5.1f}%")
            print(f"{'XGBoost (Meta)':<15} | {res['meta'][0]*100:5.1f}%   | {res['meta'][1]*100:5.1f}%   | {res['meta'][2]*100:5.1f}%")
            print(f"{'Deep Learning':<15} | {res['mlp'][0]*100:5.1f}%   | {res['mlp'][1]*100:5.1f}%   | {res['mlp'][2]*100:5.1f}%")
            print("-" * 40)
            print("Most Likely Scores:")
            for (hg, ag), prob in res['scores']:
                print(f"  {hg}-{ag} ({prob*100:.1f}%)")
            print("="*40)

if __name__ == "__main__":
    main()
