# Football Match Prediction & Value Betting Meta-Model

A machine learning system for predicting football match outcomes across the top 5 European leagues (England, Spain, Italy, Germany, France). The system combines classical statistical models (Elo Ratings, Poisson Distribution with Dixon-Coles correction) with a gradient-boosted meta-learner (XGBoost) to estimate true match probabilities and identify value bets against bookmaker odds.

> Developed as part of a Diploma Thesis in Computer & Informatics Engineering.

---

## Features

- **Automated Data Pipeline** — Downloads and syncs historical match data (2012–present) and bookmaker closing odds automatically.
- **Elo Rating System** — Dynamic team strength ratings updated after every match, with configurable home advantage and goal-margin multipliers.
- **Poisson Goal Model** — Per-team attack/defense strength estimation with exponential time-decay weighting and Dixon-Coles low-score correction.
- **XGBoost Meta-Model** — Learns non-linear patterns from Poisson+Elo probabilities, market odds, and auxiliary features (Elo differential, expected goal totals).
- **Temperature Scaling Calibration** — Post-hoc probability calibration using logit-space temperature scaling, fitted per league via NLL minimisation.
- **Walk-Forward Backtesting** — Strictly temporal train/validation/test splits with no data leakage. All predictions are made using only information available before match day.
- **Value Betting Simulation** — ROI, Net Profit, and Hit Rate breakdown by market segment (Home / Draw / Away).
- **Per-League Metrics** — LogLoss, Brier Score, and ECE reported individually for each league after evaluation.

---

## Project Structure

```
project/
│
├── main.py                  # Main pipeline: tuning → training → evaluation → predictions
│
├── src/
│   ├── update_data.py       # Downloads historical CSVs and future fixture files
│   ├── data_processing.py   # Data cleaning, odds parsing, overround removal
│   ├── elo.py               # Elo rating algorithm (dynamic init, margin multiplier)
│   ├── poisson_model.py     # Expected goals model + Dixon-Coles scoreline distribution
│   ├── calibration.py       # Temperature scaling (logit-space, scipy-optimised)
│   └── metrics.py           # Multiclass Brier Score, ECE
│
├── data/
│   └── raw/
│       ├── england/         # CSV files per season (football-data.co.uk format)
│       ├── spain/
│       ├── italy/
│       ├── germany/
│       └── france/
│
├── artifacts/               # Cached tuned parameters and trained model (auto-generated)
│
├── requirements.txt
└── setup.ps1                # Automated Windows environment setup script
```

---

## Installation

Designed for Windows. Requires Python 3.9+.

**1. Clone the repository**
```cmd
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

**2. Run the setup script** (creates virtual environment and installs dependencies)
```powershell
.\setup.ps1
```

**3. Activate the virtual environment**
```cmd
venv\Scripts\activate
```

---

## Usage

### Step 1 — Update Data
Download the latest match results and upcoming fixtures before running the model:
```bash
python src/update_data.py
```

### Step 2 — Run the Model
Trains the full pipeline, runs evaluation on the test set, and outputs predictions for the current matchday:
```bash
python main.py
```

At the end of execution the program prints:

- Per-league and aggregate evaluation metrics (LogLoss, Brier Score, ECE)
- Value betting simulation results (ROI, Net Profit, Hit Rate) broken down by Home / Draw / Away
- Predicted probabilities and value picks for upcoming fixtures across all leagues

---

## Configuration

Key constants at the top of `main.py`:

| Variable | Default | Description |
|---|---|---|
| `TRAIN_CUT` | `2024-07-01` | End of training period |
| `TEST_CUT` | `2025-07-01` | Start of test period |
| `USE_CACHED_ARTIFACTS` | `True` | Skip retuning if cached params exist |
| `FORCE_RETUNE_LEAGUES` | `False` | Force retuning of per-league Poisson/Elo params |
| `FORCE_RETUNE_META` | `False` | Force retuning of XGBoost hyperparameters |
| `FORCE_REFIT_META_MODEL` | `False` | Force retraining of the final XGBoost model |

Set any `FORCE_*` flag to `True` to re-run that stage from scratch.

---

## Key Parameters (per league)

Tuned automatically via grid search on the validation set. Stored in `artifacts/best_params_*.json`.

| Parameter | Description |
|---|---|
| `K` | Elo update rate — higher values make ratings react faster to recent results |
| `ha` | Home advantage in Elo points added to the home team before each match |
| `beta` | Controls how strongly the Elo rating difference scales the Poisson lambdas |
| `decay` | Exponential time-decay rate for match weights (per day); higher = faster forgetting |
| `rho` | Dixon-Coles dependence parameter; adjusts joint probability of low-score results (0-0, 1-0, 0-1, 1-1) |
| `T` | Temperature scaling factor; T > 1 softens overconfident probabilities toward uniform |

---

## Methodology Overview

```
Raw Data
   │
   ├─► Elo Ratings  ──────────────────────────────────────┐
   │                                                       │
   └─► Poisson Team Strengths (time-weighted)             │
           │                                              │
           └─► Expected Goals (λ_home, λ_away) ──────────┤
                       │                                  │
                       └─► Dixon-Coles Scoreline Probs    │
                                   │                      │
                                   └─► Elo Adjustment ────┘
                                               │
                                   Temperature Calibration
                                               │
                              ┌────────────────┴──────────────────┐
                              │                                   │
                        Model Probs                        Market Odds
                              │                                   │
                              └──────────────┬────────────────────┘
                                             │
                                    XGBoost Meta-Model
                                             │
                                    Final Probabilities
                                             │
                               Value Bet Detection (EV > threshold)
```

---

## Evaluation

The system is evaluated on a held-out test set using three complementary metrics:

- **Log Loss (NLL)** — Measures the accuracy of predicted probabilities. Lower is better. A perfect model scores 0; a naive uniform predictor scores ~1.099.
- **Brier Score** — Mean squared error between predicted probabilities and one-hot outcomes. Lower is better.
- **ECE (Expected Calibration Error)** — Measures how well predicted confidence matches empirical accuracy. Lower is better; 0 means perfectly calibrated.

The market odds implied probabilities serve as the primary benchmark — a model that cannot beat the market on LogLoss provides no informational edge for betting purposes.

---

## Data Sources

- **Historical match results & odds:** [football-data.co.uk](https://www.football-data.co.uk)
- **Future fixtures:** [fixturedownload.com](https://fixturedownload.com)

---

## Changelog

### v1.1.0 — Bug Fixes & Calibration Improvements

#### Bug Fix — `calibration.py`: Incorrect logit transformation
The `safe_logit` function was computing `log(p)` (log-probability) instead of the true logit `log(p / (1-p))`. This caused `temperature_scale_probs` to perform power-law scaling rather than proper temperature scaling, meaning the calibrated probabilities did not behave correctly as T varied. Fixed to `log(p) - log(1-p)` with symmetric clipping to `[eps, 1-eps]`.

```python
# Before (incorrect — log-probability, not logit)
return np.log(p)

# After (correct — proper logit)
p = np.clip(p, eps, 1.0 - eps)
return np.log(p) - np.log(1.0 - p)
```

#### Bug Fix — `poisson_model.py`: Duplicate `predict_lambdas_home_away` definition
The function `predict_lambdas_home_away` was defined twice. Python silently uses the last definition, meaning the first version (without the safety floor `max(0.05, lam)`) was dead code but created a maintenance risk. The duplicate was removed; only the correct version with the safety floor and full home/away split parameters remains.

#### Bug Fix — `poisson_model.py`: Dixon-Coles τ(0,0) could become negative
The Dixon-Coles correction factor `τ(0,0) = 1 - λ_h * λ_a * ρ` becomes negative when `ρ` is large relative to the expected goals. For example, `λ_h = λ_a = 2.5, ρ = 0.3` gives `τ(0,0) = -0.875`, producing an invalid negative joint probability. The previous code silently clamped negative probabilities to 0, causing undetected probability mass loss. Fixed by clamping `ρ` dynamically before each tau computation so all tau values are guaranteed to be positive:

```python
rho_max = min((1.0 - eps) / (lam_h * lam_a), 1.0 - eps)
rho = max(-rho_max, min(rho, rho_max))
```

#### Improvement — `calibration.py`: Robust temperature fitting
`fit_temperature` was using a fixed grid of `[0.5, 3.0]` with step 0.05. After fixing the logit bug, optimal temperatures for some leagues exceeded this range, causing the function to silently return the boundary value (3.0). Replaced with a two-stage approach:
- Wider search range `[0.1, 10.0]` with a finer grid near `T=1`
- `scipy.optimize.minimize_scalar` refinement for sub-grid precision
- Explicit boundary warnings printed if the optimum lands at either limit

#### Improvement — `main.py`: Per-league test metrics
The final evaluation section previously reported only aggregate metrics across all five leagues combined. A per-league breakdown table is now printed after the aggregate report, showing LogLoss, Brier Score, and ECE separately for each league for both the base model and the XGBoost meta-model.

#### Compatibility — `calibration.py`: Python 3.9 support
Replaced `np.ndarray | None` type hint syntax (requires Python 3.10+) with `Optional[np.ndarray]` from `typing`, ensuring compatibility with Python 3.9.