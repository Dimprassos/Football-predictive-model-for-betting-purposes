Football Match Prediction & Value Betting Meta-Model
A machine learning system for predicting football match outcomes across the top 5 European leagues (England, Spain, Italy, Germany, France). The system combines classical statistical models (Elo Ratings, Poisson Distribution with Dixon-Coles correction) with modern ensemble learning and a deep-learning baseline to estimate true match probabilities and identify value bets against bookmaker odds.
> Developed as part of a Diploma Thesis in Computer & Informatics Engineering.
---
Features
Automated Data Pipeline — Downloads and syncs historical match data and upcoming fixture files automatically.
Elo Rating System — Dynamic team strength ratings updated after every match, with configurable home advantage and goal-margin multipliers.
Poisson Goal Model — Per-team attack/defense strength estimation with exponential time-decay weighting and Dixon-Coles low-score correction.
XGBoost Meta-Model — Learns non-linear patterns from Poisson+Elo probabilities, market odds, and auxiliary features.
MLP Deep Learning Baseline — A neural-network classifier trained on the same meta-feature space and calibrated separately.
Ensemble Blender — Final weighted ensemble combining Base Model, Market, XGBoost and MLP outputs.
Temperature Scaling Calibration — Post-hoc probability calibration using logit-space temperature scaling.
Walk-Forward Backtesting — Strictly temporal train/validation/test splits with no data leakage.
Value Betting Simulation — ROI, Net Profit and Hit Rate breakdown by market segment.
Per-League Metrics — LogLoss, Brier Score and ECE reported individually for each league after evaluation.
Upcoming Matchday Picks — Predicts only the current / next available league matchday from the fixture files while preventing leakage from already-played matches.
---
Current Architecture
```text
project/
│
├── main.py
│
├── src/
│   ├── __init__.py
│   ├── update_data.py
│   ├── data_processing.py
│   ├── elo.py
│   ├── poisson_model.py
│   ├── calibration.py
│   ├── metrics.py
│   ├── artifacts.py
│   ├── evaluation.py
│   ├── fixtures.py
│   ├── meta_features.py
│   ├── streaming.py
│   ├── tuning.py
│   ├── reporting.py
│   ├── prediction_service.py
│   └── predict_match.py
│
├── data/
│   └── raw/
│       ├── england/
│       ├── spain/
│       ├── italy/
│       ├── germany/
│       └── france/
│
├── artifacts/
│   ├── best_params_*.json
│   ├── best_meta_*.json
│   ├── best_mlp_*.json
│   ├── best_blend_*.json
│   ├── meta_model_*.json
│   └── mlp_model_*.pkl
│
├── requirements.txt
├── setup.ps1
├── setup.sh
└── README.md
```
---
Installation
Requires Python 3.9+.
Windows
1. Clone the repository
```cmd
git clone https://github.com/your-username/your-repo.git
cd your-repo
```
2. Run the setup script
```powershell
.\setup.ps1
```
3. Activate the virtual environment
```cmd
venv\Scripts\activate
```
---
Linux / macOS / other Unix-like environments
1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```
2. Make the setup script executable
```bash
chmod +x setup.sh
```
3. Run the setup script
```bash
./setup.sh
```
4. Activate the virtual environment
```bash
source venv/bin/activate
```
---
Usage
Step 1 — Update Data
Download the latest historical results and upcoming fixtures before running the model:
```bash
python src/update_data.py
```
Step 2 — Run the Main Pipeline
Runs tuning/loading, training/loading, evaluation, betting simulation and upcoming matchday predictions:
```bash
python main.py
```
Step 3 — Predict a Custom Match
Run from the project root:
```bash
python -m src.predict_match
```
At the end of execution the system prints:
Aggregate evaluation metrics
Per-league evaluation metrics
Value betting simulation results
Upcoming matchday picks for all supported leagues
---
Configuration
Key constants at the top of `main.py`:
Variable	Default	Description
`EXPERIMENT_NAME`	`baseline_xgboost_v1`	Artifact namespace for cached experiments
`TRAIN_CUT`	`2024-07-01`	End of training period
`TEST_CUT`	`2025-07-01`	Start of test period
`USE_CACHED_ARTIFACTS`	`True`	Use cached models and tuned parameters when available
`FORCE_RETUNE_LEAGUES`	`False`	Retune Poisson/Elo parameters
`FORCE_RETUNE_META`	`False`	Retune XGBoost hyperparameters
`FORCE_REFIT_META_MODEL`	`False`	Refit final XGBoost model
`FORCE_RETUNE_MLP`	`False`	Retune MLP hyperparameters
`FORCE_REFIT_MLP_MODEL`	`False`	Refit final MLP model
`FORCE_RETUNE_BLEND`	`False`	Retune ensemble blend weights
This allows fast cached runs for normal use, and full retuning when you change features, models or search spaces.
---
Methodology Overview
```text
Raw Data
   │
   ├─► Elo Ratings
   │
   ├─► Poisson Team Strengths (time-weighted)
   │        │
   │        └─► Expected Goals (λ_home, λ_away)
   │                    │
   │                    └─► Dixon-Coles Scoreline Probabilities
   │
   ├─► Market Odds Probabilities
   │
   └─► Meta Features
            │
            ├─► XGBoost Meta-Model
            ├─► MLP Meta-Model
            └─► Ensemble Blender
                    │
            Final Match Probabilities
                    │
          Value Bet Detection / Matchday Picks
```
---
Evaluation
The system is evaluated on a held-out test set using:
Log Loss (NLL) — Main metric for probability quality. Lower is better.
Brier Score — Mean squared probability error. Lower is better.
ECE (Expected Calibration Error) — Measures calibration quality. Lower is better.
Accuracy — Useful as a secondary classification metric, but not the primary model-selection criterion.
Because this is a probabilistic prediction system, LogLoss, Brier and ECE are more important than plain accuracy.
---
Data Sources
Historical match results & odds: football-data.co.uk
Future fixtures: fixturedownload.com
---
Notes
Upcoming fixtures are read from dedicated fixture files, not inferred from historical result files alone.
The system predicts only the current / next available matchday window to avoid jumping too far into the future.
For custom match prediction, always run `predict_match.py` as a module from the project root:
```bash
  python -m src.predict_match
  ```
---
Thesis Context
This project is part of a diploma thesis on football match outcome prediction using:
classical statistical modelling,
machine learning meta-models,
ensemble methods,
and deep learning baselines.
The project has evolved from a single-file prototype into a more modular architecture to improve maintainability, experimentation and reproducibility.