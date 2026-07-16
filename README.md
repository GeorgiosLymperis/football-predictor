# Football Match Predictor

Match outcome prediction for six European football leagues (Greek Super League, Premier League, La Liga, Bundesliga, Serie A, Ligue 1), served through a Streamlit app.

Historical results, match statistics, and odds come from [football-data.co.uk](https://www.football-data.co.uk/), stored as one CSV per season under `data/`. `scripts/update_data.py` keeps the current season's file up to date from the same source; see [Keeping data and models fresh](#keeping-data-and-models-fresh).

## What it does

For a chosen home/away matchup, the app shows:

- Current Elo ratings and recent Elo trajectories for each league
- Win/draw/away probabilities and expected goals from a Bayesian Poisson (Dixon-Coles) model
- Win/draw/away probabilities from three additional models: Elo + XGBoost, Logistic Regression, and a small MLP
- An unweighted ensemble of the goals model, XGBoost, and Logistic Regression
- Walk-forward backtested RPS (Ranked Probability Score) for every model, benchmarked against a naive baseline and the betting market's own implied probabilities

## Models

| Model | Family | Notes |
|---|---|---|
| Bayesian Poisson (Dixon-Coles) | Goals model | Hierarchical attack/defence ratings fit with PyMC; an optional Negative Binomial variant relaxes the mean equals variance assumption |
| Elo + XGBoost | Classifier | Gradient-boosted trees over Elo, form, head-to-head, momentum, and shot-stat features |
| Logistic Regression | Classifier | Multinomial logistic regression over the same feature set |
| MLP | Classifier | Single hidden-layer neural network, L2-regularized and early-stopped |
| Ensemble | Blend | Unweighted average of the goals model, XGBoost, and Logistic Regression |

Each model is walk-forward backtested on held-out seasons and only promoted to "champion" (the version the app actually serves) if it beats a naive baseline by a configurable margin.

## Project layout

```
config/                Model, backtest, and per-league YAML configs
data/                  Raw match CSVs from football-data.co.uk, one folder per league
models/                Fitted champion artifacts the Streamlit app reads at serve time
mlruns/, mlflow.db      Local MLflow tracking store, used during training only, not needed to serve the app
scripts/                One training script per model family
src/match_predict/      The package: data loading, feature engineering, training, prediction, backtesting, mlops
tests/                  Unit tests (pytest) for the package and scripts
streamlit_app.py         App entry point
app_common.py            Shared rendering and model-loading code for the per-league pages
```

## Setup

Requires Python 3.11 or newer.

To run the app against already-fitted models in `models/`:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To train models yourself, install the heavier training-only dependencies as well (PyMC, MLflow, scikit-learn):

```bash
pip install -r requirements-dev.txt
```

## Training a model

Each script trains one model family for one league, walk-forward backtests it, and promotes it to `models/<league>/<model>/` if it beats the current champion:

```bash
python scripts/train_poisson.py --league greek
python scripts/train_xgb.py --league greek
python scripts/train_logistic.py --league greek
python scripts/train_mlp.py --league greek
python scripts/train_ensemle.py --league greek
```

Add `--dry-run` to any of the first four to see the backtest results without registering or promoting a model. `train_ensemle.py` requires the XGBoost, Logistic, and Poisson (or Negative Binomial) models to already be promoted for that league, since it blends their predictions.

Supported league keys: `greek`, `premier_league`, `la_liga`, `bundesliga`, `serie_a`, `ligue_1`.

## Using MLflow after training

Every non-dry-run training script (except `train_ensemle.py`, which does not use the registry) logs a run to a local MLflow tracking store at `mlflow.db` in the repo root, registers a new version of that model, and updates the `champion` alias if it was promoted. To browse runs, parameters, metrics, and which version is currently aliased as champion:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://localhost:5000`. This is separate from what the Streamlit app actually serves: the app only ever reads the lean artifacts already exported to `models/<league>/<model>/` (see [Project layout](#project-layout)), never the MLflow store directly.

## Keeping data and models fresh

`scripts/update_data.py` refreshes each league's current-season CSV from football-data.co.uk and validates it with the same schema checks used at training time before it replaces the local file, so a bad download can't corrupt the dataset:

```bash
python scripts/update_data.py               # all leagues
python scripts/update_data.py --league greek # one league
```

This only refreshes the season already tracked locally. When a new season starts, add its CSV and update that league's `teams:` allowlist in `config/leagues/<league>.yaml` by hand first (new, promoted clubs need to be added); after that, automatic refreshes pick it up.

Refitting a model (walk-forward backtest + PyMC sampling + feature-variant search) is much more expensive than just recomputing current Elo/form/head-to-head numbers, so those run on two separate schedules:

- `scripts/update_elo_state.py` recomputes each league's current team state (Elo, form, head-to-head, momentum, shot-roll features) and overwrites `team_state.npz` for whichever of Elo+XGBoost, Logistic, and MLP already has a promoted model, without refitting anything. It preserves any extra keys already in that file (Elo+XGBoost also stores its label encoding there) so it never has to know every model's internal layout. It skips a league/model that hasn't been trained yet. The Bayesian Poisson model has no equivalent: its attack/defence ratings are the fitted model itself, so it only updates via a full retrain.
- `.github/workflows/update-elo-weekly.yml` runs this weekly (Monday 06:00 UTC, or on demand): refresh data, run the tests, refresh Elo state, commit.
- `.github/workflows/retrain-monthly.yml` runs the full retrain (all five model scripts, every league) monthly (1st of the month, 06:00 UTC, or on demand) and commits whatever `data/` and `models/` changes result. A failure in one league does not block the others from being committed; the workflow still fails at the end so it's visible, but nothing is lost.

```bash
python scripts/update_elo_state.py               # all leagues
python scripts/update_elo_state.py --league greek # one league
```

Note: the MLflow tracking store (`mlflow.db`, `mlruns/`) is not persisted between scheduled runs, since it is gitignored. That means the "beat the current champion" comparison in `promote_if_better` has no history to compare against in CI, so in practice a scheduled retrain only gates on beating the naive baseline, not on beating last month's specific model. This is fine for how volatile these datasets are month to month, but worth knowing if you want stricter gating later.

## Configuration

- `config/backtest.yaml`: shared walk-forward settings (selection and holdout seasons, step size, promotion margin)
- `config/poisson.yaml`, `config/xgb.yaml`, `config/logistic.yaml`, `config/mlp.yaml`: per-model hyperparameters and feature variants
- `config/leagues/<league>.yaml`: per-league data location, team allowlist, and name fixes for teams that changed names across seasons

## Monitoring

Every prediction the app serves is logged to a local SQLite database (`predictions.db`) via `match_predict.mlops.monitoring`, for later comparison against actual results.
