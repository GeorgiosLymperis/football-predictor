import argparse

import numpy as np

from match_predict.config import LEAGUES_DIR, load_league_config
from match_predict.features.data import load_league_matches
from match_predict.features.elo import build_features, export_state
from match_predict.mlops.registry import MODELS_DIR

STATEFUL_MODEL_DIRS = ('elo_xgb', 'logistic', 'mlp')


def update_league(league: str) -> None:
    league_cfg = load_league_config(league)
    df = load_league_matches(league_cfg)
    _, states, h2h = build_features(df)
    current_teams = sorted(set(df[df['year'] == df['year'].max()]['team1']))

    for model_dir in STATEFUL_MODEL_DIRS:
        state_path = MODELS_DIR / league / model_dir / 'team_state.npz'
        if not state_path.exists():
            print(f'[{league}] No promoted {model_dir} model yet; skipping (run scripts/train_*.py first).')
            continue

        existing = np.load(state_path, allow_pickle=True)
        feature_names = list(existing['feature_names'])
        state = export_state(states, h2h, feature_names, current_teams)
        # elo_xgb's team_state.npz also carries an extra 'classes' key (the
        # XGBoost label encoding) that export_state() doesn't produce.
        # Start from what's already on disk and overlay the refreshed
        # fields, so any such extra key survives instead of being dropped.
        merged = {k: existing[k] for k in existing.files}
        merged.update(state)
        np.savez_compressed(state_path, **merged)
        print(f'[{league}] Refreshed {state_path} ({len(current_teams)} current teams).')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--league', choices=sorted(p.stem for p in LEAGUES_DIR.glob('*.yaml')))
    args = parser.parse_args()

    leagues = [args.league] if args.league else sorted(p.stem for p in LEAGUES_DIR.glob('*.yaml'))
    for league in leagues:
        update_league(league)


if __name__ == '__main__':
    main()
