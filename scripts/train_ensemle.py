import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from match_predict.mlops import registry
from match_predict.metrics.backtest import season_to_year, walk_forward
from match_predict.config import LEAGUES_DIR, load_config, load_league_config
from match_predict.features.data import load_league_matches
from match_predict.features.elo import FEATURE_VARIANTS, build_features
from match_predict.metrics.odds import market_rps
from match_predict.predict.poisson import predict_outcome_probs
from match_predict.training.logistic_model import hda_probs as logistic_hda_probs
from match_predict.training.logistic_model import impute
from train_poisson import _recent_window, _teams_in_seasons
from match_predict.training.poisson_model import export_posterior_params, fit_poisson_model, prepare_df


def _load_metadata(league: str, model_name: str) -> dict:
    path = registry.MODELS_DIR / league / model_name / 'metadata.yaml'
    if not path.exists():
        raise FileNotFoundError(
            f'{path} does not exist — train and promote {model_name} for {league!r} first '
            f'(train_elo_xgb.py / train_logistic.py / train_poisson.py).'
        )
    return yaml.safe_load(path.read_text())


def _feature_variant(mlflow_model_name: str, run_id: str) -> list[str]:
    client = MlflowClient()
    variant_name = client.get_run(run_id).data.params['feature_variant']
    return FEATURE_VARIANTS[variant_name]


def _choose_goals_model(league: str) -> str:
    poisson_rps = registry.get_champion_rps(f'poisson_{league}')
    negbinom_rps = registry.get_champion_rps(f'negbinom_{league}')
    if negbinom_rps is not None and negbinom_rps < poisson_rps:
        return 'negbinom'
    return 'poisson'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--league', required=True, choices=sorted(p.stem for p in LEAGUES_DIR.glob('*.yaml')))
    args = parser.parse_args()
    league = args.league

    registry.configure()
    poisson_cfg = load_config('poisson')
    elo_xgb_cfg = load_config('xgb')
    logistic_cfg = load_config('logistic')
    backtest_cfg = load_config('backtest')
    league_cfg = load_league_config(league)

    goals_short_name = _choose_goals_model(league)
    goals_model_name = f'{goals_short_name}_{league}'
    elo_xgb_meta = _load_metadata(league, 'elo_xgb')
    logistic_meta = _load_metadata(league, 'logistic')
    elo_features = _feature_variant(f'elo_xgb_{league}', elo_xgb_meta['run_id'])
    logistic_features = _feature_variant(f'logistic_{league}', logistic_meta['run_id'])
    print(f'[{league}] Ensemble = {goals_short_name} + elo_xgb ({len(elo_features)} features) '
          f'+ logistic ({len(logistic_features)} features)')

    df = load_league_matches(league_cfg)
    teams = _teams_in_seasons(df, backtest_cfg['selection_seasons'] + backtest_cfg['holdout_seasons'])
    sl_features, _, _ = build_features(df)
    n_poisson_seasons = len(poisson_cfg['train_seasons'])

    def fit_fn(train_df):
        windowed = _recent_window(train_df, n_poisson_seasons)
        _, idata = fit_poisson_model(
            prepare_df(windowed, teams), teams,
            dixon_coles=poisson_cfg['dixon_coles'], likelihood='neg_binomial' if goals_short_name == 'negbinom' else 'poisson',
            **poisson_cfg['backtest_sampling'],
        )
        goals_params = export_posterior_params(
            idata, teams, poisson_cfg['dixon_coles'],
            likelihood='neg_binomial' if goals_short_name == 'negbinom' else 'poisson',
        )

        encoder_x = LabelEncoder()
        y_x = encoder_x.fit_transform(train_df['result'])
        weights = compute_sample_weight(class_weight='balanced', y=y_x)
        xgb_model = xgb.XGBClassifier(**elo_xgb_cfg['xgb_params'])
        xgb_model.fit(train_df[elo_features], y_x, sample_weight=weights)

        encoder_l = LabelEncoder()
        y_l = encoder_l.fit_transform(train_df['result'])
        impute_means = train_df[logistic_features].mean()
        X = impute(train_df[logistic_features], impute_means)
        scaler = StandardScaler().fit(X)
        log_model = LogisticRegression(**logistic_cfg['logistic_params']).fit(scaler.transform(X), y_l)

        return {
            'goals': goals_params,
            'xgb': (xgb_model, encoder_x),
            'logistic': (log_model, scaler, encoder_l, impute_means),
        }

    def predict_fn(fitted, eval_df):
        goals_probs = np.array([
            [r['p_home'], r['p_draw'], r['p_away']]
            for r in (
                predict_outcome_probs(fitted['goals'], row.team1, row.team2) for row in eval_df.itertuples()
            )
        ])
        xgb_model, encoder_x = fitted['xgb']
        order_x = [list(encoder_x.classes_).index(c) for c in ('Home', 'Draw', 'Away')]
        xgb_probs = xgb_model.predict_proba(eval_df[elo_features])[:, order_x]

        log_model, scaler, encoder_l, impute_means = fitted['logistic']
        log_probs = logistic_hda_probs(log_model, scaler, encoder_l, eval_df[logistic_features], impute_means)

        return (goals_probs + xgb_probs + log_probs) / 3

    print(f'\n[{league}] Walk-forward backtest (ensemble, holdout seasons)...')
    fold_results, summary = walk_forward(
        sl_features, backtest_cfg['holdout_seasons'], backtest_cfg['poisson_step_matchdays'],
        fit_fn, predict_fn,
    )
    print(fold_results.to_string(index=False))

    holdout_years = {season_to_year(s) for s in backtest_cfg['holdout_seasons']}
    mkt_rps, mkt_n = market_rps(df[df['year'].isin(holdout_years)])
    print(f'\nEnsemble RPS {summary["rps"]:.4f}  baseline {summary["baseline_rps"]:.4f}  '
          f'market {mkt_rps:.4f} (n={mkt_n})  '
          f'({summary["n_matches"]} matches, {summary["n_folds"]} folds)')
    print(f'  vs. {goals_model_name} alone: {registry.get_champion_rps(goals_model_name):.4f}')
    print(f'  vs. elo_xgb_{league} alone:  {elo_xgb_meta["rps"]:.4f}')
    print(f'  vs. logistic_{league} alone: {logistic_meta["rps"]:.4f}')

    out_dir = registry.MODELS_DIR / league / 'ensemble'
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'model_name': f'ensemble_{league}',
        'version': '-',
        'constituents': [goals_model_name, f'elo_xgb_{league}', f'logistic_{league}'],
        'rps': summary['rps'],
        'baseline_rps': summary['baseline_rps'],
        'market_rps': mkt_rps if mkt_n else None,
        'market_n_matches': mkt_n or None,
        'promoted_at': pd.Timestamp.utcnow().isoformat(),
    }
    with open(out_dir / 'metadata.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    print(f'\nWrote evaluation record to {out_dir / "metadata.yaml"}')


if __name__ == '__main__':
    main()
