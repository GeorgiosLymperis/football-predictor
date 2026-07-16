import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from match_predict.mlops import registry
from match_predict.metrics.backtest import season_to_year, walk_forward
from match_predict.config import LEAGUES_DIR, load_config, load_league_config
from match_predict.features.data import load_league_matches
from match_predict.features.elo import FEATURE_VARIANTS, build_features, export_state
from match_predict.metrics.odds import market_rps
from match_predict.training.logistic_model import LogisticPyfuncModel, hda_probs, impute

MODEL_PREFIX = 'logistic'


def _make_fns(features: list[str], logistic_params: dict):
    def fit_fn(train_df):
        encoder = LabelEncoder()
        y = encoder.fit_transform(train_df['result'])
        impute_means = train_df[features].mean()
        X = impute(train_df[features], impute_means)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(**logistic_params)
        model.fit(X_scaled, y)
        return model, scaler, encoder, impute_means

    def predict_fn(fitted, eval_df):
        model, scaler, encoder, impute_means = fitted
        return hda_probs(model, scaler, encoder, eval_df[features], impute_means)

    return fit_fn, predict_fn


def select_variant(sl_features: pd.DataFrame, cfg: dict, backtest_cfg: dict) -> str:
    step = backtest_cfg['xgb_step_matchdays']
    scores = {}
    for variant in cfg['feature_variants']:
        fit_fn, predict_fn = _make_fns(FEATURE_VARIANTS[variant], cfg['logistic_params'])
        _, summary = walk_forward(sl_features, backtest_cfg['selection_seasons'], step, fit_fn, predict_fn)
        scores[variant] = summary['rps']
        print(f'  {variant:20s} RPS {summary["rps"]:.4f}')
    return min(scores, key=scores.get)


def run_backtest(sl_features: pd.DataFrame, cfg: dict, backtest_cfg: dict):
    print('Selecting feature variant via walk-forward CV (selection seasons)...')
    best_variant = select_variant(sl_features, cfg, backtest_cfg)
    print(f'Best variant: {best_variant}')

    print('\nWalk-forward backtest (holdout seasons, winning variant)...')
    features = FEATURE_VARIANTS[best_variant]
    fit_fn, predict_fn = _make_fns(features, cfg['logistic_params'])
    fold_results, summary = walk_forward(
        sl_features, backtest_cfg['holdout_seasons'], backtest_cfg['xgb_step_matchdays'],
        fit_fn, predict_fn,
    )
    return best_variant, fold_results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--league', required=True, choices=sorted(p.stem for p in LEAGUES_DIR.glob('*.yaml')))
    parser.add_argument('--dry-run', action='store_true', help='backtest only, no training/promotion')
    args = parser.parse_args()

    league_cfg = load_league_config(args.league)
    cfg = load_config('logistic')
    backtest_cfg = load_config('backtest')
    model_name = f'{MODEL_PREFIX}_{args.league}'

    df = load_league_matches(league_cfg)
    sl_features, states, h2h = build_features(df)

    best_variant, fold_results, summary = run_backtest(sl_features, cfg, backtest_cfg)
    print(fold_results.to_string(index=False))

    holdout_years = {season_to_year(s) for s in backtest_cfg['holdout_seasons']}
    mkt_rps, mkt_n = market_rps(df[df['year'].isin(holdout_years)])
    print(f'\nRPS {summary["rps"]:.4f}  baseline {summary["baseline_rps"]:.4f}  '
          f'market {mkt_rps:.4f} (n={mkt_n})  '
          f'({summary["n_matches"]} matches, {summary["n_folds"]} folds)')

    if args.dry_run:
        return

    registry.configure()
    features = FEATURE_VARIANTS[best_variant]

    print('\nFitting production model...')
    encoder = LabelEncoder()
    y = encoder.fit_transform(sl_features['result'])
    impute_means = sl_features[features].mean()
    X = impute(sl_features[features], impute_means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(**cfg['logistic_params'])
    model.fit(X_scaled, y)

    current_teams = sorted(set(df[df['year'] == df['year'].max()]['team1']))
    state = export_state(states, h2h, features, current_teams)

    logistic_params_out = dict(
        coef=model.coef_,
        intercept=model.intercept_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        impute_means=impute_means.to_numpy(),
        classes=np.array(encoder.classes_, dtype=object),
        feature_names=np.array(features, dtype=object),
    )

    tmp_dir = Path('mlruns_tmp') / model_name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    params_path = tmp_dir / 'logistic_params.npz'
    state_path = tmp_dir / 'team_state.npz'
    np.savez_compressed(params_path, **logistic_params_out)
    np.savez_compressed(state_path, **state)

    input_example = pd.DataFrame({'home_team': [current_teams[0]], 'away_team': [current_teams[1]]})
    signature = ModelSignature(
        inputs=Schema([ColSpec('string', 'home_team'), ColSpec('string', 'away_team')]),
        outputs=Schema([ColSpec('double', c) for c in ('p_home', 'p_draw', 'p_away')]),
    )

    version = registry.log_run(
        model_name,
        params={**cfg['logistic_params'], 'feature_variant': best_variant, 'league': args.league},
        metrics={'rps': summary['rps'], 'baseline_rps': summary['baseline_rps']},
        tags={'data_hash': str(pd.util.hash_pandas_object(df).sum())},
        python_model=LogisticPyfuncModel(),
        artifacts={'logistic_params': str(params_path), 'team_state': str(state_path)},
        input_example=input_example,
        signature=signature,
    )
    print(f'Registered {model_name} version {version} | RPS {summary["rps"]:.4f} '
          f'(baseline {summary["baseline_rps"]:.4f}, market {mkt_rps:.4f})')

    if registry.promote_if_better(model_name, version, summary['rps'], summary['baseline_rps'],
                                   backtest_cfg['promotion_margin']):
        out_dir = registry.export_champion(
            model_name, version, summary['rps'], summary['baseline_rps'],
            {'logistic_params.npz': str(params_path), 'team_state.npz': str(state_path)},
            out_dir=registry.MODELS_DIR / args.league / 'logistic',
            market_rps=mkt_rps if mkt_n else None, market_n_matches=mkt_n or None,
        )
        print(f'Promoted to champion; exported to {out_dir}')
    else:
        print('Did not beat the current champion -> not promoted.')


if __name__ == '__main__':
    main()
