import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema

from match_predict.mlops import registry
from match_predict.metrics.backtest import season_to_year, walk_forward
from match_predict.config import LEAGUES_DIR, load_config, load_league_config
from match_predict.features.data import load_league_matches
from match_predict.metrics.odds import market_rps
from match_predict.predict.poisson import predict_outcome_probs
from match_predict.training.poisson_model import (
    PoissonPyfuncModel,
    export_posterior_params,
    fit_poisson_model,
    prepare_df,
    run_loo,
)


def _predict_fn(params: dict, eval_df: pd.DataFrame) -> np.ndarray:
    probs = []
    for _, row in eval_df.iterrows():
        result = predict_outcome_probs(params, row['team1'], row['team2'])
        probs.append([result['p_home'], result['p_draw'], result['p_away']])
    return np.array(probs)


def _teams_in_seasons(df: pd.DataFrame, seasons: list[str]) -> list[str]:
    years = {season_to_year(s) for s in seasons}
    subset = df[df['year'].isin(years)]
    return sorted(set(subset['team1']) | set(subset['team2']))


def _recent_window(df: pd.DataFrame, n_seasons: int) -> pd.DataFrame:
    cutoff_years = sorted(df['year'].unique())[-n_seasons:]
    return df[df['year'].isin(cutoff_years)]


def run_backtest(df: pd.DataFrame, teams: list[str], cfg: dict, backtest_cfg: dict,
                  likelihood: str = 'poisson'):
    n_seasons = len(cfg['train_seasons'])

    def fit_fn(train_df):
        windowed = _recent_window(train_df, n_seasons)
        _, idata = fit_poisson_model(
            prepare_df(windowed, teams), teams,
            dixon_coles=cfg['dixon_coles'], likelihood=likelihood, **cfg['backtest_sampling'],
        )
        return export_posterior_params(idata, teams, cfg['dixon_coles'], likelihood=likelihood)

    return walk_forward(
        df, backtest_cfg['holdout_seasons'], backtest_cfg['poisson_step_matchdays'],
        fit_fn, _predict_fn,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--league', required=True, choices=sorted(p.stem for p in LEAGUES_DIR.glob('*.yaml')))
    parser.add_argument('--likelihood', choices=['poisson', 'neg_binomial'], default='poisson')
    parser.add_argument('--dry-run', action='store_true', help='backtest only, no training/promotion')
    args = parser.parse_args()

    league_cfg = load_league_config(args.league)
    cfg = load_config('poisson')
    backtest_cfg = load_config('backtest')
    short_name = 'poisson' if args.likelihood == 'poisson' else 'negbinom'
    model_name = f'{short_name}_{args.league}'
    df = load_league_matches(league_cfg)
    teams = _teams_in_seasons(df, backtest_cfg['selection_seasons'] + backtest_cfg['holdout_seasons'])

    print(f'[{args.league}] Walk-forward backtest ({args.likelihood}, holdout seasons)...')
    fold_results, summary = run_backtest(df, teams, cfg, backtest_cfg, likelihood=args.likelihood)
    print(fold_results.to_string(index=False))

    holdout_years = {season_to_year(s) for s in backtest_cfg['holdout_seasons']}
    mkt_rps, mkt_n = market_rps(df[df['year'].isin(holdout_years)])
    print(f'\nRPS {summary["rps"]:.4f}  baseline {summary["baseline_rps"]:.4f}  '
          f'market {mkt_rps:.4f} (n={mkt_n})  '
          f'({summary["n_matches"]} matches, {summary["n_folds"]} folds)')

    if args.dry_run:
        return

    registry.configure()

    print('\nFitting production model...')
    train_window = _recent_window(df, len(cfg['train_seasons']))
    train_df = prepare_df(train_window, teams)
    print(f'Training on {len(train_df)} matches from {sorted(train_window["year"].unique())}')
    model, idata = fit_poisson_model(
        train_df, teams, dixon_coles=cfg['dixon_coles'], likelihood=args.likelihood, **cfg['sampling'],
    )
    loo = run_loo(idata, model=model)
    params = export_posterior_params(idata, teams, cfg['dixon_coles'], likelihood=args.likelihood)

    tmp_dir = Path('mlruns_tmp') / model_name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    posterior_path = tmp_dir / 'posterior_params.npz'
    np.savez_compressed(posterior_path, **params)

    input_example = pd.DataFrame({'home_team': [teams[0]], 'away_team': [teams[1]]})
    signature = ModelSignature(
        inputs=Schema([ColSpec('string', 'home_team'), ColSpec('string', 'away_team')]),
        outputs=Schema([ColSpec('double', c) for c in ('p_home', 'p_draw', 'p_away', 'xg_home', 'xg_away')]),
    )

    version = registry.log_run(
        model_name,
        params={
            **cfg['sampling'], 'dixon_coles': cfg['dixon_coles'], 'decay_rate': cfg['decay_rate'],
            'likelihood': args.likelihood, 'league': args.league,
            'train_seasons': ','.join(str(y) for y in sorted(train_window['year'].unique())),
        },
        metrics={
            'rps': summary['rps'], 'baseline_rps': summary['baseline_rps'],
            'elpd_loo': float(loo.elpd_loo),
        },
        tags={'data_hash': str(pd.util.hash_pandas_object(df).sum())},
        python_model=PoissonPyfuncModel(),
        artifacts={'posterior_params': str(posterior_path)},
        input_example=input_example,
        signature=signature,
    )
    print(f'Registered {model_name} version {version} | RPS {summary["rps"]:.4f} '
          f'(baseline {summary["baseline_rps"]:.4f}, market {mkt_rps:.4f})')

    if registry.promote_if_better(model_name, version, summary['rps'], summary['baseline_rps'],
                                   backtest_cfg['promotion_margin']):
        out_dir = registry.export_champion(
            model_name, version, summary['rps'], summary['baseline_rps'],
            {'posterior_params.npz': str(posterior_path)},
            out_dir=registry.MODELS_DIR / args.league / short_name,
            market_rps=mkt_rps if mkt_n else None, market_n_matches=mkt_n or None,
        )
        print(f'Promoted to champion; exported to {out_dir}')
    else:
        print('Did not beat the current champion -> not promoted.')


if __name__ == '__main__':
    main()
