from typing import Callable

import numpy as np
import pandas as pd

from match_predict.metrics.rps import outcome_index, rps

FitFn = Callable[[pd.DataFrame], object]
PredictFn = Callable[[object, pd.DataFrame], np.ndarray]


def season_to_year(season: str) -> int:
    return int(season.split('-')[0])


def walk_forward(
    df: pd.DataFrame,
    seasons: list[str],
    step_matchdays: int,
    fit_fn: FitFn,
    predict_fn: PredictFn,
) -> tuple[pd.DataFrame, dict]:
    """Walk forward through `seasons`, refitting every `step_matchdays`.

    Args:
        df: full chronologically-sorted match history with year, match_day,
            score1, score2, result columns.
        seasons: season strings (e.g. "2024-2025") to evaluate folds within.
            Matchday chunks never cross a season boundary, so `step_matchdays`
            larger than a season's matchday count collapses to one fold per
            season.
        fit_fn: train_df -> opaque fitted-model object.
        predict_fn: (fitted-model, eval_df) -> (n, 3) array of [p_home, p_draw, p_away].

    Returns:
        (fold_results, summary): per-fold RPS/baseline_rps DataFrame, and a
        match-count-weighted summary dict.
    """
    years = {season_to_year(s) for s in seasons}
    folds = []
    for year in sorted(years):
        matchdays = sorted(df.loc[df['year'] == year, 'match_day'].unique())
        for i in range(0, len(matchdays), step_matchdays):
            chunk = matchdays[i:i + step_matchdays]
            train_df = df[(df['year'] < year) | ((df['year'] == year) & (df['match_day'] < chunk[0]))]
            eval_df = df[(df['year'] == year) & (df['match_day'].isin(chunk))]
            if train_df.empty or eval_df.empty:
                continue

            fitted = fit_fn(train_df)
            probs = predict_fn(fitted, eval_df)
            outcome = outcome_index(eval_df['score1'], eval_df['score2'])
            fold_rps = rps(probs, outcome)

            prior = train_df['result'].value_counts(normalize=True)
            baseline_probs = np.tile(
                [prior.get('Home', 0.0), prior.get('Draw', 0.0), prior.get('Away', 0.0)],
                (len(eval_df), 1),
            )
            baseline_rps = rps(baseline_probs, outcome)

            folds.append({
                'year': year, 'match_day_start': chunk[0], 'match_day_end': chunk[-1],
                'n_matches': len(eval_df), 'rps': fold_rps, 'baseline_rps': baseline_rps,
            })

    fold_results = pd.DataFrame(folds)
    if fold_results.empty:
        raise ValueError(f'No folds produced for seasons {seasons} -> check the data covers them.')

    weights = fold_results['n_matches']
    summary = {
        'rps': float(np.average(fold_results['rps'], weights=weights)),
        'baseline_rps': float(np.average(fold_results['baseline_rps'], weights=weights)),
        'n_matches': int(weights.sum()),
        'n_folds': len(fold_results),
    }
    return fold_results, summary
