import numpy as np
import pandas as pd
import pytest

from match_predict.metrics.backtest import season_to_year, walk_forward


def test_season_to_year():
    assert season_to_year('2024-2025') == 2024


def _synthetic_df() -> pd.DataFrame:
    # One prior season (2023, purely historical) plus an evaluated season
    # (2024) with 4 matchdays, so a step of 2 produces exactly two folds.
    rows = []
    for year, n_days in ((2023, 4), (2024, 4)):
        for day in range(1, n_days + 1):
            rows.append({
                'year': year, 'match_day': day,
                'team1': 'A', 'team2': 'B',
                'score1': 1, 'score2': 0, 'result': 'Home',
            })
    return pd.DataFrame(rows)


def test_walk_forward_is_causal_and_expands_the_training_window():
    df = _synthetic_df()
    train_sizes = []

    def fit_fn(train_df):
        train_sizes.append(len(train_df))
        return None

    def predict_fn(fitted, eval_df):
        return np.tile([1 / 3, 1 / 3, 1 / 3], (len(eval_df), 1))

    fold_results, summary = walk_forward(df, ['2024-2025'], step_matchdays=2,
                                          fit_fn=fit_fn, predict_fn=predict_fn)

    assert summary['n_folds'] == 2
    assert summary['n_matches'] == 4
    # The training window for fold 2 must be strictly larger than fold 1's,
    # since it now also includes fold 1's matchdays.
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[1] > train_sizes[0]
    # No training data ever includes same-or-later matchdays than eval.
    assert fold_results['match_day_start'].tolist() == [1, 3]


def test_walk_forward_raises_when_no_folds_produced():
    df = _synthetic_df()

    def fit_fn(train_df):
        return None

    def predict_fn(fitted, eval_df):
        return np.tile([1 / 3, 1 / 3, 1 / 3], (len(eval_df), 1))

    with pytest.raises(ValueError, match='No folds produced'):
        walk_forward(df, ['2099-2100'], step_matchdays=2, fit_fn=fit_fn, predict_fn=predict_fn)
