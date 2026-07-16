import numpy as np
import pandas as pd

from match_predict.metrics.odds import implied_probs, market_rps


def test_implied_probs_sums_to_one_and_devigs():
    probs = implied_probs([2.0], [3.0], [4.0])
    assert probs.shape == (1, 3)
    assert np.isclose(probs.sum(), 1.0)

    raw = np.array([1 / 2.0, 1 / 3.0, 1 / 4.0])
    np.testing.assert_allclose(probs[0], raw / raw.sum())


def test_market_rps_skips_rows_without_odds():
    df = pd.DataFrame({
        'odds_home': [2.0, np.nan],
        'odds_draw': [3.0, 3.0],
        'odds_away': [4.0, 4.0],
        'score1': [1, 0],
        'score2': [0, 0],
    })
    score, n = market_rps(df)
    assert n == 1
    assert not np.isnan(score)


def test_market_rps_all_missing_returns_nan():
    df = pd.DataFrame({
        'odds_home': [np.nan],
        'odds_draw': [np.nan],
        'odds_away': [np.nan],
        'score1': [1],
        'score2': [0],
    })
    score, n = market_rps(df)
    assert n == 0
    assert np.isnan(score)
