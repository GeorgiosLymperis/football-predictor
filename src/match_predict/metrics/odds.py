import numpy as np
import pandas as pd

from match_predict.metrics.rps import outcome_index, rps


def implied_probs(odds_home, odds_draw, odds_away) -> np.ndarray:
    """De-vigged market-implied [p_home, p_draw, p_away], one row per match.

    Args:
        odds_home, odds_draw, odds_away: array-likes of decimal odds.

    Returns:
        (n, 3) array of probabilities, each row summing to 1.
    """
    raw = np.column_stack([
        1 / np.asarray(odds_home, dtype=float),
        1 / np.asarray(odds_draw, dtype=float),
        1 / np.asarray(odds_away, dtype=float),
    ])
    return raw / raw.sum(axis=1, keepdims=True)


def market_rps(df: pd.DataFrame) -> tuple[float, int]:
    """Mean RPS of the market's own implied probabilities, over whichever
    rows of `df` have odds. `df` needs odds_home/odds_draw/odds_away and
    score1/score2 columns.

    Returns:
        (rps, n_matches).
    """
    with_odds = df.dropna(subset=['odds_home', 'odds_draw', 'odds_away'])
    if with_odds.empty:
        return float('nan'), 0

    probs = implied_probs(with_odds['odds_home'], with_odds['odds_draw'], with_odds['odds_away'])
    outcome = outcome_index(with_odds['score1'], with_odds['score2'])
    return rps(probs, outcome), len(with_odds)
