import numpy as np
import pandas as pd

from match_predict.features.elo import MATCH_STATS

ROLL_WINDOWS = (5, 10, 15)
EMA_SPANS = (5, 10, 15)


def build_feature_row(state: dict, home: str, away: str) -> pd.DataFrame:
    """One-row DataFrame of pre-match features for `home` vs `away`.

    Args:
        state: dict as saved to team_state.npz. The keys are 'teams' (array of names),
            'elo', 'pct' (arrays indexed like teams), 'roll' (n_teams, 3),
            'ema' (n_teams, 3), optionally 'form5' (n_teams,), and optionally
            'h2h_ppg'/'h2h_count' (n_teams, n_teams) pairwise matrices, plus
            'feature_names' declaring which columns (and their order) the
            trained model expects.
    """
    teams = list(state['teams'])
    h, a = teams.index(home), teams.index(away)

    f = {
        'EloDifference': state['elo'][h] - state['elo'][a],
        'HomeElo': state['elo'][h], 'AwayElo': state['elo'][a],
        'HomeEloPctChange': state['pct'][h], 'AwayEloPctChange': state['pct'][a],
        'FormDiff': state['pct'][h] - state['pct'][a],
    }
    for i, w in enumerate(ROLL_WINDOWS):
        f[f'HomeRoll{w}'] = state['roll'][h, i]
        f[f'AwayRoll{w}'] = state['roll'][a, i]
    for i, s in enumerate(EMA_SPANS):
        f[f'HomeEMA{s}'] = state['ema'][h, i]
        f[f'AwayEMA{s}'] = state['ema'][a, i]
    if 'form5' in state:
        f['HomeForm5'] = state['form5'][h]
        f['AwayForm5'] = state['form5'][a]
        f['Form5Diff'] = f['HomeForm5'] - f['AwayForm5']
    if 'h2h_ppg' in state:
        f['H2HHomePPG'] = state['h2h_ppg'][h, a]
        f['H2HAwayPPG'] = state['h2h_ppg'][a, h]
        f['H2HCount'] = state['h2h_count'][h, a]
    if 'macd' in state:
        f['HomeMACD'] = state['macd'][h]
        f['AwayMACD'] = state['macd'][a]
        f['MACDDiff'] = f['HomeMACD'] - f['AwayMACD']
        f['HomeMACDHist'] = state['macd_hist'][h]
        f['AwayMACDHist'] = state['macd_hist'][a]
        f['HomeTrendSlope'] = state['trend_slope'][h]
        f['AwayTrendSlope'] = state['trend_slope'][a]
        f['TrendSlopeDiff'] = f['HomeTrendSlope'] - f['AwayTrendSlope']
        f['HomeRSI'] = state['rsi'][h]
        f['AwayRSI'] = state['rsi'][a]
        f['RSIDiff'] = f['HomeRSI'] - f['AwayRSI']
    if 'stats_for' in state:
        for i, label in enumerate(MATCH_STATS.values()):
            f[f'Home{label}For'] = state['stats_for'][h, i]
            f[f'Away{label}For'] = state['stats_for'][a, i]
            f[f'Home{label}Against'] = state['stats_against'][h, i]
            f[f'Away{label}Against'] = state['stats_against'][a, i]
            f[f'{label}ForDiff'] = f[f'Home{label}For'] - f[f'Away{label}For']
            f[f'{label}AgainstDiff'] = f[f'Home{label}Against'] - f[f'Away{label}Against']

    cols = list(state['feature_names'])
    return pd.DataFrame([f])[cols]