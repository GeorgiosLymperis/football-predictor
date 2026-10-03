import numpy as np
import pandas as pd
import pytest

from match_predict.standings import league_table, momentum_label, with_elo


def _matches(rows, year=2026):
    df = pd.DataFrame(rows, columns=['team1', 'team2', 'score1', 'score2'])
    df['date'] = pd.date_range('2026-08-01', periods=len(df), freq='7D')
    df['year'] = year
    return df


def test_league_table_points_goals_and_order():
    old_season = _matches([('A', 'B', 0, 5)], year=2025)
    season = _matches([
        ('A', 'B', 2, 0),  # A W, B L
        ('B', 'C', 1, 1),  # B D, C D
        ('C', 'A', 3, 1),  # C W, A L
    ])
    table = league_table(pd.concat([old_season, season], ignore_index=True)).set_index('team')

    # Only the latest season counts.
    assert table['played'].tolist() == [2, 2, 2]
    assert table.loc['C', ['points', 'gf', 'ga', 'gd']].tolist() == [4, 4, 2, 2]
    assert table.loc['A', ['won', 'drawn', 'lost', 'points', 'gd']].tolist() == [1, 0, 1, 3, 0]
    assert table.loc['B', 'points'] == 1
    assert table['position'].to_dict() == {'C': 1, 'A': 2, 'B': 3}
    # Form is oldest -> newest.
    assert table.loc['A', 'form'] == ['W', 'L']


def test_ties_broken_by_goal_difference_then_goals_scored():
    season = _matches([
        ('A', 'X', 1, 0),  # A: 3 pts, gd +1, gf 1
        ('B', 'X', 3, 2),  # B: 3 pts, gd +1, gf 3
        ('C', 'X', 2, 0),  # C: 3 pts, gd +2
    ])
    order = league_table(season)['team'].tolist()
    assert order[:3] == ['C', 'B', 'A']


def test_form_keeps_last_five_results():
    season = _matches([('A', 'B', i % 2, 0) for i in range(7)])  # D W D W D W D for A
    table = league_table(season).set_index('team')
    assert table.loc['A', 'form'] == ['D', 'W', 'D', 'W', 'D']


@pytest.mark.parametrize('slope, label', [(2.0, 'up'), (-2.0, 'down'), (1.0, 'steady'), (-1.5, 'steady')])
def test_momentum_label(slope, label):
    assert momentum_label(slope) == label


def test_with_elo_ranks_compares_and_handles_unknown_teams():
    season = _matches([('A', 'B', 1, 0), ('C', 'B', 1, 0), ('A', 'C', 1, 0)])
    state = {
        'teams': np.array(['A', 'B', 'C'], dtype=object),
        'elo': np.array([1500.0, 1600.0, 1450.0]),
        'trend_slope': np.array([3.0, -0.5, -4.0]),
    }
    out = with_elo(league_table(season), state).set_index('team')

    assert out.loc['B', 'elo_rank'] == 1
    assert out.loc['B', 'vs_elo'] == 1 - 3  # 2 places lower in the table than its Elo rank
    assert out['momentum'].to_dict() == {'A': 'up', 'C': 'down', 'B': 'steady'}

    state_without_c = {k: v[:2] for k, v in state.items()}
    partial = with_elo(league_table(season), state_without_c).set_index('team')
    assert np.isnan(partial.loc['C', 'elo'])
    assert pd.isna(partial.loc['C', 'momentum'])
