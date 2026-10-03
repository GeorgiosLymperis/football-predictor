import numpy as np
import pandas as pd
import pytest

from match_predict.simulation import UnratedTeamsError, remaining_fixtures, simulate_season, summarise


def _params(teams, attack, defence=None, draws=50):
    n = len(teams)
    return dict(
        teams=np.array(teams, dtype=object),
        attack=np.tile(attack, (draws, 1)).astype(float),
        defence=np.tile(defence if defence is not None else np.zeros(n), (draws, 1)).astype(float),
        home_adv=np.full(draws, 0.2),
        intercept=np.full(draws, 0.2),
    )


def _season(rows):
    return pd.DataFrame(rows, columns=['team1', 'team2', 'score1', 'score2'])


def test_remaining_fixtures_is_the_unplayed_double_round_robin():
    season = _season([('A', 'B', 1, 0)])
    assert remaining_fixtures(season, ['A', 'B', 'C']) == [
        ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B'),
    ]


def test_finished_season_reproduces_the_final_table():
    rows = [('A', 'B', 2, 0), ('B', 'A', 0, 1), ('A', 'C', 1, 1), ('C', 'A', 0, 3),
            ('B', 'C', 1, 0), ('C', 'B', 2, 2)]
    teams, points, positions = simulate_season(_params(['A', 'B', 'C'], [0, 0, 0]), _season(rows), n_sims=20)

    assert teams == ['A', 'B', 'C']
    assert points[0].tolist() == [10, 4, 2]
    assert (positions == np.array([1, 2, 3])).all()


def test_much_stronger_team_almost_always_wins_the_league():
    season = _season([('A', 'B', 0, 0)])
    teams = ['A', 'B', 'C', 'D']
    _, points, positions = simulate_season(_params(teams, [2.0, 0, 0, 0]), season, n_sims=2000)

    assert (positions[:, 0] == 1).mean() > 0.95
    # Every team plays 2 * (4 - 1) = 6 matches, so points stay within 0..18.
    assert points.min() >= 0 and points.max() <= 18


def test_unrated_team_is_reported():
    season = _season([('A', 'Promoted', 1, 0)])
    with pytest.raises(UnratedTeamsError) as err:
        simulate_season(_params(['A', 'B'], [0, 0]), season)
    assert err.value.teams == ['Promoted']


def test_summarise_probabilities():
    teams = ['A', 'B', 'C', 'D', 'E']
    positions = np.array([[1, 2, 3, 4, 5], [2, 1, 3, 5, 4]])
    points = np.array([[12, 9, 6, 3, 0], [9, 12, 6, 0, 3]], dtype=float)
    out = summarise(teams, pd.Series({'A': 3}), points, positions,
                    relegation_spots=1, relegation_playoff=True).set_index('team')

    assert out.loc['A', 'p_title'] == 0.5
    assert out.loc['C', 'p_top4'] == 1.0
    assert out.loc['E', 'p_relegation'] == 0.5 and out.loc['D', 'p_relegation'] == 0.5
    assert out.loc['D', 'p_relegation_playoff'] == 0.5
    assert out.loc['A', 'points_now'] == 3 and out.loc['B', 'points_now'] == 0
    assert out.index[0] in ('A', 'B')
