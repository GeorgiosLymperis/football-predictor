import numpy as np
import pandas as pd
import pytest

from match_predict.features.elo import (
    FEATURE_VARIANTS,
    TeamState,
    build_features,
    elo_expected_score,
    elo_update,
    export_state,
    h2h_features,
)


def test_elo_expected_score_is_symmetric_around_half():
    assert elo_expected_score(1500, 1500) == pytest.approx(0.5)
    assert elo_expected_score(1600, 1500) > 0.5
    assert elo_expected_score(1500, 1600) < 0.5


def test_elo_update_rewards_the_winner():
    home, away = elo_update(1500, 1500, goals_home=2, goals_away=0)
    assert home > 1500
    assert away < 1500
    # zero-sum
    assert (home - 1500) == pytest.approx(-(away - 1500))


def test_elo_update_draw_with_no_home_advantage_is_a_wash():
    home, away = elo_update(1500, 1500, goals_home=1, goals_away=1, ha=0)
    assert home == pytest.approx(1500)
    assert away == pytest.approx(1500)


class TestTeamState:
    def test_form_is_nan_before_any_matches(self):
        assert np.isnan(TeamState().form())

    def test_form_is_mean_of_recent_points(self):
        state = TeamState()
        for pts in (3, 3, 0, 1, 3):
            state.points.append(pts)
        assert state.form(window=5) == pytest.approx(2.0)

    def test_form_respects_window(self):
        state = TeamState()
        for pts in (0, 0, 0, 3, 3):
            state.points.append(pts)
        assert state.form(window=2) == pytest.approx(3.0)

    def test_form_rejects_nonpositive_window(self):
        with pytest.raises(ValueError):
            TeamState().form(window=0)

    def test_roll_includes_initial_elo(self):
        state = TeamState()
        assert state.roll(5) == pytest.approx(1500.0)

    def test_pct_change_zero_with_short_history(self):
        assert TeamState().pct_change() == 0.0

    def test_trend_slope_zero_with_single_point(self):
        assert TeamState().trend_slope() == 0.0

    def test_trend_slope_detects_upward_trend(self):
        state = TeamState()
        for elo in (1500, 1520, 1540, 1560):
            state.push(elo)
        assert state.trend_slope(window=4) > 0

    def test_rsi_is_50_with_no_history(self):
        assert TeamState().rsi() == 50.0

    def test_rsi_is_100_when_only_gains(self):
        state = TeamState()
        for elo in (1510, 1520, 1530, 1540):
            state.push(elo)
        assert state.rsi(window=3) == 100.0

    def test_stat_roll_rejects_unknown_stat(self):
        with pytest.raises(ValueError):
            TeamState().stat_roll('unknown_stat', against=False)

    def test_stat_roll_accepts_match_stats_keys(self):
        state = TeamState()
        state.stats_for['shots'].append(10)
        assert state.stat_roll('shots', against=False) == pytest.approx(10.0)

    def test_stat_roll_nan_when_no_data(self):
        assert np.isnan(TeamState().stat_roll('shots', against=False))


def test_h2h_features_nan_for_unmet_teams():
    feats = h2h_features({}, 'A', 'B')
    assert np.isnan(feats['H2HHomePPG'])
    assert np.isnan(feats['H2HAwayPPG'])
    assert feats['H2HCount'] == 0.0


def test_h2h_features_uses_directional_history():
    h2h = {('A', 'B'): [3, 0], ('B', 'A'): [0, 3]}
    feats = h2h_features(h2h, 'A', 'B')
    assert feats['H2HHomePPG'] == pytest.approx(1.5)
    assert feats['H2HAwayPPG'] == pytest.approx(1.5)
    assert feats['H2HCount'] == 2.0


def _toy_matches() -> pd.DataFrame:
    return pd.DataFrame({
        'team1': ['A', 'B', 'A'],
        'team2': ['B', 'A', 'B'],
        'score1': [2, 1, 0],
        'score2': [0, 1, 0],
        'date': pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-15']),
    })


def test_build_features_runs_without_error_and_produces_all_variant_columns():
    df = _toy_matches()
    out, states, h2h = build_features(df)

    assert len(out) == len(df)
    assert set(states) == {'A', 'B'}
    all_feature_cols = {c for cols in FEATURE_VARIANTS.values() for c in cols}
    assert all_feature_cols <= set(out.columns)


def test_build_features_form_matches_points_history():
    df = _toy_matches()
    out, states, _ = build_features(df)
    # Third match (A home vs B): A's prior results were a 2-0 win (3 pts)
    # then a 1-1 draw away (1 pt) -> pre-match HomeForm5 should be their mean.
    assert out.loc[2, 'HomeForm5'] == pytest.approx(2.0)
    # B's prior results were a 0-3 loss (0 pts) then a 1-1 draw at home (1 pt).
    assert out.loc[2, 'AwayForm5'] == pytest.approx(0.5)


def test_export_state_shapes():
    df = _toy_matches()
    out, states, h2h = build_features(df)
    state = export_state(states, h2h, ['EloDifference'], current_teams=['A', 'B'])
    n = len(states)
    assert state['elo'].shape == (n,)
    assert state['h2h_ppg'].shape == (n, n)
    assert set(state['teams']) == {'A', 'B'}
