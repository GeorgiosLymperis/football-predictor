import numpy as np

from match_predict.predict.xgb import build_feature_row


def _minimal_state(**extra):
    state = dict(
        teams=np.array(['A', 'B'], dtype=object),
        elo=np.array([1550.0, 1450.0]),
        pct=np.array([1.0, -2.0]),
        roll=np.array([[1540.0, 1530.0, 1520.0], [1460.0, 1470.0, 1480.0]]),
        ema=np.array([[1545.0, 1535.0, 1525.0], [1455.0, 1465.0, 1475.0]]),
        feature_names=np.array(['EloDifference', 'HomeElo', 'AwayElo'], dtype=object),
    )
    state.update(extra)
    return state


def test_build_feature_row_basic_fields_and_column_order():
    state = _minimal_state()
    row = build_feature_row(state, 'A', 'B')
    assert list(row.columns) == ['EloDifference', 'HomeElo', 'AwayElo']
    assert row['EloDifference'].iloc[0] == 100.0
    assert row['HomeElo'].iloc[0] == 1550.0
    assert row['AwayElo'].iloc[0] == 1450.0


def test_build_feature_row_includes_form_only_when_present_in_state():
    state = _minimal_state(
        feature_names=np.array(['Form5Diff'], dtype=object),
        form5=np.array([2.0, 1.0]),
    )
    row = build_feature_row(state, 'A', 'B')
    assert row['Form5Diff'].iloc[0] == 1.0


def test_build_feature_row_h2h_is_directional():
    state = _minimal_state(
        feature_names=np.array(['H2HHomePPG', 'H2HAwayPPG'], dtype=object),
        h2h_ppg=np.array([[np.nan, 2.5], [0.5, np.nan]]),
        h2h_count=np.array([[0, 3], [3, 0]]),
    )
    row = build_feature_row(state, 'A', 'B')
    assert row['H2HHomePPG'].iloc[0] == 2.5
    assert row['H2HAwayPPG'].iloc[0] == 0.5
