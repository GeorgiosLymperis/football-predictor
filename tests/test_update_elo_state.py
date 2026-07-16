import numpy as np
import pandas as pd

import match_predict.features.data as data_module
import scripts.update_elo_state as update_elo_state


def _toy_matches_csv() -> str:
    return (
        'Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,AvgH,AvgD,AvgA\n'
        '01/08/2024,A,B,2,0,H,10,5,6,2,4,2,1.8,3.5,4.0\n'
        '08/08/2024,B,A,1,1,D,8,9,4,5,3,6,2.5,3.2,2.8\n'
        '01/08/2025,A,B,0,0,D,7,7,3,3,2,2,2.0,3.3,3.6\n'
    )


def _fake_existing_state(feature_names, extra=None):
    state = dict(
        teams=np.array(['A', 'B'], dtype=object),
        current_teams=np.array(['A', 'B'], dtype=object),
        feature_names=np.array(feature_names, dtype=object),
        elo=np.array([1500.0, 1500.0]),
        pct=np.array([0.0, 0.0]),
        roll=np.zeros((2, 3)),
        ema=np.zeros((2, 3)),
        form5=np.array([np.nan, np.nan]),
        h2h_ppg=np.full((2, 2), np.nan),
        h2h_count=np.zeros((2, 2)),
        macd=np.zeros(2),
        macd_hist=np.zeros(2),
        trend_slope=np.zeros(2),
        rsi=np.full(2, 50.0),
        stats_for=np.zeros((2, 3)),
        stats_against=np.zeros((2, 3)),
        elo_history=np.array([np.array([1500.0]), np.array([1500.0])], dtype=object),
        elo_dates=np.array([np.array([], dtype='datetime64[ns]')] * 2, dtype=object),
    )
    if extra:
        state.update(extra)
    return state


def _setup(tmp_path, monkeypatch, league='testleague', existing_models=('elo_xgb', 'logistic')):
    monkeypatch.setattr(data_module, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(update_elo_state, 'MODELS_DIR', tmp_path / 'models')

    league_dir = tmp_path / 'TESTLEAGUE'
    league_dir.mkdir()
    (league_dir / 'TL-2024-2025.csv').write_text(_toy_matches_csv())

    cfg = {
        'league_name': league, 'raw_dir': 'TESTLEAGUE', 'file_prefix': 'TL',
        'name_fixes': {}, 'teams': ['A', 'B'],
    }
    monkeypatch.setattr(update_elo_state, 'load_league_config', lambda lg: cfg)

    state_paths = {}
    for model_dir in existing_models:
        model_path = tmp_path / 'models' / league / model_dir
        model_path.mkdir(parents=True)
        state_path = model_path / 'team_state.npz'
        extra = {'classes': np.array(['Away', 'Draw', 'Home'], dtype=object)} if model_dir == 'elo_xgb' else None
        np.savez_compressed(state_path, **_fake_existing_state(['EloDifference'], extra=extra))
        state_paths[model_dir] = state_path

    return cfg, state_paths


def test_update_league_preserves_extra_keys_like_classes(tmp_path, monkeypatch):
    _, state_paths = _setup(tmp_path, monkeypatch)

    update_elo_state.update_league('testleague')

    refreshed = np.load(state_paths['elo_xgb'], allow_pickle=True)
    assert list(refreshed['classes']) == ['Away', 'Draw', 'Home']


def test_update_league_preserves_feature_names_and_recomputes_elo(tmp_path, monkeypatch):
    _, state_paths = _setup(tmp_path, monkeypatch)

    update_elo_state.update_league('testleague')

    refreshed = np.load(state_paths['elo_xgb'], allow_pickle=True)
    assert list(refreshed['feature_names']) == ['EloDifference']
    # A won its first match (2-0) so its Elo should have moved off the
    # 1500 placeholder that was in the stale fixture state.
    teams = list(refreshed['teams'])
    assert refreshed['elo'][teams.index('A')] != 1500.0


def test_update_league_skips_models_without_existing_state(tmp_path, monkeypatch):
    _, state_paths = _setup(tmp_path, monkeypatch, existing_models=('elo_xgb',))
    models_dir = state_paths['elo_xgb'].parent.parent

    update_elo_state.update_league('testleague')

    assert not (models_dir / 'mlp' / 'team_state.npz').exists()


def test_update_league_refreshes_current_teams_from_latest_season(tmp_path, monkeypatch):
    _, state_paths = _setup(tmp_path, monkeypatch)

    update_elo_state.update_league('testleague')

    refreshed = np.load(state_paths['elo_xgb'], allow_pickle=True)
    # The 2025 season (the max year in the fixture) only has A vs B.
    assert set(refreshed['current_teams']) == {'A', 'B'}
