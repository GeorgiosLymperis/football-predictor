import numpy as np
import pandas as pd
import pytest

import match_predict.features.data as data_module
from match_predict.features.data import DataValidationError, load_league_matches


def _write_season_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _base_rows():
    return {
        'Date': ['01/08/2024', '08/08/2024'],
        'HomeTeam': ['A', 'B'],
        'AwayTeam': ['B', 'A'],
        'FTHG': [2, 1],
        'FTAG': [0, 1],
        'FTR': ['H', 'D'],
        'HS': [10, 8],
        'AS': [5, 9],
        'HST': [6, 4],
        'AST': [2, 5],
        'HC': [4, 3],
        'AC': [2, 6],
        'AvgH': [1.8, 2.5],
        'AvgD': [3.5, 3.2],
        'AvgA': [4.0, 2.8],
    }


@pytest.fixture
def league_cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, 'DATA_DIR', tmp_path)
    league_dir = tmp_path / 'TESTLEAGUE'
    league_dir.mkdir()
    _write_season_csv(league_dir / 'TL-2024-2025.csv', _base_rows())
    return {
        'league_name': 'testleague',
        'raw_dir': 'TESTLEAGUE',
        'file_prefix': 'TL',
        'name_fixes': {},
        'teams': ['A', 'B'],
    }


def test_load_league_matches_extracts_match_stat_columns(league_cfg):
    df = load_league_matches(league_cfg)
    assert list(df['shots_home']) == [10, 8]
    assert list(df['shots_away']) == [5, 9]
    assert list(df['shots_target_home']) == [6, 4]
    assert list(df['corners_away']) == [2, 6]


def test_load_league_matches_missing_stat_columns_default_to_nan(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, 'DATA_DIR', tmp_path)
    league_dir = tmp_path / 'OLDLEAGUE'
    league_dir.mkdir()
    rows = _base_rows()
    for col in ('HS', 'AS', 'HST', 'AST', 'HC', 'AC'):
        del rows[col]
    _write_season_csv(league_dir / 'OL-2010-2011.csv', rows)
    cfg = {
        'league_name': 'oldleague', 'raw_dir': 'OLDLEAGUE', 'file_prefix': 'OL',
        'name_fixes': {}, 'teams': ['A', 'B'],
    }
    df = load_league_matches(cfg)
    assert df['shots_home'].isna().all()
    assert df['corners_away'].isna().all()


def test_load_league_matches_basic_fields(league_cfg):
    df = load_league_matches(league_cfg)
    assert list(df['team1']) == ['A', 'B']
    assert list(df['team2']) == ['B', 'A']
    assert list(df['result']) == ['Home', 'Draw']
    assert list(df['match_day']) == [1, 2]
    assert (df['year'] == 2024).all()


def test_load_league_matches_applies_name_fixes(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, 'DATA_DIR', tmp_path)
    league_dir = tmp_path / 'TESTLEAGUE'
    league_dir.mkdir()
    rows = _base_rows()
    rows['HomeTeam'] = ['Ajax Old Name', 'B']
    _write_season_csv(league_dir / 'TL-2024-2025.csv', rows)
    cfg = {
        'league_name': 'testleague', 'raw_dir': 'TESTLEAGUE', 'file_prefix': 'TL',
        'name_fixes': {'Ajax Old Name': 'Ajax'}, 'teams': ['Ajax', 'A', 'B'],
    }
    df = load_league_matches(cfg)
    assert df.loc[0, 'team1'] == 'Ajax'


def test_load_league_matches_rejects_unknown_team_names(league_cfg):
    league_cfg['teams'] = ['A']  # 'B' is not in the allowlist
    with pytest.raises(DataValidationError):
        load_league_matches(league_cfg)
