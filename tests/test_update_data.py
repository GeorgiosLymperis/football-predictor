from pathlib import Path
from unittest.mock import Mock

import pytest

import match_predict.features.data as data_module
import scripts.update_data as update_data


GOOD_CSV = (
    'Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,AvgH,AvgD,AvgA\n'
    '01/08/2025,A,B,2,0,H,10,5,6,2,4,2,1.8,3.5,4.0\n'
    '08/08/2025,B,A,1,1,D,8,9,4,5,3,6,2.5,3.2,2.8\n'
)


@pytest.fixture
def league_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(update_data, 'DATA_DIR', tmp_path)
    league_dir = tmp_path / 'TESTLEAGUE'
    league_dir.mkdir()
    dest = league_dir / 'TL-2025-2026.csv'
    dest.write_text(GOOD_CSV)
    cfg = {
        'league_name': 'testleague', 'raw_dir': 'TESTLEAGUE', 'file_prefix': 'TL',
        'name_fixes': {}, 'teams': ['A', 'B'],
    }
    monkeypatch.setattr(update_data, 'load_league_config', lambda league: cfg)
    monkeypatch.setitem(update_data.DIVISION_CODES, 'testleague', 'TL1')
    return dest


def test_update_league_writes_valid_download(league_setup, monkeypatch):
    extra_row = GOOD_CSV + '15/08/2025,A,B,3,1,H,12,4,7,1,5,3,1.5,4.0,5.5\n'
    monkeypatch.setattr(update_data.requests, 'get',
                         lambda *a, **k: Mock(status_code=200, content=extra_row.encode()))

    ok = update_data.update_league('testleague')

    assert ok is True
    assert league_setup.read_text() == extra_row


def test_update_league_leaves_file_untouched_on_http_error(league_setup, monkeypatch):
    original = league_setup.read_text()
    monkeypatch.setattr(update_data.requests, 'get',
                         lambda *a, **k: Mock(status_code=404, content=b''))

    ok = update_data.update_league('testleague')

    assert ok is False
    assert league_setup.read_text() == original


def test_update_league_reverts_on_validation_failure(league_setup, monkeypatch):
    original = league_setup.read_text()
    # Garbage content that will fail load_league_matches (missing columns).
    monkeypatch.setattr(update_data.requests, 'get',
                         lambda *a, **k: Mock(status_code=200, content=b'not,a,valid,csv\n1,2,3,4\n'))

    ok = update_data.update_league('testleague')

    assert ok is False
    assert league_setup.read_text() == original


def test_update_league_reverts_on_unknown_team_name(league_setup, monkeypatch):
    """A download that introduces a team outside the configured allowlist
    (e.g. a newly promoted club before the config is updated) must not
    silently corrupt the local dataset."""
    original = league_setup.read_text()
    bad = GOOD_CSV + '15/08/2025,A,NewPromotedClub,1,0,H,10,5,6,2,4,2,1.8,3.5,4.0\n'
    monkeypatch.setattr(update_data.requests, 'get',
                         lambda *a, **k: Mock(status_code=200, content=bad.encode()))

    ok = update_data.update_league('testleague')

    assert ok is False
    assert league_setup.read_text() == original
