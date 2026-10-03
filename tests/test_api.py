from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import api
from match_predict import service


@pytest.fixture(autouse=True)
def _no_prediction_logging(monkeypatch):
    monkeypatch.setattr(service, 'log_prediction_safe', lambda *a, **k: None)


@pytest.fixture
def client():
    return TestClient(api.app)


def _two_current_teams(client, league):
    teams = client.get(f'/leagues/{league}/teams').json()
    return teams[0]['team'], teams[1]['team']


def test_list_leagues_matches_model_dirs(client):
    resp = client.get('/leagues')
    assert resp.status_code == 200
    assert 'greek' in resp.json()


def test_teams_sorted_by_elo_and_ranked(client):
    teams = client.get('/leagues/greek/teams').json()
    elos = [t['elo'] for t in teams]
    assert elos == sorted(elos, reverse=True)
    assert [t['rank'] for t in teams] == list(range(1, len(teams) + 1))


def test_get_team_is_case_insensitive(client):
    name, _ = _two_current_teams(client, 'greek')
    resp = client.get(f'/leagues/greek/teams/{name.upper()}')
    assert resp.status_code == 200
    body = resp.json()
    assert body['team'] == name
    assert body['rank'] == 1


@pytest.mark.parametrize('url', [
    '/leagues/nope/teams',
    '/leagues/greek/teams/Not A Team',
    '/leagues/greek/predict?home=Not A Team&away=AEK',
])
def test_unknown_league_or_team_is_404(client, url):
    assert client.get(url).status_code == 404


def test_predict_same_team_is_422(client):
    name, _ = _two_current_teams(client, 'greek')
    assert client.get('/leagues/greek/predict', params={'home': name, 'away': name}).status_code == 422


@pytest.mark.parametrize('league', service.leagues())
def test_predict_defaults_to_lowest_rps_model(client, league):
    home, away = _two_current_teams(client, league)
    body = client.get(f'/leagues/{league}/predict', params={'home': home, 'away': away}).json()

    probs = body['probs']
    assert probs['home'] + probs['draw'] + probs['away'] == pytest.approx(1.0, abs=1e-6)
    lm = service.load_league(league)
    best_rps = min(lm.meta[m]['rps'] for m in service.OUTCOME_MODELS if lm.meta[m] is not None)
    assert body['model']['rps'] == pytest.approx(best_rps)


def test_predict_model_override(client):
    home, away = _two_current_teams(client, 'greek')
    body = client.get('/leagues/greek/predict', params={'home': home, 'away': away, 'model': 'poisson'}).json()
    assert body['model']['name'] == 'poisson'


def test_predict_unknown_model_is_422(client):
    home, away = _two_current_teams(client, 'greek')
    resp = client.get('/leagues/greek/predict', params={'home': home, 'away': away, 'model': 'gpt'})
    assert resp.status_code == 422


def test_ensemble_is_mean_of_constituents():
    lm = service.load_league('greek')
    home, away = [t for t, _ in service.current_elo_table(lm)[:2]]
    p = service.predict_match(lm, home, away).probs
    for i in range(3):
        assert p['ensemble'][i] == pytest.approx((p['poisson'][i] + p['elo_xgb'][i] + p['logistic'][i]) / 3)


FIXTURES_CSV = (
    '﻿Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA\n'
    'G1,05/10/2026,19:30,{away},{home},3.0,3.2,2.5\n'
    'G1,04/10/2026,17:00,{home},{away},2.0,3.4,4.0\n'
    'G1,04/10/2026,,{home},Brand New FC,,,\n'
    'E0,04/10/2026,15:00,Arsenal,Chelsea,2.0,3.5,3.8\n'
)


@pytest.fixture
def fixtures_csv(monkeypatch, client):
    from match_predict import fixtures
    home, away = _two_current_teams(client, 'greek')
    csv = FIXTURES_CSV.format(home=home, away=away)
    monkeypatch.setattr(fixtures.requests, 'get', lambda *a, **k: Mock(content=csv.encode('utf-8'), raise_for_status=lambda: None))
    fixtures.clear_cache()
    yield home, away
    fixtures.clear_cache()


def test_fixtures_only_league_sorted_with_model_and_market_probs(client, fixtures_csv):
    home, away = fixtures_csv
    body = client.get('/leagues/greek/fixtures').json()

    assert [(f['date'], f['home'], f['away']) for f in body] == [
        ('2026-10-04', home, away), ('2026-10-04', home, 'Brand New FC'), ('2026-10-05', away, home),
    ]
    first = body[0]
    assert first['kickoff'] == '17:00'
    assert sum(first['probs'].values()) == pytest.approx(1.0, abs=1e-6)
    assert sum(first['market_probs'].values()) == pytest.approx(1.0)
    assert first['market_probs']['home'] > first['market_probs']['away']


def test_fixture_with_unknown_team_has_no_probs(client, fixtures_csv):
    unknown = client.get('/leagues/greek/fixtures').json()[1]
    assert unknown['kickoff'] is None
    assert unknown['model'] is None and unknown['probs'] is None and unknown['market_probs'] is None


def test_fixtures_source_down_is_503(client, monkeypatch):
    from match_predict import fixtures

    def boom(*a, **k):
        raise fixtures.requests.ConnectionError('down')

    monkeypatch.setattr(fixtures.requests, 'get', boom)
    fixtures.clear_cache()
    assert client.get('/leagues/greek/fixtures').status_code == 503
