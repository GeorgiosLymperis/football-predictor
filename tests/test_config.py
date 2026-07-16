import pytest

from match_predict.config import load_config, load_league_config


@pytest.mark.parametrize('name', ['xgb', 'logistic', 'mlp', 'poisson', 'backtest'])
def test_load_config_accepts_every_config_shipped_in_the_repo(name):
    """Regression test: _REQUIRED_KEYS only listed 'xgb' and 'backtest',
    so load_config('logistic' | 'mlp' | 'poisson') raised
    'Unknown config' even though those YAML files exist and are used by
    their respective training scripts."""
    cfg = load_config(name)
    assert isinstance(cfg, dict)


def test_load_config_rejects_unknown_name():
    with pytest.raises(ValueError, match='Unknown config'):
        load_config('not_a_real_config')


@pytest.mark.parametrize('league', [
    'greek', 'premier_league', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1',
])
def test_load_league_config_accepts_every_league_shipped_in_the_repo(league):
    cfg = load_league_config(league)
    assert cfg['league_name'] == league
    assert isinstance(cfg['teams'], list) and cfg['teams']


def test_load_league_config_rejects_unknown_league():
    with pytest.raises(ValueError, match='No config'):
        load_league_config('not_a_real_league')
