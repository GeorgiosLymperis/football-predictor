from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / 'config'
LEAGUES_DIR = CONFIG_DIR / 'leagues'

_REQUIRED_KEYS = {
    'xgb': {'model_name', 'elo', 'xgb_params', 'feature_variants'},
    'logistic': {'model_name', 'logistic_params', 'feature_variants'},
    'mlp': {'model_name', 'mlp_params', 'feature_variants'},
    'poisson': {'model_name', 'train_seasons', 'dixon_coles', 'decay_rate', 'sampling', 'backtest_sampling'},
    'backtest': {
        'selection_seasons', 'holdout_seasons',
        'xgb_step_matchdays', 'promotion_margin',
        'poisson_step_matchdays',
    },
}

_REQUIRED_LEAGUE_KEYS = {
    'league_name',
    'raw_dir',
    'file_prefix',
    'name_fixes',
    'teams',
}


def load_league_config(league: str) -> dict:
    path = LEAGUES_DIR / f'{league}.yaml'
    if not path.exists():
        available = [c.stem for c in LEAGUES_DIR.glob('*.yaml')]
        available.sort()
        raise ValueError(f'No config for {league!r}. Available: {available}')

    with open(path, encoding='utf-8', mode='r') as file:
        conf = yaml.safe_load(file)

    missing = _REQUIRED_LEAGUE_KEYS - conf.keys()
    if missing:
        raise ValueError(f'{path} is missing required keys: {sorted(missing)}')
    return conf

def load_config(name: str) -> dict:
    """Load `config/<name>.yaml` (a league-agnostic model/backtest config),
    validated against its required top-level keys."""
    if name not in _REQUIRED_KEYS:
        raise ValueError(f'Unknown config {name!r}; expected one of {sorted(_REQUIRED_KEYS)}')

    path = CONFIG_DIR / f'{name}.yaml'
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    missing = _REQUIRED_KEYS[name] - cfg.keys()
    if missing:
        raise ValueError(f'{path} is missing required keys: {sorted(missing)}')
    return cfg