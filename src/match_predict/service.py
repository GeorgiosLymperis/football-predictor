"""Model loading and match prediction shared by the Streamlit app and the API,
so both serve exactly the same numbers.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import xgboost as xgb
import yaml

from match_predict.mlops.monitoring import log_prediction
from match_predict.predict.logistic import predict_outcome_probs as logistic_predict
from match_predict.predict.mlp import predict_outcome_probs as mlp_predict
from match_predict.predict.poisson import predict_outcome_probs as goals_predict
from match_predict.predict.xgb import build_feature_row

MODELS_DIR = Path(__file__).resolve().parents[2] / 'models'
METADATA_MODELS = ('poisson', 'negbinom', 'elo_xgb', 'logistic', 'mlp', 'ensemble')
# Models eligible to be served as the "best" win/draw/away prediction.
OUTCOME_MODELS = ('elo_xgb', 'logistic', 'mlp', 'ensemble')


class UnknownLeagueError(ValueError):
    pass


class UnknownTeamError(ValueError):
    pass


@dataclass(frozen=True)
class LeagueModels:
    league: str
    poisson: dict
    negbinom: dict | None
    booster: xgb.Booster
    elo_state: dict
    logistic: tuple[dict, dict] | None  # (params, team_state)
    mlp: tuple[dict, dict] | None  # (params, team_state)
    meta: dict[str, dict | None]


@dataclass(frozen=True)
class MatchPrediction:
    goals_model: str  # 'poisson' or 'negbinom'
    goals: dict | None  # full goals-model output: xG, score matrix, top scorelines
    probs: dict[str, tuple[float, float, float]]  # model -> (home, draw, away)


def leagues() -> list[str]:
    return sorted(p.name for p in MODELS_DIR.iterdir() if (p / 'elo_xgb').is_dir())


def _load_npz(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _load_metadata(path: Path) -> dict | None:
    return yaml.safe_load(path.read_text()) if path.exists() else None


def _artifacts_stamp(league: str) -> float:
    """Latest mtime of the league's model files, so a weekly Elo refresh or a
    retrain is picked up without restarting the process."""
    return max(p.stat().st_mtime for p in (MODELS_DIR / league).rglob('*') if p.is_file())


def load_league(league: str) -> LeagueModels:
    if league not in leagues():
        raise UnknownLeagueError(league)
    return _load_league(league, _artifacts_stamp(league))


@lru_cache(maxsize=16)
def _load_league(league: str, _stamp: float) -> LeagueModels:
    root = MODELS_DIR / league
    booster = xgb.Booster()
    booster.load_model(str(root / 'elo_xgb' / 'xgb_model.ubj'))

    def classifier(name: str, params_file: str) -> tuple[dict, dict] | None:
        params = _load_npz(root / name / params_file)
        return None if params is None else (params, _load_npz(root / name / 'team_state.npz'))

    return LeagueModels(
        league=league,
        poisson=_load_npz(root / 'poisson' / 'posterior_params.npz'),
        negbinom=_load_npz(root / 'negbinom' / 'posterior_params.npz'),
        booster=booster,
        elo_state=_load_npz(root / 'elo_xgb' / 'team_state.npz'),
        logistic=classifier('logistic', 'logistic_params.npz'),
        mlp=classifier('mlp', 'mlp_params.npz'),
        meta={m: _load_metadata(root / m / 'metadata.yaml') for m in METADATA_MODELS},
    )


def known_teams(lm: LeagueModels) -> list[str]:
    return sorted(set(lm.poisson['teams']) | set(lm.elo_state['teams']))


def resolve_team(lm: LeagueModels, name: str) -> str:
    """Canonical team name for a case-insensitive match."""
    by_lower = {t.lower(): t for t in known_teams(lm)}
    try:
        return by_lower[name.strip().lower()]
    except KeyError:
        raise UnknownTeamError(name) from None


def current_elo_table(lm: LeagueModels) -> list[tuple[str, float]]:
    """(team, elo) for the current season's teams, highest rated first."""
    state = lm.elo_state
    all_teams = list(state['teams'])
    rows = [(t, float(state['elo'][all_teams.index(t)])) for t in state['current_teams'] if t in all_teams]
    return sorted(rows, key=lambda kv: -kv[1])


def team_elo(lm: LeagueModels, team: str) -> dict:
    """Elo rating of a resolved team name, plus its rank among the current
    season's teams (None if it isn't in the current season)."""
    all_teams = list(lm.elo_state['teams'])
    if team not in all_teams:
        raise UnknownTeamError(team)
    table = current_elo_table(lm)
    ranked = [t for t, _ in table]
    return {
        'team': team,
        'elo': float(lm.elo_state['elo'][all_teams.index(team)]),
        'rank': ranked.index(team) + 1 if team in ranked else None,
        'n_teams': len(ranked),
    }


def best_goals_model(lm: LeagueModels) -> str:
    neg, poi = lm.meta['negbinom'], lm.meta['poisson']
    if lm.negbinom is not None and neg is not None and neg['rps'] < poi['rps']:
        return 'negbinom'
    return 'poisson'


def predict_match(lm: LeagueModels, home: str, away: str) -> MatchPrediction:
    """Every model's prediction for a matchup. Models that lack history for
    either team are left out of `probs`."""
    goals_model = best_goals_model(lm)
    goals_params = lm.negbinom if goals_model == 'negbinom' else lm.poisson
    probs: dict[str, tuple[float, float, float]] = {}

    goals = None
    if {home, away} <= set(goals_params['teams']):
        goals = goals_predict(goals_params, home, away)
        probs[goals_model] = (goals['p_home'], goals['p_draw'], goals['p_away'])

    if {home, away} <= set(lm.elo_state['teams']):
        p = lm.booster.predict(xgb.DMatrix(build_feature_row(lm.elo_state, home, away)))[0]
        classes = list(lm.elo_state['classes'])  # e.g. ['Away', 'Draw', 'Home']
        probs['elo_xgb'] = tuple(float(p[classes.index(c)]) for c in ('Home', 'Draw', 'Away'))

    for name, model, predict in (('logistic', lm.logistic, logistic_predict), ('mlp', lm.mlp, mlp_predict)):
        if model is None:
            continue
        params, state = model
        if {home, away} <= set(state['teams']):
            r = predict(params, state, home, away)
            probs[name] = (r['p_home'], r['p_draw'], r['p_away'])

    constituents = (goals_model, 'elo_xgb', 'logistic')
    if lm.meta['ensemble'] is not None and all(m in probs for m in constituents):
        probs['ensemble'] = tuple(sum(probs[m][i] for m in constituents) / 3 for i in range(3))

    return MatchPrediction(goals_model=goals_model, goals=goals, probs=probs)


def best_outcome_model(lm: LeagueModels, prediction: MatchPrediction) -> str | None:
    """Lowest walk-forward RPS among the outcome models that could predict
    this matchup, or None if none could."""
    available = [m for m in OUTCOME_MODELS if m in prediction.probs and lm.meta[m] is not None]
    return min(available, key=lambda m: lm.meta[m]['rps'], default=None)


def log_prediction_safe(model_name: str, meta: dict | None, home: str, away: str,
                         p_home: float, p_draw: float, p_away: float) -> None:
    """A monitoring-log failure must never break a prediction request."""
    try:
        log_prediction(model_name, meta['version'] if meta else None, home, away, p_home, p_draw, p_away)
    except Exception:
        pass
