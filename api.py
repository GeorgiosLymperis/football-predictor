"""REST API over the same champion models the Streamlit app serves.

Run with: uvicorn api:app --reload
"""
import datetime as dt
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from match_predict import fixtures, service

ModelName = Literal['poisson', 'negbinom', 'elo_xgb', 'logistic', 'mlp', 'ensemble']

app = FastAPI(
    title='Football Match Predictor API',
    description='Elo ratings and win/draw/away probabilities for six European football leagues.',
)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET'])


class TeamElo(BaseModel):
    team: str
    elo: float
    rank: int | None  # among the current season's teams; None if not in the current season
    n_teams: int


class ModelInfo(BaseModel):
    name: str
    version: str | None
    rps: float
    baseline_rps: float
    market_rps: float | None


class Probabilities(BaseModel):
    home: float
    draw: float
    away: float


class MatchPrediction(BaseModel):
    league: str
    home: str
    away: str
    model: ModelInfo
    probs: Probabilities


class Fixture(BaseModel):
    date: dt.date
    kickoff: str | None  # as listed by football-data.co.uk (UK time)
    home: str
    away: str
    model: ModelInfo | None  # None if no model has history for both teams
    probs: Probabilities | None
    market_probs: Probabilities | None  # de-vigged average bookmaker odds


def _league(league: str) -> service.LeagueModels:
    try:
        return service.load_league(league)
    except service.UnknownLeagueError:
        raise HTTPException(404, f'Unknown league {league!r}. Available: {service.leagues()}')


def _team(lm: service.LeagueModels, name: str) -> str:
    try:
        return service.resolve_team(lm, name)
    except service.UnknownTeamError:
        raise HTTPException(404, f'Unknown team {name!r} in {lm.league}. See /leagues/{lm.league}/teams')


@app.get('/leagues')
def list_leagues() -> list[str]:
    return service.leagues()


@app.get('/leagues/{league}/teams')
def list_teams(league: str) -> list[TeamElo]:
    """Current season's teams, highest Elo first."""
    lm = _league(league)
    table = service.current_elo_table(lm)
    return [TeamElo(team=t, elo=elo, rank=i + 1, n_teams=len(table)) for i, (t, elo) in enumerate(table)]


@app.get('/leagues/{league}/teams/{team}')
def get_team(league: str, team: str) -> TeamElo:
    lm = _league(league)
    try:
        return TeamElo(**service.team_elo(lm, _team(lm, team)))
    except service.UnknownTeamError:
        raise HTTPException(404, f'No Elo rating for {team!r} in {league}')


@app.get('/leagues/{league}/predict')
def predict(
    league: str,
    home: str = Query(description='Home team'),
    away: str = Query(description='Away team'),
    model: ModelName | None = Query(None, description='Defaults to the model with the best walk-forward RPS'),
) -> MatchPrediction:
    lm = _league(league)
    home, away = _team(lm, home), _team(lm, away)
    if home == away:
        raise HTTPException(422, 'home and away must be different teams')

    model_info, probs = _predict(lm, home, away, model)
    service.log_prediction_safe(f'{model_info.name}_{league}', lm.meta[model_info.name], home, away,
                                probs.home, probs.draw, probs.away)
    return MatchPrediction(league=league, home=home, away=away, model=model_info, probs=probs)


@app.get('/leagues/{league}/fixtures')
def list_fixtures(
    league: str,
    model: ModelName | None = Query(None, description='Defaults to the model with the best walk-forward RPS'),
) -> list[Fixture]:
    """Upcoming matches (the next few days, as published by football-data.co.uk)
    with model and market probabilities. Empty when the league has no games
    scheduled in that window, e.g. during an international break."""
    lm = _league(league)
    try:
        upcoming = service.fixture_predictions(lm, model)
    except fixtures.FixturesUnavailableError as e:
        raise HTTPException(503, str(e))

    return [
        Fixture(
            date=f['date'], kickoff=f['kickoff'], home=f['home'], away=f['away'],
            model=_model_info(lm, f['model']) if f['model'] else None,
            probs=_probabilities(f['probs']),
            market_probs=_probabilities(f['market_probs']),
        )
        for f in upcoming
    ]


def _probabilities(p: tuple[float, float, float] | None) -> Probabilities | None:
    return Probabilities(home=p[0], draw=p[1], away=p[2]) if p else None


def _model_info(lm: service.LeagueModels, name: str) -> ModelInfo:
    meta = lm.meta[name]
    return ModelInfo(
        name=name,
        version=meta.get('version'),
        rps=meta['rps'],
        baseline_rps=meta['baseline_rps'],
        market_rps=meta.get('market_rps'),
    )


def _predict(lm: service.LeagueModels, home: str, away: str,
             model: str | None) -> tuple[ModelInfo, Probabilities]:
    """The requested model's prediction, or by default the one with the best
    walk-forward RPS among those that have history for both teams."""
    served = service.served_prediction(lm, home, away, model)
    if served is None:
        detail = f'Model {model!r} is not available' if model else 'No model has feature history'
        raise HTTPException(422, f'{detail} for {home} vs {away} in {lm.league}')
    name, probs = served
    return _model_info(lm, name), _probabilities(probs)
