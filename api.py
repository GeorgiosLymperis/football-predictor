"""REST API over the same champion models the Streamlit app serves.

Run with: uvicorn api:app --reload
"""
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from match_predict import service

ModelName = Literal['poisson', 'negbinom', 'elo_xgb', 'logistic', 'mlp', 'ensemble']

app = FastAPI(
    title='Football Match Predictor API',
    description='Elo ratings and win/draw/away probabilities for six European football leagues.',
)


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

    prediction = service.predict_match(lm, home, away)
    chosen = model or service.best_outcome_model(lm, prediction)
    if chosen is None or chosen not in prediction.probs or lm.meta[chosen] is None:
        detail = f'Model {model!r} is not available' if model else 'No model has feature history'
        raise HTTPException(422, f'{detail} for {home} vs {away} in {league}')

    meta = lm.meta[chosen]
    p_home, p_draw, p_away = prediction.probs[chosen]
    service.log_prediction_safe(f'{chosen}_{league}', meta, home, away, p_home, p_draw, p_away)
    return MatchPrediction(
        league=league,
        home=home,
        away=away,
        model=ModelInfo(
            name=chosen,
            version=meta.get('version'),
            rps=meta['rps'],
            baseline_rps=meta['baseline_rps'],
            market_rps=meta.get('market_rps'),
        ),
        probs=Probabilities(home=p_home, draw=p_draw, away=p_away),
    )
