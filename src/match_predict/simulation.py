"""Monte Carlo simulation of the rest of a season with the goals model.

Each simulated season draws one set of team ratings from the posterior, so a
team the model is unsure about gets correspondingly wider outcomes, and the
uncertainty is shared across all of that team's remaining matches. Goals are
sampled from the model's Poisson (or Negative Binomial) rates; the
Dixon-Coles low-score adjustment is left out, which barely moves season-level
probabilities.
"""
import numpy as np
import pandas as pd

TOP_N = 4


class UnratedTeamsError(ValueError):
    def __init__(self, teams: list[str]):
        super().__init__(f'No goals-model rating for: {teams}')
        self.teams = teams


def remaining_fixtures(season: pd.DataFrame, teams: list[str]) -> list[tuple[str, str]]:
    """Home/away pairings of a double round-robin not yet played in `season`."""
    played = set(zip(season['team1'], season['team2']))
    return [(h, a) for h in teams for a in teams if h != a and (h, a) not in played]


def _season_totals(season: pd.DataFrame, teams: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points, goals for and goals against so far, indexed like `teams`."""
    idx = {t: i for i, t in enumerate(teams)}
    pts, gf, ga = (np.zeros(len(teams)) for _ in range(3))
    for h, a, sh, sa in season[['team1', 'team2', 'score1', 'score2']].itertuples(index=False):
        i, j = idx[h], idx[a]
        gf[i] += sh; ga[i] += sa; gf[j] += sa; ga[j] += sh
        pts[i] += 3 if sh > sa else (1 if sh == sa else 0)
        pts[j] += 3 if sa > sh else (1 if sh == sa else 0)
    return pts, gf, ga


def simulate_season(params: dict, season: pd.DataFrame, n_sims: int = 10_000,
                    seed: int = 0) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Simulate the rest of `season` (this season's played matches).

    Returns (teams, final points (n_sims, n_teams), final positions
    (n_sims, n_teams), 1 = top). Ties are broken by goal difference, then
    goals scored, then at random. Raises UnratedTeamsError if a team in
    `season` has no rating in `params`.
    """
    teams = sorted(set(season['team1']) | set(season['team2']))
    param_teams = list(params['teams'])
    unrated = [t for t in teams if t not in param_teams]
    if unrated:
        raise UnratedTeamsError(unrated)

    rng = np.random.default_rng(seed)
    n = len(teams)
    pts, gf, ga = _season_totals(season, teams)
    fixtures = remaining_fixtures(season, teams)
    pts = np.tile(pts, (n_sims, 1))
    gf = np.tile(gf, (n_sims, 1))
    ga = np.tile(ga, (n_sims, 1))

    if fixtures:
        idx = {t: i for i, t in enumerate(teams)}
        h = np.array([idx[x] for x, _ in fixtures])
        a = np.array([idx[y] for _, y in fixtures])
        cols = [param_teams.index(t) for t in teams]
        draw = rng.integers(0, len(params['intercept']), n_sims)
        attack = params['attack'][draw][:, cols]
        defence = params['defence'][draw][:, cols]
        base = params['intercept'][draw][:, None]
        lam_h = np.exp(base + params['home_adv'][draw][:, None] + attack[:, h] - defence[:, a])
        lam_a = np.exp(base + attack[:, a] - defence[:, h])
        if 'nb_alpha' in params:
            alpha = params['nb_alpha'][draw][:, None]
            lam_h = lam_h * rng.gamma(alpha, 1 / alpha, lam_h.shape)
            lam_a = lam_a * rng.gamma(alpha, 1 / alpha, lam_a.shape)
        goals_h = rng.poisson(lam_h).astype(float)
        goals_a = rng.poisson(lam_a).astype(float)

        home_onehot = np.eye(n)[h]  # (n_fixtures, n_teams)
        away_onehot = np.eye(n)[a]
        home_pts = 3.0 * (goals_h > goals_a) + (goals_h == goals_a)
        away_pts = 3.0 * (goals_a > goals_h) + (goals_h == goals_a)
        pts += home_pts @ home_onehot + away_pts @ away_onehot
        gf += goals_h @ home_onehot + goals_a @ away_onehot
        ga += goals_a @ home_onehot + goals_h @ away_onehot

    # One sortable key: points dominate, then goal difference, then goals
    # scored, then a random tie-break below one goal.
    key = pts * 1e7 + (gf - ga + 1000) * 1e3 + gf + rng.random((n_sims, n))
    order = np.argsort(-key, axis=1)
    positions = np.empty_like(order)
    np.put_along_axis(positions, order, np.arange(1, n + 1)[None, :].repeat(n_sims, 0), axis=1)
    return teams, pts, positions


def summarise(teams: list[str], current_points: pd.Series, points: np.ndarray, positions: np.ndarray,
              relegation_spots: int, relegation_playoff: bool) -> pd.DataFrame:
    """Per-team probabilities, sorted by expected finishing position."""
    n = len(teams)
    out = pd.DataFrame({
        'team': teams,
        'points_now': current_points.reindex(teams).fillna(0).astype(int).to_numpy(),
        'expected_points': points.mean(axis=0),
        'expected_position': positions.mean(axis=0),
        'p_title': (positions == 1).mean(axis=0),
        'p_top4': (positions <= TOP_N).mean(axis=0),
        'p_relegation': (positions > n - relegation_spots).mean(axis=0),
    })
    if relegation_playoff:
        out['p_relegation_playoff'] = (positions == n - relegation_spots).mean(axis=0)
    return out.sort_values('expected_position', kind='stable').reset_index(drop=True)
