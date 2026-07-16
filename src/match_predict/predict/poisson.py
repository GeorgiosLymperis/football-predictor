import numpy as np
from scipy.special import gammaln

MAX_GOALS = 10


def _poisson_pmf_matrix(lam: np.ndarray, max_goals: int) -> np.ndarray:
    """Vectorized Poisson pmf. lam: (draws,) -> (draws, max_goals + 1)."""
    k = np.arange(max_goals + 1)
    log_fact = np.cumsum(np.log(np.maximum(k, 1)))
    return np.exp(-lam[:, None] + k[None, :] * np.log(lam)[:, None] - log_fact[None, :])


def _negbinom_pmf_matrix(mu: np.ndarray, alpha: np.ndarray, max_goals: int) -> np.ndarray:
    k = np.arange(max_goals + 1)
    mu, alpha = mu[:, None], alpha[:, None]
    log_pmf = (
        gammaln(k[None, :] + alpha) - gammaln(k[None, :] + 1) - gammaln(alpha)
        + alpha * np.log(alpha / (alpha + mu))
        + k[None, :] * np.log(mu / (alpha + mu))
    )
    return np.exp(log_pmf)


def score_matrix(params: dict, home: str, away: str, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Posterior-averaged (optionally Dixon-Coles corrected) score matrix.

    Args:
        params: dict with 'teams' (array of team names) and posterior sample
            arrays 'attack', 'defence' (draws, n_teams), 'home_adv',
            'intercept' (draws,), optionally 'rho' (draws,) for the
            Dixon-Coles low-score correction, and optionally 'nb_alpha'
            (draws,) — presence selects the Negative Binomial likelihood
            instead of Poisson.
        home, away: team names, must be in params['teams'].

    Returns:
        (max_goals+1, max_goals+1) array; [h, a] = P(home scores h, away
        scores a), averaged over posterior draws.
    """
    teams = list(params['teams'])
    h, a = teams.index(home), teams.index(away)
    lam_h = np.exp(params['intercept'] + params['home_adv']
                   + params['attack'][:, h] - params['defence'][:, a])
    lam_a = np.exp(params['intercept'] + params['attack'][:, a] - params['defence'][:, h])

    if 'nb_alpha' in params:
        alpha = params['nb_alpha']
        ph = _negbinom_pmf_matrix(lam_h, alpha, max_goals)
        pa = _negbinom_pmf_matrix(lam_a, alpha, max_goals)
    else:
        ph = _poisson_pmf_matrix(lam_h, max_goals)
        pa = _poisson_pmf_matrix(lam_a, max_goals)
    grid = ph[:, :, None] * pa[:, None, :]  # (draws, home_goals, away_goals)

    if 'rho' in params:
        rho = params['rho']
        grid[:, 0, 0] *= np.clip(1 - lam_h * lam_a * rho, 1e-6, None)
        grid[:, 1, 0] *= np.clip(1 + lam_a * rho, 1e-6, None)
        grid[:, 0, 1] *= np.clip(1 + lam_h * rho, 1e-6, None)
        grid[:, 1, 1] *= np.clip(1 - rho, 1e-6, None)
        grid /= grid.sum(axis=(1, 2), keepdims=True)

    return grid.mean(axis=0)


def predict_outcome_probs(params: dict, home: str, away: str, max_goals: int = MAX_GOALS) -> dict:
    """Full prediction: outcome probabilities, expected goals, top scorelines."""
    sm = score_matrix(params, home, away, max_goals)
    p_home = float(np.tril(sm, -1).sum())
    p_draw = float(np.trace(sm))
    p_away = float(np.triu(sm, 1).sum())

    teams = list(params['teams'])
    h, a = teams.index(home), teams.index(away)
    xg_home = float(np.exp(params['intercept'] + params['home_adv']
                            + params['attack'][:, h] - params['defence'][:, a]).mean())
    xg_away = float(np.exp(params['intercept']
                            + params['attack'][:, a] - params['defence'][:, h]).mean())

    flat = [(gh, ga, sm[gh, ga]) for gh in range(max_goals + 1) for ga in range(max_goals + 1)]
    flat.sort(key=lambda t: -t[2])
    top_scorelines = [{'home_goals': gh, 'away_goals': ga, 'probability': p} for gh, ga, p in flat[:5]]

    return {
        'p_home': p_home, 'p_draw': p_draw, 'p_away': p_away,
        'xg_home': xg_home, 'xg_away': xg_away,
        'score_matrix': sm,
        'top_scorelines': top_scorelines,
    }
