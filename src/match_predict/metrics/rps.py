import numpy as np

OUTCOMES = ('home', 'draw', 'away')


def rps(probs: np.ndarray, outcome_idx: np.ndarray) -> float:
    """Mean Ranked Probability Score. Lower is better; 0 is perfect.

    Args:
        probs: (n, 3) array of [p_home, p_draw, p_away], each row summing to 1.
        outcome_idx: (n,) array of realized outcomes, 0=home, 1=draw, 2=away.

    Returns:
        Mean RPS across all n matches.
    """
    cum_pred = np.cumsum(probs, axis=1)[:, :2]
    cum_true = np.zeros((len(outcome_idx), 2))
    for i, o in enumerate(outcome_idx):
        cum_true[i, o:] = 1.0
    return float(np.mean(np.sum((cum_pred - cum_true) ** 2, axis=1) / 2))


def outcome_index(score1: np.ndarray, score2: np.ndarray) -> np.ndarray:
    """0=home win, 1=draw, 2=away win, from goal columns."""
    score1, score2 = np.asarray(score1), np.asarray(score2)
    return np.where(score1 > score2, 0, np.where(score1 == score2, 1, 2))
