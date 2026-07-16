import numpy as np

from match_predict.metrics.rps import outcome_index, rps


def test_outcome_index_maps_scores_to_labels():
    score1 = np.array([2, 0, 1])
    score2 = np.array([1, 0, 1])
    np.testing.assert_array_equal(outcome_index(score1, score2), [0, 1, 1])


def test_rps_perfect_prediction_is_zero():
    probs = np.array([[1.0, 0.0, 0.0]])
    outcome = np.array([0])
    assert rps(probs, outcome) == 0.0


def test_rps_worst_prediction_is_one():
    probs = np.array([[0.0, 0.0, 1.0]])
    outcome = np.array([0])
    assert rps(probs, outcome) == 1.0


def test_rps_uniform_prediction_is_between_perfect_and_worst():
    probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
    for outcome in (0, 1, 2):
        score = rps(probs, np.array([outcome]))
        assert 0.0 < score < 1.0


def test_rps_penalizes_far_misses_more_than_near_misses():
    """Predicting a home win when the draw happens should score better than
    predicting a home win when the away team wins (RPS respects outcome
    ordering, unlike plain classification accuracy)."""
    probs = np.array([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])
    outcome = np.array([1, 2])  # draw, then away win
    scores = [rps(probs[[i]], outcome[[i]]) for i in range(2)]
    assert scores[0] < scores[1]
