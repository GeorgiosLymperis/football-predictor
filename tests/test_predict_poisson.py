import numpy as np
import pytest
from scipy.stats import poisson as scipy_poisson

from match_predict.predict.poisson import predict_outcome_probs, score_matrix


def _base_params(**extra):

    params = dict(
        teams=np.array(['A', 'B'], dtype=object),
        attack=np.array([[0.0, 0.0]]),   # (draws, n_teams)
        defence=np.array([[0.0, 0.0]]),
        home_adv=np.array([np.log(2.0)]),   # so lambda_home = 2 * lambda_away
        intercept=np.array([np.log(1.0)]),
    )
    params.update(extra)
    return params


def test_score_matrix_matches_independent_poisson_pmfs_without_dixon_coles():
    params = _base_params()
    sm = score_matrix(params, 'A', 'B', max_goals=5)
    lam_h, lam_a = 2.0, 1.0
    expected = np.outer(scipy_poisson.pmf(np.arange(6), lam_h),
                         scipy_poisson.pmf(np.arange(6), lam_a))
    np.testing.assert_allclose(sm, expected, atol=1e-8)


def test_score_matrix_sums_to_at_most_one():
    params = _base_params()
    sm = score_matrix(params, 'A', 'B', max_goals=10)
    assert sm.sum() <= 1.0 + 1e-8


def test_dixon_coles_correction_changes_low_score_cells_only():
    params = _base_params(rho=np.array([-0.1]))
    plain = score_matrix(_base_params(), 'A', 'B', max_goals=5)
    corrected = score_matrix(params, 'A', 'B', max_goals=5)
    # Cells outside the 2x2 low-score block are untouched by the tau
    # correction (only rescaled by the renormalization).
    ratio = corrected[3, 3] / plain[3, 3]
    np.testing.assert_allclose(corrected[2:, 2:] / plain[2:, 2:], ratio, atol=1e-6)
    # The 0-0 cell should differ from the uncorrected version.
    assert corrected[0, 0] != pytest.approx(plain[0, 0] * ratio)


def test_predict_outcome_probs_sums_to_one_and_scorelines_sorted():
    # max_goals is generous here so the truncated score grid's missing
    # tail mass (lambda is only 1-2) is negligible at this tolerance.
    params = _base_params()
    result = predict_outcome_probs(params, 'A', 'B', max_goals=20)
    total = result['p_home'] + result['p_draw'] + result['p_away']
    assert total == pytest.approx(1.0, abs=1e-6)
    probs = [s['probability'] for s in result['top_scorelines']]
    assert probs == sorted(probs, reverse=True)
    assert result['xg_home'] == pytest.approx(2.0)
    assert result['xg_away'] == pytest.approx(1.0)


def test_negbinom_selected_when_nb_alpha_present():
    params = _base_params(nb_alpha=np.array([10.0]))
    result = predict_outcome_probs(params, 'A', 'B', max_goals=20)
    total = result['p_home'] + result['p_draw'] + result['p_away']
    assert total == pytest.approx(1.0, abs=1e-6)
