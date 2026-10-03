import numpy as np
import pytest
from scipy.stats import poisson as scipy_poisson

from match_predict.predict.poisson import predict_outcome_probs, score_matrix, team_ratings


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


def test_team_ratings_recentre_on_given_teams_and_skip_unknown():
    params = _base_params(
        teams=np.array(['A', 'B', 'Relegated'], dtype=object),
        attack=np.array([[np.log(1.5), 0.0, -1.0]]),
        defence=np.array([[0.0, np.log(2.0), 0.0]]),
    )
    out = team_ratings(params, ['A', 'B', 'Promoted']).set_index('team')

    assert list(out.index) == ['A', 'B']
    # Centred on A and B only: A scores sqrt(1.5) times the average, B 1/sqrt(1.5).
    assert out.loc['A', 'attack_pct'] == pytest.approx((np.sqrt(1.5) - 1) * 100)
    assert out.loc['B', 'attack_pct'] == pytest.approx((1 / np.sqrt(1.5) - 1) * 100)
    # B's defence is better, so B concedes fewer goals (positive), A more.
    assert out.loc['B', 'defence_pct'] > 0 > out.loc['A', 'defence_pct']
    assert out.loc['A', 'attack_pct_lo'] <= out.loc['A', 'attack_pct'] <= out.loc['A', 'attack_pct_hi']
