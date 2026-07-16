import numpy as np
import pytest

from match_predict.predict.logistic import predict_outcome_probs as logistic_predict
from match_predict.predict.mlp import predict_outcome_probs as mlp_predict
from match_predict.training.logistic_model import impute


def _state():
    return dict(
        teams=np.array(['A', 'B'], dtype=object),
        elo=np.array([1550.0, 1450.0]),
        pct=np.array([0.0, 0.0]),
        roll=np.array([[1500.0] * 3] * 2),
        ema=np.array([[1500.0] * 3] * 2),
        feature_names=np.array(['EloDifference'], dtype=object),
    )


def test_impute_fills_only_missing_values():
    import pandas as pd
    X = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
    out = impute(X, pd.Series({'a': 99.0}))
    assert list(out['a']) == [1.0, 99.0, 3.0]


def test_logistic_predict_outcome_probs_sums_to_one_and_favors_higher_class_logit():
    # classes deliberately not alphabetical, to make sure the function reads
    # the ordering from `classes` rather than assuming it.
    params = dict(
        coef=np.array([[0.0], [0.0], [5.0]]),  # rows follow `classes` order
        intercept=np.array([0.0, 0.0, 0.0]),
        scaler_mean=np.array([0.0]),
        scaler_scale=np.array([1.0]),
        impute_means=np.array([0.0]),
        classes=np.array(['Draw', 'Away', 'Home'], dtype=object),
    )
    result = logistic_predict(params, _state(), 'A', 'B')
    total = result['p_home'] + result['p_draw'] + result['p_away']
    assert total == pytest.approx(1.0, abs=1e-6)
    # EloDifference for A vs B is +100, times coef row for 'Home' (5.0) ->
    # large positive logit -> Home should dominate.
    assert result['p_home'] > 0.9


def test_mlp_predict_outcome_probs_sums_to_one():
    coefs = np.empty(2, dtype=object)
    coefs[0] = np.array([[1.0, -1.0]])       # (1 feature, 2 hidden units)
    coefs[1] = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])  # (2 hidden, 3 classes)
    intercepts = np.empty(2, dtype=object)
    intercepts[0] = np.zeros(2)
    intercepts[1] = np.zeros(3)
    params = dict(
        coefs=coefs,
        intercepts=intercepts,
        activation='relu',
        scaler_mean=np.array([0.0]),
        scaler_scale=np.array([1.0]),
        impute_means=np.array([0.0]),
        classes=np.array(['Away', 'Draw', 'Home'], dtype=object),
    )
    result = mlp_predict(params, _state(), 'A', 'B')
    total = result['p_home'] + result['p_draw'] + result['p_away']
    assert total == pytest.approx(1.0, abs=1e-6)
    # EloDifference for A vs B is +100 -> hidden unit 0 fires (relu), unit 1
    # is clipped to 0 -> logits are all zero except the 'Home' class.
    assert result['p_home'] > 0.9
