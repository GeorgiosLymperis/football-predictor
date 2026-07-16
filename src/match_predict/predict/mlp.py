import numpy as np

from match_predict.predict.xgb import build_feature_row

_ACTIVATIONS = {
    'relu': lambda x: np.maximum(0, x),
    'tanh': np.tanh,
    'logistic': lambda x: 1 / (1 + np.exp(-x)),
    'identity': lambda x: x,
}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _forward(x: np.ndarray, coefs: list, intercepts: list, activation: str) -> np.ndarray:
    activate = _ACTIVATIONS[activation]
    h = x
    for W, b in zip(coefs[:-1], intercepts[:-1]):
        h = activate(h @ W + b)
    logits = h @ coefs[-1] + intercepts[-1]
    return _softmax(logits)


def predict_outcome_probs(model_params: dict, state: dict, home: str, away: str) -> dict:
    feats = build_feature_row(state, home, away)
    x = feats.to_numpy(dtype=float)[0]
    x = np.where(np.isnan(x), model_params['impute_means'], x)
    x_scaled = (x - model_params['scaler_mean']) / model_params['scaler_scale']

    probs = _forward(
        x_scaled, list(model_params['coefs']), list(model_params['intercepts']),
        str(model_params['activation']),
    )

    classes = list(model_params['classes'])
    return {
        'p_home': float(probs[classes.index('Home')]),
        'p_draw': float(probs[classes.index('Draw')]),
        'p_away': float(probs[classes.index('Away')]),
    }