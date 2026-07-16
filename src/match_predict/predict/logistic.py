import numpy as np

from match_predict.predict.xgb import build_feature_row


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def predict_outcome_probs(model_params: dict, state: dict, home: str, away: str) -> dict:
    feats = build_feature_row(state, home, away)
    x = feats.to_numpy(dtype=float)[0]
    x = np.where(np.isnan(x), model_params['impute_means'], x)
    x_scaled = (x - model_params['scaler_mean']) / model_params['scaler_scale']

    logits = model_params['coef'] @ x_scaled + model_params['intercept']
    probs = _softmax(logits)

    classes = list(model_params['classes'])
    return {
        'p_home': float(probs[classes.index('Home')]),
        'p_draw': float(probs[classes.index('Draw')]),
        'p_away': float(probs[classes.index('Away')]),
    }