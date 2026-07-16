import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from match_predict.training.logistic_model import impute


def hda_probs(model: MLPClassifier, scaler: StandardScaler, encoder, X: pd.DataFrame,
              impute_means: pd.Series) -> np.ndarray:
    """Predict_proba reordered to [p_home, p_draw, p_away]."""
    X_scaled = scaler.transform(impute(X, impute_means))
    proba = model.predict_proba(X_scaled)
    order = [list(encoder.classes_).index(c) for c in ('Home', 'Draw', 'Away')]
    return proba[:, order]


class MLPPyfuncModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        data = np.load(context.artifacts['mlp_params'], allow_pickle=True)
        self.params = {k: data[k] for k in data.files}
        state_data = np.load(context.artifacts['team_state'], allow_pickle=True)
        self.state = {k: state_data[k] for k in state_data.files}

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        from match_predict.predict.mlp import predict_outcome_probs
        rows = []
        for _, r in model_input.iterrows():
            result = predict_outcome_probs(self.params, self.state, r['home_team'], r['away_team'])
            rows.append(result)
        return pd.DataFrame(rows)