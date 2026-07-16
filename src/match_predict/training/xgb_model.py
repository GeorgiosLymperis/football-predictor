import mlflow.pyfunc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from match_predict.features.elo import build_features

def fit(df: pd.DataFrame, features: list[str], xgb_params: dict):
    df, states, h2h = build_features(df)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['result'])
    weights = compute_sample_weight(class_weight='balanced', y=y)

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(df[features], y, sample_weight=weights)
    return model, encoder, df, states, h2h

def hda_xgb_probs(model: xgb.XGBClassifier, encoder, X):
    proba = model.predict_proba(X)
    order = [list(encoder.classes_).index(c) for c in ('Home', 'Draw', 'Away')]
    return proba[:, order]

class EloXGBPyfuncModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.booster = xgb.Booster()
        self.booster.load_model(context.artifacts['xgb_model'])
        data = np.load(context.artifacts['team_state'], allow_pickle=True)
        self.state = {k: data[k] for k in data.files}
        self.classes = list(self.state['classes'])

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        from match_predict.predict.xgb import build_feature_row
        rows = []
        for _, r in model_input.iterrows():
            feats = build_feature_row(self.state, r['home_team'], r['away_team'])
            probs = self.booster.predict(xgb.DMatrix(feats))[0]
            rows.append({
                'p_home': float(probs[self.classes.index('Home')]),
                'p_draw': float(probs[self.classes.index('Draw')]),
                'p_away': float(probs[self.classes.index('Away')]),
            })
        return pd.DataFrame(rows)