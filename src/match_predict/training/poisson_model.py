from pathlib import Path
from typing import List

import arviz as az
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


def prepare_df(df: pd.DataFrame, teams: List[str]) -> pd.DataFrame:
    df = df.rename(columns={
        'team1': 'home_team', 'team2': 'away_team',
        'score1': 'goals_home', 'score2': 'goals_away',
    })
    team_index = {t: i for i, t in enumerate(teams)}
    df = df.copy()
    df['home_team_index'] = df['home_team'].map(team_index)
    df['away_team_index'] = df['away_team'].map(team_index)
    return df


def build_model(df: pd.DataFrame, teams: List[str], dixon_coles: bool = False,
                 decay_rate: float = 0.0, likelihood: str = 'poisson') -> pm.Model:
    home_idx = df['home_team_index'].to_numpy(dtype=int)
    away_idx = df['away_team_index'].to_numpy(dtype=int)
    goals_home = df['goals_home'].to_numpy(dtype=int)
    goals_away = df['goals_away'].to_numpy(dtype=int)
    coords = {'team': teams, 'match': np.arange(len(df))}

    with pm.Model(coords=coords) as model:
        home_i = pm.Data('home_i', home_idx, dims='match')
        away_i = pm.Data('away_i', away_idx, dims='match')
        goals_h = pm.Data('goals_h', goals_home, dims='match')
        goals_a = pm.Data('goals_a', goals_away, dims='match')

        sigma_attack = pm.HalfNormal('sigma_attack', 0.5)
        sigma_defence = pm.HalfNormal('sigma_defence', 0.5)

        attack_coef = pm.Normal('attack_coef', mu=0, sigma=1, shape=len(teams), dims='team')
        defence_coef = pm.Normal('defence_coef', mu=0, sigma=1, shape=len(teams), dims='team')
        attack_coef = attack_coef - attack_coef.mean()
        defence_coef = defence_coef - defence_coef.mean()

        attack = pm.Deterministic('attack', attack_coef * sigma_attack, dims='team')
        defence = pm.Deterministic('defence', defence_coef * sigma_defence, dims='team')
        home_adv = pm.Normal('home_adv', mu=0, sigma=0.3)
        intercept = pm.Normal('intercept', 0.0, 1.0)

        lambda_home = pm.Deterministic(
            'lambda_home',
            pm.math.exp(intercept + home_adv + attack[home_i] - defence[away_i]),
            dims='match',
        )
        lambda_away = pm.Deterministic(
            'lambda_away',
            pm.math.exp(intercept + attack[away_i] - defence[home_i]),
            dims='match',
        )

        if likelihood == 'poisson':
            pm.Poisson('goals_home', mu=lambda_home, observed=goals_h, dims='match')
            pm.Poisson('goals_away', mu=lambda_away, observed=goals_a, dims='match')
        elif likelihood == 'neg_binomial':
            nb_alpha = pm.Gamma('nb_alpha', mu=10, sigma=5)
            pm.NegativeBinomial('goals_home', mu=lambda_home, alpha=nb_alpha, observed=goals_h, dims='match')
            pm.NegativeBinomial('goals_away', mu=lambda_away, alpha=nb_alpha, observed=goals_a, dims='match')
        else:
            raise ValueError(f'Unknown likelihood {likelihood!r}; expected "poisson" or "neg_binomial"')

        if decay_rate > 0.0:
            n = len(df)
            raw_w = np.exp(-decay_rate * np.arange(n - 1, -1, -1))
            weights = pm.Data('weights', (raw_w / raw_w.mean()).astype(np.float64), dims='match')
            logp_home = pm.logp(pm.Poisson.dist(mu=lambda_home), goals_h)
            logp_away = pm.logp(pm.Poisson.dist(mu=lambda_away), goals_a)
            pm.Potential(
                'recency_correction',
                ((weights - 1) * logp_home + (weights - 1) * logp_away).sum(),
            )

        if dixon_coles:
            rho = pm.Normal('rho', mu=0, sigma=0.1)
            is_00 = pt.eq(goals_h, 0) * pt.eq(goals_a, 0)
            is_10 = pt.eq(goals_h, 1) * pt.eq(goals_a, 0)
            is_01 = pt.eq(goals_h, 0) * pt.eq(goals_a, 1)
            is_11 = pt.eq(goals_h, 1) * pt.eq(goals_a, 1)
            log_tau = (
                is_00 * pt.log(pt.clip(1 - lambda_home * lambda_away * rho, 1e-6, np.inf))
                + is_10 * pt.log(pt.clip(1 + lambda_away * rho, 1e-6, np.inf))
                + is_01 * pt.log(pt.clip(1 + lambda_home * rho, 1e-6, np.inf))
                + is_11 * pt.log(pt.clip(1 - rho, 1e-6, np.inf))
            )
            pm.Potential('dc_correction', log_tau.sum())

    return model


def fit_poisson_model(df: pd.DataFrame, teams: List[str], draws=1500, tune=2000, chains=4,
                       target_accept=0.9, seed=42, dixon_coles: bool = False,
                       decay_rate: float = 0.0, likelihood: str = 'poisson'):
    model = build_model(df, teams, dixon_coles=dixon_coles, decay_rate=decay_rate, likelihood=likelihood)
    with model:
        idata = pm.sample(
            draws, tune=tune, chains=chains, target_accept=target_accept,
            random_seed=seed, idata_kwargs={'log_likelihood': True},
            progressbar=False,
        )
    return model, idata


def export_posterior_params(idata: az.InferenceData, teams: List[str], dixon_coles: bool,
                             likelihood: str = 'poisson') -> dict:
    post = idata.posterior
    params = dict(
        teams=np.array(teams, dtype=object),
        attack=post['attack'].stack(s=('chain', 'draw')).values.T,
        defence=post['defence'].stack(s=('chain', 'draw')).values.T,
        home_adv=post['home_adv'].stack(s=('chain', 'draw')).values,
        intercept=post['intercept'].stack(s=('chain', 'draw')).values,
    )
    if dixon_coles:
        params['rho'] = post['rho'].stack(s=('chain', 'draw')).values
    if likelihood == 'neg_binomial':
        params['nb_alpha'] = post['nb_alpha'].stack(s=('chain', 'draw')).values
    return params


def run_loo(idata: az.InferenceData, model: pm.Model | None = None):
    if 'log_likelihood' not in idata:
        if model is None:
            raise ValueError('idata has no log_likelihood; pass model= to compute it.')
        with model:
            pm.compute_log_likelihood(idata)
    idata.log_likelihood['joint'] = (
        idata.log_likelihood['goals_home'] + idata.log_likelihood['goals_away']
    )
    return az.loo(idata, var_name='joint', pointwise=True)


def plot_ppc(idata: az.InferenceData, model: pm.Model, save_path: Path):
    import matplotlib.pyplot as plt
    if 'posterior_predictive' not in idata:
        with model:
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, progressbar=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    az.plot_ppc(idata, var_names=['goals_home'], ax=axes[0])
    axes[0].set_title('PPC — Home Goals')
    az.plot_ppc(idata, var_names=['goals_away'], ax=axes[1])
    axes[1].set_title('PPC — Away Goals')
    plt.tight_layout()
    fig.savefig(save_path / 'ppc.png')
    plt.close(fig)


class PoissonPyfuncModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        data = np.load(context.artifacts['posterior_params'], allow_pickle=True)
        self.params = {k: data[k] for k in data.files}

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        from match_predict.predict.poisson import predict_outcome_probs
        rows = []
        for _, r in model_input.iterrows():
            result = predict_outcome_probs(self.params, r['home_team'], r['away_team'])
            rows.append({k: result[k] for k in ('p_home', 'p_draw', 'p_away', 'xg_home', 'xg_away')})
        return pd.DataFrame(rows)