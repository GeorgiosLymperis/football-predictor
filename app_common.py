from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import yaml
from matplotlib.colors import LinearSegmentedColormap

from match_predict.mlops.monitoring import log_prediction
from match_predict.predict.xgb import build_feature_row
from match_predict.predict.logistic import predict_outcome_probs as logistic_predict
from match_predict.predict.mlp import predict_outcome_probs as mlp_predict
from match_predict.predict.poisson import predict_outcome_probs

LEAGUE_DISPLAY_NAMES = {
    'greek': 'Greek Super League',
    'premier_league': 'Premier League',
    'la_liga': 'La Liga',
    'bundesliga': 'Bundesliga',
    'serie_a': 'Serie A',
    'ligue_1': 'Ligue 1',
}
LEAGUE_ICONS = {
    'greek': '\U0001F1EC\U0001F1F7',  # 🇬🇷
    'premier_league': '\U0001F1EC\U0001F1E7',  # 🇬🇧
    'la_liga': '\U0001F1EA\U0001F1F8',  # 🇪🇸
    'bundesliga': '\U0001F1E9\U0001F1EA',  # 🇩🇪
    'serie_a': '\U0001F1EE\U0001F1F9',  # 🇮🇹
    'ligue_1': '\U0001F1EB\U0001F1F7',  # 🇫🇷
}
DEFAULT_MATCHUPS = {
    'greek': ('AEK', 'Olympiakos'),
    'premier_league': ('Arsenal', 'Liverpool'),
    'la_liga': ('Real Madrid', 'Barcelona'),
    'bundesliga': ('Bayern Munich', 'Dortmund'),
    'serie_a': ('Milan', 'Inter'),
    'ligue_1': ('Marseille', 'Paris SG'),
}

SEQ_BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
CATEGORICAL = ['#2a78d6', "#42cc9c", '#eda100', "#005E00", '#4a3aa7']


def _model_dir(league: str, model_name: str) -> Path:
    return Path(f'models/{league}/{model_name}')


@st.cache_resource
def load_poisson(league: str) -> dict:
    d = np.load(_model_dir(league, 'poisson') / 'posterior_params.npz', allow_pickle=True)
    return {k: d[k] for k in d.files}


@st.cache_resource
def load_elo_xgb(league: str):
    model_dir = _model_dir(league, 'elo_xgb')
    booster = xgb.Booster()
    booster.load_model(str(model_dir / 'xgb_model.ubj'))
    d = np.load(model_dir / 'team_state.npz', allow_pickle=True)
    return booster, {k: d[k] for k in d.files}


@st.cache_resource
def load_negbinom(league: str) -> dict | None:
    path = _model_dir(league, 'negbinom') / 'posterior_params.npz'
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


@st.cache_resource
def load_logistic(league: str):
    """(None, None) if this league hasn't had logistic trained yet."""
    model_dir = _model_dir(league, 'logistic')
    params_path = model_dir / 'logistic_params.npz'
    state_path = model_dir / 'team_state.npz'
    if not params_path.exists():
        return None, None
    d1 = np.load(params_path, allow_pickle=True)
    d2 = np.load(state_path, allow_pickle=True)
    return {k: d1[k] for k in d1.files}, {k: d2[k] for k in d2.files}


@st.cache_resource
def load_mlp(league: str):
    """(None, None) if this league hasn't had mlp trained yet."""
    model_dir = _model_dir(league, 'mlp')
    params_path = model_dir / 'mlp_params.npz'
    state_path = model_dir / 'team_state.npz'
    if not params_path.exists():
        return None, None
    d1 = np.load(params_path, allow_pickle=True)
    d2 = np.load(state_path, allow_pickle=True)
    return {k: d1[k] for k in d1.files}, {k: d2[k] for k in d2.files}


def load_metadata(league: str, model_name: str) -> dict | None:
    path = _model_dir(league, model_name) / 'metadata.yaml'
    return yaml.safe_load(path.read_text()) if path.exists() else None


def metadata_caption(meta: dict | None, extra: str) -> str:
    if meta is None:
        return extra
    market = (
        f', market {meta["market_rps"]:.4f} (n={meta["market_n_matches"]})'
        if meta.get('market_rps') is not None else ''
    )
    return (
        f'{extra} Version {meta["version"]} (promoted {meta["promoted_at"][:10]}) | '
        f'walk-forward RPS {meta["rps"]:.4f} (baseline {meta["baseline_rps"]:.4f}{market}).'
    )


def outcome_row(p_home: float, p_draw: float, p_away: float, home: str, away: str):
    c1, c2, c3 = st.columns(3)
    c1.metric(f'{home} win', f'{p_home:.1%}')
    c2.metric('Draw', f'{p_draw:.1%}')
    c3.metric(f'{away} win', f'{p_away:.1%}')


def score_heatmap(score_matrix: np.ndarray, home: str, away: str, window: int = 6):
    m = score_matrix[: window + 1, : window + 1]
    cmap = LinearSegmentedColormap.from_list('seq_blue', SEQ_BLUES)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    im = ax.imshow(m, cmap=cmap, vmin=0)
    for i in range(window + 1):
        for j in range(window + 1):
            ink = '#ffffff' if m[i, j] > 0.55 * m.max() else '#0b0b0b'
            ax.text(j, i, f'{m[i, j]:.1%}', ha='center', va='center',
                    fontsize=8, color=ink)
    ax.set_xticks(range(window + 1))
    ax.set_yticks(range(window + 1))
    ax.set_xlabel(f'{away} goals')
    ax.set_ylabel(f'{home} goals')
    ax.tick_params()
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params()
    cbar.ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0%}')
    cbar.outline.set_visible(False)
    fig.tight_layout()
    return fig


def top_scorelines_table(top_scorelines: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {'Score': f'{s["home_goals"]} - {s["away_goals"]}', 'Probability': f'{s["probability"]:.1%}'}
        for s in top_scorelines
    ])


def elo_table_chart(state: dict):
    teams = list(state['current_teams'])
    all_teams = list(state['teams'])
    elos = sorted(
        ((t, state['elo'][all_teams.index(t)]) for t in teams if t in all_teams),
        key=lambda kv: kv[1],
    )
    names = [t for t, _ in elos]
    vals = [v for _, v in elos]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(names))))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    ax.barh(names, vals, height=0.62)
    ax.set_xlim(min(vals) - 60, max(vals) + 60)
    ax.tick_params(labelsize=9)
    ax.xaxis.grid(True, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, v in enumerate(vals):
        ax.text(v + 8, i, f'{v:.0f}', va='center', fontsize=9)
    fig.tight_layout()
    return fig


def elo_top5_progress_chart(state: dict, top_n: int = 5):
    """Recent Elo trajectory (by date) for the top_n current teams by
    rating."""
    all_teams = list(state['teams'])
    current = [t for t in state['current_teams'] if t in all_teams]
    top_teams = sorted(current, key=lambda t: -state['elo'][all_teams.index(t)])[:top_n]
    return _elo_progress_chart(state, top_teams)


def elo_matchup_progress_chart(state: dict, teams_to_plot: list[str]):
    """Recent Elo trajectory (by date) for the specific teams a user picked."""
    all_teams = list(state['teams'])
    known = [t for t in teams_to_plot if t in all_teams]
    return _elo_progress_chart(state, known)


def _elo_progress_chart(state: dict, teams: list[str]):

    all_teams = list(state['teams'])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    for i, team in enumerate(teams):
        idx = all_teams.index(team)
        ax.plot(
            state['elo_dates'][idx], state['elo_history'][idx],
            label=team, color=CATEGORICAL[i % len(CATEGORICAL)], linewidth=2.2,
        )
    ax.set_xlabel('Date')
    ax.set_ylabel('Elo rating')
    ax.tick_params()
    ax.yaxis.grid(True, linewidth=0.8)
    ax.set_axisbelow(True)
    if teams:
        ax.legend(loc='upper left')
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def log_prediction_safe(model_name: str, meta: dict | None, home: str, away: str,
                         p_home: float, p_draw: float, p_away: float) -> None:
    """A monitoring-log failure must never break a prediction request."""
    try:
        log_prediction(model_name, meta['version'] if meta else None, home, away, p_home, p_draw, p_away)
    except Exception:
        pass


def _render_goals_model_section(title: str, caption_text: str, model_name: str,
                                 params: dict, meta: dict | None, home: str, away: str) -> dict:

    st.subheader(title)
    result = predict_outcome_probs(params, home, away)
    outcome_row(result['p_home'], result['p_draw'], result['p_away'], home, away)
    c1, c2 = st.columns(2)
    c1.metric(f'Expected goals, {home}', f'{result["xg_home"]:.2f}')
    c2.metric(f'Expected goals, {away}', f'{result["xg_away"]:.2f}')
    st.markdown('Most likely scorelines')
    st.dataframe(top_scorelines_table(result['top_scorelines']), hide_index=True)
    st.markdown('Scoreline probabilities')
    st.pyplot(score_heatmap(result['score_matrix'], home, away))
    st.caption(metadata_caption(meta, caption_text))
    log_prediction_safe(model_name, meta, home, away,
                         result['p_home'], result['p_draw'], result['p_away'])
    return result


def _best_goals_model(league: str, poisson_params: dict, negbinom_params: dict | None,
                       poisson_meta: dict | None, negbinom_meta: dict | None):

    if negbinom_params is not None and negbinom_meta['rps'] < poisson_meta['rps']:
        return (
            f'negbinom_{league}', negbinom_params, negbinom_meta, 'Negative Binomial',
            'Same hierarchical attack/defence structure as the Poisson model, but '
            'relaxes its mean=variance assumption with an extra dispersion parameter. '
            'Beat Poisson on walk-forward RPS for this league.',
        )
    extra = ' Beat Negative Binomial on walk-forward RPS for this league.' if negbinom_meta else ''
    return (
        f'poisson_{league}', poisson_params, poisson_meta, 'Bayesian Poisson (Dixon-Coles)',
        f'Hierarchical Bayesian Poisson regression with Dixon-Coles low-score correction.{extra}',
    )


def model_rps_table(entries: list[tuple[str, dict | None]]) -> pd.DataFrame:
    rows = [
        {
            'Model': name,
            'RPS': round(meta['rps'], 4),
            'Baseline RPS': round(meta['baseline_rps'], 4),
            'Market RPS': round(meta['market_rps'], 4) if meta.get('market_rps') is not None else None,
        }
        for name, meta in entries if meta is not None
    ]
    return pd.DataFrame(rows).sort_values('RPS').reset_index(drop=True)


def render_league_page(league: str) -> None:
    display_name = LEAGUE_DISPLAY_NAMES[league]
    st.title(f'{display_name} match predictor')

    poisson_params = load_poisson(league)
    booster, elo_state = load_elo_xgb(league)
    negbinom_params = load_negbinom(league)
    logistic_params, logistic_state = load_logistic(league)
    mlp_params, mlp_state = load_mlp(league)
    poisson_meta = load_metadata(league, 'poisson')
    elo_meta = load_metadata(league, 'elo_xgb')
    negbinom_meta = load_metadata(league, 'negbinom')
    logistic_meta = load_metadata(league, 'logistic')
    mlp_meta = load_metadata(league, 'mlp')
    ensemble_meta = load_metadata(league, 'ensemble')

    st.subheader('Current Elo ratings')
    st.pyplot(elo_table_chart(elo_state))

    st.subheader('Elo progress — top 5 teams')
    st.pyplot(elo_top5_progress_chart(elo_state))

    st.divider()
    st.subheader('Model performance')
    st.dataframe(model_rps_table([
        ('Bayesian Poisson (Dixon-Coles)', poisson_meta),
        ('Negative Binomial', negbinom_meta),
        ('XGBoost', elo_meta),
        ('Logistic Regression', logistic_meta),
        ('MLP (Neural Network)', mlp_meta),
        ('Ensemble', ensemble_meta),
    ]), hide_index=True)
    st.caption('Walk-forward RPS on held-out seasons. Lower is better. The sections below '
               'show only the best-performing model in each family.')

    st.divider()
    st.subheader('Pick a matchup')
    poisson_teams = sorted(poisson_params['teams'])
    default_home_team, default_away_team = DEFAULT_MATCHUPS.get(league, (poisson_teams[0], poisson_teams[1]))
    default_home = poisson_teams.index(default_home_team) if default_home_team in poisson_teams else 0
    default_away = poisson_teams.index(default_away_team) if default_away_team in poisson_teams else 1

    col_home, col_away = st.columns(2)
    home = col_home.selectbox('Home team', poisson_teams, index=default_home, key=f'{league}_home')
    away = col_away.selectbox('Away team', poisson_teams, index=default_away, key=f'{league}_away')

    if home == away:
        st.warning('Pick two different teams.')
        st.stop()

    st.markdown(f'Elo progress — {home} vs {away}')
    st.pyplot(elo_matchup_progress_chart(elo_state, [home, away]))

    st.divider()
    goals_model_name, goals_params, goals_meta, goals_title, goals_caption = _best_goals_model(
        league, poisson_params, negbinom_params, poisson_meta, negbinom_meta,
    )
    goals_result = _render_goals_model_section(
        goals_title, goals_caption, goals_model_name, goals_params, goals_meta, home, away,
    )


    known = set(elo_state['teams'])
    elo_xgb_probs = None
    if home in known and away in known:
        feats = build_feature_row(elo_state, home, away)
        probs = booster.predict(xgb.DMatrix(feats))[0]
        classes = list(elo_state['classes'])  # e.g. ['Away', 'Draw', 'Home']
        elo_xgb_probs = (
            float(probs[classes.index('Home')]),
            float(probs[classes.index('Draw')]),
            float(probs[classes.index('Away')]),
        )

    logistic_probs = None
    if logistic_params is not None and home in set(logistic_state['teams']) and away in set(logistic_state['teams']):
        result = logistic_predict(logistic_params, logistic_state, home, away)
        logistic_probs = (result['p_home'], result['p_draw'], result['p_away'])

    mlp_probs = None
    if mlp_params is not None and home in set(mlp_state['teams']) and away in set(mlp_state['teams']):
        result = mlp_predict(mlp_params, mlp_state, home, away)
        mlp_probs = (result['p_home'], result['p_draw'], result['p_away'])

    ensemble_probs = None
    if ensemble_meta is not None and elo_xgb_probs is not None and logistic_probs is not None:
        ensemble_probs = tuple(
            (goals_result[f'p_{k}'] + elo_xgb_probs[i] + logistic_probs[i]) / 3
            for i, k in enumerate(('home', 'draw', 'away'))
        )

    candidates = [
        ('elo_xgb', elo_meta, elo_xgb_probs, 'XGBoost',
         'XGBoost classifier over Elo-based features (rolling/exponential moving averages, recent form).'),
        ('logistic', logistic_meta, logistic_probs, 'Logistic Regression',
         'Multinomial logistic regression over the same Elo/form/h2h/momentum features as XGBoost.'),
        ('mlp', mlp_meta, mlp_probs, 'MLP (Neural Network)',
         'Small one-hidden-layer neural network over the same features with L2-regularized and '
         'early-stopped to limit overfitting on a dataset this size.'),
        ('ensemble', ensemble_meta, ensemble_probs, 'Ensemble',
         f'Unweighted average of {", ".join(ensemble_meta["constituents"])}.' if ensemble_meta else ''),
    ]
    for short_name, meta, probs, _, _ in candidates:
        if probs is not None:
            log_prediction_safe(f'{short_name}_{league}', meta, home, away, *probs)

    available = [c for c in candidates if c[1] is not None and c[2] is not None]
    st.divider()
    if not available:
        st.info('This matchup includes a team without feature history for these models.')
    else:
        short_name, meta, probs, title, caption_extra = min(available, key=lambda c: c[1]['rps'])
        st.subheader(title)
        outcome_row(*probs, home, away)
        st.caption(
            'The model shown has the best recorded walk-forward RPS among XGBoost, '
            'Logistic Regression, MLP, and the Ensemble for this league.'
        )
        st.caption(metadata_caption(meta, caption_extra))
