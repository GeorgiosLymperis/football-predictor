from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from match_predict.config import load_league_config
from match_predict.features.data import DATA_DIR, load_league_matches
from match_predict.service import best_outcome_model, load_league, log_prediction_safe, predict_match
from match_predict.standings import MOMENTUM_THRESHOLD, league_table, with_elo

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


FORM_BADGES = {  # (background, text) - tinted so the letter, not the colour, carries the result
    'W': ('#d6efd6', '#0b5a0b'),
    'D': ('#e7e6e2', '#3d3c39'),
    'L': ('#f6d6d6', '#8f1f1f'),
}
MOMENTUM_ICONS = {  # (symbol, colour, label)
    'up': ('\u25B2', '#0b7a0b', 'Rising'),
    'steady': ('\u25BA', '#6b6a66', 'Steady'),
    'down': ('\u25BC', '#b52a2a', 'Falling'),
}
_TABLE_CSS = """
<style>
.league-table-wrap { overflow-x: auto; }
.league-table { border-collapse: collapse; width: 100%; font-size: 0.875rem; font-variant-numeric: tabular-nums; }
.league-table th { color: #52514e; font-weight: 600; text-align: right; padding: 6px 5px;
                   border-bottom: 1px solid #d8d7d2; white-space: nowrap; }
.league-table td { padding: 5px 5px; text-align: right; border-bottom: 1px solid #eeede9; white-space: nowrap; }
.league-table th.left, .league-table td.left { text-align: left; }
.league-table td.team { font-weight: 600; color: #0b0b0b; }
/* Keep position and team visible while the rest scrolls on narrow screens. */
.league-table .pos, .league-table .team { position: sticky; background: #ffffff; z-index: 1; }
.league-table .pos { left: 0; min-width: 2em; }
.league-table .team { left: 2em; padding-right: 10px; }
.league-table .badge { display: inline-block; width: 1.35em; margin-right: 2px; border-radius: 4px;
                       text-align: center; font-size: 0.8rem; font-weight: 700; line-height: 1.5em; }
</style>
"""


def _data_stamp(league: str) -> float:
    raw_dir = DATA_DIR / load_league_config(league)['raw_dir']
    return max(p.stat().st_mtime for p in raw_dir.glob('*.csv'))


@st.cache_data
def season_table(league: str, data_stamp: float) -> pd.DataFrame:
    """Current-season standings; `data_stamp` invalidates the cache when the
    weekly data refresh rewrites the CSVs."""
    return league_table(load_league_matches(load_league_config(league)))


def _signed(n) -> str:
    if pd.isna(n):
        return '\u2013'
    return f'+{n}' if n > 0 else ('0' if n == 0 else f'\u2212{-n}')


def _signed_float(x: float) -> str:
    return f'+{x:.1f}' if x >= 0 else f'\u2212{-x:.1f}'


def league_table_html(table: pd.DataFrame) -> str:
    headers = [
        ('Pos', 'left pos', ''), ('Team', 'left team', ''), ('P', '', 'Played'), ('W', '', 'Won'), ('D', '', 'Drawn'),
        ('L', '', 'Lost'), ('GF', '', 'Goals for'), ('GA', '', 'Goals against'), ('GD', '', 'Goal difference'),
        ('Pts', '', 'Points'), ('Last 5', 'left', 'Most recent on the right'), ('Elo', '', 'Current Elo rating'),
        ('vs Elo', '', "Places higher (+) or lower (\u2212) in the table than the team's Elo rank"),
        ('Trend', 'left', 'Momentum: Elo trend over the last 10 matches, in Elo points per match'),
    ]
    head = ''.join(f'<th class="{cls}" title="{tip}">{name}</th>' for name, cls, tip in headers)
    body = []
    for r in table.itertuples():
        form = ''.join(
            f'<span class="badge" style="background:{FORM_BADGES[x][0]};color:{FORM_BADGES[x][1]}">{x}</span>'
            for x in r.form
        )
        if r.momentum is None:
            momentum = '\u2013'
        else:
            symbol, colour, label = MOMENTUM_ICONS[r.momentum]
            momentum = (f'<span style="color:{colour}" title="{label}">{symbol}</span> '
                        f'<span style="color:#52514e">{_signed_float(r.trend_slope)}</span>')
        elo = '\u2013' if pd.isna(r.elo) else f'{r.elo:.0f}'
        cells = [
            f'<td class="left pos">{r.position}</td>', f'<td class="left team">{r.team}</td>',
            *(f'<td>{v}</td>' for v in (r.played, r.won, r.drawn, r.lost, r.gf, r.ga, _signed(r.gd))),
            f'<td><b>{r.points}</b></td>', f'<td class="left">{form}</td>', f'<td>{elo}</td>',
            f'<td>{_signed(r.vs_elo)}</td>', f'<td class="left">{momentum}</td>',
        ]
        body.append(f'<tr>{"".join(cells)}</tr>')
    return (f'{_TABLE_CSS}<div class="league-table-wrap"><table class="league-table">'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


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


GOALS_MODEL_INFO = {
    'negbinom': (
        'Negative Binomial',
        'Same hierarchical attack/defence structure as the Poisson model, but '
        'relaxes its mean=variance assumption with an extra dispersion parameter. '
        'Beat Poisson on walk-forward RPS for this league.',
    ),
    'poisson': (
        'Bayesian Poisson (Dixon-Coles)',
        'Hierarchical Bayesian Poisson regression with Dixon-Coles low-score correction.',
    ),
}


def _render_goals_model_section(goals_model: str, result: dict, meta: dict | None,
                                 has_negbinom: bool, home: str, away: str) -> None:
    title, caption_text = GOALS_MODEL_INFO[goals_model]
    if goals_model == 'poisson' and has_negbinom:
        caption_text += ' Beat Negative Binomial on walk-forward RPS for this league.'
    st.subheader(title)
    outcome_row(result['p_home'], result['p_draw'], result['p_away'], home, away)
    c1, c2 = st.columns(2)
    c1.metric(f'Expected goals, {home}', f'{result["xg_home"]:.2f}')
    c2.metric(f'Expected goals, {away}', f'{result["xg_away"]:.2f}')
    st.markdown('Most likely scorelines')
    st.dataframe(top_scorelines_table(result['top_scorelines']), hide_index=True)
    st.markdown('Scoreline probabilities')
    st.pyplot(score_heatmap(result['score_matrix'], home, away))
    st.caption(metadata_caption(meta, caption_text))


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


OUTCOME_MODEL_INFO = {
    'elo_xgb': ('XGBoost',
                'XGBoost classifier over Elo-based features (rolling/exponential moving averages, recent form).'),
    'logistic': ('Logistic Regression',
                 'Multinomial logistic regression over the same Elo/form/h2h/momentum features as XGBoost.'),
    'mlp': ('MLP (Neural Network)',
            'Small one-hidden-layer neural network over the same features with L2-regularized and '
            'early-stopped to limit overfitting on a dataset this size.'),
    'ensemble': ('Ensemble', ''),
}


def render_league_page(league: str) -> None:
    display_name = LEAGUE_DISPLAY_NAMES[league]
    st.title(f'{display_name} match predictor')

    lm = load_league(league)
    meta = lm.meta
    elo_state = lm.elo_state

    st.subheader('League table')
    table = with_elo(season_table(league, _data_stamp(league)), elo_state)
    st.html(league_table_html(table))
    st.caption(
        'Computed from this season\'s results; ties are broken by goal difference, then goals scored, '
        'so leagues that use head-to-head first, or have point deductions, may differ from the official table. '
        '**vs Elo**: places higher (+) or lower (\u2212) in the table than the team\'s Elo rank. '
        f'**Trend** (momentum): Elo trend over the last 10 matches in points per match; '
        f'\u25B2 above +{MOMENTUM_THRESHOLD}, \u25BC below \u2212{MOMENTUM_THRESHOLD}, \u25BA in between.'
    )

    st.subheader('Elo progress — top 5 teams')
    st.pyplot(elo_top5_progress_chart(elo_state))

    st.divider()
    st.subheader('Model performance')
    st.dataframe(model_rps_table([
        ('Bayesian Poisson (Dixon-Coles)', meta['poisson']),
        ('Negative Binomial', meta['negbinom']),
        ('XGBoost', meta['elo_xgb']),
        ('Logistic Regression', meta['logistic']),
        ('MLP (Neural Network)', meta['mlp']),
        ('Ensemble', meta['ensemble']),
    ]), hide_index=True)
    st.caption('Walk-forward RPS on held-out seasons. Lower is better. The sections below '
               'show only the best-performing model in each family.')

    st.divider()
    st.subheader('Pick a matchup')
    poisson_teams = sorted(lm.poisson['teams'])
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

    prediction = predict_match(lm, home, away)
    for model_name, probs in prediction.probs.items():
        log_prediction_safe(f'{model_name}_{league}', meta[model_name], home, away, *probs)

    st.divider()
    _render_goals_model_section(
        prediction.goals_model, prediction.goals, meta[prediction.goals_model],
        meta['negbinom'] is not None, home, away,
    )

    st.divider()
    best = best_outcome_model(lm, prediction)
    if best is None:
        st.info('This matchup includes a team without feature history for these models.')
    else:
        title, caption_extra = OUTCOME_MODEL_INFO[best]
        if best == 'ensemble':
            caption_extra = f'Unweighted average of {", ".join(meta["ensemble"]["constituents"])}.'
        st.subheader(title)
        outcome_row(*prediction.probs[best], home, away)
        st.caption(
            'The model shown has the best recorded walk-forward RPS among XGBoost, '
            'Logistic Regression, MLP, and the Ensemble for this league.'
        )
        st.caption(metadata_caption(meta[best], caption_extra))
