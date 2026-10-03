from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from match_predict.config import load_league_config
from match_predict.features.data import DATA_DIR, load_league_matches
from match_predict.fixtures import FixturesUnavailableError
from match_predict.predict.poisson import team_ratings
from match_predict.service import (
    artifacts_stamp, best_goals_model, best_outcome_model, fixture_predictions, load_league,
    log_prediction_safe, predict_match,
)
from match_predict.simulation import UnratedTeamsError, remaining_fixtures, simulate_season, summarise
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


OUTCOME_COLOURS = (('Home win', '#2a78d6'), ('Draw', '#c9c8c2'), ('Away win', '#eb6834'))
_FIXTURES_CSS = """
<style>
.fixtures-wrap { overflow-x: auto; }
.fixtures { border-collapse: collapse; width: 100%; font-size: 0.875rem; font-variant-numeric: tabular-nums; }
.fixtures th { color: #52514e; font-weight: 600; text-align: left; padding: 6px 6px;
               border-bottom: 1px solid #d8d7d2; white-space: nowrap; }
.fixtures td { padding: 7px 6px; border-bottom: 1px solid #eeede9; white-space: nowrap; vertical-align: middle; }
.fixtures .when { color: #52514e; font-size: 0.8rem; }
.fixtures .match { font-weight: 600; color: #0b0b0b; }
.fixtures .bar { display: inline-flex; gap: 2px; width: 120px; height: 10px; vertical-align: middle; margin-right: 8px; }
.fixtures .bar span:first-child { border-radius: 4px 0 0 4px; }
.fixtures .bar span:last-child { border-radius: 0 4px 4px 0; }
.fixtures .nums { color: #0b0b0b; }
.fixtures .muted { color: #52514e; }
.fixtures-legend { font-size: 0.8rem; color: #52514e; margin: 0 0 6px 0; }
.fixtures-legend .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
                           margin: 0 4px 0 12px; vertical-align: -1px; }
.fixtures-legend .swatch:first-child { margin-left: 0; }
@media (max-width: 640px) { .fixtures .bar { width: 56px; margin-right: 6px; } }
</style>
"""


def _pct_triplet(probs) -> str:
    return ' \u00b7 '.join(f'{p:.0%}' for p in probs)


def fixtures_html(rows: list[dict]) -> str:
    legend = ''.join(
        f'<span class="swatch" style="background:{colour}"></span>{label}' for label, colour in OUTCOME_COLOURS
    )
    head = '<tr><th>Match</th><th>Model: home \u00b7 draw \u00b7 away</th><th>Market</th></tr>'
    body = []
    for f in rows:
        when = f['date'].strftime('%a %d %b') + (f' \u00b7 {f["kickoff"]}' if f['kickoff'] else '')
        if f['probs'] is None:
            model = '<span class="muted">no model history for a team</span>'
        else:
            segments = ''.join(
                f'<span style="width:{p * 100:.1f}%;background:{colour}" title="{label} {p:.0%}"></span>'
                for p, (label, colour) in zip(f['probs'], OUTCOME_COLOURS)
            )
            model = f'<span class="bar">{segments}</span><span class="nums">{_pct_triplet(f["probs"])}</span>'
        market = (f'<span class="muted">{_pct_triplet(f["market_probs"])}</span>'
                  if f['market_probs'] else '<span class="muted">\u2013</span>')
        body.append(f'<tr><td><div class="match">{f["home"]} \u2013 {f["away"]}</div>'
                    f'<div class="when">{when}</div></td><td>{model}</td><td>{market}</td></tr>')
    return (f'{_FIXTURES_CSS}<div class="fixtures-legend">{legend}</div><div class="fixtures-wrap">'
            f'<table class="fixtures"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>')


def render_fixtures(lm, display_name: str) -> None:
    st.subheader('Upcoming fixtures')
    try:
        rows = fixture_predictions(lm)
    except FixturesUnavailableError:
        st.warning('Upcoming fixtures are unavailable right now (football-data.co.uk could not be reached).')
        return
    if not rows:
        st.info(f'No {display_name} matches in the fixture list right now. football-data.co.uk lists '
                'only the next few days of matches, so this is empty during international breaks '
                'and between rounds.')
        return
    st.html(fixtures_html(rows))
    served = {f['model'] for f in rows if f['model']}
    titles = ', '.join(OUTCOME_MODEL_INFO[m][0] for m in sorted(served)) or "the league's best model"
    st.caption(
        f'Model probabilities from {titles}, the model with the best '
        'walk-forward RPS for this league. Market probabilities are the average bookmaker odds with the '
        'bookmaker margin removed. Kick-off times are UK time, as listed by football-data.co.uk.'
    )


def _visible_labels(xs, ys, names, x_dom, y_dom, always=None, width=640, height=400) -> np.ndarray:
    """Which team names to print beside their dots: furthest from average
    first, skipping any label that would overlap another label or a dot.
    `always` is labelled regardless. Pixel sizes are approximate since the
    chart width follows the page; skipped teams still show on hover."""
    px_x = width / (x_dom[1] - x_dom[0])
    px_y = height / (y_dom[1] - y_dom[0])
    px = (np.asarray(xs) - x_dom[0]) * px_x
    py = (y_dom[1] - np.asarray(ys)) * px_y
    boxes = [(x - 6, x + 6, y - 6, y + 6) for x, y in zip(px, py)]  # dots
    label_box = [(x + 6, x + 10 + 6.5 * len(n), y - 7, y + 7) for x, y, n in zip(px, py, names)]
    order = sorted(range(len(names)), key=lambda i: (names[i] != always, -np.hypot(xs[i], ys[i])))
    visible = np.zeros(len(names), dtype=bool)
    for i in order:
        l, r, t, b = label_box[i]
        clear = all(r < bl or l > br or b < bt or t > bb
                    for j, (bl, br, bt, bb) in enumerate(boxes) if j != i)
        if clear or names[i] == always:
            visible[i] = True
            boxes.append(label_box[i])
    return visible


def attack_defence_chart(ratings: pd.DataFrame, highlight: str | None = None) -> alt.LayerChart:
    """Scatter of attack (x) against defence (y); right and up are better."""
    pad = 8
    x_dom = [min(ratings['attack_pct'].min(), 0) - pad, max(ratings['attack_pct'].max(), 0) + pad * 2]
    y_dom = [min(ratings['defence_pct'].min(), 0) - pad, max(ratings['defence_pct'].max(), 0) + pad]
    data = ratings.assign(
        attack_label=[f'{v:+.0f}% ({lo:+.0f} to {hi:+.0f})' for v, lo, hi in
                      ratings[['attack_pct', 'attack_pct_lo', 'attack_pct_hi']].to_numpy()],
        defence_label=[f'{v:+.0f}% ({lo:+.0f} to {hi:+.0f})' for v, lo, hi in
                       ratings[['defence_pct', 'defence_pct_lo', 'defence_pct_hi']].to_numpy()],
        labelled=_visible_labels(ratings['attack_pct'].to_numpy(), ratings['defence_pct'].to_numpy(),
                                 list(ratings['team']), x_dom, y_dom, always=highlight),
        highlighted=ratings['team'].eq(highlight),
    )
    x = alt.X('attack_pct:Q', scale=alt.Scale(domain=x_dom, nice=False),
              title='Attack: goals scored vs league average (%)')
    y = alt.Y('defence_pct:Q', scale=alt.Scale(domain=y_dom, nice=False),
              title='Defence: fewer goals conceded vs league average (%)')
    base = alt.Chart(data).encode(x=x, y=y)
    zero_x = alt.Chart(pd.DataFrame({'v': [0]})).mark_rule(color='#c9c8c2', strokeDash=[4, 4]).encode(x='v:Q')
    zero_y = alt.Chart(pd.DataFrame({'v': [0]})).mark_rule(color='#c9c8c2', strokeDash=[4, 4]).encode(y='v:Q')
    corners = pd.DataFrame([
        (x_dom[1], y_dom[1], 'Strong at both ends', 'right', 'top'),
        (x_dom[0], y_dom[1], 'Defence first', 'left', 'top'),
        (x_dom[1], y_dom[0], 'Attack first', 'right', 'bottom'),
        (x_dom[0], y_dom[0], 'Weak at both ends', 'left', 'bottom'),
    ], columns=['x', 'y', 'text', 'align', 'baseline'])
    corner_layers = [
        alt.Chart(corners[corners['align'].eq(a) & corners['baseline'].eq(b)])
        .mark_text(align=a, baseline=b, dx=6 if a == 'left' else -6, dy=6 if b == 'top' else -6,
                   color='#8a8984', fontSize=11, fontStyle='italic')
        .encode(x='x:Q', y='y:Q', text='text:N')
        for a, b in (('right', 'top'), ('left', 'top'), ('right', 'bottom'), ('left', 'bottom'))
    ]
    points = base.mark_circle(size=90, opacity=1, stroke='#ffffff', strokeWidth=2).encode(
        color=alt.condition('datum.highlighted', alt.value('#eb6834'), alt.value('#2a78d6')),
        order=alt.Order('highlighted:Q'),
        tooltip=[alt.Tooltip('team:N', title='Team'),
                 alt.Tooltip('attack_label:N', title='Attack (80% range)'),
                 alt.Tooltip('defence_label:N', title='Defence (80% range)')],
    )
    labels = base.transform_filter('datum.labelled && !datum.highlighted').mark_text(
        align='left', dx=8, fontSize=11, color='#3d3c39').encode(text='team:N')
    highlighted = base.transform_filter('datum.highlighted').mark_text(
        align='left', dx=8, fontSize=12, fontWeight='bold', color='#0b0b0b').encode(text='team:N')
    return alt.layer(zero_x, zero_y, *corner_layers, points, labels, highlighted).properties(height=440)


def render_attack_defence(lm, current_teams: list[str]) -> None:
    st.subheader('Attack vs defence')
    ratings = team_ratings(lm.poisson, current_teams)
    highlight = st.selectbox('Highlight a team', sorted(ratings['team']), index=None,
                             placeholder='Choose a team', key=f'{lm.league}_highlight')
    st.altair_chart(attack_defence_chart(ratings, highlight), use_container_width=True)
    missing = [t for t in current_teams if t not in set(ratings['team'])]
    note = (f' Not rated by the model yet: {", ".join(missing)}.'
            if missing else '')
    st.caption(
        'Team ratings from the Bayesian Poisson model, which estimates how many goals each team scores '
        'and concedes against an average opponent. Right = scores more, up = concedes fewer, relative '
        'to the average of this season\'s teams. Teams without a label are in crowded areas: hover a dot, '
        'or pick the team above. Hover also shows the 80% uncertainty range.' + note
    )


N_SIMULATIONS = 10_000
SIM_TINTS = ((0.75, '#9ec5f4'), (0.5, '#b7d3f6'), (0.25, '#cde2fb'), (0.05, '#e6f0fd'))  # blue ramp, light end


@st.cache_data(show_spinner='Simulating the rest of the season...')
def season_simulation(league: str, data_stamp: float, model_stamp: float) -> dict:
    """Simulated season summary; the stamps invalidate the cache when the
    weekly data refresh or a retrain changes the inputs."""
    lm = load_league(league)
    cfg = load_league_config(league)
    df = load_league_matches(cfg)
    season = df[df['year'] == df['year'].max()]
    params = lm.negbinom if best_goals_model(lm) == 'negbinom' else lm.poisson
    try:
        teams, points, positions = simulate_season(params, season, n_sims=N_SIMULATIONS)
    except UnratedTeamsError as e:
        return {'unrated': e.teams}
    sim = cfg.get('simulation', {})
    current = league_table(df).set_index('team')['points']
    return {
        'summary': summarise(teams, current, points, positions, sim.get('relegation_spots', 3),
                             sim.get('relegation_playoff', False)),
        'n_remaining': len(remaining_fixtures(season, teams)),
        'note': sim.get('note', ''),
    }


def _prob(p: float) -> str:
    if p == 0:
        return '\u2013'
    if p < 0.005:
        return '<1%'
    if p > 0.995 and p < 1:
        return '>99%'
    return f'{p:.0%}'


def _prob_cell(p: float) -> str:
    tint = next((colour for threshold, colour in SIM_TINTS if p >= threshold), None)
    style = f' style="background:{tint}"' if tint else ''
    return f'<td{style}>{_prob(p)}</td>'


def simulation_html(summary: pd.DataFrame) -> str:
    prob_cols = [('p_title', 'Title', 'Finish first'), ('p_top4', 'Top 4', 'Finish in the top four')]
    if 'p_relegation_playoff' in summary:
        prob_cols.append(('p_relegation_playoff', 'Playoff', 'Finish in the relegation playoff place'))
    prob_cols.append(('p_relegation', 'Relegated', 'Finish in a direct relegation place'))
    headers = [('#', 'left pos', 'Ordered by average finishing position'), ('Team', 'left team', ''),
               ('Pts', '', 'Points so far'),
               ('Proj. pts', '', 'Average final points across simulations'),
               ('Avg pos', '', 'Average finishing position'),
               *((label, '', tip) for _, label, tip in prob_cols)]
    head = ''.join(f'<th class="{cls}" title="{tip}">{name}</th>' for name, cls, tip in headers)
    body = []
    for rank, r in enumerate(summary.itertuples(), start=1):
        cells = [f'<td class="left pos">{rank}</td>', f'<td class="left team">{r.team}</td>',
                 f'<td>{r.points_now}</td>',
                 f'<td><b>{r.expected_points:.0f}</b></td>', f'<td>{r.expected_position:.1f}</td>',
                 *(_prob_cell(getattr(r, col)) for col, _, _ in prob_cols)]
        body.append(f'<tr>{"".join(cells)}</tr>')
    return (f'{_TABLE_CSS}<div class="league-table-wrap"><table class="league-table">'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def render_season_simulation(league: str) -> None:
    st.subheader('Season simulation')
    result = season_simulation(league, _data_stamp(league), artifacts_stamp(league))
    if 'unrated' in result:
        st.info('Available after the next model retrain: the goals model has no rating yet for '
                f'{", ".join(result["unrated"])}.')
        return
    st.html(simulation_html(result['summary']))
    st.caption(
        f'{N_SIMULATIONS:,} simulations of the {result["n_remaining"]} remaining matches with the goals model. '
        'Each simulated season draws team ratings from the model\'s uncertainty, so teams with few matches '
        'behind them (such as newly promoted ones) have a wider spread of outcomes. Ties are broken by goal '
        'difference, then goals scored. Shading marks probabilities of 5%, 25%, 50% and 75% and above. '
        + result['note']
    )


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

    render_fixtures(lm, display_name)

    render_attack_defence(lm, list(table['team']))

    render_season_simulation(league)

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
    season_teams = sorted(table['team'])
    default_home_team, default_away_team = DEFAULT_MATCHUPS.get(league, (season_teams[0], season_teams[1]))
    default_home = season_teams.index(default_home_team) if default_home_team in season_teams else 0
    default_away = season_teams.index(default_away_team) if default_away_team in season_teams else 1

    col_home, col_away = st.columns(2)
    home = col_home.selectbox('Home team', season_teams, index=default_home, key=f'{league}_home')
    away = col_away.selectbox('Away team', season_teams, index=default_away, key=f'{league}_away')

    if home == away:
        st.warning('Pick two different teams.')
        st.stop()

    st.markdown(f'Elo progress — {home} vs {away}')
    st.pyplot(elo_matchup_progress_chart(elo_state, [home, away]))

    prediction = predict_match(lm, home, away)
    for model_name, probs in prediction.probs.items():
        log_prediction_safe(f'{model_name}_{league}', meta[model_name], home, away, *probs)

    st.divider()
    if prediction.goals is None:
        st.subheader(GOALS_MODEL_INFO[prediction.goals_model][0])
        st.info(f'The goals model has no rating yet for {home if home not in lm.poisson["teams"] else away}; '
                'it is added at the next retrain.')
    else:
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
