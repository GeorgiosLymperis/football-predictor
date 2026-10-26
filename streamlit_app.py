import pandas as pd
import streamlit as st

from app_common import (
    LEAGUE_DISPLAY_NAMES, LEAGUE_ICONS, MODEL_TITLES, league_metadata, league_snapshot, render_league_page,
)
from match_predict.config import load_config
from match_predict.service import artifacts_stamp

REPO_URL = 'https://github.com/GeorgiosLymperis/football-predictor'
API_URL = 'https://football-predictor-api-2wwd.onrender.com/docs'

# Each league card is a bordered container whose page link is stretched over
# the whole card, so clicking anywhere on it opens the league page.
CARD_CSS = """
<style>
[class*="st-key-league-card-"] { position: relative; transition: border-color .15s, box-shadow .15s; }
[class*="st-key-league-card-"]:hover { border-color: #2a78d6; box-shadow: 0 2px 10px rgba(42, 120, 214, .15); }
[class*="st-key-league-card-"] * { position: static !important; }
[class*="st-key-league-card-"] [data-testid="stPageLink"] a::after { content: ""; position: absolute; inset: 0; }
[class*="st-key-league-card-"] [data-testid="stPageLink"] a:hover { background: none; }
</style>
"""

MODELS_SUMMARY = (
    '**Models:** Bayesian Poisson (Dixon-Coles) for scorelines, plus Elo + XGBoost, '
    'Logistic Regression, an MLP and an Ensemble for win/draw/away. '
    'All backtested walk-forward against a baseline and the betting market.'
)


def render_home(league_pages: dict, about_page) -> None:
    st.set_page_config(initial_sidebar_state='collapsed')
    st.title('Football match predictor')
    st.write('Pick a league to see its table, Elo ratings, season simulation and match predictions.')
    st.html(CARD_CSS)

    leagues = list(LEAGUE_DISPLAY_NAMES)
    for row_start in range(0, len(leagues), 3):
        for col, league in zip(st.columns(3), leagues[row_start:row_start + 3]):
            with col.container(border=True, key=f'league-card-{league}'):
                st.page_link(league_pages[league], label=f'**{LEAGUE_DISPLAY_NAMES[league]}**',
                             icon=LEAGUE_ICONS[league])
                st.caption(league_snapshot(league))

    st.caption(MODELS_SUMMARY)
    st.page_link(about_page, label='How the models are built and evaluated', icon=':material/arrow_forward:')


def best_model_summary() -> pd.DataFrame:
    """Per league, the champion with the lowest walk-forward RPS against the
    naive baseline and the de-vigged betting market."""
    rows = []
    for league, display_name in LEAGUE_DISPLAY_NAMES.items():
        meta = league_metadata(league, artifacts_stamp(league))
        model, best = min(((m, v) for m, v in meta.items() if v is not None), key=lambda mv: mv[1]['rps'])
        rows.append({
            'League': f'{LEAGUE_ICONS[league]} {display_name}',
            'Best model': MODEL_TITLES[model],
            'RPS': best['rps'],
            'Baseline': best['baseline_rps'],
            'Market': best.get('market_rps'),
            'Better than baseline': 1 - best['rps'] / best['baseline_rps'],
            'Gap to market': best['rps'] / best['market_rps'] - 1 if best.get('market_rps') else None,
        })
    return pd.DataFrame(rows)


def render_about() -> None:
    st.set_page_config(initial_sidebar_state='expanded')
    st.title('How it works')
    st.write(
        'Every model is trained on past seasons only and scored on seasons it has never seen, then compared '
        'with two benchmarks: a naive baseline and the betting market. The market is the hard one, since '
        'bookmaker odds already price in team news, injuries and a lot of money.'
    )

    summary = best_model_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric('Leagues covered', len(summary))
    c2.metric('Better than baseline', f'{summary["Better than baseline"].mean():.1%}',
              help='Average RPS reduction of the best model versus the naive baseline. Higher is better.')
    c3.metric('Gap to the market', f'{summary["Gap to market"].mean():+.1%}',
              help='Average RPS of the best model relative to the betting market. Closer to 0 is better.')

    st.dataframe(
        summary, hide_index=True,
        column_config={
            'RPS': st.column_config.NumberColumn(format='%.4f'),
            'Baseline': st.column_config.NumberColumn(format='%.4f'),
            'Market': st.column_config.NumberColumn(format='%.4f'),
            'Better than baseline': st.column_config.NumberColumn(format='percent'),
            'Gap to market': st.column_config.NumberColumn(format='percent'),
        },
    )
    st.caption('RPS (Ranked Probability Score) measures how close the predicted home/draw/away probabilities '
               'are to what happened, and penalises a home-win prediction less for a draw than for an away '
               'win. Lower is better.')

    backtest = load_config('backtest')
    st.subheader('Evaluation')
    st.markdown(
        f'- **Walk-forward backtest.** Models are refit as the season goes on and only ever predict matches '
        f'after their training data. Hyperparameters are chosen on {", ".join(backtest["selection_seasons"])}; '
        f'the scores above are on {", ".join(backtest["holdout_seasons"])}.\n'
        '- **Promotion.** A newly trained model replaces the served one only if its RPS beats both the '
        'baseline and the current model. Every version is tracked in MLflow with its git commit.\n'
        '- **Baseline** predicts every match with the historical home/draw/away frequencies. '
        '**Market** uses the average bookmaker odds with the bookmaker margin removed.'
    )

    st.subheader('Models')
    st.markdown(
        '| Model | What it does |\n|---|---|\n'
        '| Bayesian Poisson (Dixon-Coles) | Attack and defence rating per team, fit with PyMC; '
        'gives expected goals and every scoreline\'s probability |\n'
        '| Negative Binomial | Same, but allows more spread in goals than Poisson |\n'
        '| XGBoost | Gradient-boosted trees over Elo, form, head-to-head and momentum features |\n'
        '| Logistic Regression | Linear model over the same features |\n'
        '| MLP | Small neural network over the same features |\n'
        '| Ensemble | Plain average of the goals model, XGBoost and Logistic Regression |'
    )

    st.subheader('Data and code')
    st.markdown(
        'Results and odds for six leagues from [football-data.co.uk](https://www.football-data.co.uk/), '
        f'refreshed weekly. Source code on [GitHub]({REPO_URL}); the same models are served as JSON by a '
        f'[REST API]({API_URL}).'
    )


def render_league(league: str) -> None:
    st.set_page_config(initial_sidebar_state='expanded')
    render_league_page(league)


st.set_page_config(page_title='Football Match Predictor', page_icon='⚽', layout='centered')

league_pages = {
    league: st.Page(
        lambda league=league: render_league(league),
        title=display_name,
        icon=LEAGUE_ICONS[league],
        url_path=league,
    )
    for league, display_name in LEAGUE_DISPLAY_NAMES.items()
}
about = st.Page(render_about, title='About', icon=':material/info:', url_path='about')
home = st.Page(lambda: render_home(league_pages, about), title='Home', icon='⚽', url_path='home', default=True)

st.navigation({'': [home, about], 'Leagues': list(league_pages.values())}).run()
