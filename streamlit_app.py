import streamlit as st

from app_common import LEAGUE_DISPLAY_NAMES, LEAGUE_ICONS, render_league_page


def render_home() -> None:
    st.title('Super League match predictor')
    st.write(
        'Pick a league from the sidebar. Each page shows that league\'s current '
        'Elo ratings, then lets you pick a matchup and compares several independently '
        'trained models:'
    )
    st.markdown(
        '- **Bayesian Poisson (Dixon-Coles)**: a hierarchical attack/defence '
        'rating model, gives expected goals and full scoreline probabilities.\n'
        '- **Elo + XGBoost**: a classifier over Elo and recent-form features, '
        'gives win/draw/away probabilities.\n'
        '- **Logistic Regression**: multinomial logistic regression over the '
        'same Elo/form/head-to-head/momentum features as Elo + XGBoost.\n'
        '- **MLP**: a small, regularized one-hidden-layer neural network over '
        'the same feature set.\n'
        '- **Ensemble**: an unweighted average of the goals model, XGBoost, '
        'and Logistic Regression.\n\n'
        'All are backtested walk-forward against a naive baseline *and* the '
        'betting market\'s own implied probabilities.'
    )


st.set_page_config(page_title='Super League Predictor', page_icon='⚽', layout='centered')

pages = [st.Page(render_home, title='Home', icon='⚽', default=True)]
for league, display_name in LEAGUE_DISPLAY_NAMES.items():
    pages.append(st.Page(
        lambda league=league: render_league_page(league),
        title=display_name,
        icon=LEAGUE_ICONS[league],
        url_path=league,
    ))

st.navigation(pages).run()
