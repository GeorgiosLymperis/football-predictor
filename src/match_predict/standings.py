"""Current-season league table from match results, joined with each team's
Elo rating and momentum."""
import numpy as np
import pandas as pd

FORM_MATCHES = 5
# Elo points per match (trend over the last 10 matches). Historically about a
# third of team-matches fall above +1.5, a third below -1.5.
MOMENTUM_THRESHOLD = 1.5


def _team_results(season: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, oldest first."""
    home = pd.DataFrame({
        'team': season['team1'], 'gf': season['score1'], 'ga': season['score2'], 'date': season['date'],
    })
    away = pd.DataFrame({
        'team': season['team2'], 'gf': season['score2'], 'ga': season['score1'], 'date': season['date'],
    })
    rows = pd.concat([home, away], ignore_index=True).sort_values('date', kind='stable')
    rows['result'] = np.select([rows['gf'] > rows['ga'], rows['gf'] == rows['ga']], ['W', 'D'], 'L')
    rows['points'] = rows['result'].map({'W': 3, 'D': 1, 'L': 0})
    return rows


def league_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Standings for the latest season in `matches` (as returned by
    load_league_matches). Ties are broken by goal difference, then goals
    scored, then name; leagues that use head-to-head first may differ."""
    season = matches[matches['year'] == matches['year'].max()]
    results = _team_results(season)
    grouped = results.groupby('team')
    table = pd.DataFrame({
        'played': grouped.size(),
        'won': grouped['result'].apply(lambda r: (r == 'W').sum()),
        'drawn': grouped['result'].apply(lambda r: (r == 'D').sum()),
        'lost': grouped['result'].apply(lambda r: (r == 'L').sum()),
        'gf': grouped['gf'].sum(),
        'ga': grouped['ga'].sum(),
        'points': grouped['points'].sum(),
        'form': grouped['result'].apply(lambda r: list(r.iloc[-FORM_MATCHES:])),
    })
    table['gd'] = table['gf'] - table['ga']
    table = (
        table.rename_axis('team').reset_index()
        .sort_values(['points', 'gd', 'gf', 'team'], ascending=[False, False, False, True], kind='stable')
        .reset_index(drop=True)
    )
    table.insert(0, 'position', np.arange(1, len(table) + 1))
    return table


def momentum_label(trend_slope: float) -> str:
    if trend_slope > MOMENTUM_THRESHOLD:
        return 'up'
    if trend_slope < -MOMENTUM_THRESHOLD:
        return 'down'
    return 'steady'


def with_elo(table: pd.DataFrame, elo_state: dict) -> pd.DataFrame:
    """Add each team's Elo, its Elo rank within the table, how many places
    higher the team sits in the table than its Elo rank, and momentum.
    Teams missing from the Elo state (e.g. results newer than the last Elo
    refresh) get NaN/None instead of failing."""
    def lookup(key):
        return pd.Series(elo_state[key], index=list(elo_state['teams'])).reindex(table['team']).to_numpy()

    out = table.copy()
    out['elo'] = lookup('elo')
    out['elo_rank'] = out['elo'].rank(ascending=False, method='min').astype('Int64')
    out['vs_elo'] = out['elo_rank'] - out['position']
    out['trend_slope'] = lookup('trend_slope')
    # Missing momentum reads as None or NaN depending on the pandas version;
    # check it with pd.isna.
    out['momentum'] = [None if np.isnan(s) else momentum_label(s) for s in out['trend_slope']]
    return out
