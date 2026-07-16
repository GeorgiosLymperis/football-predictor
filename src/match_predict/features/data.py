from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / 'data'

_RESULT_MAP = {'H': 'Home', 'D': 'Draw', 'A': 'Away'}

_ODDS_COLUMNS = [
    # Market Average Home, Draw, Away odds. After 2019-2020
    ('AvgH', 'AvgD', 'AvgA'),
    ('B365H', 'B365D', 'B365A'),  # Bet365 Home, Draw, Away odds
]

# Match-stat columns (Home, Away), keyed to match MATCH_STATS in features/elo.py.
# Older seasons for some leagues don't carry these columns at all.
_STAT_COLUMNS = {
    'shots': ('HS', 'AS'),
    'shots_target': ('HST', 'AST'),
    'corners': ('HC', 'AC'),
}


def _extract_stats(df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns
    out = {}
    for stat_key, (home_col, away_col) in _STAT_COLUMNS.items():
        out[f'{stat_key}_home'] = df[home_col] if home_col in columns else np.nan
        out[f'{stat_key}_away'] = df[away_col] if away_col in columns else np.nan
    return pd.DataFrame(out, index=df.index)


class DataValidationError(ValueError):
    ...


def _read_season_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DataValidationError(f'Unable to read {path}: {e}')

    return df


def _extract_odds(df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns
    for h, d, a in _ODDS_COLUMNS:
        if h in columns and d in columns and a in columns:
            odds = df[[h, d, a]]
            odds = odds.rename(
                columns={
                    h: 'odds_home',
                    d: 'odds_draw',
                    a: 'odds_away',
                }
            )

            return odds

    n = df.shape[0]
    return pd.DataFrame(
        {
            'odds_home': [None] * n,
            'odds_draw': [None] * n,
            'odds_away': [None] * n,
        }
    )


def _load_season(path: Path, season: str) -> pd.DataFrame:
    raw = _read_season_csv(path)

    raw = raw.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])

    df = pd.DataFrame(
        {
            'team1': raw['HomeTeam'],
            'team2': raw['AwayTeam'],
            'score1': raw['FTHG'].astype(int),
            'score2': raw['FTAG'].astype(int),
            'result': raw['FTR'].map(_RESULT_MAP),
            'date': pd.to_datetime(raw['Date'], dayfirst=True, format='mixed'),
        }
    )

    df = pd.concat([df, _extract_odds(raw), _extract_stats(raw)], axis=1)
    df = df.sort_values(by='date', kind='stable').reset_index(drop=True)
    df['year'] = int(season.split('-')[0])
    df['match_day'] = df.index + 1
    return df


def _validate_names(
    df: pd.DataFrame, known_teams: set[str], league_name: str
) -> None:
    unknown = sorted((set(df['team1']) | set(df['team2'])) - known_teams)
    if unknown:
        seasons = sorted(
            df.loc[
                df['team1'].isin(unknown) | df['team2'].isin(unknown), 'year'
            ].unique()
        )
        raise DataValidationError(
            f'[{league_name}] Unrecognized team name(s) {unknown} in season(s) {seasons}. '
            f'Add a mapping under name_fixes in config/leagues/{league_name}.yaml (or, if this '
            f'is a genuinely new club, add it to the teams: allowlist).'
        )


def _validate_schema(df: pd.DataFrame) -> None:
    for col in ('score1', 'score2', 'odds_home', 'odds_draw', 'odds_away'):
        if (df[col] < 0).any():
            raise DataValidationError(f'Negative values found in {col!r}.')

    dup_key = ['year', 'match_day', 'team1', 'team2']
    dupes = df[df.duplicated(dup_key, keep=False)]
    if not dupes.empty:
        raise DataValidationError(
            f'Duplicate (year, match_day, team1, team2) rows found:\n{dupes[dup_key]}'
        )


def load_league_matches(
    league_cfg: dict, seasons: List[str] | None = None
) -> pd.DataFrame:
    raw_dir: Path = DATA_DIR / league_cfg['raw_dir']
    prefix = league_cfg['file_prefix']

    if seasons is None:
        files = sorted(raw_dir.glob(f'{prefix}-*.csv'))
        seasons = [f.stem.removeprefix(f'{prefix}-') for f in files]
    else:
        files = [raw_dir / f'{prefix}-{s}.csv' for s in seasons]

    frames = [_load_season(p, s) for p, s in zip(files, seasons)]
    df = pd.concat(frames, ignore_index=True)
    df = df.replace(
        {'team1': league_cfg['name_fixes'], 'team2': league_cfg['name_fixes']}
    )
    df = df.sort_values(['year', 'match_day'], kind='stable').reset_index(
        drop=True
    )
    _validate_names(df, set(league_cfg['teams']), league_cfg['league_name'])
    _validate_schema(df)
    return df
