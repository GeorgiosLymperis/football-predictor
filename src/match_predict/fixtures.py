"""Upcoming fixtures from football-data.co.uk's rolling fixtures.csv, which
lists the next few days of matches across all divisions it covers.
"""
import io
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from match_predict.config import DIVISION_CODES, load_league_config
from match_predict.metrics.odds import implied_probs

FIXTURES_URL = 'https://www.football-data.co.uk/fixtures.csv'
CACHE_SECONDS = 3600

_cache: tuple[float, pd.DataFrame] | None = None


class FixturesUnavailableError(RuntimeError):
    pass


def _download() -> pd.DataFrame:
    resp = requests.get(FIXTURES_URL, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.content.decode('utf-8-sig')))


def all_fixtures() -> pd.DataFrame:
    """The whole fixtures file, re-downloaded at most once per CACHE_SECONDS.
    If a refresh fails, the last good copy is served instead."""
    global _cache
    now = time.monotonic()
    if _cache is None or now - _cache[0] > CACHE_SECONDS:
        try:
            _cache = (now, _download())
        except (requests.RequestException, ValueError) as e:
            if _cache is None:
                raise FixturesUnavailableError(f'Could not fetch {FIXTURES_URL}: {e}') from e
    return _cache[1]


def _market_probs(row: dict) -> tuple[float, float, float] | None:
    odds = [row.get(c) for c in ('AvgH', 'AvgD', 'AvgA')]
    try:
        odds = [float(o) for o in odds]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(o) and o > 1 for o in odds):
        return None
    return tuple(float(p) for p in implied_probs(*([o] for o in odds))[0])


def _kickoff(value) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def league_fixtures(league: str) -> list[dict]:
    """Upcoming matches for one league, earliest first. Team names are mapped
    through the league's name_fixes so they match the models' names."""
    fixes = load_league_config(league)['name_fixes'] or {}
    df = all_fixtures()
    rows = df[df['Div'] == DIVISION_CODES[league]].to_dict('records')
    fixtures = [
        {
            'date': datetime.strptime(r['Date'], '%d/%m/%Y').date(),
            'kickoff': _kickoff(r.get('Time')),
            'home': fixes.get(r['HomeTeam'], r['HomeTeam']),
            'away': fixes.get(r['AwayTeam'], r['AwayTeam']),
            'market_probs': _market_probs(r),
        }
        for r in rows
    ]
    return sorted(fixtures, key=lambda f: (f['date'], f['kickoff'] is None, f['kickoff'] or ''))


def clear_cache() -> None:
    global _cache
    _cache = None

