import argparse
import sys

import requests

from match_predict.config import load_league_config
from match_predict.features.data import DATA_DIR, load_league_matches

BASE_URL = 'https://www.football-data.co.uk/mmz4281'

DIVISION_CODES = {
    'greek': 'G1',
    'premier_league': 'E0',
    'la_liga': 'SP1',
    'bundesliga': 'D1',
    'serie_a': 'I1',
    'ligue_1': 'F1',
}


def _season_code(season: str) -> str:
    """'2025-2026' -> '2526', matching football-data.co.uk's URL scheme."""
    start, end = season.split('-')
    return start[-2:] + end[-2:]


def _latest_local_season(raw_dir, prefix: str) -> str:
    files = sorted(raw_dir.glob(f'{prefix}-*.csv'))
    if not files:
        raise FileNotFoundError(f'No existing season files found in {raw_dir}')
    return files[-1].stem.removeprefix(f'{prefix}-')


def update_league(league: str) -> bool:
    cfg = load_league_config(league)
    raw_dir = DATA_DIR / cfg['raw_dir']
    prefix = cfg['file_prefix']
    season = _latest_local_season(raw_dir, prefix)
    code = DIVISION_CODES[league]
    url = f'{BASE_URL}/{_season_code(season)}/{code}.csv'
    dest = raw_dir / f'{prefix}-{season}.csv'

    resp = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
    if resp.status_code != 200 or not resp.content.strip():
        print(f'[{league}] Fetch failed ({resp.status_code}) from {url}; leaving local file untouched.')
        return False

    backup = dest.read_bytes()
    old_rows = backup.count(b'\n')
    dest.write_bytes(resp.content)
    try:
        df = load_league_matches(cfg)
    except Exception as e:
        dest.write_bytes(backup)
        print(f'[{league}] Downloaded file failed validation ({e}); reverted {dest}.')
        return False

    new_rows = resp.content.count(b'\n')
    added = max(new_rows - old_rows, 0)
    season_matches = int((df['year'] == int(season.split('-')[0])).sum())
    print(f'[{league}] Refreshed {dest} from {url}: {season_matches} matches for {season} '
          f'({added} new rows this run).')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--league', choices=sorted(DIVISION_CODES),
        help='Only refresh this league; default refreshes all leagues.',
    )
    args = parser.parse_args()

    leagues = [args.league] if args.league else sorted(DIVISION_CODES)
    all_ok = True
    for league in leagues:
        try:
            all_ok = update_league(league) and all_ok
        except Exception as e:
            print(f'[{league}] ERROR: {e}')
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
