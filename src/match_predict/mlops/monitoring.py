'''
Prediction logging to sqlite, so streamlit app
can log every prediction it serves and track performance
'''

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[3] / 'predictions.db'

_SCHEMA = '''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    p_home REAL NOT NULL,
    p_draw REAL NOT NULL,
    p_away REAL NOT NULL
)
'''

def log_prediction(model_name: str, model_version: Optional[str],
                   home_team: str, away_team: str,
                   p_home: float, p_draw: float, p_away: float) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_SCHEMA)
        conn.execute(

            'INSERT INTO predictions (ts, model_name, model_version, '
            'home_team, away_team, p_home, p_draw, p_away) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (datetime.now(timezone.utc).isoformat(), model_name, model_version,
             home_team, away_team, p_home, p_draw, p_away),
        )

def fetch_predictions() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_SCHEMA)
        df = pd.read_sql('SELECT * FROM predictions ORDER BY ts', conn)

    return df