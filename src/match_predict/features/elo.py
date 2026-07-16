import numpy as np
import pandas as pd

K = 20
HOME_ADVANTAGE = 100
INITIAL_ELO = 1500.0
EMA_SPANS = (5, 10, 15)
ROLL_WINDOWS = (5, 10, 15)
PCT_PERIODS = 10
FORM_MATCHES = 5
H2H_MEETINGS = 8
ELO_HISTORY_LENGTH = 100 

# Momentum/technical-analysis features, applied to the Elo series the same
# way they'd apply to a price series.
MACD_FAST_SPAN = 5
MACD_SLOW_SPAN = 15
MACD_SIGNAL_SPAN = 9
TREND_SLOPE_WINDOW = 10
RSI_WINDOW = 14 

STAT_ROLL_WINDOW = 5
MATCH_STATS = {'shots': 'Shots', 'shots_target': 'ShotsTarget', 'corners': 'Corners'}

BASE_FEATURES = [
    'EloDifference', 'HomeElo', 'AwayElo',
    'HomeEloPctChange', 'AwayEloPctChange', 'FormDiff',
    'HomeRoll5', 'AwayRoll5', 'HomeRoll10', 'AwayRoll10',
    'HomeRoll15', 'AwayRoll15', 'HomeEMA5', 'AwayEMA5',
    'HomeEMA10', 'AwayEMA10', 'HomeEMA15', 'AwayEMA15',
]
FORM_FEATURES = ['HomeForm5', 'AwayForm5', 'Form5Diff']
H2H_FEATURES = ['H2HHomePPG', 'H2HAwayPPG', 'H2HCount']
MOMENTUM_FEATURES = [
    'HomeMACD', 'AwayMACD', 'MACDDiff',
    'HomeMACDHist', 'AwayMACDHist',
    'HomeTrendSlope', 'AwayTrendSlope', 'TrendSlopeDiff',
    'HomeRSI', 'AwayRSI', 'RSIDiff',
]
SHOTS_FEATURES = [
    col
    for label in MATCH_STATS.values()
    for col in (
        f'Home{label}For', f'Away{label}For', f'Home{label}Against', f'Away{label}Against',
        f'{label}ForDiff', f'{label}AgainstDiff',
    )
]

FEATURE_VARIANTS = {
    'base': BASE_FEATURES,
    'base+form': BASE_FEATURES + FORM_FEATURES,
    'base+h2h': BASE_FEATURES + H2H_FEATURES,
    'base+form+h2h': BASE_FEATURES + FORM_FEATURES + H2H_FEATURES,
    'base+momentum': BASE_FEATURES + MOMENTUM_FEATURES,
    'base+form+momentum': BASE_FEATURES + FORM_FEATURES + MOMENTUM_FEATURES,
    'base+shots': BASE_FEATURES + SHOTS_FEATURES,
    'base+form+shots': BASE_FEATURES + FORM_FEATURES + SHOTS_FEATURES,
}


def elo_expected_score(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def elo_update(elo_home, elo_away, goals_home, goals_away, k=K, ha=HOME_ADVANTAGE):
    expected_home = elo_expected_score(elo_home + ha, elo_away)
    score_home = 1.0 if goals_home > goals_away else 0.0 if goals_home < goals_away else 0.5
    goal_diff = abs(goals_home - goals_away)
    if goal_diff <= 1:
        g = 1.0
    elif goal_diff == 2:
        g = 1.5
    else:
        g = (goal_diff + 11) / 8
    delta = k * g * (score_home - expected_home)
    return elo_home + delta, elo_away - delta

class TeamState:
    def __init__(self):
        self.history = [INITIAL_ELO]
        self.ema = {span: INITIAL_ELO for span in EMA_SPANS}
        self.macd_signal = 0.0
        self.points: list[int] = []  # league points earned per game
        self.dates: list = []  # match date
        self.stats_for: dict[str, list] = {stat: [] for stat in MATCH_STATS}
        self.stats_against: dict[str, list] = {stat: [] for stat in MATCH_STATS}

    @property
    def elo(self):
        return self.history[-1]
    
    def pct_change(self) -> float:
        if len(self.history) <= PCT_PERIODS:
            return 0.0
        past = self.history[-1 - PCT_PERIODS]
        now = self.history[-1]
        return (now - past) / past * 100
    
    def roll(self, window: int) -> float:
        if window <= 0:
            raise ValueError(f'window must be > 0. Got: {window}')
        return float(np.mean(self.history[-window:]))

    def form(self, window: int = FORM_MATCHES) -> float:
        if window <= 0:
            raise ValueError(f'window must be > 0. Got: {window}')
        recent = self.points[-window:]
        return float(np.mean(recent)) if recent else np.nan

    def macd(self) -> float:
        return self.ema[MACD_FAST_SPAN] - self.ema[MACD_SLOW_SPAN]
    
    def macd_histogram(self) -> float:
        return self.macd() - self.macd_signal
    
    def trend_slope(self, window: int = TREND_SLOPE_WINDOW) -> float:
        if window <= 0:
            raise ValueError(f'window should be positive. Got: {window}')
        y = self.history[-window:]
        if len(y) < 2:
            return 0.0
        x = np.arange(len(y))
        return float(np.polyfit(x, y, deg=1)[0])

    def rsi(self, window: int = RSI_WINDOW) -> float:
        '''Relative Strength Index'''
        if window <= 0:
            raise ValueError(f'window should be positive. Got: {window}')
        deltas = np.diff(self.history[-1 - window:])
        gains = deltas[deltas > 0]
        losses = - deltas[deltas < 0]
        avg_gain = gains.mean() if len(gains) else 0.0
        avg_losses = losses.mean() if len(losses) else 0.0

        if avg_losses == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        ratio = avg_gain / avg_losses
        return 100 - 100/(1 + ratio)
    
    def stat_roll(self, stat: str, against: bool, window: int = STAT_ROLL_WINDOW) -> float:
        if window <= 0:
            raise ValueError(f'window should be positive. Got: {window}')
        if stat not in MATCH_STATS:
            raise ValueError(
                f'{stat} not supported. '
                f'Available stats: {list(MATCH_STATS)}'
                )
        
        history = (self.stats_against if against else self.stats_for)[stat]
        recent = [v for v in history[-window:] if pd.notna(v)]
        return float(np.mean(recent)) if recent else np.nan
    
    def features(self, side: str) -> dict:
        out = {
            f'{side}Elo': self.elo,
            f'{side}EloPctChange': self.pct_change(),
        }
        
        for window in ROLL_WINDOWS:
            out[f'{side}Roll{window}'] = self.roll(window)

        for span in EMA_SPANS:
            out[f'{side}EMA{span}'] = self.ema[span]

        return out
    
    def push(self, new_elo: float):
        self.history.append(new_elo)
        for span in EMA_SPANS:
            alpha = 2 / (span + 1)
            self.ema[span] = alpha * new_elo + (1 - alpha) * self.ema[span]

        alpha_signal = 2 / (MACD_SIGNAL_SPAN + 1)
        self.macd_signal = (
            alpha_signal * self.macd() +
            (1 - alpha_signal) * self.macd_signal
            )
        
def h2h_features(h2h: dict, home_name: str, away_name: str) -> dict:
    """Points per game each side earned in their last previous meetings.

    NaN when the clubs have never met.
    """
    home_pts = h2h.get((home_name, away_name), [])[-H2H_MEETINGS:]
    away_pts = h2h.get((away_name, home_name), [])[-H2H_MEETINGS:]
    return {
        'H2HHomePPG': float(np.mean(home_pts)) if home_pts else np.nan,
        'H2HAwayPPG': float(np.mean(away_pts)) if away_pts else np.nan,
        'H2HCount': float(len(home_pts)),
    }

def build_features(df: pd.DataFrame):
    states: dict[str, TeamState] = {}
    h2h: dict[tuple, list] = {}
    rows = []
    for row in df.itertuples():
        home = states.setdefault(row.team1, TeamState())
        away = states.setdefault(row.team2, TeamState())
        feats = home.features('Home') | away.features('Away')
        feats['EloDifference'] = feats['HomeElo'] - feats['AwayElo']
        feats['FormDiff'] = feats['HomeEloPctChange'] - feats['AwayEloPctChange']
        feats['HomeForm5'] = home.form()
        feats['AwayForm5'] = away.form()
        feats['Form5Diff'] = feats['HomeForm5'] - feats['AwayForm5']
        feats |= h2h_features(h2h, row.team1, row.team2)
        feats['HomeMACD'] = home.macd()
        feats['AwayMACD'] = away.macd()
        feats['MACDDiff'] = feats['HomeMACD'] - feats['AwayMACD']
        feats['HomeMACDHist'] = home.macd_histogram()
        feats['AwayMACDHist'] = away.macd_histogram()
        feats['HomeTrendSlope'] = home.trend_slope()
        feats['AwayTrendSlope'] = away.trend_slope()
        feats['TrendSlopeDiff'] = feats['HomeTrendSlope'] - feats['AwayTrendSlope']
        feats['HomeRSI'] = home.rsi()
        feats['AwayRSI'] = away.rsi()
        feats['RSIDiff'] = feats['HomeRSI'] - feats['AwayRSI']
        for stat_key, label in MATCH_STATS.items():
            feats[f'Home{label}For'] = home.stat_roll(stat_key, against=False)
            feats[f'Away{label}For'] = away.stat_roll(stat_key, against=False)
            feats[f'Home{label}Against'] = home.stat_roll(stat_key, against=True)
            feats[f'Away{label}Against'] = away.stat_roll(stat_key, against=True)
            feats[f'{label}ForDiff'] = feats[f'Home{label}For'] - feats[f'Away{label}For']
            feats[f'{label}AgainstDiff'] = feats[f'Home{label}Against'] - feats[f'Away{label}Against']
        rows.append(feats)

        new_home, new_away = elo_update(home.elo, away.elo, row.score1, row.score2)
        home.push(new_home)
        away.push(new_away)
        home.dates.append(row.date)
        away.dates.append(row.date)
        for stat_key in MATCH_STATS:
            home_val = getattr(row, f'{stat_key}_home', np.nan)
            away_val = getattr(row, f'{stat_key}_away', np.nan)
            home.stats_for[stat_key].append(home_val)
            home.stats_against[stat_key].append(away_val)
            away.stats_for[stat_key].append(away_val)
            away.stats_against[stat_key].append(home_val)
        home_pts = 3 if row.score1 > row.score2 else 0 if row.score1 < row.score2 else 1
        home.points.append(home_pts)
        away.points.append(3 - home_pts if home_pts != 1 else 1)
        h2h.setdefault((row.team1, row.team2), []).append(home_pts)
        h2h.setdefault((row.team2, row.team1), []).append(
            3 - home_pts if home_pts != 1 else 1
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1), states, h2h

def export_state(states: dict, h2h: dict, feature_names: list[str], current_teams: list[str]) -> dict:
    team_list = list(states)
    n = len(team_list)
    h2h_ppg = np.full((n, n), np.nan)
    h2h_count = np.zeros((n, n))
    for i, x in enumerate(team_list):
        for j, y in enumerate(team_list):
            pts = h2h.get((x, y), [])[-H2H_MEETINGS:]
            if pts:
                h2h_ppg[i, j] = np.mean(pts)
                h2h_count[i, j] = len(pts)

    return dict(
        teams=np.array(team_list, dtype=object),
        current_teams=np.array(current_teams, dtype=object),
        feature_names=np.array(feature_names, dtype=object),
        elo=np.array([s.elo for s in states.values()]),
        pct=np.array([s.pct_change() for s in states.values()]),
        roll=np.array([[s.roll(w) for w in ROLL_WINDOWS] for s in states.values()]),
        ema=np.array([[s.ema[sp] for sp in EMA_SPANS] for s in states.values()]),
        form5=np.array([s.form() for s in states.values()]),
        h2h_ppg=h2h_ppg,
        h2h_count=h2h_count,
        macd=np.array([s.macd() for s in states.values()]),
        macd_hist=np.array([s.macd_histogram() for s in states.values()]),
        trend_slope=np.array([s.trend_slope() for s in states.values()]),
        rsi=np.array([s.rsi() for s in states.values()]),
        stats_for=np.array([[s.stat_roll(k, against=False) for k in MATCH_STATS] for s in states.values()]),
        stats_against=np.array([[s.stat_roll(k, against=True) for k in MATCH_STATS] for s in states.values()]),
        elo_history=np.array(
            [np.array(s.history[1:][-ELO_HISTORY_LENGTH:]) for s in states.values()], dtype=object,
        ),
        elo_dates=np.array(
            [np.array(s.dates[-ELO_HISTORY_LENGTH:], dtype='datetime64[ns]') for s in states.values()],
            dtype=object,
        ),
    )