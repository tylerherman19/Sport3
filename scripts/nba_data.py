"""
NBA Data Pipeline — Comprehensive feature engineering using nba_api.
Ingests game metadata, schedule/fatigue, travel, rolling performance metrics,
situational context, H2H history, and roster availability for every NBA game.

Outputs:
  data/nba_features.json  — engineered features per game
  data/nba_results.json   — game results only
  data/nba_fetch_log.txt  — timestamped error log

Known limitations:
  - top3_available defaults to 1.0 for upcoming games (no real-time injury API)
  - is_national_tv defaults to 0 (no free schedule API; placeholder)
  - nba_api rate-limits to ~1 req/sec; 0.6s sleep between calls is enforced
  - COVID bubble (Mar–Oct 2020) games flagged: travel/attendance features unreliable
"""

import os
import sys
import json
import math
import time
import logging
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
FETCH_LOG = DATA_DIR / "nba_fetch_log.txt"

# ─── Arena Lookup (hardcoded static data for all 30 NBA teams) ─────────────────
# lat/lon: arena coordinates  |  utc_offset: standard time (non-DST)
ARENA_INFO = {
    "ATL": {"name": "State Farm Arena",        "city": "Atlanta",          "lat": 33.7573,  "lon": -84.3963,  "capacity": 18118, "utc_offset": -5},
    "BOS": {"name": "TD Garden",               "city": "Boston",           "lat": 42.3662,  "lon": -71.0621,  "capacity": 19156, "utc_offset": -5},
    "BKN": {"name": "Barclays Center",         "city": "Brooklyn",         "lat": 40.6826,  "lon": -73.9754,  "capacity": 17732, "utc_offset": -5},
    "CHA": {"name": "Spectrum Center",         "city": "Charlotte",        "lat": 35.2251,  "lon": -80.8392,  "capacity": 19077, "utc_offset": -5},
    "CHI": {"name": "United Center",           "city": "Chicago",          "lat": 41.8807,  "lon": -87.6742,  "capacity": 20917, "utc_offset": -6},
    "CLE": {"name": "Rocket Mortgage FieldHouse","city": "Cleveland",      "lat": 41.4964,  "lon": -81.6882,  "capacity": 19432, "utc_offset": -5},
    "DAL": {"name": "American Airlines Center","city": "Dallas",           "lat": 32.7905,  "lon": -96.8103,  "capacity": 19200, "utc_offset": -6},
    "DEN": {"name": "Ball Arena",              "city": "Denver",           "lat": 39.7487,  "lon": -105.0077, "capacity": 19520, "utc_offset": -7},
    "DET": {"name": "Little Caesars Arena",    "city": "Detroit",          "lat": 42.3410,  "lon": -83.0554,  "capacity": 20332, "utc_offset": -5},
    "GSW": {"name": "Chase Center",            "city": "San Francisco",    "lat": 37.7680,  "lon": -122.3877, "capacity": 18064, "utc_offset": -8},
    "HOU": {"name": "Toyota Center",           "city": "Houston",          "lat": 29.7508,  "lon": -95.3621,  "capacity": 18055, "utc_offset": -6},
    "IND": {"name": "Gainbridge Fieldhouse",   "city": "Indianapolis",     "lat": 39.7640,  "lon": -86.1555,  "capacity": 17923, "utc_offset": -5},
    "LAC": {"name": "Intuit Dome",             "city": "Inglewood",        "lat": 33.9578,  "lon": -118.3417, "capacity": 18000, "utc_offset": -8},
    "LAL": {"name": "Crypto.com Arena",        "city": "Los Angeles",      "lat": 34.0430,  "lon": -118.2673, "capacity": 19068, "utc_offset": -8},
    "MEM": {"name": "FedExForum",              "city": "Memphis",          "lat": 35.1383,  "lon": -90.0505,  "capacity": 17794, "utc_offset": -6},
    "MIA": {"name": "Kaseya Center",           "city": "Miami",            "lat": 25.7814,  "lon": -80.1870,  "capacity": 19600, "utc_offset": -5},
    "MIL": {"name": "Fiserv Forum",            "city": "Milwaukee",        "lat": 43.0450,  "lon": -87.9170,  "capacity": 17341, "utc_offset": -6},
    "MIN": {"name": "Target Center",           "city": "Minneapolis",      "lat": 44.9795,  "lon": -93.2762,  "capacity": 18978, "utc_offset": -6},
    "NOP": {"name": "Smoothie King Center",    "city": "New Orleans",      "lat": 29.9490,  "lon": -90.0821,  "capacity": 17791, "utc_offset": -6},
    "NYK": {"name": "Madison Square Garden",   "city": "New York",         "lat": 40.7505,  "lon": -73.9934,  "capacity": 19812, "utc_offset": -5},
    "OKC": {"name": "Paycom Center",           "city": "Oklahoma City",    "lat": 35.4634,  "lon": -97.5151,  "capacity": 18203, "utc_offset": -6},
    "ORL": {"name": "Kia Center",              "city": "Orlando",          "lat": 28.5392,  "lon": -81.3839,  "capacity": 18846, "utc_offset": -5},
    "PHI": {"name": "Wells Fargo Center",      "city": "Philadelphia",     "lat": 39.9012,  "lon": -75.1720,  "capacity": 20478, "utc_offset": -5},
    "PHX": {"name": "Footprint Center",        "city": "Phoenix",          "lat": 33.4457,  "lon": -112.0712, "capacity": 17072, "utc_offset": -7},
    "POR": {"name": "Moda Center",             "city": "Portland",         "lat": 45.5316,  "lon": -122.6668, "capacity": 19393, "utc_offset": -8},
    "SAC": {"name": "Golden 1 Center",         "city": "Sacramento",       "lat": 38.5490,  "lon": -121.5002, "capacity": 17608, "utc_offset": -8},
    "SAS": {"name": "Frost Bank Center",       "city": "San Antonio",      "lat": 29.4270,  "lon": -98.4375,  "capacity": 18418, "utc_offset": -6},
    "TOR": {"name": "Scotiabank Arena",        "city": "Toronto",          "lat": 43.6435,  "lon": -79.3791,  "capacity": 19800, "utc_offset": -5},
    "UTA": {"name": "Delta Center",            "city": "Salt Lake City",   "lat": 40.7683,  "lon": -111.9011, "capacity": 18306, "utc_offset": -7},
    "WAS": {"name": "Capital One Arena",       "city": "Washington",       "lat": 38.8981,  "lon": -77.0209,  "capacity": 20356, "utc_offset": -5},
}

NBA_DIVISIONS = {
    "Atlantic":  ["BOS", "BKN", "NYK", "PHI", "TOR"],
    "Central":   ["CHI", "CLE", "DET", "IND", "MIL"],
    "Southeast": ["ATL", "CHA", "MIA", "ORL", "WAS"],
    "Northwest": ["DEN", "MIN", "OKC", "POR", "UTA"],
    "Pacific":   ["GSW", "LAC", "LAL", "PHX", "SAC"],
    "Southwest": ["DAL", "HOU", "MEM", "NOP", "SAS"],
}

NBA_CONFERENCES = {
    "East": ["BOS", "BKN", "NYK", "PHI", "TOR", "CHI", "CLE", "DET", "IND", "MIL",
             "ATL", "CHA", "MIA", "ORL", "WAS"],
    "West": ["DEN", "MIN", "OKC", "POR", "UTA", "GSW", "LAC", "LAL", "PHX", "SAC",
             "DAL", "HOU", "MEM", "NOP", "SAS"],
}

# Build reverse lookup: team → division and conference
TEAM_DIVISION = {}
TEAM_CONFERENCE = {}
for div, teams in NBA_DIVISIONS.items():
    for t in teams:
        TEAM_DIVISION[t] = div
for conf, teams in NBA_CONFERENCES.items():
    for t in teams:
        TEAM_CONFERENCE[t] = conf

# COVID bubble period — all games played at ESPN Wide World of Sports Complex, Orlando
BUBBLE_START = date(2020, 3, 11)
BUBBLE_END   = date(2020, 10, 12)

# NBA abbreviation normalization (ESPN sometimes uses short forms)
ESPN_TO_NBA = {
    "GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS",
    "WSH": "WAS", "UTAH": "UTA",
}

NBA_TEAMS = list(ARENA_INFO.keys())


# ─── Logging helper ───────────────────────────────────────────────────────────

def log_fetch_error(msg: str):
    ts = datetime.now().isoformat()
    with open(FETCH_LOG, "a") as f:
        f.write(f"[{ts}] {msg}\n")
    log.warning(msg)


# ─── Helper functions ─────────────────────────────────────────────────────────

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in miles using haversine formula."""
    R = 3959.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def nba_season_str(season_year: int) -> str:
    """Convert season year (e.g. 2025) → '2024-25' nba_api format."""
    return f"{season_year - 1}-{str(season_year)[2:]}"


def abbrev_norm(abbrev: str) -> str:
    a = abbrev.upper().strip()
    return ESPN_TO_NBA.get(a, a)


def is_bubble(game_date) -> int:
    """Return 1 if game was played during the COVID bubble (Mar–Oct 2020)."""
    if isinstance(game_date, str):
        try:
            game_date = date.fromisoformat(game_date[:10])
        except ValueError:
            return 0
    return 1 if BUBBLE_START <= game_date <= BUBBLE_END else 0


def is_division_rival(team_a: str, team_b: str) -> int:
    return 1 if TEAM_DIVISION.get(team_a) == TEAM_DIVISION.get(team_b) else 0


def is_interconference(team_a: str, team_b: str) -> int:
    return 1 if TEAM_CONFERENCE.get(team_a) != TEAM_CONFERENCE.get(team_b) else 0


def parse_min(min_val) -> float:
    """Parse MIN field which may be '240:00' (string) or 240.0 (float)."""
    if min_val is None:
        return 240.0
    if isinstance(min_val, (int, float)):
        return float(min_val)
    s = str(min_val)
    if ":" in s:
        parts = s.split(":")
        return float(parts[0]) + float(parts[1]) / 60.0
    try:
        return float(s)
    except ValueError:
        return 240.0


# ─── nba_api fetch functions ──────────────────────────────────────────────────

def _sleep():
    """Rate-limit nba_api calls to ~1 req/sec."""
    time.sleep(0.65)


def fetch_league_game_log(season_year: int, season_type: str = "Regular Season") -> pd.DataFrame:
    """Fetch all games for a season (one row per team-game)."""
    try:
        from nba_api.stats.endpoints import LeagueGameLog
        _sleep()
        gl = LeagueGameLog(
            season=nba_season_str(season_year),
            season_type_all_star=season_type,
            timeout=60,
        )
        df = gl.get_data_frames()[0]
        if df.empty:
            return pd.DataFrame()
        df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].apply(abbrev_norm)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df
    except Exception as e:
        log_fetch_error(f"LeagueGameLog {season_year} {season_type}: {e}")
        return pd.DataFrame()


def fetch_league_dash_stats(season_year: int, last_n_games: int = 0) -> pd.DataFrame:
    """
    Fetch team advanced stats (off_rating, def_rating, pace, ts_pct, etc.)
    last_n_games=0 → season average; last_n_games=5/10/20 → rolling window.
    """
    try:
        from nba_api.stats.endpoints import LeagueDashTeamStats
        _sleep()
        kwargs = dict(
            season=nba_season_str(season_year),
            per_mode_simple="Per100Possessions",
            measure_type_simple="Advanced",
            last_n_games=last_n_games,
            timeout=60,
        )
        ds = LeagueDashTeamStats(**kwargs)
        df = ds.get_data_frames()[0]
        if not df.empty:
            df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].apply(abbrev_norm)
        return df
    except Exception as e:
        log_fetch_error(f"LeagueDashTeamStats {season_year} last_n={last_n_games}: {e}")
        return pd.DataFrame()


def fetch_standings(season_year: int) -> pd.DataFrame:
    """Fetch current standings."""
    try:
        from nba_api.stats.endpoints import LeagueStandingsV3
        _sleep()
        st = LeagueStandingsV3(
            season=nba_season_str(season_year),
            season_type="Regular Season",
            timeout=60,
        )
        df = st.get_data_frames()[0]
        if not df.empty and "TeamAbbreviation" in df.columns:
            df["TeamAbbreviation"] = df["TeamAbbreviation"].apply(abbrev_norm)
        return df
    except Exception as e:
        log_fetch_error(f"LeagueStandingsV3 {season_year}: {e}")
        return pd.DataFrame()


def fetch_roster(team_id: int) -> pd.DataFrame:
    """Fetch current roster for a team."""
    try:
        from nba_api.stats.endpoints import CommonTeamRoster
        _sleep()
        r = CommonTeamRoster(team_id=team_id, timeout=60)
        return r.get_data_frames()[0]
    except Exception as e:
        log_fetch_error(f"CommonTeamRoster team_id={team_id}: {e}")
        return pd.DataFrame()


def fetch_box_score_summary(game_id: str) -> dict:
    """Fetch box score summary for attendance data (completed games)."""
    try:
        from nba_api.stats.endpoints import BoxScoreSummaryV2
        _sleep()
        bs = BoxScoreSummaryV2(game_id=game_id, timeout=60)
        dfs = bs.get_data_frames()
        # dfs[1] is GameSummary which has ATTENDANCE
        summary = dfs[1] if len(dfs) > 1 else pd.DataFrame()
        if summary.empty or "ATTENDANCE" not in summary.columns:
            return {}
        row = summary.iloc[0]
        return {"attendance": int(row.get("ATTENDANCE", 0) or 0)}
    except Exception as e:
        log_fetch_error(f"BoxScoreSummaryV2 game_id={game_id}: {e}")
        return {}


# ─── Game pairing & advanced stats computation ───────────────────────────────

def pair_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pair home and away team rows for each GAME_ID.
    Returns DataFrame with one row per game: home_* and away_* columns.
    """
    if games_df.empty:
        return pd.DataFrame()

    games_df = games_df.copy()
    games_df["IS_HOME"] = games_df["MATCHUP"].str.contains("vs\.")

    home_df = games_df[games_df["IS_HOME"]].copy()
    away_df = games_df[~games_df["IS_HOME"]].copy()

    home_df = home_df.add_prefix("home_")
    away_df = away_df.add_prefix("away_")

    merged = pd.merge(
        home_df.rename(columns={"home_GAME_ID": "GAME_ID"}),
        away_df.rename(columns={"away_GAME_ID": "GAME_ID"}),
        on="GAME_ID",
        how="inner",
    )
    return merged


def compute_possessions(fga, fta, oreb, tov) -> float:
    """Estimate possessions from box score components."""
    return float(fga) + 0.44 * float(fta) - float(oreb) + float(tov)


def compute_advanced(team_row, opp_row) -> dict:
    """
    Compute advanced per-game stats from paired box score rows.
    team_row / opp_row: Series with FGA, FTA, FGM, FG3A, FG3M, OREB, DREB, TOV, PTS, MIN.
    """
    def safe(x):
        try:
            return float(x) if x is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    pts     = safe(team_row.get("PTS", 0))
    opp_pts = safe(opp_row.get("PTS", 0))
    fga     = max(safe(team_row.get("FGA", 1)), 1.0)
    fta     = safe(team_row.get("FTA", 0))
    oreb    = safe(team_row.get("OREB", 0))
    tov     = safe(team_row.get("TOV", 0))
    opp_fga  = max(safe(opp_row.get("FGA", 1)), 1.0)
    opp_fta  = safe(opp_row.get("FTA", 0))
    opp_oreb = safe(opp_row.get("OREB", 0))
    opp_tov  = safe(opp_row.get("TOV", 0))
    opp_dreb = safe(opp_row.get("DREB", 0))
    min_played = max(parse_min(team_row.get("MIN", 240)), 1.0)

    team_poss = compute_possessions(fga, fta, oreb, tov)
    opp_poss  = compute_possessions(opp_fga, opp_fta, opp_oreb, opp_tov)
    avg_poss  = max((team_poss + opp_poss) / 2, 1.0)

    off_rtg = (pts / avg_poss) * 100.0
    def_rtg = (opp_pts / avg_poss) * 100.0
    pace    = (avg_poss / min_played) * 48.0
    ts_pct  = pts / (2.0 * (fga + 0.44 * fta)) if (fga + 0.44 * fta) > 0 else 0.5
    tov_rate = tov / (fga + 0.44 * fta + tov) if (fga + 0.44 * fta + tov) > 0 else 0.12
    denom_oreb = oreb + opp_dreb
    oreb_pct = oreb / denom_oreb if denom_oreb > 0 else 0.25
    ft_rate  = fta / fga if fga > 0 else 0.2
    win      = 1 if pts > opp_pts else 0

    return {
        "off_rating":  round(off_rtg,  2),
        "def_rating":  round(def_rtg,  2),
        "net_rating":  round(off_rtg - def_rtg, 2),
        "pace":        round(pace,     2),
        "ts_pct":      round(ts_pct,   4),
        "tov_rate":    round(tov_rate, 4),
        "oreb_pct":    round(oreb_pct, 4),
        "ft_rate":     round(ft_rate,  4),
        "point_diff":  int(pts - opp_pts),
        "win":         win,
    }


# ─── Schedule / fatigue features ─────────────────────────────────────────────

def compute_schedule_features(team_game_log: pd.DataFrame, team: str, game_date: date) -> dict:
    """
    Compute schedule/fatigue features for a team relative to game_date.
    team_game_log: DataFrame with columns GAME_DATE (datetime), TEAM_ABBREVIATION, MATCHUP.
    """
    prior = team_game_log[
        (team_game_log["TEAM_ABBREVIATION"] == team) &
        (team_game_log["GAME_DATE"].dt.date < game_date)
    ].sort_values("GAME_DATE")

    if prior.empty:
        return {
            "days_rest":         7,
            "is_back_to_back":   0,
            "games_last_7":      0,
            "games_last_14":     0,
            "schedule_density":  0.0,
            "home_stand_length": 0,
            "road_trip_length":  0,
        }

    last_date   = prior.iloc[-1]["GAME_DATE"].date()
    days_rest   = (game_date - last_date).days
    is_b2b      = 1 if days_rest == 1 else 0

    tgt         = pd.Timestamp(game_date)
    games_7     = int((prior["GAME_DATE"] >= tgt - pd.Timedelta(days=7)).sum())
    games_14    = int((prior["GAME_DATE"] >= tgt - pd.Timedelta(days=14)).sum())
    games_10    = int((prior["GAME_DATE"] >= tgt - pd.Timedelta(days=10)).sum())
    density     = round(games_10 / 10.0, 3)

    # Home stand / road trip length: count consecutive home (vs.) or away (@) games
    home_stand  = 0
    road_trip   = 0
    last_is_home = "vs." in str(prior.iloc[-1].get("MATCHUP", ""))
    for _, row in prior.iloc[::-1].iterrows():
        row_is_home = "vs." in str(row.get("MATCHUP", ""))
        if row_is_home == last_is_home:
            if last_is_home:
                home_stand += 1
            else:
                road_trip += 1
        else:
            break

    return {
        "days_rest":         max(days_rest, 0),
        "is_back_to_back":   is_b2b,
        "games_last_7":      games_7,
        "games_last_14":     games_14,
        "schedule_density":  density,
        "home_stand_length": home_stand,
        "road_trip_length":  road_trip,
    }


# ─── Travel features (away team) ─────────────────────────────────────────────

def _last_arena(prior_games: pd.DataFrame, team: str) -> str:
    """Determine the arena team played at in their last game."""
    if prior_games.empty:
        return team  # assume started at home
    last = prior_games.iloc[-1]
    matchup = str(last.get("MATCHUP", ""))
    if "@" in matchup:
        # Away game: "TEAM @ OPP" → arena is OPP's home
        parts = matchup.split("@")
        opp_abbrev = abbrev_norm(parts[1].strip()) if len(parts) > 1 else team
        return opp_abbrev
    else:
        # Home game
        return team


def compute_travel_features(
    team_game_log: pd.DataFrame,
    team: str,
    game_date: date,
    home_team: str,       # home team for THIS game (= where away is traveling to)
) -> dict:
    """Compute travel features for an away team arriving at home_team's arena."""
    prior = team_game_log[
        (team_game_log["TEAM_ABBREVIATION"] == team) &
        (team_game_log["GAME_DATE"].dt.date < game_date)
    ].sort_values("GAME_DATE")

    dest_info  = ARENA_INFO.get(home_team, {})
    prev_arena = _last_arena(prior, team)
    prev_info  = ARENA_INFO.get(prev_arena, {})

    travel_miles = 0.0
    tz_change    = 0
    if dest_info and prev_info:
        travel_miles = haversine_miles(
            prev_info["lat"], prev_info["lon"],
            dest_info["lat"], dest_info["lon"],
        )
        tz_change = dest_info["utc_offset"] - prev_info["utc_offset"]

    travel_direction = 1 if tz_change > 0 else (-1 if tz_change < 0 else 0)

    # Cumulative travel last 7 days
    tgt      = pd.Timestamp(game_date)
    recent   = prior[prior["GAME_DATE"] >= tgt - pd.Timedelta(days=7)]
    cum_travel = 0.0
    prev_loc = prev_arena
    for _, row in recent.iterrows():
        curr_loc = abbrev_norm(row.get("TEAM_ABBREVIATION", team))
        matchup  = str(row.get("MATCHUP", ""))
        if "@" in matchup:
            parts    = matchup.split("@")
            curr_loc = abbrev_norm(parts[1].strip()) if len(parts) > 1 else prev_loc
        else:
            curr_loc = team
        p = ARENA_INFO.get(prev_loc, {})
        c = ARENA_INFO.get(curr_loc, {})
        if p and c:
            cum_travel += haversine_miles(p["lat"], p["lon"], c["lat"], c["lon"])
        prev_loc = curr_loc

    return {
        "travel_miles":              round(travel_miles, 1),
        "timezone_change":           int(tz_change),
        "travel_direction":          travel_direction,
        "cumulative_travel_miles_7d": round(cum_travel, 1),
    }


# ─── Rolling performance metrics ─────────────────────────────────────────────

def compute_rolling_metrics(
    paired_df: pd.DataFrame,
    team: str,
    game_date: date,
    n: int,
) -> dict:
    """
    Compute rolling performance metrics over last N completed games for a team.
    paired_df: output of pair_games() with columns home_TEAM_ABBREVIATION,
               away_TEAM_ABBREVIATION, home_GAME_DATE, home_/away_ stat columns.
    """
    empty = {
        f"off_rating_l{n}":  None,
        f"def_rating_l{n}":  None,
        f"net_rating_l{n}":  None,
        f"pace_l{n}":        None,
        f"ts_pct_l{n}":      None,
        f"tov_rate_l{n}":    None,
        f"oreb_pct_l{n}":    None,
        f"ft_rate_l{n}":     None,
        f"point_diff_l{n}":  None,
        f"win_pct_l{n}":     None,
    }
    if paired_df.empty:
        return empty

    # Determine which rows involve this team (home or away)
    def _is_home_col(col):
        return f"home_{col}"
    def _is_away_col(col):
        return f"away_{col}"

    game_date_ts = pd.Timestamp(game_date)
    home_mask = paired_df.get("home_TEAM_ABBREVIATION", pd.Series(dtype=str)) == team
    away_mask = paired_df.get("away_TEAM_ABBREVIATION", pd.Series(dtype=str)) == team

    date_col = "home_GAME_DATE" if "home_GAME_DATE" in paired_df.columns else None
    if date_col is None:
        return empty

    team_rows_h = paired_df[home_mask & (paired_df[date_col] < game_date_ts)]
    team_rows_a = paired_df[away_mask & (paired_df[date_col] < game_date_ts)]

    adv_rows = []
    stat_cols = ["PTS", "FGA", "FTA", "OREB", "DREB", "TOV", "MIN"]

    for _, row in team_rows_h.iterrows():
        team_s = {c: row.get(f"home_{c}", 0) for c in stat_cols}
        opp_s  = {c: row.get(f"away_{c}", 0) for c in stat_cols}
        adv    = compute_advanced(team_s, opp_s)
        adv["date"] = row[date_col]
        adv_rows.append(adv)

    for _, row in team_rows_a.iterrows():
        team_s = {c: row.get(f"away_{c}", 0) for c in stat_cols}
        opp_s  = {c: row.get(f"home_{c}", 0) for c in stat_cols}
        adv    = compute_advanced(team_s, opp_s)
        adv["date"] = row[date_col]
        adv_rows.append(adv)

    if not adv_rows:
        return empty

    adv_df = pd.DataFrame(adv_rows).sort_values("date")
    window = adv_df.tail(n)

    def avg(col):
        vals = window[col].dropna()
        return round(float(vals.mean()), 4) if len(vals) > 0 else None

    return {
        f"off_rating_l{n}":  avg("off_rating"),
        f"def_rating_l{n}":  avg("def_rating"),
        f"net_rating_l{n}":  avg("net_rating"),
        f"pace_l{n}":        avg("pace"),
        f"ts_pct_l{n}":      avg("ts_pct"),
        f"tov_rate_l{n}":    avg("tov_rate"),
        f"oreb_pct_l{n}":    avg("oreb_pct"),
        f"ft_rate_l{n}":     avg("ft_rate"),
        f"point_diff_l{n}":  avg("point_diff"),
        f"win_pct_l{n}":     avg("win"),
    }


def rolling_delta(home_metrics: dict, away_metrics: dict, n: int) -> dict:
    """Compute home minus away delta for each rolling metric window n."""
    keys = ["off_rating", "def_rating", "net_rating", "pace",
            "ts_pct", "tov_rate", "oreb_pct", "ft_rate", "point_diff", "win_pct"]
    result = {}
    for k in keys:
        hk = f"{k}_l{n}"
        h = home_metrics.get(hk)
        a = away_metrics.get(hk)
        result[f"delta_{k}_l{n}"] = round(h - a, 4) if (h is not None and a is not None) else None
    return result


# ─── Head-to-head history ─────────────────────────────────────────────────────

def compute_h2h(home: str, away: str, all_games: list, season_year: int) -> dict:
    """
    Compute H2H features between two teams.
    all_games: list of dicts with keys team1(home), team2(away), score1, score2, season.
    """
    matchups = [g for g in all_games
                if (g["team1"] == home and g["team2"] == away) or
                   (g["team1"] == away and g["team2"] == home)]

    if not matchups:
        return {"h2h_win_pct_season": 0.5, "h2h_win_pct_3yr": 0.5, "h2h_avg_margin_5": 0.0}

    def home_win(g):
        if g["team1"] == home:
            return 1 if g["score1"] > g["score2"] else 0
        else:
            return 1 if g["score2"] > g["score1"] else 0

    def margin(g):
        if g["team1"] == home:
            return g["score1"] - g["score2"]
        else:
            return g["score2"] - g["score1"]

    season_games = [g for g in matchups if g.get("season") == season_year]
    h2h_season   = round(sum(home_win(g) for g in season_games) / len(season_games), 4) if season_games else 0.5

    recent_3yr   = [g for g in matchups if g.get("season", 0) >= season_year - 2]
    h2h_3yr      = round(sum(home_win(g) for g in recent_3yr) / len(recent_3yr), 4) if recent_3yr else 0.5

    last5        = sorted(matchups, key=lambda g: g.get("date", ""))[-5:]
    h2h_margin5  = round(sum(margin(g) for g in last5) / len(last5), 2) if last5 else 0.0

    return {
        "h2h_win_pct_season": h2h_season,
        "h2h_win_pct_3yr":    h2h_3yr,
        "h2h_avg_margin_5":   h2h_margin5,
    }


# ─── Roster availability ─────────────────────────────────────────────────────

def fetch_roster_availability(team_abbrev: str) -> dict:
    """
    Estimate roster availability using CommonTeamRoster.
    top3_available defaults to 1.0 (no real-time injury API).
    roster_depth = active players / 13.
    """
    try:
        from nba_api.stats.static import teams as nba_teams_static
        all_teams = nba_teams_static.get_teams()
        team_map  = {abbrev_norm(t["abbreviation"]): t["id"] for t in all_teams}
        team_id   = team_map.get(team_abbrev)
        if not team_id:
            return {"top3_available": 1.0, "roster_depth": 1.0}
        roster_df = fetch_roster(team_id)
        if roster_df.empty:
            return {"top3_available": 1.0, "roster_depth": 1.0}
        n_players     = min(len(roster_df), 15)
        roster_depth  = round(n_players / 13.0, 3)
        return {"top3_available": 1.0, "roster_depth": roster_depth}
    except Exception as e:
        log_fetch_error(f"Roster availability {team_abbrev}: {e}")
        return {"top3_available": 1.0, "roster_depth": 1.0}


# ─── Situational context ──────────────────────────────────────────────────────

def compute_game_importance(team: str, standings_df: pd.DataFrame, season_year: int) -> float:
    """
    game_importance = (1 / games_remaining) * (1 / max(1, games_back_from_8th_seed))
    Higher = more important game (team fighting for playoff spot late in season).
    """
    if standings_df.empty:
        return 0.0
    try:
        col_abbrev = "TeamAbbreviation" if "TeamAbbreviation" in standings_df.columns else None
        if col_abbrev is None:
            return 0.0
        row = standings_df[standings_df[col_abbrev] == team]
        if row.empty:
            return 0.0
        row = row.iloc[0]
        wins   = float(row.get("WINS", row.get("W", 0)) or 0)
        losses = float(row.get("LOSSES", row.get("L", 0)) or 0)
        gp     = wins + losses
        games_remaining = max(82 - gp, 1)

        # Games back from 8th seed in conference
        conf  = TEAM_CONFERENCE.get(team, "East")
        conf_teams = NBA_CONFERENCES.get(conf, [])
        conf_rows = standings_df[standings_df[col_abbrev].isin(conf_teams)].copy()

        win_pct_col = "WinPct" if "WinPct" in conf_rows.columns else None
        if win_pct_col:
            conf_rows = conf_rows.sort_values(win_pct_col, ascending=False)
        else:
            conf_rows = conf_rows.sort_values("W", ascending=False) if "W" in conf_rows.columns else conf_rows

        if len(conf_rows) >= 8:
            eighth = conf_rows.iloc[7]
            e_wins   = float(eighth.get("WINS", eighth.get("W", 0)) or 0)
            e_losses = float(eighth.get("LOSSES", eighth.get("L", 0)) or 0)
            team_pct = wins / max(gp, 1)
            e_pct    = e_wins / max(e_wins + e_losses, 1)
            # Convert to "games back" from 8th
            games_back = max(0, (e_wins - wins + losses - e_losses) / 2.0)
        else:
            games_back = 0.0

        return round((1.0 / games_remaining) * (1.0 / max(1.0, games_back)), 6)
    except Exception as e:
        log_fetch_error(f"compute_game_importance {team}: {e}")
        return 0.0


# ─── Season average attendance ────────────────────────────────────────────────

def compute_avg_attendance(team: str, game_logs_df: pd.DataFrame) -> float:
    """Approximate season average attendance as 85% of capacity (fallback)."""
    cap = ARENA_INFO.get(team, {}).get("capacity", 18000)
    return round(cap * 0.85)


# ─── Main feature builder ─────────────────────────────────────────────────────

def build_features(season_years: list) -> tuple:
    """
    Build feature set for all games in season_years.
    Returns (features_list, results_list).
    """
    log.info(f"Building NBA features for seasons {season_years}...")

    # Determine current season
    now          = datetime.now()
    current_year = now.year + (1 if now.month >= 10 else 0)
    main_season  = season_years[-1] if season_years else current_year

    # ── 1. Fetch game logs ──────────────────────────────────────────────────
    log.info("Fetching LeagueGameLog data...")
    all_dfs = []
    for yr in season_years:
        df = fetch_league_game_log(yr, "Regular Season")
        if not df.empty:
            df["SEASON_YEAR"] = yr
            all_dfs.append(df)
        df_po = fetch_league_game_log(yr, "Playoffs")
        if not df_po.empty:
            df_po["SEASON_YEAR"] = yr
            df_po["IS_PLAYOFF"]  = True
            all_dfs.append(df_po)

    if not all_dfs:
        log.error("No game log data fetched. Aborting feature build.")
        return [], []

    combined_log = pd.concat(all_dfs, ignore_index=True)
    combined_log["GAME_DATE"] = pd.to_datetime(combined_log["GAME_DATE"])
    combined_log.sort_values("GAME_DATE", inplace=True)

    # ── 2. Pair games for advanced stats ────────────────────────────────────
    log.info("Pairing game rows for advanced stat computation...")
    paired_df = pair_games(combined_log)

    # ── 3. Fetch standings (current season) ─────────────────────────────────
    log.info("Fetching standings...")
    standings_df = fetch_standings(main_season)

    # ── 4. Fetch rolling dash stats (for cross-check / supplement) ───────────
    log.info("Fetching LeagueDashTeamStats (season, L5, L10, L20)...")
    dash_season = fetch_league_dash_stats(main_season, last_n_games=0)
    dash_l5     = fetch_league_dash_stats(main_season, last_n_games=5)
    dash_l10    = fetch_league_dash_stats(main_season, last_n_games=10)
    dash_l20    = fetch_league_dash_stats(main_season, last_n_games=20)

    def dash_row(df, team):
        if df.empty or "TEAM_ABBREVIATION" not in df.columns:
            return {}
        r = df[df["TEAM_ABBREVIATION"] == team]
        return r.iloc[0].to_dict() if not r.empty else {}

    # ── 5. Build historical game list for H2H ────────────────────────────────
    all_game_results = []
    if not paired_df.empty:
        for _, row in paired_df.iterrows():
            all_game_results.append({
                "date":   str(row.get("home_GAME_DATE", ""))[:10],
                "season": int(row.get("home_SEASON_YEAR", main_season)),
                "team1":  row.get("home_TEAM_ABBREVIATION", ""),
                "team2":  row.get("away_TEAM_ABBREVIATION", ""),
                "score1": float(row.get("home_PTS", 0) or 0),
                "score2": float(row.get("away_PTS", 0) or 0),
            })

    # ── 6. Identify current-season games to featurize ────────────────────────
    current_log = combined_log[combined_log["SEASON_YEAR"] == main_season].copy()
    if current_log.empty:
        log.warning("No games found for current season.")
        return [], []

    current_paired = pair_games(current_log) if not current_log.empty else pd.DataFrame()

    # ── 7. Build features per game ───────────────────────────────────────────
    log.info(f"Featurizing {len(current_paired)} games...")
    features_list = []
    results_list  = []
    seen_ids      = set()

    roster_cache = {}

    for _, row in current_paired.iterrows():
        game_id  = str(row.get("GAME_ID", ""))
        if game_id in seen_ids:
            continue
        seen_ids.add(game_id)

        home = row.get("home_TEAM_ABBREVIATION", "")
        away = row.get("away_TEAM_ABBREVIATION", "")
        if not home or not away or home not in ARENA_INFO or away not in ARENA_INFO:
            continue

        game_date_ts = row.get("home_GAME_DATE", pd.NaT)
        if pd.isnull(game_date_ts):
            continue
        game_date_obj = game_date_ts.date()
        game_date_str = game_date_ts.strftime("%Y-%m-%d")
        season_year_g = int(row.get("home_SEASON_YEAR", main_season))
        is_playoff    = bool(row.get("home_IS_PLAYOFF", False))

        home_score = float(row.get("home_PTS", 0) or 0)
        away_score = float(row.get("away_PTS", 0) or 0)
        is_completed = home_score > 0 or away_score > 0

        # ── Game metadata ─────────────────────────────────────────────────
        arena       = ARENA_INFO.get(home, {})
        day_of_week = game_date_ts.day_name()
        month       = game_date_ts.month
        bubble      = is_bubble(game_date_obj)

        # ── Schedule / fatigue ───────────────────────────────────────────
        sched_home = compute_schedule_features(combined_log, home, game_date_obj)
        sched_away = compute_schedule_features(combined_log, away, game_date_obj)
        rest_diff  = sched_home["days_rest"] - sched_away["days_rest"]

        # ── Travel (away team) ───────────────────────────────────────────
        if not bubble:
            travel = compute_travel_features(combined_log, away, game_date_obj, home)
        else:
            travel = {
                "travel_miles": 0, "timezone_change": 0,
                "travel_direction": 0, "cumulative_travel_miles_7d": 0,
            }

        # ── Rolling metrics from paired DF ───────────────────────────────
        home_roll = {}
        away_roll = {}
        deltas    = {}
        for n in [5, 10, 20]:
            hm = compute_rolling_metrics(paired_df, home, game_date_obj, n)
            am = compute_rolling_metrics(paired_df, away, game_date_obj, n)
            home_roll.update(hm)
            away_roll.update(am)
            deltas.update(rolling_delta(hm, am, n))

        # Supplement with LeagueDashTeamStats for current season accuracy
        def _dash_adv(df_dash, team, suffix):
            r = dash_row(df_dash, team)
            if not r:
                return {}
            return {
                f"off_rtg_dash{suffix}": r.get("OFF_RATING"),
                f"def_rtg_dash{suffix}": r.get("DEF_RATING"),
                f"net_rtg_dash{suffix}": r.get("NET_RATING"),
                f"pace_dash{suffix}":    r.get("PACE"),
                f"ts_pct_dash{suffix}":  r.get("TS_PCT"),
            }

        # ── Situational context ──────────────────────────────────────────
        div_rival = is_division_rival(home, away)
        interconf = is_interconference(home, away)

        cap_home = ARENA_INFO.get(home, {}).get("capacity", 18000)
        if is_completed:
            # Try to get actual attendance from BoxScoreSummaryV2 (skipped here for speed;
            # use capacity-based fallback — can be enabled per-game if needed)
            att = compute_avg_attendance(home, combined_log)
        else:
            att = compute_avg_attendance(home, combined_log)
        attendance_pct = round(att / max(cap_home, 1), 4)

        game_importance_home = compute_game_importance(home, standings_df, season_year_g)
        game_importance_away = compute_game_importance(away, standings_df, season_year_g)

        # ── H2H features ─────────────────────────────────────────────────
        h2h = compute_h2h(home, away, all_game_results, season_year_g)

        # ── Roster availability ──────────────────────────────────────────
        if home not in roster_cache:
            roster_cache[home] = fetch_roster_availability(home)
        if away not in roster_cache:
            roster_cache[away] = fetch_roster_availability(away)
        roster_home = roster_cache[home]
        roster_away = roster_cache[away]

        # ── Assemble feature record ──────────────────────────────────────
        feature = {
            # Metadata
            "game_id":      game_id,
            "game_date":    game_date_str,
            "season":       season_year_g,
            "season_type":  "Playoffs" if is_playoff else "Regular Season",
            "home_team":    home,
            "away_team":    away,
            "home_score":   int(home_score) if is_completed else None,
            "away_score":   int(away_score) if is_completed else None,
            "is_completed": is_completed,
            # Arena
            "arena_name":     arena.get("name", ""),
            "arena_city":     arena.get("city", ""),
            "arena_capacity": arena.get("capacity", 0),
            # Time context
            "day_of_week": day_of_week,
            "month":       month,
            # Bubble flag
            "is_bubble": bubble,
            # Schedule / fatigue — home
            "home_days_rest":         sched_home["days_rest"],
            "home_is_back_to_back":   sched_home["is_back_to_back"],
            "home_games_last_7":      sched_home["games_last_7"],
            "home_games_last_14":     sched_home["games_last_14"],
            "home_schedule_density":  sched_home["schedule_density"],
            "home_stand_length":      sched_home["home_stand_length"],
            # Schedule / fatigue — away
            "away_days_rest":         sched_away["days_rest"],
            "away_is_back_to_back":   sched_away["is_back_to_back"],
            "away_games_last_7":      sched_away["games_last_7"],
            "away_games_last_14":     sched_away["games_last_14"],
            "away_schedule_density":  sched_away["schedule_density"],
            "away_road_trip_length":  sched_away["road_trip_length"],
            "rest_differential":      rest_diff,
            # Travel — away team
            "travel_miles":               travel["travel_miles"],
            "timezone_change":            travel["timezone_change"],
            "travel_direction":           travel["travel_direction"],
            "cumulative_travel_miles_7d": travel["cumulative_travel_miles_7d"],
            # Situational context
            "is_national_tv":   0,     # placeholder — no free API
            "is_division_rival":    div_rival,
            "is_interconference":   interconf,
            "attendance_pct":       attendance_pct,
            "game_importance_home": game_importance_home,
            "game_importance_away": game_importance_away,
            # H2H
            **h2h,
            # Roster
            "home_top3_available": roster_home["top3_available"],
            "home_roster_depth":   roster_home["roster_depth"],
            "away_top3_available": roster_away["top3_available"],
            "away_roster_depth":   roster_away["roster_depth"],
        }

        # Rolling metrics — home, away, deltas (L5/L10/L20)
        feature.update({f"home_{k}": v for k, v in home_roll.items()})
        feature.update({f"away_{k}": v for k, v in away_roll.items()})
        feature.update(deltas)

        # LeagueDash supplemental metrics
        for suffix, df_dash in [("_season", dash_season), ("_l5", dash_l5),
                                  ("_l10", dash_l10), ("_l20", dash_l20)]:
            feature.update({f"home_{k}": v for k, v in _dash_adv(df_dash, home, suffix).items()})
            feature.update({f"away_{k}": v for k, v in _dash_adv(df_dash, away, suffix).items()})

        features_list.append(feature)

        # Results record (completed games only)
        if is_completed:
            results_list.append({
                "game_id":    game_id,
                "game_date":  game_date_str,
                "season":     season_year_g,
                "home_team":  home,
                "away_team":  away,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "home_win":   1 if home_score > away_score else 0,
            })

    log.info(f"Built {len(features_list)} feature records ({len(results_list)} completed games)")
    return features_list, results_list


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    now = datetime.now()
    current_year = now.year + (1 if now.month >= 10 else 0)
    season_years = [current_year - 2, current_year - 1, current_year]

    log.info(f"nba_data.py starting — seasons {season_years}")

    try:
        features, results = build_features(season_years)
    except Exception as e:
        log_fetch_error(f"build_features failed: {e}")
        raise

    ts = now.isoformat()

    features_out = DATA_DIR / "nba_features.json"
    results_out  = DATA_DIR / "nba_results.json"

    features_out.write_text(json.dumps({
        "updated":  ts,
        "seasons":  season_years,
        "count":    len(features),
        "features": features,
    }, indent=2, default=str))

    results_out.write_text(json.dumps({
        "updated": ts,
        "count":   len(results),
        "results": results,
    }, indent=2, default=str))

    log.info(f"Wrote {len(features)} records to {features_out}")
    log.info(f"Wrote {len(results)} results to {results_out}")


if __name__ == "__main__":
    main()
