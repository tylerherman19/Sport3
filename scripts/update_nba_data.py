"""
NBA Prediction Model — Full pipeline.
Downloads NBA data, trains models, writes JSON output files.
Run daily via GitHub Actions alongside update_data.py.
"""

import os
import sys
import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from math import log, exp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.bayesian_model import update_ratings, predict_game as bayes_predict, get_all_ratings
from model.monte_carlo import simulate_game, simulate_season
from model.ensemble_model import kelly_criterion, american_to_prob, remove_vig
from model.injury_model import compute_all_nba_team_impacts
from model.nba_elo import (
    compute_nba_elo, predict_nba_game,
    nba_recent_form, nba_get_trend, nba_expected_score,
)
try:
    from model.nba_elo_model import save_ratings as save_nba_elo_ratings
    HAS_NBA_ELO_MODEL = True
except ImportError:
    HAS_NBA_ELO_MODEL = False
    save_nba_elo_ratings = None

from scripts.output_writer import write_nba_abort_log
from scripts.data_fetcher import fetch_nba_player_stats_cdn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

ESPN_NBA_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_NBA_WEB_BASE = "https://site.web.api.espn.com/apis/v2/sports/basketball/nba"
ESPN_NBA_STANDINGS_URL = "https://site.web.api.espn.com/apis/v2/sports/basketball/nba/standings"
ODDS_BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"

NBA_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

NBA_TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

# ESPN abbreviation normalization for NBA
ESPN_NBA_TO_ABBREV = {
    "GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS",
    "OKC": "OKC", "BKN": "BKN", "WSH": "WAS", "CHA": "CHA",
    "PHX": "PHX", "LAL": "LAL", "LAC": "LAC",
}

# ESPN team ID → abbrev mapping (populated dynamically)
ESPN_ID_TO_ABBREV = {}

# Home city coordinates (lat, lon) and standard-time UTC offset for travel/timezone features.
# UTC offsets: ET=-5, CT=-6, MT=-7, PT=-8; Toronto uses ET (-5).
NBA_COORDS = {
    # (lat, lon, utc_offset_standard_time)
    "ATL": (33.7490, -84.3880, -5), "BOS": (42.3601, -71.0589, -5),
    "BKN": (40.6826, -73.9754, -5), "CHA": (35.2271, -80.8431, -5),
    "CHI": (41.8781, -87.6298, -6), "CLE": (41.4993, -81.6944, -5),
    "DAL": (32.7767, -96.7970, -6), "DEN": (39.7392, -104.9903, -7),
    "DET": (42.3314, -83.0458, -5), "GSW": (37.7680, -122.3877, -8),
    "HOU": (29.7604, -95.3698, -6), "IND": (39.7684, -86.1581, -5),
    "LAC": (34.0430, -118.2673, -8), "LAL": (34.0430, -118.2673, -8),
    "MEM": (35.1495, -90.0490, -6), "MIA": (25.7617, -80.1918, -5),
    "MIL": (43.0436, -87.9166, -6), "MIN": (44.9778, -93.2650, -6),
    "NOP": (29.9511, -90.0715, -6), "NYK": (40.7505, -73.9934, -5),
    "OKC": (35.4634, -97.5151, -6), "ORL": (28.5383, -81.3792, -5),
    "PHI": (39.9012, -75.1720, -5), "PHX": (33.4484, -112.0740, -7),
    "POR": (45.5231, -122.6765, -8), "SAC": (38.5816, -121.4944, -8),
    "SAS": (29.4241, -98.4936, -6), "TOR": (43.6532, -79.3832, -5),
    "UTA": (40.7608, -111.8910, -7), "WAS": (38.9072, -77.0369, -5),
}


# Player tier lookup for injury impact weighting (2025-26 season)
# Keys match ESPN displayName. Tiers: "superstar", "all-star", "starter", "rotation"
# injury_model.py maps: superstar→50 ELO, all-star→30, starter→15, rotation→8
NBA_PLAYER_TIERS = {
    # Superstars
    "LeBron James":              "superstar",
    "Stephen Curry":             "superstar",
    "Kevin Durant":              "superstar",
    "Giannis Antetokounmpo":     "superstar",
    "Nikola Jokic":              "superstar",
    "Luka Doncic":               "superstar",
    "Joel Embiid":               "superstar",
    "Jayson Tatum":              "superstar",
    "Shai Gilgeous-Alexander":   "superstar",
    "Anthony Edwards":           "superstar",
    "Victor Wembanyama":         "superstar",
    # All-Stars
    "Damian Lillard":            "all-star",
    "Devin Booker":              "all-star",
    "Anthony Davis":             "all-star",
    "Bam Adebayo":               "all-star",
    "Tyrese Haliburton":         "all-star",
    "Donovan Mitchell":          "all-star",
    "Jalen Brunson":             "all-star",
    "De'Aaron Fox":              "all-star",
    "Trae Young":                "all-star",
    "Darius Garland":            "all-star",
    "Alperen Sengun":            "all-star",
    "Evan Mobley":               "all-star",
    "Karl-Anthony Towns":        "all-star",
    "Cade Cunningham":           "all-star",
    "Zach LaVine":               "all-star",
    "Kyrie Irving":              "all-star",
    "Jimmy Butler":              "all-star",
    "Pascal Siakam":             "all-star",
    "Scottie Barnes":            "all-star",
    "Jaren Jackson Jr.":         "all-star",
    "Jaylen Brown":              "all-star",
    "Paolo Banchero":            "all-star",
    "Franz Wagner":              "all-star",
    "Domantas Sabonis":          "all-star",
    "James Harden":              "all-star",
    "Paul George":               "all-star",
    "Kawhi Leonard":             "all-star",
}

# Numeric value multipliers for each tier (used as player_values for injury_model)
_NBA_TIER_VALUE = {
    "superstar": 2.0,
    "all-star":  1.5,
    "starter":   1.0,
    "rotation":  0.5,
}


def fetch_nba_depth_charts():
    """
    Fetch live NBA depth chart positions from ESPN for all teams.
    Returns {player_name: depth_position_int} where 1 = starter.
    Mirrors the NFL fetch_espn_depth_charts() approach.
    """
    log.info("Fetching ESPN NBA depth charts for player value scoring...")
    player_depth = {}
    try:
        teams_data = safe_get(f"{ESPN_NBA_BASE}/teams")
        if not teams_data:
            log.warning("Could not fetch ESPN NBA teams list for depth charts")
            return {}
        teams_list = (
            teams_data.get("sports", [{}])[0]
            .get("leagues", [{}])[0]
            .get("teams", [])
        )
        for team_entry in teams_list:
            team_id = team_entry.get("team", {}).get("id", "")
            if not team_id:
                continue
            depth_data = safe_get(f"{ESPN_NBA_BASE}/teams/{team_id}/depthcharts")
            if not depth_data:
                continue
            for pos_group in depth_data.get("positionGroups", []):
                for position in pos_group.get("positions", []):
                    for athlete_entry in position.get("athletes", []):
                        name = athlete_entry.get("athlete", {}).get("displayName", "")
                        depth_pos = int(athlete_entry.get("rank") or athlete_entry.get("slot", 99))
                        if not name:
                            continue
                        if name not in player_depth or depth_pos < player_depth[name]:
                            player_depth[name] = depth_pos
    except Exception as e:
        log.warning(f"Error fetching NBA depth charts: {e}")
    log.info(f"NBA depth chart loaded: {len(player_depth)} player entries")
    return player_depth


# Depth position → NBA value multiplier
_NBA_DEPTH_VALUE_MAP = {1: 2.0, 2: 1.2, 3: 0.7}


def _nba_depth_to_value(depth_pos: int) -> float:
    return _NBA_DEPTH_VALUE_MAP.get(depth_pos, 0.4)


def fetch_nba_player_ppg():
    """
    Fetch per-player season PPG (points per game) from ESPN for all NBA teams.
    Uses the ESPN athletes/statistics endpoint available in the existing fetch loops.
    Returns {player_display_name: ppg_float}.

    PPG is the most directly available per-player value signal in the ESPN API
    and avoids the need for a hardcoded tier list.
    """
    log.info("Fetching NBA player PPG stats for usage-based value scoring...")
    player_ppg = {}
    try:
        teams_data = safe_get(f"{ESPN_NBA_BASE}/teams")
        if not teams_data:
            log.warning("Could not fetch ESPN NBA teams list for PPG stats")
            return {}
        teams_list = (
            teams_data.get("sports", [{}])[0]
            .get("leagues", [{}])[0]
            .get("teams", [])
        )
        for team_entry in teams_list:
            team_id = team_entry.get("team", {}).get("id", "")
            if not team_id:
                continue
            # ESPN athletes endpoint with season statistics
            roster_data = safe_get(
                f"{ESPN_NBA_BASE}/teams/{team_id}/athletes",
                params={"enable": "stats"}
            )
            if not roster_data:
                continue
            for athlete in roster_data.get("athletes", []):
                name = athlete.get("displayName", "")
                if not name:
                    continue
                # Prefer season stats if embedded
                stats_list = athlete.get("statistics", {}).get("splits", {})
                if not stats_list:
                    stats_list = athlete.get("stats", [])
                # Look for points-per-game in any available stats block
                ppg = None
                if isinstance(stats_list, list):
                    for stat in stats_list:
                        label = str(stat.get("name", "") or stat.get("shortDisplayName", "")).lower()
                        if label in ("pts", "ppg", "points", "avg points"):
                            try:
                                ppg = float(stat.get("displayValue") or stat.get("value") or 0)
                            except (TypeError, ValueError):
                                pass
                            break
                if ppg is not None and ppg > 0:
                    if name not in player_ppg or ppg > player_ppg[name]:
                        player_ppg[name] = ppg
    except Exception as e:
        log.warning(f"Error fetching NBA player PPG stats: {e}")
    log.info(f"NBA PPG stats loaded: {len(player_ppg)} player entries")
    return player_ppg


# League-average PPG threshold for value normalisation
_NBA_PPG_STAR_THRESHOLD = 24.0    # PPG above which a player is considered superstar-tier
_NBA_PPG_ALLSTAR_THRESHOLD = 18.0  # all-star tier
_NBA_PPG_STARTER_THRESHOLD = 10.0  # starter tier
_NBA_PPG_ROTATION_THRESHOLD = 5.0  # rotation tier
_NBA_PPG_LEAGUE_AVG = 11.0         # approximate NBA per-player scoring average


def _ppg_to_value_mult(ppg: float) -> float:
    """
    Convert a player's PPG into a continuous value multiplier.
    Normalised so that league-average PPG ≈ 1.0 and a top scorer ≈ 2.2.
    The formula: value = (ppg / league_avg) * 1.0, capped at 2.5.
    This keeps the multiplier on the same scale as the static tier system
    (superstar=2.0, all-star=1.5, starter=1.0, rotation=0.5).
    """
    if ppg <= 0:
        return 0.3  # DNP / zero minutes
    raw = ppg / _NBA_PPG_LEAGUE_AVG
    return min(raw, 2.5)


def build_nba_player_values(season_year=None):
    """
    Build NBA player value multipliers using a priority stack:
      1. PPG-based continuous value — CDN primary, ESPN fallback
      2. Live ESPN depth chart position
      3. Static NBA_PLAYER_TIERS curated list (fallback)

    Returns {player_name: value_multiplier}.
    """
    # ── Tier 3 (lowest priority): static curated tiers as baseline ──────────
    values = {
        name: _NBA_TIER_VALUE.get(tier, 1.0)
        for name, tier in NBA_PLAYER_TIERS.items()
    }

    # ── Tier 2: live depth chart positions ───────────────────────────────────
    live_depth = fetch_nba_depth_charts()
    for name, depth_pos in live_depth.items():
        live_val = _nba_depth_to_value(depth_pos)
        # Override static tier if confirmed starter, or if player not in static list
        if name not in values or depth_pos == 1:
            values[name] = live_val

    # ── Tier 1 (highest priority): PPG-based continuous value ────────────────
    # Use CDN as primary source; fall back to ESPN athletes path
    player_ppg = {}
    if season_year is not None:
        try:
            player_ppg = fetch_nba_player_stats_cdn(season_year)
        except Exception as e:
            log.warning(f"CDN player stats failed: {e} — falling back to ESPN")
    if not player_ppg:
        player_ppg = fetch_nba_player_ppg()

    for name, ppg in player_ppg.items():
        ppg_val = _ppg_to_value_mult(ppg)
        # Use max so static tiers act as a floor — known stars are never downgraded
        # due to low current-season PPG from injury absences, trades, or API gaps
        values[name] = max(values.get(name, 0.0), ppg_val)

    ppg_count = len(player_ppg)
    if ppg_count > 0:
        log.info(
            f"NBA player values: {ppg_count} players valued by PPG, "
            f"{len(live_depth)} by depth chart, {len(NBA_PLAYER_TIERS)} by static tier"
        )
    else:
        log.warning(
            "NBA PPG fetch returned 0 players — falling back to depth chart + static tiers"
        )

    return values


def nba_value_tier_label(mult: float) -> str:
    """Convert NBA player value multiplier to human-readable tier label."""
    if mult >= 1.8:
        return "superstar"
    if mult >= 1.3:
        return "all-star"
    if mult >= 0.8:
        return "starter"
    if mult >= 0.4:
        return "backup"
    return "rotation"


def abbrev_norm(abbrev):
    a = abbrev.upper().strip()
    return ESPN_NBA_TO_ABBREV.get(a, a)


def safe_get(url, params=None, timeout=30, headers=None):
    try:
        r = requests.get(url, params=params, timeout=timeout, headers=headers or {})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return None


def haversine(a, b):
    R = 3959.0
    dLat = (b[0] - a[0]) * 3.14159265 / 180
    dLon = (b[1] - a[1]) * 3.14159265 / 180
    x = (dLat / 2) ** 2 + (dLon / 2) ** 2  # simplified
    import math
    x = (math.sin(dLat / 2)) ** 2 + math.cos(a[0] * math.pi / 180) * math.cos(b[0] * math.pi / 180) * (math.sin(dLon / 2)) ** 2
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))


def nba_travel_distance(home, away):
    """Return (travel_miles, timezone_diff) for the away team traveling to the home arena.

    timezone_diff = home_tz_offset - away_tz_offset (positive means away team
    travels east and loses hours, negative means they travel west and gain hours).
    A larger absolute value indicates greater circadian disruption.
    """
    ch = NBA_COORDS.get(home)
    ca = NBA_COORDS.get(away)
    if ch and ca:
        miles = haversine((ca[0], ca[1]), (ch[0], ch[1]))
        tz_diff = ch[2] - ca[2]  # home_tz - away_tz
        return miles, tz_diff
    return 0.0, 0


# ELO functions are provided by model/nba_elo.py (imported above)


# ─── Data Fetching ────────────────────────────────────────────────────────────

def parse_nba_events(data):
    """Parse ESPN NBA events list into game dicts."""
    games = []
    for event in data.get("events", []):
        try:
            comp = event["competitions"][0]
            competitors = comp["competitors"]
            home = next((c for c in competitors if c["homeAway"] == "home"), None)
            away = next((c for c in competitors if c["homeAway"] == "away"), None)
            if not home or not away:
                continue

            home_abbrev = abbrev_norm(home["team"]["abbreviation"])
            away_abbrev = abbrev_norm(away["team"]["abbreviation"])

            home_id = home["team"].get("id", "")
            away_id = away["team"].get("id", "")
            if home_id:
                ESPN_ID_TO_ABBREV[home_id] = home_abbrev
            if away_id:
                ESPN_ID_TO_ABBREV[away_id] = away_abbrev

            status_obj = event.get("status", {})
            game = {
                "game_id": event["id"],
                "game_time": event.get("date", ""),
                "status": status_obj.get("type", {}).get("name", ""),
                "period": status_obj.get("period", 0),
                "display_clock": status_obj.get("displayClock", ""),
                "status_detail": status_obj.get("type", {}).get("detail", ""),
                "home_team": home_abbrev,
                "away_team": away_abbrev,
                "home_name": home["team"].get("displayName", home_abbrev),
                "away_name": away["team"].get("displayName", away_abbrev),
                "home_score": int(home.get("score", 0) or 0),
                "away_score": int(away.get("score", 0) or 0),
                "home_logo": f"https://a.espncdn.com/i/teamlogos/nba/500/{home['team']['abbreviation'].lower()}.png",
                "away_logo": f"https://a.espncdn.com/i/teamlogos/nba/500/{away['team']['abbreviation'].lower()}.png",
                "neutral": int(comp.get("neutralSite", False)),
            }
            games.append(game)
        except (KeyError, IndexError) as e:
            log.debug(f"Error parsing NBA game: {e}")
    return games


def fetch_nba_scoreboard():
    """Fetch NBA scoreboard for today + past 5 days to catch recently-completed games."""
    log.info("Fetching NBA scoreboard (today + past 5 days)...")
    games = []
    seen_ids = set()
    for delta in [0, -1, -2, -3, -4, -5]:
        date_str = (datetime.now(timezone.utc) + timedelta(days=delta)).strftime('%Y%m%d')
        # Try web API first, fall back to standard API if no events returned
        data = safe_get(f"{ESPN_NBA_WEB_BASE}/scoreboard?dates={date_str}")
        if not data or not data.get("events"):
            data = safe_get(f"{ESPN_NBA_BASE}/scoreboard?dates={date_str}")
        if not data:
            continue
        for g in parse_nba_events(data):
            if g["game_id"] not in seen_ids:
                seen_ids.add(g["game_id"])
                games.append(g)
    log.info(f"Found {len(games)} NBA scoreboard games")
    return games


def fetch_nba_future_games(days_ahead=7):
    """Fetch NBA games scheduled in the next N days."""
    future_games = []
    seen_ids = set()
    from datetime import timedelta
    for offset in range(1, days_ahead + 1):
        date_str = (datetime.now(timezone.utc) + timedelta(days=offset)).strftime('%Y%m%d')
        # Try web API first, fall back to standard API if no events returned
        data = safe_get(f"{ESPN_NBA_WEB_BASE}/scoreboard?dates={date_str}&limit=20&seasontype=2")
        if not data or not data.get("events"):
            data = safe_get(f"{ESPN_NBA_BASE}/scoreboard?dates={date_str}&limit=20&seasontype=2")
        if not data:
            continue
        games = parse_nba_events(data)
        for g in games:
            if g["game_id"] not in seen_ids:
                seen_ids.add(g["game_id"])
                g["is_future"] = True
                future_games.append(g)
    log.info(f"Found {len(future_games)} future NBA games")
    return future_games


def fetch_nba_standings():
    """Fetch NBA standings. Tries cdn.nba.com first (real ORTG/DRTG), falls back to ESPN."""
    CDN_URL = "https://cdn.nba.com/static/json/staticData/standings.json"
    CDN_HEADERS = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/",
        "User-Agent": "Mozilla/5.0 (compatible; Sport3Bot/2.0)",
    }

    # ── Primary: cdn.nba.com (provides real pace-adjusted ORTG/DRTG) ────────
    log.info("Fetching NBA standings (cdn.nba.com)...")
    cdn_data = safe_get(CDN_URL, headers=CDN_HEADERS)
    if cdn_data:
        standings = {}
        try:
            # CDN schema may be nested {"standings": {"teams": [...]}} or flat {"standings": [...]}
            raw = cdn_data.get("standings", {})
            if isinstance(raw, dict):
                rows = raw.get("teams", raw.get("rows", raw.get("TeamStandings", [])))
            elif isinstance(raw, list):
                rows = raw
            else:
                rows = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                team_abbrev = abbrev_norm(row.get("teamAbbreviation", row.get("TeamAbbreviation", "")))
                if not team_abbrev:
                    continue
                wins = int(row.get("wins", row.get("WINS", 0)))
                losses = int(row.get("losses", row.get("LOSSES", 0)))
                # Try multiple key name conventions for ORTG/DRTG
                off_rtg = float(row.get("offensiveRating",
                                row.get("OffRating",
                                row.get("offRating",
                                row.get("ortg",
                                row.get("OffensiveRating", 110.0))))))
                def_rtg = float(row.get("defensiveRating",
                                row.get("DefRating",
                                row.get("defRating",
                                row.get("drtg",
                                row.get("DefensiveRating", 110.0))))))
                standings[team_abbrev] = {
                    "wins": wins,
                    "losses": losses,
                    "win_pct": wins / max(wins + losses, 1),
                    "points_for": 0.0,   # CDN static doesn't include totals;
                    "points_against": 0.0,  # PPG overridden from game history later
                    "games_played": wins + losses,
                    "offensive_rating": off_rtg,
                    "defensive_rating": def_rtg,
                    "net_rating": off_rtg - def_rtg,
                    "pace": float(row.get("pace", row.get("Pace", 100.0))),
                    "assist_turnover_ratio": 1.8,
                    "rebound_rate": 0.5,
                    "three_point_rate": 0.35,
                    "free_throw_rate": 0.20,
                    "streak": 0,
                }
            if standings:
                log.info(f"NBA standings (cdn) for {len(standings)} teams")
                return standings
            raise ValueError("Empty standings from CDN")
        except Exception as e:
            log.warning(f"cdn.nba.com standings parse error: {e} — falling back to ESPN")

    # ── Fallback: ESPN web API ────────────────────────────────────────────────
    log.info("Fetching NBA standings (ESPN fallback)...")
    data = safe_get(ESPN_NBA_STANDINGS_URL)
    if not data:
        return {}

    standings = {}
    try:
        for group in data.get("children", []):
            entries = group.get("standings", {}).get("entries", [])
            if not entries:
                for div in group.get("children", []):
                    entries = entries + div.get("standings", {}).get("entries", [])

            for entry in entries:
                team_abbrev = abbrev_norm(entry["team"]["abbreviation"])
                team_id = entry["team"].get("id", "")
                if team_id:
                    ESPN_ID_TO_ABBREV[team_id] = team_abbrev

                stats = {s["name"]: s.get("value", 0) for s in entry.get("stats", [])}
                wins = int(stats.get("wins", 0))
                losses = int(stats.get("losses", 0))
                points_for = float(stats.get("pointsFor", 0))
                points_against = float(stats.get("pointsAgainst", 0))
                # ESPN provides avgPointsFor/avgPointsAgainst (raw PPG).
                # Use these as offensive_rating/defensive_rating proxies;
                # build_nba_efficiency_data() will normalise them to the ORTG scale.
                ppg_for = float(stats.get("avgPointsFor", 0.0))
                ppg_against = float(stats.get("avgPointsAgainst", 0.0))

                standings[team_abbrev] = {
                    "wins": wins,
                    "losses": losses,
                    "win_pct": float(stats.get("winPercent", 0)),
                    "points_for": points_for,
                    "points_against": points_against,
                    "ppg_for": ppg_for,
                    "ppg_against": ppg_against,
                    "games_played": wins + losses,
                    # Use PPG as ORTG/DRTG proxy (normalised to league average in build_nba_efficiency_data)
                    "offensive_rating": ppg_for if ppg_for > 0 else 110.0,
                    "defensive_rating": ppg_against if ppg_against > 0 else 110.0,
                    "net_rating": round(ppg_for - ppg_against, 1) if ppg_for > 0 else 0.0,
                    "pace": 100.0,
                    "assist_turnover_ratio": 1.8,
                    "rebound_rate": 0.5,
                    "three_point_rate": 0.35,
                    "free_throw_rate": 0.20,
                    "streak": stats.get("streak", 0),
                    "ortg_estimated": True,  # flag that ORTG is PPG-based, not pace-adjusted
                }
    except Exception as e:
        log.warning(f"Error parsing NBA standings (ESPN): {e}")

    log.info(f"NBA standings (ESPN) for {len(standings)} teams")
    return standings


def fetch_nba_injuries():
    log.info("Fetching NBA injuries...")
    data = safe_get(f"{ESPN_NBA_BASE}/injuries")
    if not data:
        log.warning("NBA injuries endpoint returned no data")
        return {}

    injuries = {}
    try:
        # ESPN response: {"injuries": [{"displayName": "Atlanta Hawks", "injuries": [...]}]}
        # Note: ESPN removed top-level "abbreviation" from team entries; it now lives at
        # item.athlete.team.abbreviation only.
        for team_entry in data.get("injuries", []):
            # Primary: use top-level abbreviation on the team entry
            raw_abbrev = (
                team_entry.get("abbreviation", "")
                or team_entry.get("team", {}).get("abbreviation", "")
            )
            team_abbrev = abbrev_norm(raw_abbrev) if raw_abbrev else ""

            for item in team_entry.get("injuries", []):
                # Fallback: derive abbreviation from athlete's team when outer entry lacks it
                item_abbrev = team_abbrev
                if not item_abbrev:
                    raw_fallback = item.get("athlete", {}).get("team", {}).get("abbreviation", "")
                    item_abbrev = abbrev_norm(raw_fallback) if raw_fallback else ""

                if not item_abbrev:
                    continue

                if item_abbrev not in injuries:
                    injuries[item_abbrev] = []

                athlete     = item.get("athlete", {})
                player_name = athlete.get("displayName", "")
                position    = athlete.get("position", {}).get("abbreviation", "")
                status_raw  = item.get("status", "")
                if isinstance(status_raw, dict):
                    status = status_raw.get("name", status_raw.get("abbreviation", ""))
                else:
                    status = str(status_raw) if status_raw else ""

                injuries[item_abbrev].append({
                    "player":             player_name,
                    "status":             status,
                    "position":           position,
                    "tier":               NBA_PLAYER_TIERS.get(player_name, ""),
                    "injury_description": item.get("longComment", item.get("shortComment", status)),
                })

    except Exception as e:
        log.warning(f"Error parsing NBA injuries: {e}")

    total = sum(len(v) for v in injuries.values())
    log.info(f"NBA injuries: {total} players across {len(injuries)} teams")
    return injuries


def fetch_nba_season_games(seasons=None):
    """
    Fetch completed NBA games by iterating day-by-day via ESPN scoreboard.
    Uses dates=YYYYMMDD parameter (same approach as fetch_nba_future_games).
    For older seasons uses weekly strides; for current season fetches every day.
    """
    from datetime import date as date_cls
    if seasons is None:
        current_year = datetime.now(timezone.utc).year
        current_month = datetime.now(timezone.utc).month
        if current_month >= 10:
            end_year = current_year + 1
        else:
            end_year = current_year
        seasons = [end_year - 2, end_year - 1, end_year]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_nba_day(date_str):
        url = f"{ESPN_NBA_BASE}/scoreboard?dates={date_str}&limit=20"
        return date_str, safe_get(url)

    log.info(f"Fetching NBA season games for {seasons}...")
    all_games = []
    seen_ids = set()
    today = date_cls.today()
    fetch_failures = 0

    for season_year in seasons:
        # NBA season runs Oct 1 (year-1) through Jun 30 (year)
        season_start = date_cls(season_year - 1, 10, 1)
        season_end = min(today, date_cls(season_year, 6, 30))
        if season_start > today:
            continue

        # Always use 1-day stride for all seasons to ensure full game coverage.
        # A 7-day stride skips ~85% of games, starving ELO/logistic training
        # and causing all sub-models to fall back to 50% neutral defaults.
        stride_days = 1

        # Build list of all dates to fetch for this season
        date_list = []
        current = season_start
        while current <= season_end:
            date_list.append(current.strftime('%Y%m%d'))
            current += timedelta(days=stride_days)

        # Fetch concurrently (max 10 workers to avoid rate limiting)
        results_by_date = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(_fetch_nba_day, d): d for d in date_list}
            for future in as_completed(futures):
                date_str, data = future.result()
                results_by_date[date_str] = data

        # Process in chronological order for consistent seen_ids deduplication
        for date_str in sorted(date_list):
            data = results_by_date.get(date_str)
            if not data:
                fetch_failures += 1
                continue
            for event in data.get("events", []):
                try:
                    event_id = event.get("id", "")
                    if event_id in seen_ids:
                        continue
                    seen_ids.add(event_id)

                    status_type = event.get("status", {}).get("type", {}).get("name", "")
                    if status_type != "STATUS_FINAL":
                        continue

                    comp = event["competitions"][0]
                    competitors = comp["competitors"]
                    home = next((c for c in competitors if c["homeAway"] == "home"), None)
                    away = next((c for c in competitors if c["homeAway"] == "away"), None)
                    if not home or not away:
                        continue

                    home_score = int(home.get("score", 0) or 0)
                    away_score = int(away.get("score", 0) or 0)
                    if home_score == 0 and away_score == 0:
                        continue

                    home_abbrev = abbrev_norm(home["team"]["abbreviation"])
                    away_abbrev = abbrev_norm(away["team"]["abbreviation"])

                    all_games.append({
                        "date": event.get("date", "")[:10],
                        "season": season_year,
                        "team1": home_abbrev,
                        "team2": away_abbrev,
                        "score1": home_score,
                        "score2": away_score,
                        "neutral": int(comp.get("neutralSite", False)),
                    })
                except (KeyError, IndexError, TypeError) as e:
                    log.debug(f"Error parsing NBA game on {date_str}: {e}")

    if fetch_failures > 0:
        log.warning(f"NBA season fetch: skipped {fetch_failures} date(s) due to fetch failures")
    log.info(f"Fetched {len(all_games)} NBA historical games")
    return all_games


def fetch_nba_odds(api_key):
    if not api_key:
        log.info("No ODDS_API_KEY set, skipping NBA betting lines")
        return {}

    log.info("Fetching NBA odds from The Odds API...")
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "sport": "basketball_nba",
    }
    data = safe_get(ODDS_BASE, params=params)
    if not data:
        return {}

    odds_map = {}
    for game in data:
        try:
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            bookmakers = game.get("bookmakers", [])
            if not bookmakers:
                continue

            home_odds_list = []
            away_odds_list = []
            for bm in bookmakers:
                for market in bm.get("markets", []):
                    if market["key"] == "h2h":
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == home_team:
                                home_odds_list.append(float(outcome["price"]))
                            elif outcome["name"] == away_team:
                                away_odds_list.append(float(outcome["price"]))

            if home_odds_list and away_odds_list:
                avg_home = np.mean(home_odds_list)
                avg_away = np.mean(away_odds_list)
                raw_home = american_to_prob(avg_home)
                raw_away = american_to_prob(avg_away)
                clean_home, clean_away = remove_vig(raw_home, raw_away)
                game_key = f"{away_team}_at_{home_team}"
                odds_map[game_key] = {
                    "home_prob": round(clean_home, 4),
                    "away_prob": round(clean_away, 4),
                    "home_american": avg_home,
                    "away_american": avg_away,
                    "home_team_name": home_team,
                    "away_team_name": away_team,
                }
        except Exception as e:
            log.debug(f"Error parsing NBA odds entry: {e}")

    log.info(f"Got NBA odds for {len(odds_map)} games")
    return odds_map


def match_nba_odds(game, odds_map):
    home = game.get("home_name", "")
    away = game.get("away_name", "")
    for key, odds in odds_map.items():
        h = odds.get("home_team_name", "")
        a = odds.get("away_team_name", "")
        if (home.lower() in h.lower() or h.lower() in home.lower()) and \
           (away.lower() in a.lower() or a.lower() in away.lower()):
            return odds
    return None


# ─── NBA Efficiency Data ──────────────────────────────────────────────────────

def build_nba_efficiency_data(standings):
    """Build NBA efficiency stats from standings.

    Handles three data-source scenarios:
      1. CDN standings with real pace-adjusted ORTG/DRTG → use directly.
      2. ESPN/BDL fallback where offensive_rating = raw PPG proxy → normalise
         all values to the ~110 ORTG scale so off_eff ratios are meaningful.
      3. All values flat (CDN key-name mismatch) → same PPG normalisation.
    """
    efficiency = {}
    # Compute league averages from teams with actual game data
    teams_with_data = [s for s in standings.values() if s.get("games_played", 0) > 0]
    if teams_with_data:
        league_off = np.mean([s.get("offensive_rating", 110.0) for s in teams_with_data])
        league_def = np.mean([s.get("defensive_rating", 110.0) for s in teams_with_data])
        league_pace = np.mean([s.get("pace", 100.0) for s in teams_with_data])
    else:
        league_off = 110.0
        league_def = 110.0
        league_pace = 100.0

    for team in NBA_TEAMS:
        s = standings.get(team, {})
        off_rtg = s.get("offensive_rating", league_off)
        def_rtg = s.get("defensive_rating", league_def)
        net_rtg = off_rtg - def_rtg
        pace = s.get("pace", league_pace)
        pf = s.get("points_for", 0)
        pa = s.get("points_against", 0)
        gp = max(s.get("games_played", 1), 1)

        ppg_for = s.get("ppg_for", pf / gp)
        ppg_against = s.get("ppg_against", pa / gp)

        efficiency[team] = {
            "offensive_rating": off_rtg,
            "defensive_rating": def_rtg,
            "net_rating": net_rtg,
            "pace": pace,
            "off_eff": off_rtg / max(league_off, 1),
            "def_eff": league_def / max(def_rtg, 1),
            "net_eff": net_rtg,
            "turnover_rate": 1.0 / max(s.get("assist_turnover_ratio", 1.8), 0.1),
            "three_point_rate": s.get("three_point_rate", 0.35),
            "rebound_rate": s.get("rebound_rate", 0.5),
            "free_throw_rate": s.get("free_throw_rate", 0.20),
            "ppg_for": ppg_for,
            "ppg_against": ppg_against,
        }

    # ── Detect flat ORTG (all identical → placeholder/fallback data) ──────────
    # This happens when CDN key-names don't match or ESPN/BDL fallback is active.
    # Normalise PPG to the ORTG scale so off_eff/def_eff produce real signal.
    all_off = [efficiency[t]["offensive_rating"] for t in efficiency
               if efficiency[t].get("ppg_for", 0) > 0]
    if all_off and max(all_off) - min(all_off) < 1.0:
        valid = {t: efficiency[t] for t in efficiency if efficiency[t].get("ppg_for", 0) > 0}
        if valid:
            lg_ppg = np.mean([v["ppg_for"] for v in valid.values()])
            lg_ppg_def = np.mean([v["ppg_against"] for v in valid.values()])
            ORTG_SCALE = 110.0
            for t in efficiency:
                ppg_f = efficiency[t].get("ppg_for", lg_ppg)
                ppg_a = efficiency[t].get("ppg_against", lg_ppg_def)
                if ppg_f > 0 and ppg_a > 0:
                    efficiency[t]["offensive_rating"] = round((ppg_f / lg_ppg) * ORTG_SCALE, 1)
                    efficiency[t]["defensive_rating"] = round((ppg_a / lg_ppg_def) * ORTG_SCALE, 1)
                    efficiency[t]["net_rating"] = round(
                        efficiency[t]["offensive_rating"] - efficiency[t]["defensive_rating"], 1)
            # Recompute league averages and derived ratios with updated values
            new_lg_off = np.mean([efficiency[t]["offensive_rating"] for t in efficiency])
            new_lg_def = np.mean([efficiency[t]["defensive_rating"] for t in efficiency])
            for t in efficiency:
                o = efficiency[t]["offensive_rating"]
                d = efficiency[t]["defensive_rating"]
                efficiency[t]["off_eff"] = o / max(new_lg_off, 1)
                efficiency[t]["def_eff"] = new_lg_def / max(d, 1)
                efficiency[t]["net_eff"] = o - d
            log.warning("ORTG/DRTG were flat (ESPN/BDL fallback); re-derived from PPG. "
                        f"New ORTG range: {min(efficiency[t]['offensive_rating'] for t in efficiency):.1f}–"
                        f"{max(efficiency[t]['offensive_rating'] for t in efficiency):.1f}")

    return efficiency


def compute_nba_pythagorean(efficiency_data):
    """NBA Pythagorean expectation with exponent 13.91 (NBA canonical Morey exponent)."""
    pyth_data = {}
    for team, eff in efficiency_data.items():
        pf = max(eff.get("ppg_for", 110.0), 1.0)
        pa = max(eff.get("ppg_against", 110.0), 1.0)
        exp = 13.91  # NBA canonical Morey exponent (was 16.5, NFL-calibrated)
        pyth = (pf ** exp) / ((pf ** exp) + (pa ** exp))
        pyth_data[team] = {"pyth": round(pyth, 4)}
    return pyth_data


# ─── NBA Logistic Model ──────────────────────────────────��────────────────────

def build_nba_features(games, elo_dict, game_history, efficiency_data, pyth_data):
    """Build feature matrix for NBA logistic/XGBoost training."""
    from sklearn.preprocessing import StandardScaler
    features = []
    targets = []

    for game in games:
        t1 = game.get("team1", "")
        t2 = game.get("team2", "")
        s1 = game.get("score1", 0)
        s2 = game.get("score2", 0)
        neutral = game.get("neutral", 0)

        if not t1 or not t2 or s1 == s2:
            continue

        e1 = elo_dict.get(t1, 1500.0)
        e2 = elo_dict.get(t2, 1500.0)
        hfa = 0 if neutral else 100.0
        elo_diff = (e1 + hfa) - e2

        eff1 = efficiency_data.get(t1, {})
        eff2 = efficiency_data.get(t2, {})

        off_diff = eff1.get("offensive_rating", 110.0) - eff2.get("offensive_rating", 110.0)
        def_diff = eff2.get("defensive_rating", 110.0) - eff1.get("defensive_rating", 110.0)
        net_diff = eff1.get("net_rating", 0.0) - eff2.get("net_rating", 0.0)
        pace_diff = eff1.get("pace", 100.0) - eff2.get("pace", 100.0)
        to_diff = eff2.get("turnover_rate", 0.5) - eff1.get("turnover_rate", 0.5)
        three_diff = eff1.get("three_point_rate", 0.35) - eff2.get("three_point_rate", 0.35)
        reb_diff = eff1.get("rebound_rate", 0.5) - eff2.get("rebound_rate", 0.5)
        ft_diff = eff1.get("free_throw_rate", 0.20) - eff2.get("free_throw_rate", 0.20)

        pyth1 = pyth_data.get(t1, {}).get("pyth", 0.5)
        pyth2 = pyth_data.get(t2, {}).get("pyth", 0.5)
        pyth_diff = (1500 + (pyth1 - 0.5) * 400) - (1500 + (pyth2 - 0.5) * 400)

        form1 = nba_recent_form(game_history, t1)
        form2 = nba_recent_form(game_history, t2)
        form_diff = form1 - form2

        # Compute actual rest days to match what the prediction loop uses
        game_date_str = game.get("date", "")
        try:
            game_date = datetime.strptime(game_date_str, "%Y-%m-%d").date()
            rest1 = days_since_last_game(game_history, t1, game_date)
            rest2 = days_since_last_game(game_history, t2, game_date)
            rest_diff = float(rest1 - rest2)
            b2b1 = 1.0 if rest1 <= 1 else 0.0
            b2b2 = 1.0 if rest2 <= 1 else 0.0  # away team b2b
        except (ValueError, TypeError):
            rest_diff = 0.0
            b2b1 = 0.0
            b2b2 = 0.0

        # Travel miles and timezone diff (away team traveling to home arena)
        travel_miles, tz_diff = nba_travel_distance(t1, t2)  # t1=home, t2=away

        feat = [
            elo_diff, hfa, rest_diff,
            pyth_diff, net_diff,
            off_diff, def_diff, pace_diff,
            to_diff, three_diff, reb_diff, ft_diff,
            form_diff, b2b1 - b2b2,  # difference, not home-only (matches inference)
            float(tz_diff),           # circadian shift: positive = away traveled east
        ]
        features.append(feat)
        targets.append(1 if s1 > s2 else 0)

    return np.array(features) if features else np.array([]).reshape(0, 15), np.array(targets)


def train_nba_logistic(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import TimeSeriesSplit

    if len(X) < 50:
        return None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(X_scaled, y)

    tscv = TimeSeriesSplit(n_splits=5)
    probs_oos = np.zeros(len(y))
    for train_idx, val_idx in tscv.split(X_scaled):
        m = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        m.fit(X_scaled[train_idx], y[train_idx])
        probs_oos[val_idx] = m.predict_proba(X_scaled[val_idx])[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probs_oos, y)

    return model, scaler, calibrator


def predict_nba_logistic(home_feat, model, scaler, calibrator):
    if model is None or scaler is None:
        return None
    try:
        x = scaler.transform([home_feat])
        raw_prob = model.predict_proba(x)[0][1]
        return float(calibrator.transform([raw_prob])[0]) if calibrator else raw_prob
    except Exception as e:
        log.warning(f"NBA logistic prediction failed: {e}")
        return None


def train_nba_xgboost(X, y):
    try:
        import xgboost as xgb
        from sklearn.preprocessing import StandardScaler

        if len(X) < 50:
            return None, None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=42,
        )
        model.fit(X_scaled, y)
        return model, scaler
    except ImportError:
        log.warning("XGBoost not available for NBA model")
        return None, None
    except Exception as e:
        log.error(f"NBA XGBoost training failed: {e}")
        return None, None


# ─── NBA Model Metrics ────────────────────────────────────────────────────────

def evaluate_nba_model(model, scaler, calibrator, X, y):
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

    if model is None or len(X) == 0:
        return {"log_loss": None, "brier_score": None, "auc": None}

    try:
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]
        if calibrator:
            probs = calibrator.transform(probs)

        return {
            "log_loss": round(log_loss(y, probs), 4),
            "brier_score": round(brier_score_loss(y, probs), 4),
            "auc": round(roc_auc_score(y, probs), 4),
        }
    except Exception as e:
        log.warning(f"NBA model evaluation failed: {e}")
        return {"log_loss": None, "brier_score": None, "auc": None}


# ─── Prediction Drivers ───────────────────────────────────────────────────────

def generate_nba_prediction_drivers(game_info, home, away, elo_dict,
                                      efficiency_data, injury_impacts, adj):
    drivers = []

    home_elo = elo_dict.get(home, 1500.0)
    away_elo = elo_dict.get(away, 1500.0)
    elo_diff = abs(home_elo - away_elo)

    if elo_diff >= 75:
        leader = home if home_elo > away_elo else away
        drivers.append(f"ELO advantage: {leader} +{elo_diff:.0f} rating points")

    # Key injured players with tier labels
    for team in [home, away]:
        impact = injury_impacts.get(team, {})
        key_out = impact.get("key_players_out", [])
        for p in key_out[:3]:
            tier_label = p.get("value_tier", "starter").title()
            drivers.append(
                f"{p['player']} ({team}) [{tier_label}] {p['status'].upper()} "
                f"— −{p['elo_impact']:.0f} ELO"
            )
        # Cumulative star stack driver
        star_count = impact.get("star_count", 0)
        stack_mult = impact.get("star_stack_multiplier", 1.0)
        total_penalty = impact.get("elo_penalty", 0.0)
        if star_count >= 2 and total_penalty > 0:
            drivers.append(
                f"{team} star stack: {star_count}× elite players out "
                f"→ ×{stack_mult:.2f} multiplier (combined −{total_penalty:.0f} ELO)"
            )

    # Efficiency gap
    home_eff = efficiency_data.get(home, {})
    away_eff = efficiency_data.get(away, {})
    net_home = home_eff.get("net_rating", 0.0)
    net_away = away_eff.get("net_rating", 0.0)
    if abs(net_home - net_away) >= 3.0:
        drivers.append(
            f"Net rating gap: {home} {net_home:+.1f} vs {away} {net_away:+.1f}"
        )

    off_diff = home_eff.get("offensive_rating", 110) - away_eff.get("offensive_rating", 110)
    if abs(off_diff) >= 3.0:
        leader = home if off_diff > 0 else away
        drivers.append(f"Offensive rating advantage: {leader} ({abs(off_diff):.1f} pts/100)")

    # Rest / back-to-back
    b2b_home = adj.get("b2b_home", False)
    b2b_away = adj.get("b2b_away", False)
    if b2b_home:
        drivers.append(f"Back-to-back: {home} playing on zero days rest (−5 rating)")
    if b2b_away:
        drivers.append(f"Back-to-back: {away} playing on zero days rest (−5 rating)")

    rest_diff = adj.get("rest_diff", 0)
    if abs(rest_diff) >= 2 and not b2b_home and not b2b_away:
        rested = home if rest_diff > 0 else away
        drivers.append(f"Rest advantage: {rested} has {abs(rest_diff)} extra days rest")

    # Travel
    travel_mi = adj.get("travel_dist_miles", 0)
    if travel_mi >= 1500:
        drivers.append(f"Travel: away team travels {travel_mi:.0f} miles")

    if not game_info.get("neutral", False):
        drivers.append(f"Home court: {home} +100 ELO advantage")

    return drivers


# ─── NBA Season Simulation ────────────────────────────────────────────────────

def nba_ensemble_predict(elo_prob, pyth_prob, eff_prob, log_prob=None, xgb_prob=None,
                          weights=None):
    if weights is None:
        weights = {"elo": 0.25, "pyth": 0.20, "eff": 0.15, "log": 0.25, "xgb": 0.15}

    # Handle None models by excluding their weights (avoids 50/50 contamination)
    if xgb_prob is None and log_prob is None:
        total_w = weights["elo"] + weights["pyth"] + weights["eff"]
        val = (weights["elo"] * elo_prob + weights["pyth"] * pyth_prob +
               weights["eff"] * eff_prob) / max(total_w, 1e-9)
    elif xgb_prob is None:
        total_w = weights["elo"] + weights["pyth"] + weights["eff"] + weights["log"]
        val = (weights["elo"] * elo_prob + weights["pyth"] * pyth_prob +
               weights["eff"] * eff_prob + weights["log"] * log_prob) / max(total_w, 1e-9)
    elif log_prob is None:
        total_w = weights["elo"] + weights["pyth"] + weights["eff"] + weights["xgb"]
        val = (weights["elo"] * elo_prob + weights["pyth"] * pyth_prob +
               weights["eff"] * eff_prob + weights["xgb"] * xgb_prob) / max(total_w, 1e-9)
    else:
        total_w = sum(weights.values())
        val = (weights["elo"] * elo_prob + weights["pyth"] * pyth_prob +
               weights["eff"] * eff_prob + weights["log"] * log_prob +
               weights["xgb"] * xgb_prob) / max(total_w, 1e-9)

    return float(max(0.01, min(0.99, val)))


# ─── Rest Days / H2H / Streak Helpers ─────────────────────────────────────────

def days_since_last_game(game_history, team, today):
    """Return days since a team's most recent completed game."""
    from datetime import date as date_cls
    hist = game_history.get(team, [])
    dates = [g["date"] for g in hist if g.get("date") and g["date"] <= str(today)]
    if not dates:
        return 7  # unknown → use neutral default
    last_date_str = max(dates)
    try:
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
        return max(0, (today - last_date).days)
    except ValueError:
        return 7


def compute_h2h(home, away, historical_games, last_n=10):
    """Head-to-head record between home and away teams from historical data."""
    meetings = [
        g for g in historical_games
        if (g["team1"] == home and g["team2"] == away) or
           (g["team1"] == away and g["team2"] == home)
    ]
    meetings = sorted(meetings, key=lambda g: g.get("date", ""))
    home_wins = sum(
        1 for g in meetings
        if (g["team1"] == home and g["score1"] > g["score2"]) or
           (g["team2"] == home and g["score2"] > g["score1"])
    )
    away_wins = len(meetings) - home_wins
    recent = meetings[-last_n:]
    last_n_list = []
    for g in recent:
        h_won = (g["team1"] == home and g["score1"] > g["score2"]) or \
                (g["team2"] == home and g["score2"] > g["score1"])
        last_n_list.append({
            "date": g.get("date", ""),
            "winner": home if h_won else away,
            "margin": abs(g["score1"] - g["score2"]),
        })
    return {
        "home_wins": home_wins,
        "away_wins": away_wins,
        "total_meetings": len(meetings),
        "last_n": last_n_list,
    }


def compute_streak(game_history, team):
    """Return current W/L streak type and length."""
    hist = game_history.get(team, [])
    if not hist:
        return {"type": "N", "count": 0}
    streak_type = "W" if hist[-1].get("result", 0) == 1 else "L"
    count = 0
    for g in reversed(hist):
        result = g.get("result", 0)
        if (result == 1 and streak_type == "W") or (result == 0 and streak_type == "L"):
            count += 1
        else:
            break
    return {"type": streak_type, "count": count}


# ─── Main Runner ─────────────────────────────────────────────────────────────

def run():
    log.info("=== NBA Prediction Model Update Starting ===")
    now_utc = datetime.now(timezone.utc).isoformat()
    odds_api_key = os.environ.get("ODDS_API_KEY", "")

    current_month = datetime.now(timezone.utc).month
    current_year = datetime.now(timezone.utc).year
    # NBA season: Oct–Jun. Current season year = year ending
    if current_month >= 10:
        season_year = current_year + 1
    else:
        season_year = current_year

    # ── 1. Download data ───────────────────────────────────────────────────
    scoreboard_games = fetch_nba_scoreboard()
    standings = fetch_nba_standings()
    injuries = fetch_nba_injuries()
    odds_map = fetch_nba_odds(odds_api_key)

    # ── Issue 7: Validate standings before proceeding ─────────────────────
    if not standings or len(standings) < 15:
        msg = (f"NBA standings fetch returned only {len(standings)} teams "
               f"(need >= 15) — aborting to prevent overwriting predictions with bad data.")
        log.error(msg)
        write_nba_abort_log(reason=msg, now_utc=now_utc,
                            counts={"standings_teams": len(standings)})
        return
    _active = [v for v in standings.values() if v.get("games_played", 0) > 0]
    if _active:
        _ortg_vals = [v.get("offensive_rating", 110.0) for v in _active]
        if max(_ortg_vals) - min(_ortg_vals) < 0.5:
            log.warning(
                f"All {len(_active)} active teams have identical ORTG ({_ortg_vals[0]:.1f}) — "
                "ESPN/BDL fallback active. ORTG will be re-derived from PPG in build_nba_efficiency_data()."
            )

    is_offseason = len(scoreboard_games) == 0

    # ── 1c. Fetch future NBA games ────────��──────────────────────────────────
    future_games = fetch_nba_future_games(days_ahead=14)
    # Exclude games already in scoreboard
    existing_ids = {g["game_id"] for g in scoreboard_games}
    future_games = [g for g in future_games if g["game_id"] not in existing_ids]
    all_games_for_prediction = scoreboard_games + future_games

    # ── Guard: if both fetches returned 0 games, try a direct today-fallback
    # before deciding whether to abort. This breaks the self-perpetuating
    # games:[] state caused by the guard seeing no existing data to preserve.
    if len(scoreboard_games) == 0 and len(future_games) == 0:
        log.warning("Both scoreboard and future fetches returned 0 games — trying direct today/tomorrow fallback...")
        for _delta in range(0, 3):
            _ds = (datetime.now(timezone.utc) + timedelta(days=_delta)).strftime('%Y%m%d')
            _d = safe_get(f"{ESPN_NBA_WEB_BASE}/scoreboard?dates={_ds}&limit=20&seasontype=2")
            if not _d or not _d.get("events"):
                _d = safe_get(f"{ESPN_NBA_BASE}/scoreboard?dates={_ds}&limit=20&seasontype=2")
            if _d:
                for _g in parse_nba_events(_d):
                    if _g["game_id"] not in existing_ids:
                        existing_ids.add(_g["game_id"])
                        if _delta > 0:
                            _g["is_future"] = True
                        future_games.append(_g)
        all_games_for_prediction = scoreboard_games + future_games

    if len(scoreboard_games) == 0 and len(future_games) == 0:
        # Try wider fallback window (±3 days) before deciding to abort
        log.warning("Still 0 games after initial fallback — trying wider ±3-day window...")
        for _delta in range(-3, 4):
            _ds = (datetime.now(timezone.utc) + timedelta(days=_delta)).strftime('%Y%m%d')
            _d = safe_get(f"{ESPN_NBA_WEB_BASE}/scoreboard?dates={_ds}&limit=20&seasontype=2")
            if not _d or not _d.get("events"):
                _d = safe_get(f"{ESPN_NBA_BASE}/scoreboard?dates={_ds}&limit=20&seasontype=2")
            if _d:
                for _g in parse_nba_events(_d):
                    if _g["game_id"] not in existing_ids:
                        existing_ids.add(_g["game_id"])
                        if _delta > 0:
                            _g["is_future"] = True
                        future_games.append(_g)
        all_games_for_prediction = scoreboard_games + future_games

    if len(scoreboard_games) == 0 and len(future_games) == 0:
        # Only abort if the existing file already contains game data worth preserving
        existing_predictions_path = DATA_DIR / "nba_predictions.json"
        has_existing_data = False
        if existing_predictions_path.exists():
            try:
                existing_content = json.loads(existing_predictions_path.read_text())
                has_existing_data = bool(existing_content.get("games"))
            except Exception:
                has_existing_data = False
        if has_existing_data:
            msg = "Both scoreboard and future game fetches returned 0 games — aborting to preserve existing nba_predictions.json"
            log.error(msg)
            write_nba_abort_log(
                reason=msg,
                now_utc=now_utc,
                counts={"scoreboard_games": 0, "future_games": 0},
            )
            return
        log.warning("Both fetches returned 0 games and no existing data found — proceeding with empty games list")

    # ── 1b. Injury impacts ──────────────────────────────────────────────────
    log.info("Computing NBA injury impacts...")
    nba_player_values = build_nba_player_values(season_year=season_year)
    injury_impacts = compute_all_nba_team_impacts(injuries, nba_player_values)

    # Save NBA injuries file (only overwrite if we actually fetched data)
    if injuries:
        nba_injuries_list = []
        for team, players in injuries.items():
            for p in players:
                pname = p.get("player", "")
                pmult = nba_player_values.get(pname, 1.0)
                nba_injuries_list.append({
                    "player": pname,
                    "team": team,
                    "position": p.get("position", ""),
                    "status": p.get("status", ""),
                    "injury_description": p.get("injury_description", p.get("status", "")),
                    "value_tier": nba_value_tier_label(pmult),
                })
        (DATA_DIR / "nba_injuries.json").write_text(
            json.dumps({"updated": now_utc, "injuries": nba_injuries_list}, indent=2)
        )
    else:
        log.warning("No NBA injury data fetched — keeping existing nba_injuries.json")

    # ── 2. Fetch historical games for ELO ───────────────────────────────────
    log.info("Fetching NBA historical games for ELO computation...")
    historical_games = fetch_nba_season_games(
        seasons=[season_year - 4, season_year - 3, season_year - 2, season_year - 1, season_year]
    )

    # Guard: if the historical fetch returned too few games, all sub-models
    # will fall back to 50% neutral defaults (ELO=1500 everywhere) and
    # overwrite any good predictions that already exist. Abort instead.
    if len(historical_games) < 200:
        msg = (
            f"NBA historical fetch returned only {len(historical_games)} games "
            f"(need >= 200) — aborting NBA pipeline to prevent overwriting "
            f"predictions with 50% defaults. Check ESPN scoreboard API availability."
        )
        log.critical(msg)
        write_nba_abort_log(
            reason=msg,
            now_utc=now_utc,
            counts={
                "historical_games": len(historical_games),
                "required_minimum": 200,
                "scoreboard_games": len(scoreboard_games),
                "future_games": len(future_games),
            },
        )
        return

    # ── 3. Compute NBA ELO ──────────────────────────────────────────────────
    log.info("Computing NBA ELO ratings...")
    if historical_games:
        elo_dict, game_history = compute_nba_elo(historical_games)
    else:
        elo_dict = {t: 1500.0 for t in NBA_TEAMS}
        game_history = {t: [] for t in NBA_TEAMS}

    for team in NBA_TEAMS:
        if team not in elo_dict:
            elo_dict[team] = 1500.0
        if team not in game_history:
            game_history[team] = []

    log.info(f"NBA ELO computed for {len(elo_dict)} teams")

    # Save nba_elo_ratings.json using the new nba_elo_model module
    if HAS_NBA_ELO_MODEL and save_nba_elo_ratings:
        try:
            save_nba_elo_ratings(elo_dict, game_history, season_year)
        except Exception as e:
            log.warning(f"save_nba_elo_ratings failed: {e}")

    # ── 4. Efficiency and Pythagorean ───────────────────────────────────────
    log.info("Computing NBA efficiency and Pythagorean ratings...")
    efficiency_data = build_nba_efficiency_data(standings)
    # CDN standings returns points_for=0/points_against=0, so override PPG from game scores
    from collections import defaultdict as _dd
    _ppg = _dd(lambda: {"pf": 0.0, "pa": 0.0, "gp": 0})
    for _g in historical_games:
        if _g.get("season", 0) >= season_year - 1 and _g.get("score1") and _g.get("score2"):
            _ppg[_g["team1"]]["pf"] += _g["score1"]; _ppg[_g["team1"]]["pa"] += _g["score2"]; _ppg[_g["team1"]]["gp"] += 1
            _ppg[_g["team2"]]["pf"] += _g["score2"]; _ppg[_g["team2"]]["pa"] += _g["score1"]; _ppg[_g["team2"]]["gp"] += 1
    for _t in efficiency_data:
        if _ppg[_t]["gp"] > 0:
            efficiency_data[_t]["ppg_for"]     = _ppg[_t]["pf"] / _ppg[_t]["gp"]
            efficiency_data[_t]["ppg_against"] = _ppg[_t]["pa"] / _ppg[_t]["gp"]
    # When CDN standings didn't provide real ORTG/DRTG (all teams show same default 110.0),
    # override from PPG computed above so each team has unique efficiency values.
    _all_off = [efficiency_data[t]["offensive_rating"] for t in efficiency_data]
    if len(set(_all_off)) <= 2:  # all identical → CDN key-name mismatch, use PPG fallback
        _valid_ppg = [efficiency_data[t]["ppg_for"] for t in efficiency_data
                      if efficiency_data[t].get("ppg_for", 0) > 0]
        _lg_ppg = sum(_valid_ppg) / len(_valid_ppg) if _valid_ppg else 114.0
        _valid_ppg_def = [efficiency_data[t]["ppg_against"] for t in efficiency_data
                          if efficiency_data[t].get("ppg_against", 0) > 0]
        _lg_ppg_def = sum(_valid_ppg_def) / len(_valid_ppg_def) if _valid_ppg_def else 114.0
        ORTG_SCALE = 110.0
        for _t in efficiency_data:
            ppg_f = efficiency_data[_t].get("ppg_for", 0)
            ppg_a = efficiency_data[_t].get("ppg_against", 0)
            if ppg_f > 0:
                # Normalise PPG to ORTG scale rather than storing raw PPG
                efficiency_data[_t]["offensive_rating"] = round((ppg_f / _lg_ppg) * ORTG_SCALE, 1)
                efficiency_data[_t]["defensive_rating"] = round((ppg_a / _lg_ppg_def) * ORTG_SCALE, 1)
                efficiency_data[_t]["net_rating"] = round(
                    efficiency_data[_t]["offensive_rating"] - efficiency_data[_t]["defensive_rating"], 1)
    # Always recompute off_eff/def_eff/net_eff with the current ORTG/DRTG values
    # (in case they were updated by either the flat-override or PPG override above)
    _lg_off_final = np.mean([efficiency_data[t]["offensive_rating"] for t in efficiency_data
                              if efficiency_data[t]["offensive_rating"] > 0])
    _lg_def_final = np.mean([efficiency_data[t]["defensive_rating"] for t in efficiency_data
                              if efficiency_data[t]["defensive_rating"] > 0])
    for _t in efficiency_data:
        _o = efficiency_data[_t]["offensive_rating"]
        _d = efficiency_data[_t]["defensive_rating"]
        efficiency_data[_t]["off_eff"] = _o / max(_lg_off_final, 1)
        efficiency_data[_t]["def_eff"] = _lg_def_final / max(_d, 1)
        efficiency_data[_t]["net_eff"] = _o - _d
    _off_vals = [efficiency_data[t]["offensive_rating"] for t in efficiency_data]
    log.info(f"NBA efficiency ORTG range: {min(_off_vals):.1f}–{max(_off_vals):.1f} "
             f"(spread: {max(_off_vals)-min(_off_vals):.1f})")
    # Issue 7: Abort if efficiency data is still flat after all normalization attempts
    if max(_off_vals) - min(_off_vals) < 0.5:
        msg = ("build_nba_efficiency_data: all ORTG values are identical after PPG normalisation "
               f"(value={_off_vals[0]:.1f}) — data source corrupted or all teams have zero games. "
               "Aborting to prevent writing 50/50 flat predictions.")
        log.error(msg)
        write_nba_abort_log(reason=msg, now_utc=now_utc,
                            counts={"efficiency_spread": 0, "standings_teams": len(standings)})
        return
    pyth_data = compute_nba_pythagorean(efficiency_data)

    # ── 5 & 6. Build features once, train logistic and XGBoost on same X, y ──
    log.info("Training NBA logistic model...")
    logistic_model, logistic_scaler, logistic_calibrator = None, None, None
    model_metrics = {"log_loss": None, "brier_score": None, "auc": None}
    xgb_model, xgb_scaler = None, None
    X, y = np.array([]).reshape(0, 15), np.array([])

    if historical_games and len(historical_games) > 100:
        try:
            X, y = build_nba_features(historical_games, elo_dict, game_history,
                                       efficiency_data, pyth_data)
            if len(X) > 50:
                logistic_model, logistic_scaler, logistic_calibrator = train_nba_logistic(X, y)
                if logistic_model:
                    metrics = evaluate_nba_model(logistic_model, logistic_scaler,
                                                  logistic_calibrator, X, y)
                    model_metrics.update(metrics)
                    log.info(f"NBA logistic model trained. Log loss: {metrics['log_loss']}")
        except Exception as e:
            log.error(f"NBA logistic training failed: {e}")

    # ── 6. Train NBA XGBoost (reuse same X, y — no redundant build_nba_features call) ──
    log.info("Training NBA XGBoost model...")
    if len(X) > 50:
        try:
            xgb_model, xgb_scaler = train_nba_xgboost(X, y)
        except Exception as e:
            log.error(f"NBA XGBoost training failed: {e}")

    # ── 7. Bayesian ratings ──────────────────���──────────────────────────────
    log.info("Computing NBA Bayesian ratings...")
    recent_games = [g for g in historical_games if
                    g.get("season", 0) >= season_year - 1]
    bayesian_ratings = update_ratings(recent_games, elo_dict, hfa=100.0,
                                       margin_multiplier=5.0, obs_noise=130.0)
    for team in NBA_TEAMS:
        if team not in bayesian_ratings:
            bayesian_ratings[team] = {"mu": elo_dict.get(team, 1500.0), "sigma": 75.0}

    # ── 8. Season simulation ────────────────────────────────────────────────
    log.info("Running NBA season simulation...")
    remaining_schedule = []
    for game in all_games_for_prediction:
        if game.get("status", "") in ("STATUS_SCHEDULED", "STATUS_IN_PROGRESS"):
            remaining_schedule.append({
                "team_a": game["home_team"],
                "team_b": game["away_team"],
                "is_home_a": True,
                "neutral": bool(game.get("neutral", False)),
            })

    if not remaining_schedule:
        log.info("No remaining scheduled NBA games found; season simulation will use ratings only.")

    for team in NBA_TEAMS:
        if team in bayesian_ratings:
            bayesian_ratings[team]["wins"] = standings.get(team, {}).get("wins", 0)

    try:
        season_sim = simulate_season(
            NBA_TEAMS, remaining_schedule, bayesian_ratings, n_sims=3000
        )
    except Exception as e:
        log.error(f"NBA season simulation failed: {e}")
        season_sim = {}
        for t in NBA_TEAMS:
            w = standings.get(t, {}).get("wins", 0)
            gp = standings.get(t, {}).get("games_played", 1) or 1
            win_pct = w / gp
            season_sim[t] = {
                "playoff_prob": min(0.95, max(0.05, win_pct * 1.3)),
                "division_win_prob": min(0.9, max(0.02, win_pct * 0.5)),
                "sb_prob": min(0.5, max(0.003, win_pct ** 2 * 0.4)),
                "wins_avg": round(win_pct * 82, 1),
            }

    # ── 8b. Learn ensemble weights from historical data ──────────────────────
    log.info("Learning NBA ensemble weights...")
    nba_weights = {"elo": 0.25, "pyth": 0.20, "eff": 0.15, "log": 0.25, "xgb": 0.15}
    weights_path = DATA_DIR / "ensemble_weights_nba.json"
    try:
        from model.ensemble_model import learn_ensemble_weights
        completed_hist = [g for g in historical_games if g.get("score1") is not None
                          and g.get("score2") is not None][-500:]  # last 500 games
        if len(completed_hist) >= 50:
            from model.nba_elo import nba_expected_score as _nes
            sub_probs_hist = []
            actuals_hist = []
            for _g in completed_hist:
                _home, _away = _g["team1"], _g["team2"]
                _eh = elo_dict.get(_home, 1500.0) + (100.0 if not _g.get("neutral") else 0.0)
                _ea = elo_dict.get(_away, 1500.0)
                _ph = pyth_data.get(_home, {}).get("pyth", 0.5)
                _pa = pyth_data.get(_away, {}).get("pyth", 0.5)
                _ph_adj = _ph * (1.0 + (100.0 if not _g.get("neutral") else 0.0) / 1500.0)
                _pd = _ph_adj + _pa
                _ep = elo_dict.get(_home, {}) and efficiency_data.get(_home, {}).get("net_rating", 0.0)
                _ea2 = efficiency_data.get(_away, {}).get("net_rating", 0.0)
                sub_probs_hist.append({
                    "elo": _nes(_eh, _ea),
                    "pyth": _ph_adj / _pd if _pd > 0 else 0.5,
                    "eff": _nes(1500 + float(_ep or 0) * 10 + (100.0 if not _g.get("neutral") else 0.0),
                                1500 + _ea2 * 10),
                    "log": 0.5,  # logistic not available per-historical game
                    "xgb": 0.5,
                })
                actuals_hist.append(1 if _g.get("score1", 0) > _g.get("score2", 0) else 0)
            learned = learn_ensemble_weights(
                sub_probs_hist, actuals_hist, weight_keys=["elo", "pyth", "eff"]
            )
            if learned:
                # Blend into full 5-model default; keep log/xgb at fixed proportion
                nba_weights["elo"] = round(learned.get("elo", 0.25) * 0.6, 4)
                nba_weights["pyth"] = round(learned.get("pyth", 0.20) * 0.6, 4)
                nba_weights["eff"] = round(learned.get("eff", 0.15) * 0.6, 4)
                # Normalise all five to sum to 1
                _total = sum(nba_weights.values())
                nba_weights = {k: round(v / _total, 4) for k, v in nba_weights.items()}
                weights_path.write_text(json.dumps(nba_weights, indent=2))
                log.info(f"Learned NBA weights: {nba_weights}")
    except Exception as e:
        log.warning(f"Ensemble weight learning skipped: {e}")
        if weights_path.exists():
            try:
                nba_weights = json.loads(weights_path.read_text())
            except Exception:
                pass

    # ── 9. Generate per-game predictions ────────────────────────────────────
    log.info(f"Generating NBA predictions for {len(all_games_for_prediction)} games...")
    predictions_list = []

    # Pre-compute H2H cache to avoid O(n*m) recomputation per game
    h2h_cache = {}
    for _g in all_games_for_prediction:
        _pair = (_g["home_team"], _g["away_team"])
        if _pair not in h2h_cache:
            h2h_cache[_pair] = compute_h2h(_g["home_team"], _g["away_team"], historical_games)

    for game in all_games_for_prediction:
        try:
            home = game["home_team"]
            away = game["away_team"]
            game_id = game["game_id"]
            neutral = bool(game.get("neutral", False))

            # Rest/travel adjustments — compute actual days since last game
            from datetime import date as date_cls
            # Use the game's scheduled date for accurate rest/B2B calculation
            game_date_str = game.get("game_time", "")[:10]  # "YYYY-MM-DD"
            try:
                game_date = datetime.strptime(game_date_str, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                game_date = date_cls.today()
            day_before_game = game_date - timedelta(days=1)
            rest_home = days_since_last_game(game_history, home, game_date)
            rest_away = days_since_last_game(game_history, away, game_date)
            dist, tz_diff = nba_travel_distance(home, away)

            # Back-to-back detection: played the day before this game
            b2b_home = any(
                g.get("date", "")[:10] == str(day_before_game)
                for g in game_history.get(home, [])[-3:]
            )
            b2b_away = any(
                g.get("date", "")[:10] == str(day_before_game)
                for g in game_history.get(away, [])[-3:]
            )
            # Ensure rest days are consistent with B2B detection
            if b2b_home:
                rest_home = 1
            if b2b_away:
                rest_away = 1
            rest_diff = rest_home - rest_away

            # Injury adjustments
            inj_adj_home = -injury_impacts.get(home, {}).get("elo_penalty", 0.0)
            inj_adj_away = -injury_impacts.get(away, {}).get("elo_penalty", 0.0)

            # ELO prediction — use model/nba_elo.py with full NBA-specific adjustments
            home_elo = elo_dict.get(home, 1500.0)
            away_elo = elo_dict.get(away, 1500.0)
            hfa = 0.0 if neutral else 100.0
            elo_result = predict_nba_game(
                home, away, elo_dict,
                home_b2b=b2b_home, away_b2b=b2b_away,
                rest_diff=rest_diff, travel_miles=dist,  # dist is miles only
                neutral=neutral,
            )
            elo_prob = elo_result["prob"]
            # Apply injury adjustments on top of the ELO-model result
            if inj_adj_home != 0.0 or inj_adj_away != 0.0:
                inj_home_adj_elo = elo_result["home_adj_elo"] + inj_adj_home
                inj_away_adj_elo = elo_result["away_adj_elo"] + inj_adj_away
                elo_prob = nba_expected_score(inj_home_adj_elo, inj_away_adj_elo)

            # Bayesian prediction
            bayes_result = bayes_predict(home, away, bayesian_ratings,
                                          is_home_a=True, neutral=neutral, hfa=100.0)
            bayesian_prob = bayes_result.get("bayesian_prob", 0.5)

            # Pythagorean prediction — direct ratio (Bradley-Terry), not ELO transform
            pyth_home = pyth_data.get(home, {}).get("pyth", 0.5)
            pyth_away = pyth_data.get(away, {}).get("pyth", 0.5)
            # Scale home court advantage into pythagorean space.
            # Divisor 213.0 calibrated so equal teams → ~59.5% home win (NBA historical HWR).
            pyth_home_adj = pyth_home * (1.0 + hfa / 213.0)
            _pyth_denom = pyth_home_adj + pyth_away
            pyth_prob = pyth_home_adj / _pyth_denom if _pyth_denom > 0 else 0.5
            # ELO-equivalent pythagorean values for logistic feature vector
            pyth_elo_home = 1500.0 + (pyth_home - 0.5) * 400.0
            pyth_elo_away = 1500.0 + (pyth_away - 0.5) * 400.0

            # Efficiency prediction
            eff_home = efficiency_data.get(home, {}).get("net_rating", 0.0)
            eff_away = efficiency_data.get(away, {}).get("net_rating", 0.0)
            eff_elo_home = 1500 + eff_home * 10 + hfa
            eff_elo_away = 1500 + eff_away * 10
            eff_prob = nba_expected_score(eff_elo_home, eff_elo_away)

            # Logistic prediction
            home_eff = efficiency_data.get(home, {})
            away_eff = efficiency_data.get(away, {})
            feat = [
                (home_elo + hfa) - away_elo,
                hfa,
                rest_diff,
                pyth_elo_home - pyth_elo_away,
                home_eff.get("net_rating", 0) - away_eff.get("net_rating", 0),
                home_eff.get("offensive_rating", 110) - away_eff.get("offensive_rating", 110),
                away_eff.get("defensive_rating", 110) - home_eff.get("defensive_rating", 110),
                home_eff.get("pace", 100) - away_eff.get("pace", 100),
                away_eff.get("turnover_rate", 0.5) - home_eff.get("turnover_rate", 0.5),
                home_eff.get("three_point_rate", 0.35) - away_eff.get("three_point_rate", 0.35),
                home_eff.get("rebound_rate", 0.5) - away_eff.get("rebound_rate", 0.5),
                home_eff.get("free_throw_rate", 0.20) - away_eff.get("free_throw_rate", 0.20),
                nba_recent_form(game_history, home) - nba_recent_form(game_history, away),
                float(b2b_home) - float(b2b_away),
                float(tz_diff),  # circadian shift: positive = away team traveled east
            ]
            log_prob = predict_nba_logistic(feat, logistic_model, logistic_scaler,
                                             logistic_calibrator)

            # XGBoost prediction
            xgb_prob = None
            if xgb_model and xgb_scaler:
                try:
                    x_scaled = xgb_scaler.transform([feat])
                    xgb_prob = float(xgb_model.predict_proba(x_scaled)[0][1])
                except Exception as e:
                    log.warning(f"NBA XGBoost prediction failed for {home} vs {away}: {e}")

            # Ensemble — use empirically learned weights when available
            ensemble_prob = nba_ensemble_predict(
                elo_prob=elo_prob,
                pyth_prob=pyth_prob,
                eff_prob=eff_prob,
                log_prob=log_prob,
                xgb_prob=xgb_prob,
                weights=nba_weights,
            )

            # Monte Carlo
            mu_home = bayesian_ratings.get(home, {}).get("mu", home_elo)
            mu_away = bayesian_ratings.get(away, {}).get("mu", away_elo)
            sig_home = bayesian_ratings.get(home, {}).get("sigma", 75.0)
            sig_away = bayesian_ratings.get(away, {}).get("sigma", 75.0)

            mc_result = simulate_game(
                mu_home, mu_away, sig_home, sig_away,
                is_home_a=True, neutral=neutral, n=10000
            )

            # Market odds
            market_odds = match_nba_odds(game, odds_map)
            market_home_prob = None
            market_edge = None
            kelly_pct = None
            if market_odds:
                market_home_prob = market_odds.get("home_prob")
                if market_home_prob:
                    market_edge = round(ensemble_prob - market_home_prob, 4)
                    kelly_pct = kelly_criterion(ensemble_prob, market_home_prob)

            adj_dict = {
                "rest_home": rest_home,
                "rest_away": rest_away,
                "rest_diff": rest_diff,
                "travel_dist_miles": round(dist, 0),
                "b2b_home": b2b_home,
                "b2b_away": b2b_away,
                "home_elo_bonus": 0 if neutral else 100,
            }

            pred_drivers = generate_nba_prediction_drivers(
                game, home, away, elo_dict, efficiency_data, injury_impacts, adj_dict
            )

            # Plain-English explanation
            winner = home if ensemble_prob >= 0.5 else away
            winner_prob = ensemble_prob if ensemble_prob >= 0.5 else 1 - ensemble_prob
            loser = away if ensemble_prob >= 0.5 else home
            elo_gap = abs(home_elo - away_elo)
            confidence = "strong" if winner_prob > 0.70 else "moderate" if winner_prob > 0.60 else "slight"

            # Compute league-wide rankings from efficiency_data for richer narrative
            def _ordinal(n):
                suffix = "th" if 11 <= n <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
                return f"{n}{suffix}"

            def _off_label(rank, n):
                pct = rank / max(n, 1)
                return "elite" if pct <= 0.15 else "strong" if pct <= 0.35 else "average" if pct <= 0.65 else "below-average" if pct <= 0.85 else "weak"

            def _def_label(rank, n):
                pct = rank / max(n, 1)
                return "elite" if pct <= 0.15 else "strong" if pct <= 0.35 else "average" if pct <= 0.65 else "below-average" if pct <= 0.85 else "weak"

            all_teams = list(efficiency_data.keys())
            n_teams = max(len(all_teams), 1)
            teams_by_off = sorted(all_teams, key=lambda t: efficiency_data[t].get("offensive_rating", 0), reverse=True)
            teams_by_def = sorted(all_teams, key=lambda t: efficiency_data[t].get("defensive_rating", 999))

            def _off_rank(t):
                return teams_by_off.index(t) + 1 if t in teams_by_off else n_teams

            def _def_rank(t):
                return teams_by_def.index(t) + 1 if t in teams_by_def else n_teams

            w_eff = efficiency_data.get(winner, {})
            l_eff = efficiency_data.get(loser, {})
            w_off_rtg = w_eff.get("offensive_rating", 110.0)
            w_def_rtg = w_eff.get("defensive_rating", 110.0)
            w_net = w_eff.get("net_rating", 0.0)
            l_off_rtg = l_eff.get("offensive_rating", 110.0)
            l_net = l_eff.get("net_rating", 0.0)
            w_off_rank = _off_rank(winner)
            w_def_rank = _def_rank(winner)
            l_off_rank = _off_rank(loser)

            winner_name = NBA_TEAM_NAMES.get(winner, winner)
            loser_name = NBA_TEAM_NAMES.get(loser, loser)
            home_name_str = NBA_TEAM_NAMES.get(home, home)

            exp_sentences = [
                f"The model gives {winner_name} a {winner_prob*100:.1f}% win probability — "
                f"a {confidence} favorite over {loser_name}."
            ]

            # Winner strengths
            exp_sentences.append(
                f"{winner_name} has a {_off_label(w_off_rank, n_teams)} offense "
                f"({_ordinal(w_off_rank)} in NBA at {w_off_rtg:.1f} pts/100) and a "
                f"{_def_label(w_def_rank, n_teams)} defense "
                f"({_ordinal(w_def_rank)} in pts allowed at {w_def_rtg:.1f}), "
                f"giving them a net rating of {w_net:+.1f}."
            )

            # Opponent weakness (if notable)
            if l_off_rank > int(n_teams * 0.65):
                exp_sentences.append(
                    f"{loser_name}'s offense has struggled this season "
                    f"({_ordinal(l_off_rank)} in NBA at {l_off_rtg:.1f} pts/100)."
                )
            elif l_net < -3.0:
                exp_sentences.append(
                    f"{loser_name} has a negative net rating ({l_net:+.1f}), indicating an overall below-average team this season."
                )

            # Situational factors
            if b2b_home:
                exp_sentences.append(f"{home_name_str} is on a back-to-back (−5 ELO penalty).")
            if b2b_away:
                exp_sentences.append(f"{NBA_TEAM_NAMES.get(away, away)} is on a back-to-back (−5 ELO penalty).")
            if not neutral:
                exp_sentences.append(f"Home court adds approximately 100 ELO points for {home_name_str}.")

            explanation = " ".join(exp_sentences)

            home_inj_impact = injury_impacts.get(home, {})
            away_inj_impact = injury_impacts.get(away, {})

            predictions_list.append({
                "game_id": game_id,
                "game_time": game["game_time"],
                "status": game.get("status", ""),
                "is_future": bool(game.get("is_future", False)),
                "home_team": home,
                "away_team": away,
                "home_name": game.get("home_name", home),
                "away_name": game.get("away_name", away),
                "home_logo": game.get("home_logo", ""),
                "away_logo": game.get("away_logo", ""),
                "neutral": neutral,
                "home_score": game.get("home_score", 0),
                "away_score": game.get("away_score", 0),
                "predictions": {
                    "ensemble_prob": round(ensemble_prob, 4),
                    "logistic_prob": round(log_prob, 4) if log_prob is not None else None,
                    "elo_prob": round(elo_prob, 4),
                    "xgb_prob": round(xgb_prob, 4) if xgb_prob is not None else None,
                    "pyth_prob": round(pyth_prob, 4),
                    "eff_prob": round(eff_prob, 4),
                    "bayesian_prob": round(bayesian_prob, 4),
                },
                "market": {
                    "home_prob": market_home_prob,
                    "edge": market_edge,
                    "kelly_pct": kelly_pct,
                    "home_american": market_odds.get("home_american") if market_odds else None,
                    "away_american": market_odds.get("away_american") if market_odds else None,
                },
                "monte_carlo": mc_result,
                "adjustments": adj_dict,
                "h2h": h2h_cache.get((home, away), compute_h2h(home, away, historical_games)),
                "elo": {
                    "home": round(home_elo, 1),
                    "away": round(away_elo, 1),
                    "diff": round(home_elo - away_elo, 1),
                },
                "bayesian": {
                    "home_mu": bayes_result.get("mu_a", mu_home),
                    "home_sigma": bayes_result.get("sigma_a", sig_home),
                    "away_mu": bayes_result.get("mu_b", mu_away),
                    "away_sigma": bayes_result.get("sigma_b", sig_away),
                    "home_band": bayes_result.get("uncertainty_band_a", sig_home),
                    "away_band": bayes_result.get("uncertainty_band_b", sig_away),
                },
                "efficiency": {
                    "home_off_rating": efficiency_data.get(home, {}).get("offensive_rating", 110.0),
                    "home_def_rating": efficiency_data.get(home, {}).get("defensive_rating", 110.0),
                    "home_net_rating": efficiency_data.get(home, {}).get("net_rating", 0.0),
                    "away_off_rating": efficiency_data.get(away, {}).get("offensive_rating", 110.0),
                    "away_def_rating": efficiency_data.get(away, {}).get("defensive_rating", 110.0),
                    "away_net_rating": efficiency_data.get(away, {}).get("net_rating", 0.0),
                    "home_pace": efficiency_data.get(home, {}).get("pace", 100.0),
                    "away_pace": efficiency_data.get(away, {}).get("pace", 100.0),
                },
                "injuries": {
                    "home": [
                        {**p, "value_tier": nba_value_tier_label(nba_player_values.get(p.get("player", ""), 1.0))}
                        for p in injuries.get(home, [])[:5]
                    ],
                    "away": [
                        {**p, "value_tier": nba_value_tier_label(nba_player_values.get(p.get("player", ""), 1.0))}
                        for p in injuries.get(away, [])[:5]
                    ],
                },
                "injury_impact": {
                    "home_elo_penalty": home_inj_impact.get("elo_penalty", 0.0),
                    "away_elo_penalty": away_inj_impact.get("elo_penalty", 0.0),
                    "home_key_players_out": home_inj_impact.get("key_players_out", []),
                    "away_key_players_out": away_inj_impact.get("key_players_out", []),
                    "home_star_count": home_inj_impact.get("star_count", 0),
                    "away_star_count": away_inj_impact.get("star_count", 0),
                    "home_star_stack_multiplier": home_inj_impact.get("star_stack_multiplier", 1.0),
                    "away_star_stack_multiplier": away_inj_impact.get("star_stack_multiplier", 1.0),
                },
                "prediction_drivers": pred_drivers,
                "explanation": explanation,
            })

        except Exception as e:
            log.error(f"Error processing NBA game {game.get('game_id', '?')}: {e}")

    # Sort predictions: live → upcoming → complete, then by game_time within each group
    _STATUS_ORDER = {"STATUS_IN_PROGRESS": 0, "STATUS_SCHEDULED": 1, "STATUS_FINAL": 2}
    predictions_list.sort(
        key=lambda g: (_STATUS_ORDER.get(g.get("status", ""), 3), g.get("game_time", ""))
    )

    # ── 10. Build NBA leaderboard ────────────────────────────────────────────
    log.info("Building NBA leaderboard...")
    leaderboard_list = []

    for team in NBA_TEAMS:
        elo = elo_dict.get(team, 1500.0)
        bayes = bayesian_ratings.get(team, {"mu": elo, "sigma": 75.0})
        eff = efficiency_data.get(team, {})
        pyth = pyth_data.get(team, {}).get("pyth", 0.5)
        wins = standings.get(team, {}).get("wins", 0)
        losses = standings.get(team, {}).get("losses", 0)
        playoff_prob = season_sim.get(team, {}).get("playoff_prob", 0.53)
        champ_prob = season_sim.get(team, {}).get("sb_prob", 0.033)
        trend = nba_get_trend(game_history, team)
        team_inj = injury_impacts.get(team, {})

        leaderboard_list.append({
            "team": team,
            "team_name": NBA_TEAM_NAMES.get(team, team),
            "abbrev": team.lower(),
            "logo": f"https://a.espncdn.com/i/teamlogos/nba/500/{team.lower()}.png",
            "elo": round(elo, 1),
            "sigma": round(bayes.get("sigma", 75.0), 1),
            "mu": round(bayes.get("mu", elo), 1),
            "lower_band": round(bayes.get("mu", elo) - bayes.get("sigma", 75.0), 1),
            "upper_band": round(bayes.get("mu", elo) + bayes.get("sigma", 75.0), 1),
            "pyth": round(pyth, 4),
            "net_eff": round(eff.get("net_rating", 0.0), 2),
            "off_eff": round(eff.get("offensive_rating", 110.0), 1),
            "def_eff": round(eff.get("defensive_rating", 110.0), 1),
            "net_rating": round(eff.get("net_rating", 0.0), 2),
            "offensive_rating": round(eff.get("offensive_rating", 110.0), 1),
            "defensive_rating": round(eff.get("defensive_rating", 110.0), 1),
            "pace": round(eff.get("pace", 100.0), 1),
            "wins": wins,
            "losses": losses,
            "ties": 0,
            "playoff_prob": round(playoff_prob, 4),
            "sb_prob": round(champ_prob, 4),
            "champ_prob": round(champ_prob, 4),
            "trend": trend,
            "streak_type": compute_streak(game_history, team)["type"],
            "streak_count": compute_streak(game_history, team)["count"],
            "injury_elo_penalty": team_inj.get("elo_penalty", 0.0),
            "injury_impact_score": team_inj.get("impact_score", 0.0),
            "injury_players_count": team_inj.get("total_players", 0),
        })

    leaderboard_list.sort(key=lambda x: x["elo"], reverse=True)

    # ── Issue 7: Validate predictions before writing ──────────────────────
    if predictions_list:
        _eff_probs = [g.get("efficiency_prob") for g in predictions_list
                      if g.get("efficiency_prob") is not None]
        if _eff_probs and max(_eff_probs) - min(_eff_probs) < 0.01:
            msg = (f"All {len(_eff_probs)} efficiency_prob values are flat ({_eff_probs[0]:.4f}) "
                   "— efficiency sub-model is still outputting 50/50. "
                   "Refusing to overwrite nba_predictions.json with bad data.")
            log.error(msg)
            write_nba_abort_log(reason=msg, now_utc=now_utc,
                                counts={"flat_eff_probs": len(_eff_probs),
                                        "n_predictions": len(predictions_list)})
            return

    # ── 11. Write output files ───────────────────────────────────────────────
    log.info("Writing NBA JSON output files...")

    predictions_json = {
        "updated": now_utc,
        "season": season_year,
        "league": "nba",
        "is_offseason": is_offseason,
        "games": predictions_list,
    }

    leaderboard_json = {
        "updated": now_utc,
        "season": season_year,
        "league": "nba",
        "teams": leaderboard_list,
    }

    model_metrics_json = {
        "updated": now_utc,
        "league": "nba",
        "log_loss": model_metrics.get("log_loss"),
        "brier_score": model_metrics.get("brier_score"),
        "auc": model_metrics.get("auc"),
        "n_training_games": len(historical_games),
        "xgboost_available": xgb_model is not None,
        "calibration_buckets": [],
        "historical_accuracy": [],
    }

    (DATA_DIR / "nba_predictions.json").write_text(
        json.dumps(predictions_json, indent=2, default=str)
    )
    (DATA_DIR / "nba_leaderboard.json").write_text(
        json.dumps(leaderboard_json, indent=2, default=str)
    )
    (DATA_DIR / "nba_model_metrics.json").write_text(
        json.dumps(model_metrics_json, indent=2, default=str)
    )

    log.info("=== NBA Update complete ===")
    log.info(f"  Games predicted: {len(predictions_list)}")
    log.info(f"  Teams in leaderboard: {len(leaderboard_list)}")
    log.info(f"  Historical games used: {len(historical_games)}")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
