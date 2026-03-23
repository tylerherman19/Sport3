"""
data_fetcher.py — All external data fetching for Sport3.
Covers NFL (ESPN) and NBA (ESPN + cdn.nba.com) endpoints.
No math/model logic; only HTTP requests and raw parsing.

NBA endpoints (all stable, no auth required):
  - ESPN scoreboard: site.api.espn.com (primary)
  - ESPN game summary (boxscore): site.api.espn.com/apis/site/v2/sports/basketball/nba/summary
    Used by fetch_nba_depth_charts_espn() to extract per-player starter flags.
    Iterates boxscore.players -> statistics -> athletes, checks 'starter' boolean.
    Replaces the old team-roster/depthcharts endpoint approach.
  - cdn.nba.com/stats/leaguedashplayerstats (replaces stats.nba.com)
  - cdn.nba.com/static/json/staticData/standings.json

Fallback:
  - balldontlie.io (All-Star tier) used only when primary sources
    return None or raise errors. Endpoints used: /v1/games, /v1/players,
    /v1/teams. Stats and injury endpoints are NOT used (blocked on this tier).
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

log = logging.getLogger(__name__)

ESPN_NFL_BASE      = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ESPN_NFL_STANDINGS = "https://site.web.api.espn.com/apis/v2/sports/football/nfl/standings"
ESPN_NBA_BASE      = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_NBA_WEB_BASE  = "https://site.web.api.espn.com/apis/v2/sports/basketball/nba"
ESPN_NBA_STANDINGS = "https://site.web.api.espn.com/apis/v2/sports/basketball/nba/standings"
CDN_NBA_STATS      = "https://cdn.nba.com/stats"
CDN_NBA_STATIC     = "https://cdn.nba.com/static/json"
FTE_URL            = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nfl-elo/nfl_elo.csv"
ODDS_BASE_NFL      = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
ODDS_BASE_NBA      = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"

# ---- balldontlie.io (All-Star tier) ----
# Used as fallback only when ESPN / cdn.nba.com primary sources fail.
# Permitted endpoints: /v1/games, /v1/players, /v1/teams
# Blocked (do not call): /v1/stats, /v1/injuries
BDL_BASE    = "https://api.balldontlie.io/v1"
BDL_API_KEY = "3f8c3073-796d-4226-a8dc-4784afb14287"
_BDL_HEADERS = {"Authorization": BDL_API_KEY}

# Mapping from balldontlie team abbreviation → internal abbreviation
_BDL_NBA_ABBREV = {
    "GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS",
    "OKC": "OKC", "BKN": "BKN", "WSH": "WAS", "CHA": "CHA",
    "PHX": "PHX", "LAL": "LAL", "LAC": "LAC",
}
_BDL_NFL_ABBREV = {
    "WSH": "WAS", "JAC": "JAX", "LV": "LV", "LA": "LAR",
}

def _bdl_nba_abbrev(a):
    a = a.upper().strip()
    return _BDL_NBA_ABBREV.get(a, a)

def _bdl_nfl_abbrev(a):
    a = a.upper().strip()
    return _BDL_NFL_ABBREV.get(a, a)

_CDN_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": "Mozilla/5.0 (compatible; Sport3Bot/2.0)",
}

_ESPN_NFL_TO_ABBREV = {
    "WSH": "WAS", "JAC": "JAX", "LVR": "LV", "LA": "LAR", "LAR": "LAR", "LAC": "LAC",
}
_ESPN_NBA_TO_ABBREV = {
    "GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS",
    "OKC": "OKC", "BKN": "BKN", "WSH": "WAS", "CHA": "CHA", "PHX": "PHX", "LAL": "LAL", "LAC": "LAC",
}
ESPN_NBA_ID_TO_ABBREV = {}


def nfl_abbrev_norm(abbrev):
    return _ESPN_NFL_TO_ABBREV.get(abbrev.upper(), abbrev.upper())


def nba_abbrev_norm(abbrev):
    a = abbrev.upper().strip()
    return _ESPN_NBA_TO_ABBREV.get(a, a)


def safe_get(url, params=None, timeout=30, headers=None):
    try:
        r = requests.get(url, params=params, timeout=timeout, headers=headers or {})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return None
