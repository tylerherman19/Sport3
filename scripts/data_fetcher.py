"""
Data Fetcher Module - Sport3 Refactor
Consolidated fetching for NFL and NBA data from ESPN and cdn.nba.com endpoints.
Replaces stats.nba.com (nba_api) with stable ESPN and CDN endpoints.
"""

import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
import re
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────────────────

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ESPN_NFL_STANDINGS_URL = "https://site.web.api.espn.com/apis/v2/sports/football/nfl/standings"
FTE_URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nfl-elo/nfl_elo.csv"
ODDS_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"

# Stable ESPN NBA endpoints (replaces stats.nba.com)
ESPN_NBA_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_NBA_WEB_BASE = "https://site.web.api.espn.com/apis/v2/sports/basketball/nba"
ESPN_NBA_STANDINGS_URL = "https://site.web.api.espn.com/apis/v2/sports/basketball/nba/standings"
NBA_ODDS_BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
# cdn.nba.com endpoints (stable alternative to stats.nba.com)
NBA_CDN_BASE = "https://cdn.nba.com/static/json"
NBA_CDN_SCOREBOARD = f"{NBA_CDN_BASE}/liveData/scoreboard/todaysScoreboard_00.json"

NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
]

NFL_TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders"
}

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

ESPN_TO_ABBREV = {"WSH": "WAS", "JAC": "JAX", "LVR": "LV", "LA": "LAR", "LAR": "LAR", "LAC": "LAC"}
ESPN_NBA_TO_ABBREV = {
    "GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS",
    "OKC": "OKC", "BKN": "BKN", "WSH": "WAS", "CHA": "CHA", "PHX": "PHX", "LAL": "LAL", "LAC": "LAC",
}

# ─── Helper Functions ──────────────────────────────────────────────────────────────────────

def safe_get(url, params=None, timeout=30):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return None


def abbrev_norm(abbrev):
    """NFL abbreviation normalization."""
    return ESPN_TO_ABBREV.get(abbrev.upper(), abbrev.upper())


def nba_abbrev_norm(abbrev):
    """NBA abbreviation normalization."""
    a = abbrev.upper().strip()
    return ESPN_NBA_TO_ABBREV.get(a, a)


def nfl_days_since_last_game(game_history, team, before_date):
    """Return days since team's last completed game strictly before before_date."""
    from datetime import date as date_cls
    dates = [g["date"] for g in game_history.get(team, []) if g.get("date", "") < str(before_date)]
    if not dates:
        return 7
    try:
        last = max(dates)
        return max(0, (before_date - datetime.strptime(last, "%Y-%m-%d").date()).days)
    except ValueError:
        return 7


def normalize_player_name(name: str) -> str:
    """Normalize player name: strip Jr/Sr/II/III/IV suffixes and normalize apostrophes."""
    n = name.lower().strip()
    n = n.replace("\u2019", "'").replace("`", "'")
    n = re.sub(r"\s+\b(jr\.?|sr\.?|ii|iii|iv)\b\.?$", "", n).strip()
    return n


def days_since_last_game(game_history, team, today):
    """Return days since NBA team's most recent completed game."""
    from datetime import date as date_cls
    hist = game_history.get(team, [])
    dates = [g["date"] for g in hist if g.get("date") and g["date"] <= str(today)]
    if not dates:
        return 7
    last_date_str = max(dates)
    try:
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
        return max(0, (today - last_date).days)
    except ValueError:
        return 7


# ─── NFL Data Fetching ─────────────────────────────────────────────────────────────────────

def download_fte_data():
    """Download FiveThirtyEight NFL ELO data."""
    log.info("Downloading FiveThirtyEight NFL ELO data...")
    try:
        df = pd.read_csv(FTE_URL)
        log.info(f"FTE data: {len(df)} rows, seasons {df['season'].min()}–{df['season'].max()}")
        return df
    except Exception as e:
        log.error(f"Failed to download FTE data: {e}")
        return pd.DataFrame()


def parse_espn_events(data, default_week=0):
    """Parse ESPN events list into game dicts."""
    games = []
    week_num = data.get("week", {}).get("number", default_week)
    season_year = data.get("season", {}).get("year", datetime.now().year)
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
            status = event.get("status", {}).get("type", {}).get("name", "")
            game = {
                "game_id": event["id"],
                "game_time": event.get("date", ""),
                "status": status,
                "week": week_num,
                "season": season_year,
                "home_team": home_abbrev,
                "away_team": away_abbrev,
                "home_name": home["team"].get("displayName", home_abbrev),
                "away_name": away["team"].get("displayName", away_abbrev),
                "home_score": int(home.get("score", 0) or 0),
                "away_score": int(away.get("score", 0) or 0),
                "home_logo": f"https://a.espncdn.com/i/teamlogos/nfl/500/{home['team']['abbreviation'].lower()}.png",
                "away_logo": f"https://a.espncdn.com/i/teamlogos/nfl/500/{away['team']['abbreviation'].lower()}.png",
                "neutral": int(comp.get("neutralSite", False)),
            }
            games.append(game)
        except (KeyError, IndexError) as e:
            log.debug(f"Error parsing game: {e}")
    return games


def fetch_espn_scoreboard():
    """Fetch ESPN NFL scoreboard (current week)."""
    log.info("Fetching ESPN scoreboard (current week)...")
    data = safe_get(f"{ESPN_BASE}/scoreboard")
    if not data:
        return [], 0
    games = parse_espn_events(data)
    current_week = data.get("week", {}).get("number", 0)
    log.info(f"Found {len(games)} scoreboard games (week {current_week})")
    return games, current_week


def fetch_espn_future_games(current_week, season_year, weeks_ahead=3):
    """Fetch upcoming scheduled games for the next N weeks."""
    future_games = []
    seen_ids = set()
    fetch_failures = 0
    for offset in range(1, weeks_ahead + 1):
        week = current_week + offset
        if week > 22:
            break
        url = f"{ESPN_BASE}/scoreboard?seasontype=2&week={week}"
        data = safe_get(url)
        if not data:
            url = f"{ESPN_BASE}/scoreboard?seasontype=3&week={week - 18}"
            data = safe_get(url)
        if not data:
            fetch_failures += 1
            continue
        games = parse_espn_events(data, default_week=week)
        for g in games:
            if g["game_id"] not in seen_ids:
                seen_ids.add(g["game_id"])
                g["is_future"] = True
                future_games.append(g)
    if fetch_failures > 0:
        log.warning(f"NFL future-games fetch: skipped {fetch_failures} week(s) due to fetch failures")
    log.info(f"Found {len(future_games)} future scheduled games")
    return future_games


def fetch_espn_standings():
    """Fetch ESPN NFL standings."""
    log.info("Fetching ESPN standings...")
    data = safe_get(ESPN_NFL_STANDINGS_URL)
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
                stats = {s["name"]: s.get("value", 0) for s in entry.get("stats", [])}
                wins = int(stats.get("wins", 0))
                losses = int(stats.get("losses", 0))
                standings[team_abbrev] = {
                    "wins": wins, "losses": losses, "ties": int(stats.get("ties", 0)),
                    "win_pct": float(stats.get("winPercent", 0)),
                    "points_for": float(stats.get("pointsFor", 0)),
                    "points_against": float(stats.get("pointsAgainst", 0)),
                    "streak": stats.get("streak", 0), "games_played": wins + losses,
                }
    except Exception as e:
        log.warning(f"Error parsing standings: {e}")
    log.info(f"Standings for {len(standings)} teams")
    return standings


def fetch_espn_injuries():
    """Fetch ESPN NFL injuries."""
    log.info("Fetching ESPN injuries...")
    data = safe_get(f"{ESPN_BASE}/injuries")
    if not data:
        return {}
    injuries = {}
    try:
        for team_entry in data.get("injuries", []):
            raw_abbrev = team_entry.get("abbreviation", "") or team_entry.get("team", {}).get("abbreviation", "")
            team_abbrev = abbrev_norm(raw_abbrev) if raw_abbrev else ""
            for item in team_entry.get("injuries", []):
                item_abbrev = team_abbrev
                if not item_abbrev:
                    raw_fallback = item.get("athlete", {}).get("team", {}).get("abbreviation", "")
                    item_abbrev = abbrev_norm(raw_fallback) if raw_fallback else ""
                if not item_abbrev:
                    continue
                if item_abbrev not in injuries:
                    injuries[item_abbrev] = []
                status_raw = item.get("status", "")
                if isinstance(status_raw, dict):
                    status = status_raw.get("name", status_raw.get("abbreviation", ""))
                else:
                    status = str(status_raw) if status_raw else ""
                injuries[item_abbrev].append({
                    "player": item.get("athlete", {}).get("displayName", ""),
                    "status": status,
                    "position": item.get("athlete", {}).get("position", {}).get("abbreviation", ""),
                    "injury_description": item.get("longComment", item.get("shortComment", status)),
                })
    except Exception as e:
        log.warning(f"Error parsing injuries: {e}")
    return injuries


def fetch_espn_depth_charts():
    """Fetch NFL depth chart positions for all teams from ESPN."""
    log.info("Fetching ESPN NFL depth charts for player value scoring...")
    player_depth = {}
    try:
        teams_data = safe_get(f"{ESPN_BASE}/teams")
        if not teams_data:
            return {}
        teams_list = teams_data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
        for team_entry in teams_list:
            team_id = team_entry.get("team", {}).get("id", "")
            if not team_id:
                continue
            depth_data = safe_get(f"{ESPN_BASE}/teams/{team_id}/depthcharts")
            if not depth_data:
                continue
            for pos_group in depth_data.get("positionGroups", []):
                for position in pos_group.get("positions", []):
                    for athlete_entry in position.get("athletes", []):
                        name = athlete_entry.get("athlete", {}).get("displayName", "")
                        depth_pos = int(athlete_entry.get("rank") or athlete_entry.get("slot", 99))
                        if not name:
                            continue
                        for key in [name, normalize_player_name(name)]:
                            if key and (key not in player_depth or depth_pos < player_depth[key]):
                                player_depth[key] = depth_pos
    except Exception as e:
        log.warning(f"Error fetching depth charts: {e}")
    log.info(f"Depth chart loaded: {len(player_depth)} player entries")
    return player_depth


def fetch_espn_completed_games(season_year):
    """Fetch completed NFL games from ESPN for the given season year."""
    log.info(f"Fetching ESPN completed games for {season_year} season...")
    completed = []
    seen_ids = set()
    fetch_failures = 0
    season_types = [(2, range(1, 23)), (3, range(1, 6))]
    for season_type, weeks in season_types:
        for week in weeks:
            params = {"seasontype": season_type, "week": week, "dates": season_year, "limit": 100}
            data = safe_get(f"{ESPN_BASE}/scoreboard", params=params)
            if not data:
                fetch_failures += 1
                continue
            for event in data.get("events", []):
                try:
                    event_id = event.get("id", "")
                    if event_id in seen_ids:
                        continue
                    seen_ids.add(event_id)
                    if event.get("status", {}).get("type", {}).get("name", "") != "STATUS_FINAL":
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
                    completed.append({
                        "date": event.get("date", "")[:10], "season": season_year,
                        "team1": abbrev_norm(home["team"]["abbreviation"]),
                        "team2": abbrev_norm(away["team"]["abbreviation"]),
                        "score1": home_score, "score2": away_score,
                        "neutral": int(comp.get("neutralSite", False)),
                    })
                except (KeyError, IndexError, TypeError) as e:
                    log.debug(f"Error parsing ESPN game: {e}")
    if fetch_failures > 0:
        log.warning(f"NFL completed-games fetch: skipped {fetch_failures} week(s)")
    log.info(f"Found {len(completed)} completed ESPN games for {season_year}")
    return completed


def fetch_betting_odds(api_key, sport="americanfootball_nfl"):
    """Fetch NFL/NBA odds from The Odds API."""
    if not api_key:
        log.info(f"No ODDS_API_KEY set, skipping {sport} betting lines")
        return {}
    log.info(f"Fetching {sport} odds from The Odds API...")
    params = {"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "american", "sport": sport}
    odds_base = ODDS_BASE if sport == "americanfootball_nfl" else NBA_ODDS_BASE
    data = safe_get(odds_base, params=params)
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
            home_odds_list, away_odds_list = [], []
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
                # Simple vig removal
                raw_home = 100 / (avg_home + 100) if avg_home > 0 else (-avg_home) / (-avg_home + 100)
                raw_away = 100 / (avg_away + 100) if avg_away > 0 else (-avg_away) / (-avg_away + 100)
                total = raw_home + raw_away
                clean_home, clean_away = raw_home / total, raw_away / total
                game_key = f"{away_team}_at_{home_team}"
                odds_map[game_key] = {
                    "home_prob": round(clean_home, 4), "away_prob": round(clean_away, 4),
                    "home_american": avg_home, "away_american": avg_away,
                    "home_team_name": home_team, "away_team_name": away_team,
                }
        except Exception as e:
            log.debug(f"Error parsing odds entry: {e}")
    log.info(f"Got odds for {len(odds_map)} games")
    return odds_map


# ─── NBA Data Fetching (ESPN endpoints, replacing stats.nba.com) ───────────────────────────

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
            home_abbrev = nba_abbrev_norm(home["team"]["abbreviation"])
            away_abbrev = nba_abbrev_norm(away["team"]["abbreviation"])
            status_obj = event.get("status", {})
            game = {
                "game_id": event["id"],
                "game_time": event.get("date", ""),
                "status": status_obj.get("type", {}).get("name", ""),
                "period": status_obj.get("period", 0),
                "display_clock": status_obj.get("displayClock", ""),
                "status_detail": status_obj.get("type", {}).get("detail", ""),
                "home_team": home_abbrev, "away_team": away_abbrev,
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
    """Fetch NBA scoreboard for today + past 5 days."""
    log.info("Fetching NBA scoreboard (today + past 5 days)...")
    games = []
    seen_ids = set()
    for delta in [0, -1, -2, -3, -4, -5]:
        date_str = (datetime.now(timezone.utc) + timedelta(days=delta)).strftime('%Y%m%d')
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
    for offset in range(1, days_ahead + 1):
        date_str = (datetime.now(timezone.utc) + timedelta(days=offset)).strftime('%Y%m%d')
        data = safe_get(f"{ESPN_NBA_WEB_BASE}/scoreboard?dates={date_str}&limit=20&seasontype=2")
        if not data or not data.get("events"):
            data = safe_get(f"{ESPN_NBA_BASE}/scoreboard?dates={date_str}&limit=20&seasontype=2")
        if not data:
            continue
        for g in parse_nba_events(data):
            if g["game_id"] not in seen_ids:
                seen_ids.add(g["game_id"])
                g["is_future"] = True
                future_games.append(g)
    log.info(f"Found {len(future_games)} future NBA games")
    return future_games


def fetch_nba_standings():
    """Fetch NBA standings from ESPN."""
    log.info("Fetching NBA standings...")
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
                team_abbrev = nba_abbrev_norm(entry["team"]["abbreviation"])
                stats = {s["name"]: s.get("value", 0) for s in entry.get("stats", [])}
                wins = int(stats.get("wins", 0))
                losses = int(stats.get("losses", 0))
                off_rating = float(stats.get("avgPointsFor", 110.0))
                def_rating = float(stats.get("avgPointsAgainst", 110.0))
                standings[team_abbrev] = {
                    "wins": wins, "losses": losses, "win_pct": float(stats.get("winPercent", 0)),
                    "points_for": float(stats.get("pointsFor", 0)),
                    "points_against": float(stats.get("pointsAgainst", 0)),
                    "games_played": wins + losses, "offensive_rating": off_rating,
                    "defensive_rating": def_rating, "net_rating": off_rating - def_rating,
                    "pace": 100.0, "assist_turnover_ratio": 1.8, "rebound_rate": 0.5,
                    "three_point_rate": 0.35, "free_throw_rate": 0.20, "streak": stats.get("streak", 0),
                }
    except Exception as e:
        log.warning(f"Error parsing NBA standings: {e}")
    log.info(f"NBA standings for {len(standings)} teams")
    return standings


def fetch_nba_injuries():
    """Fetch NBA injuries from ESPN."""
    log.info("Fetching NBA injuries...")
    data = safe_get(f"{ESPN_NBA_BASE}/injuries")
    if not data:
        return {}
    injuries = {}
    try:
        for team_entry in data.get("injuries", []):
            raw_abbrev = team_entry.get("abbreviation", "") or team_entry.get("team", {}).get("abbreviation", "")
            team_abbrev = nba_abbrev_norm(raw_abbrev) if raw_abbrev else ""
            for item in team_entry.get("injuries", []):
                item_abbrev = team_abbrev
                if not item_abbrev:
                    raw_fallback = item.get("athlete", {}).get("team", {}).get("abbreviation", "")
                    item_abbrev = nba_abbrev_norm(raw_fallback) if raw_fallback else ""
                if not item_abbrev:
                    continue
                if item_abbrev not in injuries:
                    injuries[item_abbrev] = []
                athlete = item.get("athlete", {})
                status_raw = item.get("status", "")
                status = status_raw.get("name", status_raw.get("abbreviation", "")) if isinstance(status_raw, dict) else str(status_raw) if status_raw else ""
                injuries[item_abbrev].append({
                    "player": athlete.get("displayName", ""),
                    "status": status,
                    "position": athlete.get("position", {}).get("abbreviation", ""),
                    "injury_description": item.get("longComment", item.get("shortComment", status)),
                })
    except Exception as e:
        log.warning(f"Error parsing NBA injuries: {e}")
    total = sum(len(v) for v in injuries.values())
    log.info(f"NBA injuries: {total} players across {len(injuries)} teams")
    return injuries


def fetch_nba_season_games_espn(seasons=None):
    """
    Fetch completed NBA games by iterating day-by-day via ESPN scoreboard.
    Replaces nba_api LeagueGameLog which hits the unreachable stats.nba.com.
    """
    from datetime import date as date_cls
    if seasons is None:
        current_year = datetime.now(timezone.utc).year
        current_month = datetime.now(timezone.utc).month
        end_year = current_year + 1 if current_month >= 10 else current_year
        seasons = [end_year - 2, end_year - 1, end_year]

    def _fetch_nba_day(date_str):
        url = f"{ESPN_NBA_BASE}/scoreboard?dates={date_str}&limit=20"
        return date_str, safe_get(url)

    log.info(f"Fetching NBA season games for {seasons}...")
    all_games = []
    seen_ids = set()
    today = date_cls.today()
    fetch_failures = 0

    for season_year in seasons:
        season_start = date_cls(season_year - 1, 10, 1)
        season_end = min(today, date_cls(season_year, 6, 30))
        if season_start > today:
            continue
        is_current = (season_year == seasons[-1])
        stride_days = 1 if is_current else 7
        date_list = []
        current = season_start
        while current <= season_end:
            date_list.append(current.strftime('%Y%m%d'))
            current += timedelta(days=stride_days)

        results_by_date = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(_fetch_nba_day, d): d for d in date_list}
            for future in as_completed(futures):
                date_str, data = future.result()
                results_by_date[date_str] = data

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
                    if event.get("status", {}).get("type", {}).get("name", "") != "STATUS_FINAL":
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
                    all_games.append({
                        "date": event.get("date", "")[:10], "season": season_year,
                        "team1": nba_abbrev_norm(home["team"]["abbreviation"]),
                        "team2": nba_abbrev_norm(away["team"]["abbreviation"]),
                        "score1": home_score, "score2": away_score,
                        "neutral": int(comp.get("neutralSite", False)),
                    })
                except (KeyError, IndexError, TypeError) as e:
                    log.debug(f"Error parsing NBA game on {date_str}: {e}")

    if fetch_failures > 0:
        log.warning(f"NBA season fetch: skipped {fetch_failures} date(s) due to fetch failures")
    log.info(f"Fetched {len(all_games)} NBA historical games")
    return all_games


def fetch_nba_depth_charts():
    """Fetch live NBA depth chart positions from ESPN (replaces nba_api roster endpoints)."""
    log.info("Fetching ESPN NBA depth charts for player value scoring...")
    player_depth = {}
    try:
        teams_data = safe_get(f"{ESPN_NBA_BASE}/teams")
        if not teams_data:
            return {}
        teams_list = teams_data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
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


def fetch_nba_player_ppg():
    """Fetch per-player season PPG from ESPN for all NBA teams."""
    log.info("Fetching NBA player PPG stats for usage-based value scoring...")
    player_ppg = {}
    try:
        teams_data = safe_get(f"{ESPN_NBA_BASE}/teams")
        if not teams_data:
            return {}
        teams_list = teams_data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
        for team_entry in teams_list:
            team_id = team_entry.get("team", {}).get("id", "")
            if not team_id:
                continue
            roster_data = safe_get(f"{ESPN_NBA_BASE}/teams/{team_id}/athletes", params={"enable": "stats"})
            if not roster_data:
                continue
            for athlete in roster_data.get("athletes", []):
                name = athlete.get("displayName", "")
                if not name:
                    continue
                stats_list = athlete.get("statistics", {}).get("splits", {})
                if not stats_list:
                    stats_list = athlete.get("stats", [])
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


# ─── cdn.nba.com Endpoints (stable alternative to stats.nba.com) ────────────────────────

def fetch_cdn_nba_scoreboard(date_str=""):
    """
    Fetch today's NBA scoreboard from cdn.nba.com.
    Endpoint: cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json
    Stable replacement for stats.nba.com Scoreboard endpoint.
    """
    data = safe_get(NBA_CDN_SCOREBOARD)
    if not data:
        return []
    games = []
    for game in data.get("scoreboard", {}).get("games", []):
        try:
            home_team = game.get("homeTeam", {})
            away_team = game.get("awayTeam", {})
            games.append({
                "game_id": game.get("gameId", ""),
                "game_date": game.get("gameTimeUTC", "")[:10],
                "home_team": nba_abbrev_norm(home_team.get("teamTricode", "")),
                "away_team": nba_abbrev_norm(away_team.get("teamTricode", "")),
                "home_score": home_team.get("score", 0),
                "away_score": away_team.get("score", 0),
                "status": game.get("gameStatus", 1),  # 1=scheduled, 2=live, 3=final
                "period": game.get("period", 0),
            })
        except Exception as e:
            log.debug(f"cdn.nba.com game parse error: {e}")
    log.info(f"cdn.nba.com scoreboard: found {len(games)} games")
    return games


def fetch_cdn_nba_team_stats(season="2024-25"):
    """
    Fetch team stats from cdn.nba.com.
    Stable replacement for nba_api LeagueDashTeamStats (stats.nba.com).
    Endpoint: cdn.nba.com/static/json/stats/leagueTeamStats/...
    """
    season_compact = season.replace("-", "")
    url = f"{NBA_CDN_BASE}/stats/leagueTeamStats/00_{season_compact}_02_RS_team_traditional.json"
    data = safe_get(url)
    if not data:
        return {}
    team_stats = {}
    for team in data.get("leagueTeamStats", []):
        try:
            abbrev = team.get("teamTricode", "")
            if not abbrev:
                continue
            team_stats[nba_abbrev_norm(abbrev)] = {
                "games": team.get("gamesPlayed", 0),
                "wins": team.get("wins", 0),
                "losses": team.get("losses", 0),
                "win_pct": team.get("winPct", 0.0),
                "points": team.get("points", 0.0),
                "opp_points": team.get("oppPoints", 0.0),
                "field_goals_made": team.get("fieldGoalsMade", 0),
                "field_goals_attempted": team.get("fieldGoalsAttempted", 0),
                "three_pointers_made": team.get("threePointersMade", 0),
                "free_throws_made": team.get("freeThrowsMade", 0),
                "rebounds_offensive": team.get("reboundsOffensive", 0),
                "rebounds_defensive": team.get("reboundsDefensive", 0),
                "assists": team.get("assists", 0),
                "turnovers": team.get("turnovers", 0),
            }
        except Exception as e:
            log.debug(f"cdn.nba.com team stats parse error: {e}")
    log.info(f"cdn.nba.com team stats: fetched {len(team_stats)} teams")
    return team_stats
