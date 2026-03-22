"""
data_fetcher.py — All external data fetching for Sport3.
Covers NFL (ESPN) and NBA (ESPN + cdn.nba.com) endpoints.
No math/model logic; only HTTP requests and raw parsing.

NBA endpoints (all stable, no auth required):
  - ESPN scoreboard: site.api.espn.com (primary)
  - cdn.nba.com/stats/leaguedashplayerstats (replaces stats.nba.com)
  - cdn.nba.com/static/json/staticData/standings.json
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


# ---- FTE ----

def fetch_fte_data():
    import pandas as pd
    log.info("Downloading FiveThirtyEight NFL ELO data...")
    try:
        df = pd.read_csv(FTE_URL)
        log.info(f"FTE data: {len(df)} rows, seasons {df['season'].min()}–{df['season'].max()}")
        return df
    except Exception as e:
        log.error(f"Failed to download FTE data: {e}")
        return pd.DataFrame()


# ---- NFL ESPN ----

def _parse_nfl_events(data, default_week=0):
    games = []
    week_num    = data.get("week", {}).get("number", default_week)
    season_year = data.get("season", {}).get("year", datetime.now().year)
    for event in data.get("events", []):
        try:
            comp  = event["competitions"][0]
            comps = comp["competitors"]
            home  = next((c for c in comps if c["homeAway"] == "home"), None)
            away  = next((c for c in comps if c["homeAway"] == "away"), None)
            if not home or not away:
                continue
            ha = nfl_abbrev_norm(home["team"]["abbreviation"])
            aa = nfl_abbrev_norm(away["team"]["abbreviation"])
            games.append({
                "game_id": event["id"], "game_time": event.get("date", ""),
                "status": event.get("status", {}).get("type", {}).get("name", ""),
                "week": week_num, "season": season_year,
                "home_team": ha, "away_team": aa,
                "home_name": home["team"].get("displayName", ha),
                "away_name": away["team"].get("displayName", aa),
                "home_score": int(home.get("score", 0) or 0),
                "away_score": int(away.get("score", 0) or 0),
                "home_logo": f"https://a.espncdn.com/i/teamlogos/nfl/500/{home['team']['abbreviation'].lower()}.png",
                "away_logo": f"https://a.espncdn.com/i/teamlogos/nfl/500/{away['team']['abbreviation'].lower()}.png",
                "neutral": int(comp.get("neutralSite", False)),
            })
        except (KeyError, IndexError) as e:
            log.debug(f"Error parsing NFL game: {e}")
    return games


def fetch_nfl_scoreboard():
    log.info("Fetching NFL scoreboard...")
    data = safe_get(f"{ESPN_NFL_BASE}/scoreboard")
    if not data:
        return [], 0
    games = _parse_nfl_events(data)
    week  = data.get("week", {}).get("number", 0)
    log.info(f"Found {len(games)} scoreboard games (week {week})")
    return games, week


def fetch_nfl_future_games(current_week, season_year, weeks_ahead=3):
    future = []; seen = set(); fails = 0
    for offset in range(1, weeks_ahead + 1):
        week = current_week + offset
        if week > 22:
            break
        data = safe_get(f"{ESPN_NFL_BASE}/scoreboard?seasontype=2&week={week}")
        if not data:
            data = safe_get(f"{ESPN_NFL_BASE}/scoreboard?seasontype=3&week={week-18}")
        if not data:
            fails += 1; continue
        for g in _parse_nfl_events(data, default_week=week):
            if g["game_id"] not in seen:
                seen.add(g["game_id"]); g["is_future"] = True; future.append(g)
    if fails:
        log.warning(f"NFL future-games: skipped {fails} week(s)")
    log.info(f"Found {len(future)} future NFL games")
    return future


def fetch_nfl_completed_games(season_year):
    log.info(f"Fetching ESPN completed NFL games for {season_year}...")
    completed = []; seen = set(); fails = 0
    for stype, weeks in [(2, range(1, 23)), (3, range(1, 6))]:
        for week in weeks:
            data = safe_get(f"{ESPN_NFL_BASE}/scoreboard",
                            params={"seasontype": stype, "week": week, "dates": season_year, "limit": 100})
            if not data:
                fails += 1; continue
            for event in data.get("events", []):
                try:
                    eid = event.get("id", "")
                    if eid in seen: continue
                    seen.add(eid)
                    if event.get("status", {}).get("type", {}).get("name", "") != "STATUS_FINAL":
                        continue
                    comp  = event["competitions"][0]
                    comps = comp["competitors"]
                    home  = next((c for c in comps if c["homeAway"] == "home"), None)
                    away  = next((c for c in comps if c["homeAway"] == "away"), None)
                    if not home or not away: continue
                    hs = int(home.get("score", 0) or 0)
                    as_ = int(away.get("score", 0) or 0)
                    if hs == 0 and as_ == 0: continue
                    completed.append({
                        "date": event.get("date", "")[:10], "season": season_year,
                        "team1": nfl_abbrev_norm(home["team"]["abbreviation"]),
                        "team2": nfl_abbrev_norm(away["team"]["abbreviation"]),
                        "score1": hs, "score2": as_,
                        "neutral": int(comp.get("neutralSite", False)),
                    })
                except (KeyError, IndexError, TypeError) as e:
                    log.debug(f"Error parsing NFL game: {e}")
    if fails: log.warning(f"NFL completed-games: skipped {fails} week(s)")
    log.info(f"Found {len(completed)} completed NFL games for {season_year}")
    return completed


def fetch_nfl_standings():
    log.info("Fetching NFL standings...")
    data = safe_get(ESPN_NFL_STANDINGS)
    if not data: return {}
    standings = {}
    try:
        for group in data.get("children", []):
            entries = group.get("standings", {}).get("entries", [])
            if not entries:
                for div in group.get("children", []):
                    entries = entries + div.get("standings", {}).get("entries", [])
            for entry in entries:
                abbrev = nfl_abbrev_norm(entry["team"]["abbreviation"])
                stats  = {s["name"]: s.get("value", 0) for s in entry.get("stats", [])}
                wins   = int(stats.get("wins", 0)); losses = int(stats.get("losses", 0))
                standings[abbrev] = {
                    "wins": wins, "losses": losses, "ties": int(stats.get("ties", 0)),
                    "win_pct": float(stats.get("winPercent", 0)),
                    "points_for": float(stats.get("pointsFor", 0)),
                    "points_against": float(stats.get("pointsAgainst", 0)),
                    "streak": stats.get("streak", 0), "games_played": wins + losses,
                }
    except Exception as e:
        log.warning(f"Error parsing NFL standings: {e}")
    log.info(f"NFL standings for {len(standings)} teams")
    return standings


def fetch_nfl_injuries():
    log.info("Fetching NFL injuries...")
    data = safe_get(f"{ESPN_NFL_BASE}/injuries")
    if not data: return {}
    injuries = {}
    try:
        for team_entry in data.get("injuries", []):
            raw    = team_entry.get("abbreviation", "") or team_entry.get("team", {}).get("abbreviation", "")
            abbrev = nfl_abbrev_norm(raw) if raw else ""
            for item in team_entry.get("injuries", []):
                ia = abbrev
                if not ia:
                    fb = item.get("athlete", {}).get("team", {}).get("abbreviation", "")
                    ia = nfl_abbrev_norm(fb) if fb else ""
                if not ia: continue
                injuries.setdefault(ia, [])
                sr = item.get("status", "")
                st = sr.get("name", sr.get("abbreviation", "")) if isinstance(sr, dict) else str(sr or "")
                injuries[ia].append({
                    "player": item.get("athlete", {}).get("displayName", ""),
                    "status": st,
                    "position": item.get("athlete", {}).get("position", {}).get("abbreviation", ""),
                    "injury_description": item.get("longComment", item.get("shortComment", st)),
                })
    except Exception as e:
        log.warning(f"Error parsing NFL injuries: {e}")
    return injuries


def _nfl_normalize_name(name):
    n = name.lower().strip().replace("\u2019", "'").replace("`", "'")
    n = re.sub(r"\s+\b(jr\.?|sr\.?|ii|iii|iv)\b\.?$", "", n).strip()
    return n


def fetch_nfl_depth_charts():
    log.info("Fetching NFL depth charts...")
    player_depth = {}
    try:
        teams_data = safe_get(f"{ESPN_NFL_BASE}/teams")
        if not teams_data: return {}
        teams_list = teams_data.get("sports",[{}])[0].get("leagues",[{}])[0].get("teams",[])
        for te in teams_list:
            team_id = te.get("team", {}).get("id", "")
            if not team_id: continue
            dd = safe_get(f"{ESPN_NFL_BASE}/teams/{team_id}/depthcharts")
            if not dd: continue
            for pg in dd.get("positionGroups", []):
                for pos in pg.get("positions", []):
                    for ae in pos.get("athletes", []):
                        name = ae.get("athlete", {}).get("displayName", "")
                        dp   = int(ae.get("rank") or ae.get("slot", 99))
                        if not name: continue
                        for key in [name, _nfl_normalize_name(name)]:
                            if key and (key not in player_depth or dp < player_depth[key]):
                                player_depth[key] = dp
    except Exception as e:
        log.warning(f"Error fetching NFL depth charts: {e}")
    log.info(f"NFL depth chart: {len(player_depth)} entries")
    return player_depth


def fetch_nfl_betting_odds(api_key):
    if not api_key:
        log.info("No ODDS_API_KEY, skipping NFL odds"); return {}
    log.info("Fetching NFL betting odds...")
    data = safe_get(ODDS_BASE_NFL, params={"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "american"})
    if not data: return {}
    import numpy as np
    from model.ensemble_model import american_to_prob, remove_vig
    odds_map = {}
    for game in data:
        try:
            ht = game.get("home_team", ""); at = game.get("away_team", "")
            hl, al = [], []
            for bm in game.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if mkt["key"] == "h2h":
                        for out in mkt.get("outcomes", []):
                            if out["name"] == ht: hl.append(float(out["price"]))
                            elif out["name"] == at: al.append(float(out["price"]))
            if hl and al:
                ah, aa = np.mean(hl), np.mean(al)
                ch, ca = remove_vig(american_to_prob(ah), american_to_prob(aa))
                odds_map[f"{at}_at_{ht}"] = {"home_prob": round(ch,4), "away_prob": round(ca,4),
                                              "home_american": ah, "away_american": aa,
                                              "home_team_name": ht, "away_team_name": at}
        except Exception as e:
            log.debug(f"Error parsing NFL odds: {e}")
    log.info(f"NFL odds for {len(odds_map)} games"); return odds_map


# ---- NBA ESPN / cdn.nba.com ----

def _parse_nba_events(data):
    games = []
    for event in data.get("events", []):
        try:
            comp  = event["competitions"][0]
            comps = comp["competitors"]
            home  = next((c for c in comps if c["homeAway"] == "home"), None)
            away  = next((c for c in comps if c["homeAway"] == "away"), None)
            if not home or not away: continue
            ha = nba_abbrev_norm(home["team"]["abbreviation"])
            aa = nba_abbrev_norm(away["team"]["abbreviation"])
            for tid, abbrev in [(home["team"].get("id"), ha), (away["team"].get("id"), aa)]:
                if tid: ESPN_NBA_ID_TO_ABBREV[tid] = abbrev
            so = event.get("status", {})
            games.append({
                "game_id": event["id"], "game_time": event.get("date", ""),
                "status": so.get("type", {}).get("name", ""),
                "period": so.get("period", 0), "display_clock": so.get("displayClock", ""),
                "status_detail": so.get("type", {}).get("detail", ""),
                "home_team": ha, "away_team": aa,
                "home_name": home["team"].get("displayName", ha),
                "away_name": away["team"].get("displayName", aa),
                "home_score": int(home.get("score", 0) or 0),
                "away_score": int(away.get("score", 0) or 0),
                "home_logo": f"https://a.espncdn.com/i/teamlogos/nba/500/{home['team']['abbreviation'].lower()}.png",
                "away_logo": f"https://a.espncdn.com/i/teamlogos/nba/500/{away['team']['abbreviation'].lower()}.png",
                "neutral": int(comp.get("neutralSite", False)),
            })
        except (KeyError, IndexError) as e:
            log.debug(f"Error parsing NBA game: {e}")
    return games


def fetch_nba_scoreboard():
    log.info("Fetching NBA scoreboard (today + past 5 days)...")
    games = []; seen = set()
    for delta in range(0, -6, -1):
        ds = (datetime.now(timezone.utc) + timedelta(days=delta)).strftime("%Y%m%d")
        data = safe_get(f"{ESPN_NBA_WEB_BASE}/scoreboard?dates={ds}")
        if not data or not data.get("events"):
            data = safe_get(f"{ESPN_NBA_BASE}/scoreboard?dates={ds}")
        if not data: continue
        for g in _parse_nba_events(data):
            if g["game_id"] not in seen:
                seen.add(g["game_id"]); games.append(g)
    log.info(f"Found {len(games)} NBA scoreboard games"); return games


def fetch_nba_future_games(days_ahead=14):
    future = []; seen = set()
    for offset in range(0, days_ahead + 1):
        ds = (datetime.now(timezone.utc) + timedelta(days=offset)).strftime("%Y%m%d")
        data = safe_get(f"{ESPN_NBA_WEB_BASE}/scoreboard?dates={ds}&limit=20&seasontype=2")
        if not data or not data.get("events"):
            data = safe_get(f"{ESPN_NBA_BASE}/scoreboard?dates={ds}&limit=20&seasontype=2")
        if not data: continue
        for g in _parse_nba_events(data):
            if g["game_id"] not in seen:
                seen.add(g["game_id"]); g["is_future"] = True; future.append(g)
    log.info(f"Found {len(future)} future NBA games"); return future


def fetch_nba_season_games_espn(seasons=None):
    """
    Fetch completed NBA games day-by-day via ESPN scoreboard.
    Replaces nba_api LeagueGameLog (stats.nba.com) with ESPN endpoints.

    FIX: stride_days is now always 1 for ALL seasons to ensure full
    historical coverage for model training. Previously, older seasons
    used a 7-day stride which skipped ~85% of games, starving the model
    and causing it to fall back to 50% neutral defaults.
    """
    from datetime import date as date_cls
    if seasons is None:
        cy = datetime.now(timezone.utc).year
        cm = datetime.now(timezone.utc).month
        ey = cy + 1 if cm >= 10 else cy
        seasons = [ey - 2, ey - 1, ey]

    def _fetch_day(ds):
        return ds, safe_get(f"{ESPN_NBA_BASE}/scoreboard?dates={ds}&limit=20")

    log.info(f"Fetching NBA historical games (ESPN) for {seasons}...")
    all_games = []; seen = set(); today = date_cls.today(); fails = 0
    for season_year in seasons:
        season_start = date_cls(season_year - 1, 10, 1)
        season_end   = min(today, date_cls(season_year, 6, 30))
        if season_start > today: continue
        # stride_days is always 1 for all seasons — ensures full game coverage
        # for model training. Do not revert to a 7-day stride.
        stride_days  = 1
        date_list    = []
        current      = season_start
        while current <= season_end:
            date_list.append(current.strftime("%Y%m%d"))
            current += timedelta(days=stride_days)
        results_by_date = {}
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(_fetch_day, ds): ds for ds in date_list}
            for fut in as_completed(futs):
                ds, data = fut.result(); results_by_date[ds] = data
        for ds in sorted(date_list):
            data = results_by_date.get(ds)
            if not data: fails += 1; continue
            for event in data.get("events", []):
                try:
                    eid = event.get("id", "")
                    if eid in seen: continue
                    seen.add(eid)
                    if event.get("status", {}).get("type", {}).get("name", "") != "STATUS_FINAL": continue
                    comp  = event["competitions"][0]
                    comps = comp["competitors"]
                    home  = next((c for c in comps if c["homeAway"] == "home"), None)
                    away  = next((c for c in comps if c["homeAway"] == "away"), None)
                    if not home or not away: continue
                    hs = int(home.get("score", 0) or 0)
                    as_ = int(away.get("score", 0) or 0)
                    if hs == 0 and as_ == 0: continue
                    all_games.append({
                        "date": event.get("date", "")[:10], "season": season_year,
                        "team1": nba_abbrev_norm(home["team"]["abbreviation"]),
                        "team2": nba_abbrev_norm(away["team"]["abbreviation"]),
                        "score1": hs, "score2": as_,
                        "neutral": int(comp.get("neutralSite", False)),
                    })
                except (KeyError, IndexError, TypeError) as e:
                    log.debug(f"Error parsing NBA game: {e}")
    if fails: log.warning(f"NBA season fetch: skipped {fails} date(s)")
    log.info(f"Fetched {len(all_games)} NBA historical games (ESPN)")
    return all_games


def fetch_nba_standings_espn():
    log.info("Fetching NBA standings (ESPN)...")
    data = safe_get(ESPN_NBA_STANDINGS)
    if not data: return {}
    standings = {}
    try:
        for group in data.get("children", []):
            entries = group.get("standings", {}).get("entries", [])
            if not entries:
                for div in group.get("children", []):
                    entries = entries + div.get("standings", {}).get("entries", [])
            for entry in entries:
                abbrev = nba_abbrev_norm(entry["team"]["abbreviation"])
                tid = entry["team"].get("id", "")
                if tid: ESPN_NBA_ID_TO_ABBREV[tid] = abbrev
                stats   = {s["name"]: s.get("value", 0) for s in entry.get("stats", [])}
                wins    = int(stats.get("wins", 0)); losses = int(stats.get("losses", 0))
                off_rtg = float(stats.get("avgPointsFor", 110.0))
                def_rtg = float(stats.get("avgPointsAgainst", 110.0))
                standings[abbrev] = {
                    "wins": wins, "losses": losses,
                    "win_pct": float(stats.get("winPercent", 0)),
                    "points_for": float(stats.get("pointsFor", 0)),
                    "points_against": float(stats.get("pointsAgainst", 0)),
                    "games_played": wins + losses,
                    "offensive_rating": off_rtg, "defensive_rating": def_rtg,
                    "net_rating": off_rtg - def_rtg,
                    "pace": 100.0, "assist_turnover_ratio": 1.8,
                    "rebound_rate": 0.5, "three_point_rate": 0.35, "free_throw_rate": 0.20,
                    "streak": stats.get("streak", 0),
                }
    except Exception as e:
        log.warning(f"Error parsing NBA standings: {e}")
    log.info(f"NBA standings for {len(standings)} teams"); return standings


def fetch_nba_standings_cdn():
    """
    Fetch NBA standings from cdn.nba.com/static/json/staticData/standings.json.
    Falls back to ESPN fetch on failure.
    """
    log.info("Fetching NBA standings (cdn.nba.com)...")
    data = safe_get(f"{CDN_NBA_STATIC}/staticData/standings.json", headers=_CDN_HEADERS)
    if not data:
        log.warning("cdn.nba.com standings unavailable — falling back to ESPN")
        return fetch_nba_standings_espn()
    standings = {}
    try:
        rows = data.get("standings", [])
        if rows and isinstance(rows[0], dict):
            for row in rows:
                abbrev  = nba_abbrev_norm(row.get("teamAbbreviation", ""))
                if not abbrev: continue
                wins    = int(row.get("wins", 0)); losses = int(row.get("losses", 0))
                off_rtg = float(row.get("offensiveRating", row.get("OffRating", 110.0)))
                def_rtg = float(row.get("defensiveRating", row.get("DefRating", 110.0)))
                standings[abbrev] = {
                    "wins": wins, "losses": losses,
                    "win_pct": wins / max(wins + losses, 1),
                    "points_for": 0.0, "points_against": 0.0,
                    "games_played": wins + losses,
                    "offensive_rating": off_rtg, "defensive_rating": def_rtg,
                    "net_rating": off_rtg - def_rtg,
                    "pace": float(row.get("pace", 100.0)),
                    "assist_turnover_ratio": 1.8, "rebound_rate": 0.5,
                    "three_point_rate": 0.35, "free_throw_rate": 0.20, "streak": 0,
                }
        if not standings:
            raise ValueError("No usable standings in cdn response")
    except Exception as e:
        log.warning(f"cdn.nba.com standings parse error: {e} — falling back to ESPN")
        return fetch_nba_standings_espn()
    log.info(f"NBA standings (cdn) for {len(standings)} teams"); return standings


def fetch_nba_injuries_espn():
    log.info("Fetching NBA injuries (ESPN)...")
    data = safe_get(f"{ESPN_NBA_BASE}/injuries")
    if not data: return {}
    injuries = {}
    try:
        for team_entry in data.get("injuries", []):
            raw    = team_entry.get("abbreviation", "") or team_entry.get("team", {}).get("abbreviation", "")
            abbrev = nba_abbrev_norm(raw) if raw else ""
            for item in team_entry.get("injuries", []):
                ia = abbrev
                if not ia:
                    fb = item.get("athlete", {}).get("team", {}).get("abbreviation", "")
                    ia = nba_abbrev_norm(fb) if fb else ""
                if not ia: continue
                injuries.setdefault(ia, [])
                athlete = item.get("athlete", {})
                sr      = item.get("status", "")
                st      = sr.get("name", sr.get("abbreviation", "")) if isinstance(sr, dict) else str(sr or "")
                injuries[ia].append({
                    "player": athlete.get("displayName", ""),
                    "status": st,
                    "position": athlete.get("position", {}).get("abbreviation", ""),
                    "injury_description": item.get("longComment", item.get("shortComment", st)),
                })
    except Exception as e:
        log.warning(f"Error parsing NBA injuries: {e}")
    total = sum(len(v) for v in injuries.values())
    log.info(f"NBA injuries: {total} players across {len(injuries)} teams")
    return injuries


def fetch_nba_depth_charts_espn():
    log.info("Fetching ESPN NBA depth charts...")
    player_depth = {}
    try:
        teams_data = safe_get(f"{ESPN_NBA_BASE}/teams")
        if not teams_data: return {}
        teams_list = teams_data.get("sports",[{}])[0].get("leagues",[{}])[0].get("teams",[])
        for te in teams_list:
            team_id = te.get("team", {}).get("id", "")
            if not team_id: continue
            dd = safe_get(f"{ESPN_NBA_BASE}/teams/{team_id}/depthcharts")
            if not dd: continue
            for pg in dd.get("positionGroups", []):
                for pos in pg.get("positions", []):
                    for ae in pos.get("athletes", []):
                        name = ae.get("athlete", {}).get("displayName", "")
                        dp   = int(ae.get("rank") or ae.get("slot", 99))
                        if not name: continue
                        if name not in player_depth or dp < player_depth[name]:
                            player_depth[name] = dp
    except Exception as e:
        log.warning(f"Error fetching NBA depth charts: {e}")
    log.info(f"NBA depth chart: {len(player_depth)} entries")
    return player_depth


def fetch_nba_player_ppg_espn():
    """Fetch per-player PPG from ESPN team athlete endpoints."""
    log.info("Fetching NBA player PPG (ESPN)...")
    player_ppg = {}
    try:
        teams_data = safe_get(f"{ESPN_NBA_BASE}/teams")
        if not teams_data: return {}
        teams_list = teams_data.get("sports",[{}])[0].get("leagues",[{}])[0].get("teams",[])
        for te in teams_list:
            team_id = te.get("team", {}).get("id", "")
            if not team_id: continue
            rd = safe_get(f"{ESPN_NBA_BASE}/teams/{team_id}/athletes", params={"enable": "stats"})
            if not rd: continue
            for athlete in rd.get("athletes", []):
                name = athlete.get("displayName", "")
                if not name: continue
                stats_list = athlete.get("statistics", {}).get("splits", {})
                if not stats_list: stats_list = athlete.get("stats", [])
                ppg = None
                if isinstance(stats_list, list):
                    for stat in stats_list:
                        label = str(stat.get("name", "") or stat.get("shortDisplayName", "")).lower()
                        if label in ("pts", "ppg", "points", "avg points"):
                            try: ppg = float(stat.get("displayValue") or stat.get("value") or 0)
                            except (TypeError, ValueError): pass
                            break
                if ppg and ppg > 0:
                    if name not in player_ppg or ppg > player_ppg[name]:
                        player_ppg[name] = ppg
    except Exception as e:
        log.warning(f"Error fetching NBA player PPG: {e}")
    log.info(f"NBA PPG stats: {len(player_ppg)} entries")
    return player_ppg


def fetch_nba_player_stats_cdn(season_year):
    """
    Fetch per-player PPG from cdn.nba.com/stats/leaguedashplayerstats.
    Replaces stats.nba.com endpoint. Falls back to ESPN on failure.
    """
    log.info(f"Fetching NBA player stats (cdn.nba.com) for {season_year}...")
    season_str = f"{season_year - 1}-{str(season_year)[2:]}"
    url    = f"{CDN_NBA_STATS}/leaguedashplayerstats"
    params = {
        "MeasureType": "Base", "PerMode": "PerGame",
        "Season": season_str, "SeasonType": "Regular Season", "LeagueID": "00",
    }
    data = safe_get(url, params=params, headers=_CDN_HEADERS, timeout=20)
    if not data:
        log.warning("cdn.nba.com player stats unavailable — falling back to ESPN")
        return fetch_nba_player_ppg_espn()
    player_ppg = {}
    try:
        rs = data.get("resultSets", [{}])[0]
        headers_list = rs.get("headers", []); rows = rs.get("rowSet", [])
        ni = headers_list.index("PLAYER_NAME") if "PLAYER_NAME" in headers_list else None
        pi = headers_list.index("PTS")         if "PTS"         in headers_list else None
        if ni is not None and pi is not None:
            for row in rows:
                try:
                    name = str(row[ni]); ppg = float(row[pi] or 0)
                    if name and ppg > 0: player_ppg[name] = ppg
                except (IndexError, TypeError, ValueError):
                    pass
    except Exception as e:
        log.warning(f"cdn.nba.com player stats parse error: {e}")
    log.info(f"NBA player stats (cdn): {len(player_ppg)} players")
    return player_ppg if player_ppg else fetch_nba_player_ppg_espn()


def fetch_nba_odds_api(api_key):
    if not api_key:
        log.info("No ODDS_API_KEY, skipping NBA odds"); return {}
    log.info("Fetching NBA odds...")
    data = safe_get(ODDS_BASE_NBA, params={"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "american"})
    if not data: return {}
    import numpy as np
    from model.ensemble_model import american_to_prob, remove_vig
    odds_map = {}
    for game in data:
        try:
            ht = game.get("home_team", ""); at = game.get("away_team", "")
            hl, al = [], []
            for bm in game.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if mkt["key"] == "h2h":
                        for out in mkt.get("outcomes", []):
                            if out["name"] == ht: hl.append(float(out["price"]))
                            elif out["name"] == at: al.append(float(out["price"]))
            if hl and al:
                ah, aa = np.mean(hl), np.mean(al)
                ch, ca = remove_vig(american_to_prob(ah), american_to_prob(aa))
                odds_map[f"{at}_at_{ht}"] = {"home_prob": round(ch,4), "away_prob": round(ca,4),
                                              "home_american": ah, "away_american": aa,
                                              "home_team_name": ht, "away_team_name": at}
        except Exception as e:
            log.debug(f"Error parsing NBA odds: {e}")
    log.info(f"NBA odds for {len(odds_map)} games"); return odds_map
