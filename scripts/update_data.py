"""
Main orchestration script for NFL prediction model.
Downloads data, runs all models, writes JSON output files.
Run daily via GitHub Actions.
Includes live ESPN continuation past FiveThirtyEight 2024 cutoff.
"""

import os
import re
import sys
import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.elo_model import compute_elo, predict_game as elo_predict_game, get_trend
from model.logistic_model import (build_features, train_logistic, evaluate_model,
                                   calibration_buckets, predict_matchups,
                                   historical_accuracy_by_year)
from model.bayesian_model import update_ratings, predict_game as bayes_predict, get_all_ratings
from model.efficiency_model import (compute_pythagorean, compute_efficiency,
                                     travel_distance, travel_elo_adjustment,
                                     rest_elo_adjustment, turnover_regression_adjustment,
                                     efficiency_predict_game, hierarchical_team_rating)
from model.monte_carlo import simulate_game, simulate_season
from model.ensemble_model import (build_xgb_features, train_xgboost, predict_xgboost,
                                   ensemble_predict, kelly_criterion,
                                   american_to_prob, remove_vig, DEFAULT_WEIGHTS)
from model.nfl_ensemble_weights import learn_nfl_weights
from model.injury_model import compute_all_team_impacts, injury_elo_adjustment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ESPN_NFL_STANDINGS_URL = "https://site.web.api.espn.com/apis/v2/sports/football/nfl/standings"
FTE_URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nfl-elo/nfl_elo.csv"
ODDS_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"

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

# ESPN abbrev → our abbrev mapping
ESPN_TO_ABBREV = {
    "WSH": "WAS", "JAC": "JAX", "LVR": "LV",
    "LA": "LAR", "LAR": "LAR", "LAC": "LAC",
}


def abbrev_norm(abbrev):
    return ESPN_TO_ABBREV.get(abbrev.upper(), abbrev.upper())


def nfl_days_since_last_game(game_history, team, before_date):
    """Return days since team's last completed game strictly before before_date."""
    from datetime import date as date_cls
    dates = [
        g["date"] for g in game_history.get(team, [])
        if g.get("date", "") < str(before_date)
    ]
    if not dates:
        return 7  # default bye-equivalent
    try:
        last = max(dates)
        return max(0, (before_date - datetime.strptime(last, "%Y-%m-%d").date()).days)
    except ValueError:
        return 7


def safe_get(url, params=None, timeout=30):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return None


def download_fte_data():
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
        # Regular season
        url = f"{ESPN_BASE}/scoreboard?seasontype=2&week={week}"
        data = safe_get(url)
        if not data:
            # Try postseason (type 3)
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
    log.info("Fetching ESPN standings...")
    # Use web API which returns conference-level entries with wins/losses
    data = safe_get(ESPN_NFL_STANDINGS_URL)
    if not data:
        return {}

    standings = {}
    try:
        for group in data.get("children", []):
            # Try conference-level entries first (web API structure)
            entries = group.get("standings", {}).get("entries", [])
            # Fallback: division children (site.api structure)
            if not entries:
                for div in group.get("children", []):
                    entries = entries + div.get("standings", {}).get("entries", [])

            for entry in entries:
                team_abbrev = abbrev_norm(entry["team"]["abbreviation"])
                stats = {s["name"]: s.get("value", 0)
                         for s in entry.get("stats", [])}
                wins = int(stats.get("wins", 0))
                losses = int(stats.get("losses", 0))
                points_for = float(stats.get("pointsFor", 0))
                points_against = float(stats.get("pointsAgainst", 0))

                standings[team_abbrev] = {
                    "wins": wins,
                    "losses": losses,
                    "ties": int(stats.get("ties", 0)),
                    "win_pct": float(stats.get("winPercent", 0)),
                    "points_for": points_for,
                    "points_against": points_against,
                    "streak": stats.get("streak", 0),
                    "games_played": wins + losses,
                }
    except Exception as e:
        log.warning(f"Error parsing standings: {e}")

    log.info(f"Standings for {len(standings)} teams")
    return standings


def fetch_espn_injuries():
    log.info("Fetching ESPN injuries...")
    data = safe_get(f"{ESPN_BASE}/injuries")
    if not data:
        return {}

    injuries = {}
    try:
        # Response structure: {"injuries": [{"displayName": "Team", "injuries": [{player entries}]}]}
        # Note: ESPN removed top-level "abbreviation" from team entries; it now lives at
        # item.athlete.team.abbreviation only.
        for team_entry in data.get("injuries", []):
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


def normalize_player_name(name: str) -> str:
    """Normalize player name for cross-source matching.

    Strips name suffixes (Jr/Sr/II/III/IV) and normalizes apostrophes so that
    names from ESPN injuries ('Mack Wilson Sr.') match depth chart entries
    ('Mack Wilson').
    """
    n = name.lower().strip()
    n = n.replace("\u2019", "'").replace("`", "'")  # curly → straight apostrophe
    n = re.sub(r"\s+\b(jr\.?|sr\.?|ii|iii|iv)\b\.?$", "", n).strip()
    return n


def fetch_espn_depth_charts():
    """
    Fetch NFL depth chart positions for all teams from ESPN.
    Returns {player_name: depth_position_int} where 1 = first string (starter).
    """
    log.info("Fetching ESPN NFL depth charts for player value scoring...")
    player_depth = {}
    try:
        teams_data = safe_get(f"{ESPN_BASE}/teams")
        if not teams_data:
            log.warning("Could not fetch ESPN teams list for depth charts")
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
            depth_data = safe_get(f"{ESPN_BASE}/teams/{team_id}/depthcharts")
            if not depth_data:
                continue
            for pos_group in depth_data.get("positionGroups", []):
                for position in pos_group.get("positions", []):
                    for athlete_entry in position.get("athletes", []):
                        name = athlete_entry.get("athlete", {}).get("displayName", "")
                        # ESPN changed "rank" to "slot" — check both fields
                        depth_pos = int(athlete_entry.get("rank") or athlete_entry.get("slot", 99))
                        if not name:
                            continue
                        # Keep the best (lowest) depth position seen for this player.
                        # Store under both the raw display name and the normalized form
                        # so lookups succeed regardless of suffix differences (Jr/Sr/etc.).
                        for key in [name, normalize_player_name(name)]:
                            if key and (key not in player_depth or depth_pos < player_depth[key]):
                                player_depth[key] = depth_pos
    except Exception as e:
        log.warning(f"Error fetching depth charts: {e}")
    log.info(f"Depth chart loaded: {len(player_depth)} player entries")
    return player_depth


# Depth position → value multiplier (how impactful vs an average starter)
_DEPTH_VALUE_MAP = {1: 1.0, 2: 0.55, 3: 0.25}


def _depth_to_value_mult(depth_pos: int) -> float:
    return _DEPTH_VALUE_MAP.get(depth_pos, 0.15)


def build_player_values(depth_charts: dict) -> dict:
    """
    Convert depth chart positions to value multipliers.
    Returns {player_name: multiplier} where 1.0 = first-string starter.
    """
    return {name: _depth_to_value_mult(pos) for name, pos in depth_charts.items()}


def value_tier_label(mult: float) -> str:
    """Return tier label for a player value multiplier."""
    if mult >= 1.8:
        return "superstar"
    if mult >= 1.3:
        return "all-star"
    if mult >= 0.8:
        return "starter"
    if mult >= 0.4:
        return "backup"
    return "rotation"


def fetch_espn_completed_games(season_year):
    """
    Fetch completed NFL games from ESPN for the given season year.
    Used to extend ELO past the FiveThirtyEight 2024 cutoff.
    Returns list of dicts: {date, team1, team2, score1, score2, neutral, season}
    """
    log.info(f"Fetching ESPN completed games for {season_year} season...")
    completed = []
    seen_ids = set()
    fetch_failures = 0

    # Fetch regular season weeks 1-22 and postseason weeks 1-5
    season_types = [(2, range(1, 23)), (3, range(1, 6))]

    for season_type, weeks in season_types:
        for week in weeks:
            url = f"{ESPN_BASE}/scoreboard"
            params = {
                "seasontype": season_type,
                "week": week,
                "dates": season_year,
                "limit": 100,
            }
            data = safe_get(url, params=params)
            if not data:
                fetch_failures += 1
                continue

            events = data.get("events", [])
            if not events:
                continue

            for event in events:
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
                    game_date = event.get("date", "")[:10]
                    neutral = int(comp.get("neutralSite", False))

                    completed.append({
                        "date": game_date,
                        "season": season_year,
                        "team1": home_abbrev,
                        "team2": away_abbrev,
                        "score1": home_score,
                        "score2": away_score,
                        "neutral": neutral,
                    })
                except (KeyError, IndexError, TypeError) as e:
                    log.debug(f"Error parsing ESPN game: {e}")

    if fetch_failures > 0:
        log.warning(f"NFL completed-games fetch: skipped {fetch_failures} week(s) due to fetch failures")
    log.info(f"Found {len(completed)} completed ESPN games for {season_year}")
    return completed


def extend_elo_with_espn(elo_dict, game_history, espn_games,
                          fte_cutoff_date=None, k_base=20.0, hfa=65.0):
    """
    Continue ELO computation with ESPN live results after FTE dataset ends.
    Skips games already covered by FTE (before cutoff date).
    """
    if not espn_games:
        return elo_dict, game_history

    # Determine cutoff: skip games on or before last FTE date
    if fte_cutoff_date:
        cutoff = pd.to_datetime(fte_cutoff_date)
    else:
        cutoff = pd.Timestamp("2024-02-12")  # Super Bowl LVIII (end of 2023 season)

    from model.elo_model import expected_score, mov_multiplier

    new_games = sorted(
        [g for g in espn_games if pd.to_datetime(g["date"]) > cutoff],
        key=lambda g: g["date"]
    )

    log.info(f"Extending ELO with {len(new_games)} ESPN games after {cutoff.date()}")

    for game in new_games:
        team1 = game["team1"]
        team2 = game["team2"]
        score1 = game["score1"]
        score2 = game["score2"]
        neutral = game.get("neutral", 0)

        if team1 not in elo_dict:
            elo_dict[team1] = 1500.0
            game_history[team1] = []
        if team2 not in elo_dict:
            elo_dict[team2] = 1500.0
            game_history[team2] = []

        e1 = elo_dict[team1]
        e2 = elo_dict[team2]

        hfa_adj = 0 if neutral else hfa
        adj_e1 = e1 + hfa_adj

        exp1 = expected_score(adj_e1, e2)
        exp2 = 1.0 - exp1

        actual1 = 1.0 if score1 > score2 else (0.5 if score1 == score2 else 0.0)
        actual2 = 1.0 - actual1

        point_diff = abs(score1 - score2)
        elo_diff_abs = abs(adj_e1 - e2)
        mov = mov_multiplier(point_diff, elo_diff_abs) if point_diff > 0 else 1.0

        elo_dict[team1] = e1 + k_base * mov * (actual1 - exp1)
        elo_dict[team2] = e2 + k_base * mov * (actual2 - exp2)

        game_history.setdefault(team1, []).append({
            "result": actual1,
            "elo_diff": adj_e1 - e2,
            "date": game["date"],
        })
        game_history.setdefault(team2, []).append({
            "result": actual2,
            "elo_diff": e2 - adj_e1,
            "date": game["date"],
        })

    return elo_dict, game_history


def generate_prediction_drivers(game_info, home, away, elo_dict,
                                  efficiency_data, injury_impacts, adj):
    """
    Generate a list of plain-English prediction driver strings for a game.
    """
    drivers = []

    # 1. ELO advantage
    home_elo = elo_dict.get(home, 1500.0)
    away_elo = elo_dict.get(away, 1500.0)
    elo_diff = abs(home_elo - away_elo)
    if elo_diff >= 50:
        leader = home if home_elo > away_elo else away
        drivers.append(f"ELO advantage: {leader} +{elo_diff:.0f} rating points")

    # 2. Key injured players (Out / Doubtful) with tier labels
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

    # 3. Efficiency mismatch
    net_home = efficiency_data.get(home, {}).get("net_eff", 0.0)
    net_away = efficiency_data.get(away, {}).get("net_eff", 0.0)
    if abs(net_home - net_away) >= 0.10:
        leaders = home if net_home > net_away else away
        drivers.append(
            f"Efficiency gap: {home} net {net_home:+.3f} vs {away} net {net_away:+.3f}"
        )

    # 4. Rest advantage
    rest_diff = adj.get("rest_diff", 0)
    if abs(rest_diff) >= 3:
        rested = home if rest_diff > 0 else away
        drivers.append(
            f"Rest advantage: {rested} has {abs(rest_diff)} extra days rest"
        )

    # 5. Travel penalty
    travel_mi = adj.get("travel_dist_miles", 0)
    if travel_mi >= 1500:
        drivers.append(
            f"Travel penalty: away team travels {travel_mi:.0f} miles"
        )

    # 6. Home field
    if not game_info.get("neutral", False):
        drivers.append(f"Home field: {home} +65 ELO home advantage")

    return drivers


def fetch_betting_odds(api_key):
    """Fetch NFL odds from The Odds API."""
    if not api_key:
        log.info("No ODDS_API_KEY set, skipping betting lines")
        return {}

    log.info("Fetching betting odds from The Odds API...")
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "sport": "americanfootball_nfl"
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

            # Average across bookmakers
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
            log.debug(f"Error parsing odds entry: {e}")

    log.info(f"Got odds for {len(odds_map)} games")
    return odds_map


def build_team_efficiency_data(standings, fte_df):
    """Build efficiency data from standings (using points) and FTE for YPP."""
    efficiency_data = {}

    for team, s in standings.items():
        pf = s.get("points_for", 350)
        pa = s.get("points_against", 350)
        gp = max(s.get("games_played", 1), 1)

        # Approximate YPP from points per game (rough conversion)
        ppg_off = pf / gp
        ppg_def = pa / gp
        league_ppg = 22.0  # roughly 22 ppg average

        ypp_off = 5.5 * (ppg_off / league_ppg)  # scale around league avg YPP
        ypp_def = 5.5 * (ppg_def / league_ppg)

        efficiency_data[team] = {
            "ypp_offense": ypp_off,
            "ypp_allowed": ypp_def,
            "off_eff": 0.0,  # will be filled by compute_efficiency
            "def_eff": 0.0,
            "net_eff": 0.0,
            "passing_eff": 1.0 + (ppg_off - league_ppg) / league_ppg * 0.6,
            "rushing_eff": 1.0 + (ppg_off - league_ppg) / league_ppg * 0.4,
            "pass_def_eff": 1.0 - (ppg_def - league_ppg) / league_ppg * 0.6,
            "rush_def_eff": 1.0 - (ppg_def - league_ppg) / league_ppg * 0.4,
        }

    computed = compute_efficiency(
        {t: {"ypp_offense": d["ypp_offense"], "ypp_allowed": d["ypp_allowed"]}
         for t, d in efficiency_data.items()}
    )

    for team in efficiency_data:
        if team in computed:
            efficiency_data[team].update(computed[team])

    # Fill missing teams
    for team in NFL_TEAMS:
        if team not in efficiency_data:
            efficiency_data[team] = {
                "ypp_offense": 5.5, "ypp_allowed": 5.5,
                "off_eff": 1.0, "def_eff": 1.0, "net_eff": 0.0,
                "elo_equiv": 1500.0,
                "passing_eff": 1.0, "rushing_eff": 1.0,
                "pass_def_eff": 1.0, "rush_def_eff": 1.0,
            }

    return efficiency_data


def match_odds_to_game(game, odds_map):
    """Try to find odds for a game by matching team names."""
    home = game.get("home_name", "")
    away = game.get("away_name", "")
    for key, odds in odds_map.items():
        h = odds.get("home_team_name", "")
        a = odds.get("away_team_name", "")
        if (home.lower() in h.lower() or h.lower() in home.lower()) and \
           (away.lower() in a.lower() or a.lower() in away.lower()):
            return odds
    return None


def run():
    log.info("=== NFL Prediction Model Update Starting ===")
    now_utc = datetime.now(timezone.utc).isoformat()
    odds_api_key = os.environ.get("ODDS_API_KEY", "")

    # ── 1. Download data ────────────────────────────────────────────────────
    fte_df = download_fte_data()
    scoreboard_games, current_week = fetch_espn_scoreboard()
    standings = fetch_espn_standings()
    injuries = fetch_espn_injuries()
    odds_map = fetch_betting_odds(odds_api_key)

    # ── 1a. Fallback standings from FTE data when ESPN returns empty (offseason) ──
    if not standings and not fte_df.empty:
        log.info("ESPN standings empty (offseason), computing W-L from FTE data...")
        last_season = fte_df[fte_df["season"] == fte_df["season"].max()]
        completed = last_season.dropna(subset=["score1", "score2"])
        for _, row in completed.iterrows():
            for team_col, opp_col, score_col, opp_score_col in [
                ("team1", "team2", "score1", "score2"),
                ("team2", "team1", "score2", "score1"),
            ]:
                team = abbrev_norm(str(row[team_col]))
                if team not in standings:
                    standings[team] = {"wins": 0, "losses": 0, "ties": 0,
                                       "points_for": 0.0, "points_against": 0.0,
                                       "games_played": 0}
                won = float(row[score_col]) > float(row[opp_score_col])
                tied = float(row[score_col]) == float(row[opp_score_col])
                if won:
                    standings[team]["wins"] += 1
                elif tied:
                    standings[team]["ties"] += 1
                else:
                    standings[team]["losses"] += 1
                standings[team]["points_for"] += float(row[score_col])
                standings[team]["points_against"] += float(row[opp_score_col])
                standings[team]["games_played"] += 1
        log.info(f"Built standings from FTE data for {len(standings)} teams")

    # ── 1b. Fetch depth charts and compute player values ────────────────────
    depth_charts = fetch_espn_depth_charts()
    player_values = build_player_values(depth_charts)

    # ── 1c. Compute injury impacts (using player-specific value multipliers) ─
    log.info("Computing injury impact scores...")
    injury_impacts = compute_all_team_impacts(injuries, player_values)

    # Save NFL injuries file (only overwrite if we actually fetched data)
    if injuries:
        nfl_injuries_list = []
        for team, players in injuries.items():
            for p in players:
                pname = p.get("player", "")
                pmult = player_values.get(pname) or player_values.get(normalize_player_name(pname), 1.0)
                nfl_injuries_list.append({
                    "player": pname,
                    "team": team,
                    "position": p.get("position", ""),
                    "status": p.get("status", ""),
                    "injury_description": p.get("injury_description", p.get("status", "")),
                    "value_tier": value_tier_label(pmult),
                })
        (DATA_DIR / "nfl_injuries.json").write_text(
            json.dumps({"updated": now_utc, "injuries": nfl_injuries_list}, indent=2)
        )
    else:
        log.warning("No NFL injury data fetched — keeping existing nfl_injuries.json")

    season_year = datetime.now().year
    current_month = datetime.now().month
    # NFL season runs Sep–Feb; if before September treat as prior season
    if current_month < 8:
        season_year -= 1

    # Detect offseason: no games, or all games are final and most recent was >21 days ago
    is_offseason = len(scoreboard_games) == 0
    if not is_offseason and scoreboard_games and all(
        g.get("status") == "STATUS_FINAL" for g in scoreboard_games
    ):
        try:
            recent_times = [g["game_time"] for g in scoreboard_games if g.get("game_time")]
            if recent_times:
                latest_str = max(recent_times)
                latest_dt = datetime.fromisoformat(latest_str.replace("Z", "+00:00"))
                days_since = (datetime.now(timezone.utc) - latest_dt).days
                if days_since > 21:
                    is_offseason = True
                    log.info(f"Detected NFL offseason: most recent game was {days_since} days ago")
        except Exception as e:
            log.warning(f"Could not check offseason recency: {e}")

    # ── 1c. Fetch future scheduled games ────────────────────────────────────
    future_games = fetch_espn_future_games(current_week, season_year, weeks_ahead=3)
    all_games_for_prediction = scoreboard_games + future_games

    # ── 2. Build ELO ratings (FTE historical + ESPN live continuation) ─────
    log.info("Computing ELO ratings from FTE data...")
    fte_cutoff_date = None
    if not fte_df.empty:
        elo_dict, game_history = compute_elo(fte_df)
        # Determine last date covered by FTE dataset
        completed_fte = fte_df.dropna(subset=["score1", "score2"])
        if not completed_fte.empty:
            fte_cutoff_date = pd.to_datetime(completed_fte["date"]).max()
            log.info(f"FTE data ends at {fte_cutoff_date.date()}")
    else:
        elo_dict = {t: 1500.0 for t in NFL_TEAMS}
        game_history = {t: [] for t in NFL_TEAMS}

    # Fill missing teams
    for team in NFL_TEAMS:
        if team not in elo_dict:
            elo_dict[team] = 1500.0
        if team not in game_history:
            game_history[team] = []

    log.info(f"ELO computed for {len(elo_dict)} teams")

    # ── 2b. Extend ELO with live ESPN results after FTE cutoff ─────────────
    log.info("Extending ELO with live ESPN results...")
    espn_completed = fetch_espn_completed_games(season_year)
    if espn_completed:
        elo_dict, game_history = extend_elo_with_espn(
            elo_dict, game_history, espn_completed,
            fte_cutoff_date=fte_cutoff_date
        )
    log.info(f"ELO updated for {len(elo_dict)} teams after live extension")

    # ── 3. Build efficiency & pythagorean data ──────────────────────────────
    log.info("Computing efficiency and pythagorean ratings...")
    efficiency_data = build_team_efficiency_data(standings, fte_df)
    teams_pts_data = {
        team: {
            "points_for": standings.get(team, {}).get("points_for", 350),
            "points_against": standings.get(team, {}).get("points_against", 350)
        }
        for team in NFL_TEAMS
    }
    pythagorean_data = compute_pythagorean(teams_pts_data)

    # ── 4. Train logistic model ─────────────────────────────────────────────
    log.info("Training logistic regression model...")
    logistic_model, logistic_scaler, logistic_calibrator = None, None, None
    model_metrics = {
        "log_loss": None, "brier_score": None, "auc": None,
        "calibration_buckets": [], "historical_accuracy": []
    }

    if not fte_df.empty and len(fte_df) > 100:
        try:
            X, y = build_features(fte_df, elo_dict, game_history,
                                   efficiency_data, pythagorean_data)
            if len(X) > 50:
                logistic_model, logistic_scaler, logistic_calibrator = train_logistic(X, y)
                metrics = evaluate_model(logistic_model, logistic_scaler,
                                         logistic_calibrator, X, y)
                model_metrics.update(metrics)
                model_metrics["calibration_buckets"] = calibration_buckets(
                    logistic_model, logistic_scaler, logistic_calibrator, X, y
                )
                model_metrics["historical_accuracy"] = historical_accuracy_by_year(
                    fte_df, logistic_model, logistic_scaler, logistic_calibrator,
                    elo_dict, game_history, efficiency_data, pythagorean_data
                )
                log.info(f"Logistic model trained. Log loss: {metrics['log_loss']:.4f}")
        except Exception as e:
            log.error(f"Logistic model training failed: {e}")

    # ── 5. Train XGBoost model ──────────────────────────────────────────────
    log.info("Training XGBoost model...")
    xgb_model, xgb_scaler = None, None
    if not fte_df.empty and len(fte_df) > 100:
        try:
            X_xgb, y_xgb = build_xgb_features(fte_df, elo_dict, game_history,
                                                efficiency_data, pythagorean_data)
            if len(X_xgb) > 50:
                xgb_model, xgb_scaler = train_xgboost(X_xgb, y_xgb)
                if xgb_model:
                    log.info("XGBoost model trained successfully")
                else:
                    log.warning("XGBoost not available, skipping")
        except Exception as e:
            log.error(f"XGBoost training failed: {e}")

    # ── 6. Bayesian ratings ─────────────────────────────────────────────────
    log.info("Computing Bayesian ratings...")
    games_list = []
    if not fte_df.empty:
        recent_fte = fte_df[fte_df["season"] >= season_year - 3].copy()
        recent_fte = recent_fte.dropna(subset=["score1", "score2"])
        games_list = recent_fte[["team1", "team2", "score1", "score2", "date", "neutral", "season"]].to_dict("records")

    bayesian_ratings = update_ratings(games_list, elo_dict)
    for team in NFL_TEAMS:
        if team not in bayesian_ratings:
            bayesian_ratings[team] = {"mu": elo_dict.get(team, 1500.0), "sigma": 75.0}

    # ── 7. Season simulation ────────────────────────────────────────────────
    log.info("Running season simulation...")
    remaining_schedule = []
    for game in all_games_for_prediction:
        if game.get("status", "") in ("STATUS_SCHEDULED", "STATUS_IN_PROGRESS"):
            remaining_schedule.append({
                "team_a": game["home_team"],
                "team_b": game["away_team"],
                "is_home_a": True,
                "neutral": bool(game.get("neutral", False)),
                "week": game.get("week", 0),
                "game_id": game.get("game_id", ""),
            })

    if not remaining_schedule:
        log.info("No remaining scheduled NFL games found; season simulation will use ratings only.")

    # Augment Bayesian ratings with win data
    for team in NFL_TEAMS:
        if team in bayesian_ratings:
            bayesian_ratings[team]["wins"] = standings.get(team, {}).get("wins", 0)

    try:
        season_sim = simulate_season(
            NFL_TEAMS, remaining_schedule, bayesian_ratings, n_sims=5000
        )
    except Exception as e:
        log.error(f"Season simulation failed: {e}")
        season_sim = {}
        for t in NFL_TEAMS:
            w = standings.get(t, {}).get("wins", 0)
            gp = standings.get(t, {}).get("games_played", 1) or 1
            win_pct = w / gp
            season_sim[t] = {
                "playoff_prob": min(0.95, max(0.05, win_pct * 1.2)),
                "division_win_prob": min(0.9, max(0.02, win_pct * 0.6)),
                "sb_prob": min(0.5, max(0.005, win_pct ** 2 * 0.5)),
                "wins_avg": round(win_pct * 17, 1),
            }

    # ── 8. Generate per-game predictions ────────────────────────────────────
    log.info(f"Generating predictions for {len(all_games_for_prediction)} games...")
    predictions_list = []

    for game in all_games_for_prediction:
        try:
            home = game["home_team"]
            away = game["away_team"]
            game_id = game["game_id"]
            neutral = bool(game.get("neutral", False))

            # Rest/travel adjustments — compute actual days since last game
            from datetime import date as _date_cls
            game_date_str = game.get("game_time", "")[:10]
            try:
                game_date = datetime.strptime(game_date_str, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                game_date = _date_cls.today()
            rest_home = nfl_days_since_last_game(game_history, home, game_date)
            rest_away = nfl_days_since_last_game(game_history, away, game_date)
            dist = travel_distance(home, away, is_home_a=True)
            rest_adj_home = rest_elo_adjustment(rest_home, rest_away)
            rest_adj_away = -rest_adj_home
            travel_adj_away = travel_elo_adjustment(dist)

            # Injury adjustments
            inj_adj_home, inj_adj_away = injury_elo_adjustment(
                home, away, injury_impacts
            )
            home_inj_impact = injury_impacts.get(home, {})
            away_inj_impact = injury_impacts.get(away, {})

            matchup_data = [{
                "game_id": game_id,
                "team_a": home, "team_b": away,
                "is_home_a": True, "neutral": neutral,
                "rest_diff": rest_home - rest_away,
                "travel_diff": dist,
                "turnover_diff": 0,
            }]

            # ELO prediction (includes rest, travel, and injury adjustments)
            elo_result = elo_predict_game(
                home, away, elo_dict, game_history,
                is_home_a=True, neutral=neutral,
                rest_adj_a=rest_adj_home + inj_adj_home,
                rest_adj_b=rest_adj_away + travel_adj_away + inj_adj_away
            )

            # Bayesian prediction
            bayes_result = bayes_predict(home, away, bayesian_ratings,
                                          is_home_a=True, neutral=neutral)

            # Efficiency prediction
            eff_result = efficiency_predict_game(home, away, efficiency_data,
                                                  pythagorean_data, not neutral, neutral)

            # Logistic prediction — fall back to ELO when model not trained
            log_prob = elo_result["prob"]
            if logistic_model and logistic_scaler and logistic_calibrator:
                log_preds = predict_matchups(
                    matchup_data, logistic_model, logistic_scaler, logistic_calibrator,
                    elo_dict, game_history, efficiency_data, pythagorean_data
                )
                if log_preds:
                    log_prob = log_preds[0]["logistic_prob"]

            # XGBoost prediction
            xgb_prob = None
            if xgb_model and xgb_scaler:
                xgb_preds = predict_xgboost(
                    matchup_data, xgb_model, xgb_scaler,
                    elo_dict, game_history, efficiency_data, pythagorean_data
                )
                if xgb_preds and xgb_preds[0]["xgb_prob"] is not None:
                    xgb_prob = xgb_preds[0]["xgb_prob"]

            # Ensemble
            # Issue 5: Learn + apply NFL ensemble weights
            if "nfl_weights" not in dir():
                nfl_weights = learn_nfl_weights(fte_df, pythagorean_data, efficiency_data, DEFAULT_WEIGHTS)
            ensemble_prob = ensemble_predict(
                logistic_prob=log_prob,
                xgb_prob=xgb_prob,
                elo_prob=elo_result["prob"],
                pyth_prob=eff_result["pyth_prob"],
                eff_prob=eff_result["eff_prob"],
                weights=nfl_weights,
            )

            # Monte Carlo
            mu_home = bayesian_ratings.get(home, {}).get("mu", elo_dict.get(home, 1500.0))
            mu_away = bayesian_ratings.get(away, {}).get("mu", elo_dict.get(away, 1500.0))
            sig_home = bayesian_ratings.get(home, {}).get("sigma", 75.0)
            sig_away = bayesian_ratings.get(away, {}).get("sigma", 75.0)

            mc_result = simulate_game(
                mu_home, mu_away, sig_home, sig_away,
                is_home_a=True, neutral=neutral, n=10000
            )

            # Market odds
            market_odds = match_odds_to_game(game, odds_map)
            market_home_prob = None
            market_edge = None
            kelly_pct = None

            if market_odds:
                market_home_prob = market_odds.get("home_prob")
                market_edge = round(ensemble_prob - market_home_prob, 4) if market_home_prob else None
                if market_home_prob:
                    kelly_pct = kelly_criterion(ensemble_prob, market_home_prob)

            adj_dict = {
                "rest_home": rest_home,
                "rest_away": rest_away,
                "rest_diff": rest_home - rest_away,
                "travel_dist_miles": round(dist, 0),
                "travel_adj": travel_adj_away,
                "home_elo_bonus": 0 if neutral else 65,
            }

            # Prediction drivers
            pred_drivers = generate_prediction_drivers(
                game, home, away, elo_dict, efficiency_data, injury_impacts, adj_dict
            )

            # Plain-English explanation
            winner = home if ensemble_prob >= 0.5 else away
            winner_prob = ensemble_prob if ensemble_prob >= 0.5 else 1 - ensemble_prob
            loser = away if ensemble_prob >= 0.5 else home
            elo_gap = abs(elo_dict.get(home, 1500) - elo_dict.get(away, 1500))
            confidence = "strong" if winner_prob > 0.70 else "moderate" if winner_prob > 0.60 else "slight"
            explanation = (
                f"The model gives {NFL_TEAM_NAMES.get(winner, winner)} a "
                f"{winner_prob*100:.1f}% win probability — a {confidence} favorite over "
                f"{NFL_TEAM_NAMES.get(loser, loser)}. "
                f"The ELO gap is {elo_gap:.0f} points"
                + (f" with home field adding approximately 65 ELO points." if not neutral else " at a neutral site.")
                + (f" Rest advantage favors {home if adj_dict['rest_diff'] > 0 else away}." if abs(adj_dict.get('rest_diff', 0)) >= 3 else "")
                + (f" Travel distance {round(dist)} miles factors into the away team rating." if dist > 1000 else "")
            )

            predictions_list.append({
                "game_id": game_id,
                "game_time": game["game_time"],
                "week": game.get("week", 0),
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
                    "ensemble_prob": ensemble_prob,
                    "logistic_prob": round(log_prob, 4),
                    "elo_prob": round(elo_result["prob"], 4),
                    "xgb_prob": xgb_prob,
                    "pyth_prob": eff_result["pyth_prob"],
                    "eff_prob": eff_result["eff_prob"],
                    "bayesian_prob": bayes_result["bayesian_prob"],
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
                "elo": {
                    "home": round(elo_dict.get(home, 1500.0), 1),
                    "away": round(elo_dict.get(away, 1500.0), 1),
                    "diff": round(elo_result["elo_diff"], 1),
                },
                "bayesian": {
                    "home_mu": bayes_result["mu_a"],
                    "home_sigma": bayes_result["sigma_a"],
                    "away_mu": bayes_result["mu_b"],
                    "away_sigma": bayes_result["sigma_b"],
                    "home_band": bayes_result["uncertainty_band_a"],
                    "away_band": bayes_result["uncertainty_band_b"],
                },
                "injuries": {
                    "home": [
                        {**p, "value_tier": value_tier_label(player_values.get(p.get("player", ""), 1.0))}
                        for p in injuries.get(home, [])[:5]
                    ],
                    "away": [
                        {**p, "value_tier": value_tier_label(player_values.get(p.get("player", ""), 1.0))}
                        for p in injuries.get(away, [])[:5]
                    ],
                },
                "injury_impact": {
                    "home_elo_penalty": home_inj_impact.get("elo_penalty", 0.0),
                    "away_elo_penalty": away_inj_impact.get("elo_penalty", 0.0),
                    "home_impact_score": home_inj_impact.get("impact_score", 0.0),
                    "away_impact_score": away_inj_impact.get("impact_score", 0.0),
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
            log.error(f"Error processing game {game.get('game_id', '?')}: {e}")

    # ── 9. Build leaderboard / ELO ratings ──────────────────────────────────
    log.info("Building leaderboard...")
    elo_ratings_list = []

    for team in NFL_TEAMS:
        elo = elo_dict.get(team, 1500.0)
        bayes = bayesian_ratings.get(team, {"mu": elo, "sigma": 75.0})
        pyth = pythagorean_data.get(team, {}).get("pyth", 0.5)
        net_eff = efficiency_data.get(team, {}).get("net_eff", 0.0)
        wins = standings.get(team, {}).get("wins", 0)
        losses = standings.get(team, {}).get("losses", 0)
        ties = standings.get(team, {}).get("ties", 0)
        playoff_prob = season_sim.get(team, {}).get("playoff_prob", 0.5)
        sb_prob = season_sim.get(team, {}).get("sb_prob", 0.03)
        trend = get_trend(game_history, team)
        hier = hierarchical_team_rating(team, efficiency_data)

        team_inj = injury_impacts.get(team, {})

        elo_ratings_list.append({
            "team": team,
            "team_name": NFL_TEAM_NAMES.get(team, team),
            "abbrev": team.lower(),
            "logo": f"https://a.espncdn.com/i/teamlogos/nfl/500/{team.lower()}.png",
            "elo": round(elo, 1),
            "sigma": round(bayes["sigma"], 1),
            "mu": round(bayes["mu"], 1),
            "lower_band": round(bayes["mu"] - bayes["sigma"], 1),
            "upper_band": round(bayes["mu"] + bayes["sigma"], 1),
            "pyth": round(pyth, 4),
            "net_eff": round(net_eff, 4),
            "off_eff": round(efficiency_data.get(team, {}).get("off_eff", 1.0), 4),
            "def_eff": round(efficiency_data.get(team, {}).get("def_eff", 1.0), 4),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "playoff_prob": round(playoff_prob, 4),
            "sb_prob": round(sb_prob, 4),
            "trend": trend,
            "offensive_rating": round(hier["offensive_rating"], 4),
            "defensive_rating": round(hier["defensive_rating"], 4),
            "injury_elo_penalty": team_inj.get("elo_penalty", 0.0),
            "injury_impact_score": team_inj.get("impact_score", 0.0),
            "injury_players_count": team_inj.get("total_players", 0),
        })

    elo_ratings_list.sort(key=lambda x: x["elo"], reverse=True)
    leaderboard_list = list(elo_ratings_list)  # same data, sorted by ELO

    # ── 10. Write JSON files ─────────────────────────────────────────────────
    log.info("Writing JSON output files...")

    week_num = current_week if current_week else (scoreboard_games[0].get("week", 0) if scoreboard_games else 0)

    predictions_json = {
        "updated": now_utc,
        "season": season_year,
        "week": week_num,
        "is_offseason": is_offseason,
        "games": predictions_list,
    }

    elo_ratings_json = {
        "updated": now_utc,
        "season": season_year,
        "ratings": elo_ratings_list,
    }

    leaderboard_json = {
        "updated": now_utc,
        "season": season_year,
        "teams": leaderboard_list,
    }

    model_metrics_json = {
        "updated": now_utc,
        "log_loss": model_metrics.get("log_loss"),
        "brier_score": model_metrics.get("brier_score"),
        "auc": model_metrics.get("auc"),
        "calibration_buckets": model_metrics.get("calibration_buckets", []),
        "historical_accuracy": model_metrics.get("historical_accuracy", []),
        "n_training_games": len(fte_df) if not fte_df.empty else 0,
        "xgboost_available": xgb_model is not None,
    }

    (DATA_DIR / "predictions.json").write_text(
        json.dumps(predictions_json, indent=2, default=str)
    )
    (DATA_DIR / "elo_ratings.json").write_text(
        json.dumps(elo_ratings_json, indent=2, default=str)
    )
    (DATA_DIR / "leaderboard.json").write_text(
        json.dumps(leaderboard_json, indent=2, default=str)
    )
    (DATA_DIR / "model_metrics.json").write_text(
        json.dumps(model_metrics_json, indent=2, default=str)
    )

    # Also write league-namespaced files for multi-league dashboard
    (DATA_DIR / "nfl_predictions.json").write_text(
        json.dumps(predictions_json, indent=2, default=str)
    )
    (DATA_DIR / "nfl_leaderboard.json").write_text(
        json.dumps(leaderboard_json, indent=2, default=str)
    )

    log.info("=== Update complete ===")
    log.info(f"  Games predicted: {len(predictions_list)}")
    log.info(f"  Teams in leaderboard: {len(leaderboard_list)}")
    log.info(f"  Model metrics: {model_metrics.get('log_loss', 'N/A')}")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
