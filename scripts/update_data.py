"""
Main orchestration script for NFL prediction model.
Downloads data, runs all models, writes JSON output files.
Run daily via GitHub Actions.
"""

import os
import sys
import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.elo_model import (compute_elo, predict_game as elo_predict_game,
                             get_trend, update_elo_with_result)
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
                                   american_to_prob, remove_vig)
from model.injury_model import compute_all_team_impacts, injury_elo_adjustment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
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


def fetch_espn_scoreboard():
    log.info("Fetching ESPN scoreboard...")
    data = safe_get(f"{ESPN_BASE}/scoreboard")
    if not data:
        return []

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

            game = {
                "game_id": event["id"],
                "game_time": event.get("date", ""),
                "status": event.get("status", {}).get("type", {}).get("name", ""),
                "week": data.get("week", {}).get("number", 0),
                "season": data.get("season", {}).get("year", datetime.now().year),
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

    log.info(f"Found {len(games)} scoreboard games")
    return games


def fetch_espn_standings():
    log.info("Fetching ESPN standings...")
    data = safe_get(f"{ESPN_BASE}/standings")
    if not data:
        return {}

    standings = {}
    try:
        for group in data.get("children", []):
            for div in group.get("children", []):
                for entry in div.get("standings", {}).get("entries", []):
                    team_abbrev = abbrev_norm(
                        entry["team"]["abbreviation"]
                    )
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
        for item in data.get("injuries", []):
            team_abbrev = abbrev_norm(
                item.get("team", {}).get("abbreviation", "")
            )
            if team_abbrev:
                if team_abbrev not in injuries:
                    injuries[team_abbrev] = []
                injuries[team_abbrev].append({
                    "player": item.get("athlete", {}).get("displayName", ""),
                    "status": item.get("status", ""),
                    "position": item.get("athlete", {}).get("position", {}).get("abbreviation", "")
                })
    except Exception as e:
        log.warning(f"Error parsing injuries: {e}")

    return injuries


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


def fetch_espn_season_results(season_year):
    """
    Fetch all completed NFL games for a season from ESPN team schedules.
    Returns a list of dicts: {game_id, date, team_a, team_b, score_a, score_b,
                               is_home_a, neutral}
    """
    log.info(f"Fetching ESPN NFL season {season_year} results for live ELO update...")
    games = {}  # game_id → game dict (dedup since each game appears for both teams)

    # ESPN team IDs for NFL — abbrev → numeric ESPN id
    ESPN_NFL_TEAM_IDS = {
        "ARI": 22, "ATL": 1, "BAL": 33, "BUF": 2, "CAR": 29, "CHI": 3,
        "CIN": 4, "CLE": 5, "DAL": 6, "DEN": 7, "DET": 8, "GB": 9,
        "HOU": 34, "IND": 11, "JAX": 30, "KC": 12, "LAC": 24, "LAR": 14,
        "LV": 13, "MIA": 15, "MIN": 16, "NE": 17, "NO": 18, "NYG": 19,
        "NYJ": 20, "PHI": 21, "PIT": 23, "SEA": 26, "SF": 25, "TB": 27,
        "TEN": 10, "WAS": 28,
    }

    for abbrev, team_id in ESPN_NFL_TEAM_IDS.items():
        url = f"{ESPN_BASE}/teams/{team_id}/schedule"
        data = safe_get(url, params={"season": season_year})
        if not data:
            continue
        try:
            for event in data.get("events", []):
                comp_list = event.get("competitions", [])
                if not comp_list:
                    continue
                comp = comp_list[0]
                status_name = comp.get("status", {}).get("type", {}).get("name", "")
                if status_name != "STATUS_FINAL":
                    continue

                game_id = event.get("id", "")
                if game_id in games:
                    continue

                competitors = comp.get("competitors", [])
                if len(competitors) < 2:
                    continue

                home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home or not away:
                    continue

                home_abbrev = abbrev_norm(home["team"]["abbreviation"])
                away_abbrev = abbrev_norm(away["team"]["abbreviation"])
                home_score = int(home.get("score", 0) or 0)
                away_score = int(away.get("score", 0) or 0)
                neutral = int(comp.get("neutralSite", False))
                date_str = event.get("date", "")

                games[game_id] = {
                    "game_id": game_id,
                    "date": date_str,
                    "team_a": home_abbrev,
                    "team_b": away_abbrev,
                    "score_a": home_score,
                    "score_b": away_score,
                    "is_home_a": True,
                    "neutral": bool(neutral),
                }
        except Exception as e:
            log.debug(f"Error fetching schedule for team {abbrev}: {e}")

    result = sorted(games.values(), key=lambda g: g["date"])
    log.info(f"Found {len(result)} completed ESPN games for season {season_year}")
    return result


def update_elo_from_espn(elo_dict, game_history, fte_df, season_year):
    """
    Continue updating ELO with ESPN live results not yet in FTE data.
    Only processes games dated after the FTE dataset's last entry.
    """
    if fte_df.empty:
        fte_cutoff = "2000-01-01"
    else:
        fte_cutoff = pd.to_datetime(fte_df["date"]).max().isoformat()[:10]

    log.info(f"FTE data ends at {fte_cutoff}. Fetching newer ESPN results...")
    espn_games = fetch_espn_season_results(season_year)

    new_count = 0
    for game in espn_games:
        game_date = game["date"][:10]
        if game_date <= fte_cutoff:
            continue
        update_elo_with_result(
            elo_dict, game_history,
            game["team_a"], game["team_b"],
            game["score_a"], game["score_b"],
            is_home_a=game["is_home_a"],
            neutral=game["neutral"],
        )
        new_count += 1

    log.info(f"Updated ELO with {new_count} live ESPN games past FTE cutoff")
    return espn_games  # return for Bayesian update use


def compute_prediction_drivers(home, away, elo_result, eff_result, adj,
                                home_inj_impact, away_inj_impact,
                                home_name, away_name):
    """
    Build a ranked list of the top prediction drivers for a game.
    Each driver: {type, team, description, impact_pct}
    impact_pct is the estimated ±% change on home win probability.
    """
    from scipy.special import expit, logit as sp_logit

    drivers = []

    base_prob = elo_result.get("prob", 0.5)

    def prob_delta_from_elo(elo_change):
        """Estimate probability change from adding `elo_change` to home ELO."""
        elo_diff = elo_result.get("elo_diff", 0.0)
        new_prob = 1.0 / (1.0 + 10 ** (-(elo_diff + elo_change) / 400))
        return round((new_prob - base_prob) * 100, 1)

    # 1 — ELO gap
    elo_diff = elo_result.get("elo_diff", 0.0)
    if abs(elo_diff) > 30:
        favored = home_name if elo_diff > 0 else away_name
        impact = round((base_prob - 0.5) * 100, 1)
        drivers.append({
            "type": "elo",
            "description": f"ELO advantage: {favored} +{abs(elo_diff):.0f} rating pts",
            "impact_pct": impact,
        })

    # 2 — Home injury penalty
    home_pen = home_inj_impact.get("elo_penalty", 0.0)
    if home_pen > 10:
        key_out = home_inj_impact.get("key_players_out", [])
        detail = key_out[0]["player"] if key_out else "key player"
        drivers.append({
            "type": "injury",
            "team": home,
            "description": f"{home_name}: {detail} OUT (−{home_pen:.0f} ELO impact)",
            "impact_pct": prob_delta_from_elo(-home_pen),
        })

    # 3 — Away injury penalty
    away_pen = away_inj_impact.get("elo_penalty", 0.0)
    if away_pen > 10:
        key_out = away_inj_impact.get("key_players_out", [])
        detail = key_out[0]["player"] if key_out else "key player"
        drivers.append({
            "type": "injury",
            "team": away,
            "description": f"{away_name}: {detail} OUT (−{away_pen:.0f} ELO impact)",
            "impact_pct": prob_delta_from_elo(away_pen),
        })

    # 4 — Rest advantage
    rest_diff = adj.get("rest_diff", 0)
    if abs(rest_diff) >= 3:
        rest_team = home_name if rest_diff > 0 else away_name
        rest_adj_pts = rest_diff * 1.5
        drivers.append({
            "type": "rest",
            "description": f"Rest advantage: {rest_team} +{abs(rest_diff)} days",
            "impact_pct": prob_delta_from_elo(rest_adj_pts if rest_diff > 0 else -rest_adj_pts),
        })

    # 5 — Travel penalty
    travel_mi = adj.get("travel_dist_miles", 0)
    travel_adj = adj.get("travel_adj", 0)
    if travel_mi > 1500:
        drivers.append({
            "type": "travel",
            "description": f"Travel penalty: away team {travel_mi:.0f} mi ({travel_adj:+.0f} ELO)",
            "impact_pct": prob_delta_from_elo(-travel_adj),
        })

    # 6 — Efficiency mismatch
    net_eff_home = eff_result.get("home_net_eff", 0.0)
    net_eff_away = eff_result.get("away_net_eff", 0.0)
    eff_gap = net_eff_home - net_eff_away
    if abs(eff_gap) > 0.08:
        eff_team = home_name if eff_gap > 0 else away_name
        drivers.append({
            "type": "efficiency",
            "description": (f"Efficiency gap: {home_name} net {net_eff_home:+.3f} "
                            f"vs {away_name} net {net_eff_away:+.3f}"),
            "impact_pct": round(eff_gap * 30, 1),
        })

    # 7 — Pythagorean signal
    pyth_prob = eff_result.get("pyth_prob", 0.5)
    if abs(pyth_prob - 0.5) > 0.08:
        pyth_team = home_name if pyth_prob > 0.5 else away_name
        drivers.append({
            "type": "pythagorean",
            "description": f"Pythagorean expectation favors {pyth_team} ({pyth_prob*100:.1f}%)",
            "impact_pct": round((pyth_prob - 0.5) * 100, 1),
        })

    # Sort by absolute impact, return top 5
    drivers.sort(key=lambda d: abs(d.get("impact_pct", 0)), reverse=True)
    return drivers[:5]


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
    scoreboard_games = fetch_espn_scoreboard()
    standings = fetch_espn_standings()
    injuries = fetch_espn_injuries()
    odds_map = fetch_betting_odds(odds_api_key)

    # ── 1b. Compute injury impacts ──────────────────────────────────────────
    log.info("Computing injury impact scores...")
    injury_impacts = compute_all_team_impacts(injuries)

    season_year = datetime.now().year
    current_month = datetime.now().month
    # NFL season runs Sep–Feb; if before September treat as prior season
    if current_month < 8:
        season_year -= 1

    is_offseason = len(scoreboard_games) == 0

    # ── 2. Build ELO ratings (FTE historical + ESPN live continuation) ───────
    log.info("Computing ELO ratings from FTE data...")
    if not fte_df.empty:
        elo_dict, game_history = compute_elo(fte_df)
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

    # ── 2b. Continue ELO with live ESPN results past FTE cutoff ─────────────
    try:
        espn_completed = update_elo_from_espn(elo_dict, game_history, fte_df, season_year)
    except Exception as e:
        log.warning(f"Live ELO update failed (non-fatal): {e}")
        espn_completed = []

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
                    game_history, efficiency_data, pythagorean_data
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

    # Append ESPN live completed games to Bayesian training set
    for g in espn_completed:
        game_date = g.get("date", "")[:10]
        if not fte_df.empty:
            fte_cutoff = pd.to_datetime(fte_df["date"]).max().isoformat()[:10]
            if game_date <= fte_cutoff:
                continue
        games_list.append({
            "team1": g["team_a"], "team2": g["team_b"],
            "score1": g["score_a"], "score2": g["score_b"],
            "date": g["date"], "neutral": int(g.get("neutral", False)),
            "season": season_year,
        })

    bayesian_ratings = update_ratings(games_list, elo_dict)
    for team in NFL_TEAMS:
        if team not in bayesian_ratings:
            bayesian_ratings[team] = {"mu": elo_dict.get(team, 1500.0), "sigma": 75.0}

    # ── 7. Season simulation ────────────────────────────────────────────────
    log.info("Running season simulation...")
    remaining_schedule = []  # Would populate from ESPN schedule API
    for game in scoreboard_games:
        if game.get("status", "") in ("STATUS_SCHEDULED", "STATUS_IN_PROGRESS"):
            remaining_schedule.append({
                "team_a": game["home_team"],
                "team_b": game["away_team"],
                "is_home_a": True,
                "neutral": bool(game.get("neutral", False)),
                "week": game.get("week", 0),
                "game_id": game.get("game_id", ""),
            })

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
        season_sim = {t: {"playoff_prob": 0.5, "division_win_prob": 0.25,
                          "sb_prob": 0.03, "wins_avg": 8.5} for t in NFL_TEAMS}

    # ── 8. Generate per-game predictions ────────────────────────────────────
    log.info(f"Generating predictions for {len(scoreboard_games)} games...")
    predictions_list = []

    for game in scoreboard_games:
        try:
            home = game["home_team"]
            away = game["away_team"]
            game_id = game["game_id"]
            neutral = bool(game.get("neutral", False))

            # Rest/travel adjustments
            rest_home = 7  # default
            rest_away = 7
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
            # Augment eff_result with net efficiency values for prediction drivers
            eff_result["home_net_eff"] = efficiency_data.get(home, {}).get("net_eff", 0.0)
            eff_result["away_net_eff"] = efficiency_data.get(away, {}).get("net_eff", 0.0)

            # Logistic prediction
            log_prob = 0.5
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
            ensemble_prob = ensemble_predict(
                logistic_prob=log_prob,
                xgb_prob=xgb_prob,
                elo_prob=elo_result["prob"],
                pyth_prob=eff_result["pyth_prob"],
                eff_prob=eff_result["eff_prob"]
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

            predictions_list.append({
                "game_id": game_id,
                "game_time": game["game_time"],
                "week": game.get("week", 0),
                "status": game.get("status", ""),
                "home_team": home,
                "away_team": away,
                "home_name": game.get("home_name", home),
                "away_name": game.get("away_name", away),
                "home_logo": game.get("home_logo", ""),
                "away_logo": game.get("away_logo", ""),
                "neutral": neutral,
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
                "adjustments": {
                    "rest_home": rest_home,
                    "rest_away": rest_away,
                    "rest_diff": rest_home - rest_away,
                    "travel_dist_miles": round(dist, 0),
                    "travel_adj": travel_adj_away,
                    "home_elo_bonus": 0 if neutral else 65,
                },
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
                    "home": injuries.get(home, [])[:5],
                    "away": injuries.get(away, [])[:5],
                },
                "injury_impact": {
                    "home_elo_penalty": home_inj_impact.get("elo_penalty", 0.0),
                    "away_elo_penalty": away_inj_impact.get("elo_penalty", 0.0),
                    "home_impact_score": home_inj_impact.get("impact_score", 0.0),
                    "away_impact_score": away_inj_impact.get("impact_score", 0.0),
                    "home_key_players_out": home_inj_impact.get("key_players_out", []),
                    "away_key_players_out": away_inj_impact.get("key_players_out", []),
                },
                "prediction_drivers": compute_prediction_drivers(
                    home, away, elo_result, eff_result,
                    {
                        "rest_diff": rest_home - rest_away,
                        "travel_dist_miles": round(dist, 0),
                        "travel_adj": travel_adj_away,
                    },
                    home_inj_impact, away_inj_impact,
                    game.get("home_name", home), game.get("away_name", away),
                ),
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

    week_num = scoreboard_games[0].get("week", 0) if scoreboard_games else 0

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

    # Build flat injuries list for nfl_injuries.json
    injuries_flat = []
    for team_abbrev, player_list in injuries.items():
        for p in player_list:
            injuries_flat.append({
                "team": team_abbrev,
                "player_name": p.get("player", ""),
                "position": p.get("position", ""),
                "status": p.get("status", ""),
                "injury_description": "",
            })

    (DATA_DIR / "nfl_predictions.json").write_text(
        json.dumps(predictions_json, indent=2, default=str)
    )
    (DATA_DIR / "nfl_elo_ratings.json").write_text(
        json.dumps(elo_ratings_json, indent=2, default=str)
    )
    (DATA_DIR / "nfl_leaderboard.json").write_text(
        json.dumps(leaderboard_json, indent=2, default=str)
    )
    (DATA_DIR / "nfl_model_metrics.json").write_text(
        json.dumps(model_metrics_json, indent=2, default=str)
    )
    (DATA_DIR / "nfl_injuries.json").write_text(
        json.dumps({"updated": now_utc, "injuries": injuries_flat}, indent=2, default=str)
    )

    log.info("=== Update complete ===")
    log.info(f"  Games predicted: {len(predictions_list)}")
    log.info(f"  Teams in leaderboard: {len(leaderboard_list)}")
    log.info(f"  Model metrics: {model_metrics.get('log_loss', 'N/A')}")


if __name__ == "__main__":
    run()
