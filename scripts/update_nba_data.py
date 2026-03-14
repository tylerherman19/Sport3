"""
NBA Prediction Model Orchestrator.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.nba_model import (
    NBA_TEAMS, NBA_TEAM_NAMES, ESPN_NBA_TEAM_IDS,
    compute_nba_elo, nba_predict_game, nba_get_trend,
    nba_pythagorean, compute_nba_efficiency, nba_efficiency_predict,
    nba_travel_distance, nba_travel_adjustment, nba_rest_adjustment,
    nba_update_bayesian, nba_bayes_predict,
    nba_simulate_game, nba_ensemble_predict,
    nba_injury_impact, compute_all_nba_team_impacts, nba_injury_elo_adjustment,
    nba_abbrev_norm, build_nba_features, NBA_DEFAULT_WEIGHTS,
)
from model.ensemble_model import train_xgboost, american_to_prob, remove_vig, kelly_criterion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

ESPN_NBA_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ODDS_NBA_BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"


def safe_get(url, params=None, timeout=30):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return None


def fetch_nba_scoreboard():
    log.info("Fetching NBA scoreboard...")
    data = safe_get(f"{ESPN_NBA_BASE}/scoreboard")
    if not data:
        return []

    games = []
    for event in data.get("events", []):
        try:
            comp = event["competitions"][0]
            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue

            home_abbrev = nba_abbrev_norm(home["team"]["abbreviation"])
            away_abbrev = nba_abbrev_norm(away["team"]["abbreviation"])

            # Parse record from competitors records array
            def parse_record(comp_entry):
                for rec in comp_entry.get("records", []):
                    if rec.get("name") in ("overall", "total"):
                        summary = rec.get("summary", "0-0")
                        parts = summary.split("-")
                        if len(parts) >= 2:
                            return int(parts[0]), int(parts[1])
                return 0, 0

            home_wins, home_losses = parse_record(home)
            away_wins, away_losses = parse_record(away)

            game = {
                "game_id": event["id"],
                "game_time": event.get("date", ""),
                "status": comp.get("status", {}).get("type", {}).get("name", ""),
                "home_team": home_abbrev,
                "away_team": away_abbrev,
                "home_name": home["team"].get("displayName", home_abbrev),
                "away_name": away["team"].get("displayName", away_abbrev),
                "home_logo": f"https://a.espncdn.com/i/teamlogos/nba/500/{home['team']['abbreviation'].lower()}.png",
                "away_logo": f"https://a.espncdn.com/i/teamlogos/nba/500/{away['team']['abbreviation'].lower()}.png",
                "home_score": int(home.get("score", 0) or 0),
                "away_score": int(away.get("score", 0) or 0),
                "home_wins": home_wins, "home_losses": home_losses,
                "away_wins": away_wins, "away_losses": away_losses,
                "neutral": int(comp.get("neutralSite", False)),
            }
            games.append(game)
        except Exception as e:
            log.debug(f"Error parsing NBA game: {e}")

    log.info(f"Found {len(games)} NBA scoreboard games")
    return games


def fetch_nba_standings(scoreboard_games):
    """Build standings from scoreboard and team schedule records."""
    standings = {}
    # Seed from scoreboard data
    for g in scoreboard_games:
        for side in ("home", "away"):
            abbrev = g[f"{side}_team"]
            wins   = g.get(f"{side}_wins", 0)
            losses = g.get(f"{side}_losses", 0)
            if abbrev and abbrev not in standings:
                standings[abbrev] = {
                    "wins": wins, "losses": losses,
                    "games_played": wins + losses,
                    "points_for": 0, "points_against": 0,
                }
    return standings


def fetch_nba_injuries():
    log.info("Fetching NBA injuries...")
    data = safe_get(f"{ESPN_NBA_BASE}/injuries")
    if not data:
        return {}

    injuries = {}
    try:
        for team_entry in data.get("injuries", []):
            for player_inj in team_entry.get("injuries", []):
                athlete = player_inj.get("athlete", {})
                team_abbrev = nba_abbrev_norm(
                    athlete.get("team", {}).get("abbreviation", "") or
                    team_entry.get("team", {}).get("abbreviation", "")
                )
                if not team_abbrev:
                    continue
                if team_abbrev not in injuries:
                    injuries[team_abbrev] = []
                injuries[team_abbrev].append({
                    "player": athlete.get("displayName", ""),
                    "status": player_inj.get("status", ""),
                    "position": athlete.get("position", {}).get("abbreviation", ""),
                    "injury_description": player_inj.get("shortComment", ""),
                })
    except Exception as e:
        log.warning(f"Error parsing NBA injuries: {e}")

    return injuries


def fetch_nba_season_games(season_year):
    """
    Fetch completed NBA games for a season from ESPN team schedules.
    Returns deduped list of {team_a, team_b, score_a, score_b, is_home_a, neutral, season, date}.
    """
    log.info(f"Fetching NBA season {season_year} games from ESPN schedules...")
    games = {}

    for abbrev, team_id in ESPN_NBA_TEAM_IDS.items():
        url = f"{ESPN_NBA_BASE}/teams/{team_id}/schedule"
        data = safe_get(url, params={"season": season_year})
        if not data:
            continue
        try:
            for event in data.get("events", []):
                comp_list = event.get("competitions", [])
                if not comp_list:
                    continue
                comp = comp_list[0]
                status = comp.get("status", {}).get("type", {}).get("name", "")
                if status != "STATUS_FINAL":
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

                home_abbrev = nba_abbrev_norm(home["team"]["abbreviation"])
                away_abbrev = nba_abbrev_norm(away["team"]["abbreviation"])
                home_score = int(home.get("score", 0) or 0)
                away_score = int(away.get("score", 0) or 0)
                neutral = bool(comp.get("neutralSite", False))
                date_str = event.get("date", "")

                games[game_id] = {
                    "game_id": game_id,
                    "date": date_str,
                    "team_a": home_abbrev,
                    "team_b": away_abbrev,
                    "score_a": home_score,
                    "score_b": away_score,
                    "is_home_a": True,
                    "neutral": neutral,
                    "season": season_year,
                }
        except Exception as e:
            log.debug(f"Error fetching NBA schedule for {abbrev}: {e}")

    result = sorted(games.values(), key=lambda g: g["date"])
    log.info(f"Found {len(result)} completed NBA games for season {season_year}")
    return result


def fetch_nba_betting_odds(api_key):
    if not api_key:
        log.info("No ODDS_API_KEY, skipping NBA betting lines")
        return {}

    log.info("Fetching NBA betting odds...")
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    }
    data = safe_get(ODDS_NBA_BASE, params=params)
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

            home_list, away_list = [], []
            for bm in bookmakers:
                for market in bm.get("markets", []):
                    if market["key"] == "h2h":
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == home_team:
                                home_list.append(float(outcome["price"]))
                            elif outcome["name"] == away_team:
                                away_list.append(float(outcome["price"]))

            if home_list and away_list:
                avg_h = np.mean(home_list)
                avg_a = np.mean(away_list)
                raw_h = american_to_prob(avg_h)
                raw_a = american_to_prob(avg_a)
                clean_h, clean_a = remove_vig(raw_h, raw_a)
                odds_map[f"{away_team}_at_{home_team}"] = {
                    "home_prob": round(clean_h, 4),
                    "away_prob": round(clean_a, 4),
                    "home_american": avg_h,
                    "away_american": avg_a,
                    "home_team_name": home_team,
                    "away_team_name": away_team,
                }
        except Exception as e:
            log.debug(f"Error parsing NBA odds: {e}")

    log.info(f"Got NBA odds for {len(odds_map)} games")
    return odds_map


def match_odds_to_game(game, odds_map):
    home = game.get("home_name", "")
    away = game.get("away_name", "")
    for key, odds in odds_map.items():
        h = odds.get("home_team_name", "")
        a = odds.get("away_team_name", "")
        if (home.lower() in h.lower() or h.lower() in home.lower()) and \
           (away.lower() in a.lower() or a.lower() in away.lower()):
            return odds
    return None


def compute_nba_prediction_drivers(home, away, elo_result, eff_result,
                                    adj, home_inj, away_inj,
                                    home_name, away_name):
    """Build ranked prediction drivers for an NBA game."""
    drivers = []
    base_prob = elo_result.get("prob", 0.5)

    def prob_delta(elo_change):
        elo_diff = elo_result.get("elo_diff", 0.0)
        new_prob = 1.0 / (1.0 + 10 ** (-(elo_diff + elo_change) / 400))
        return round((new_prob - base_prob) * 100, 1)

    elo_diff = elo_result.get("elo_diff", 0.0)
    if abs(elo_diff) > 30:
        favored = home_name if elo_diff > 0 else away_name
        drivers.append({
            "type": "elo",
            "description": f"ELO advantage: {favored} +{abs(elo_diff):.0f} rating pts",
            "impact_pct": round((base_prob - 0.5) * 100, 1),
        })

    home_pen = home_inj.get("elo_penalty", 0.0)
    if home_pen > 10:
        key_out = home_inj.get("key_players_out", [])
        detail = key_out[0]["player"] if key_out else "key player"
        drivers.append({
            "type": "injury", "team": home,
            "description": f"{home_name}: {detail} OUT (−{home_pen:.0f} ELO impact)",
            "impact_pct": prob_delta(-home_pen),
        })

    away_pen = away_inj.get("elo_penalty", 0.0)
    if away_pen > 10:
        key_out = away_inj.get("key_players_out", [])
        detail = key_out[0]["player"] if key_out else "key player"
        drivers.append({
            "type": "injury", "team": away,
            "description": f"{away_name}: {detail} OUT (−{away_pen:.0f} ELO impact)",
            "impact_pct": prob_delta(away_pen),
        })

    rest_diff = adj.get("rest_diff", 0)
    if abs(rest_diff) >= 1:
        rest_team = home_name if rest_diff > 0 else away_name
        rest_adj_pts = rest_diff * 2.0
        drivers.append({
            "type": "rest",
            "description": f"Rest advantage: {rest_team} +{abs(rest_diff)} days",
            "impact_pct": prob_delta(rest_adj_pts if rest_diff > 0 else -rest_adj_pts),
        })

    travel_mi = adj.get("travel_dist_miles", 0)
    travel_adj = adj.get("travel_adj", 0)
    if travel_mi > 1500:
        drivers.append({
            "type": "travel",
            "description": f"Travel penalty: away team {travel_mi:.0f} mi ({travel_adj:+.0f} ELO)",
            "impact_pct": prob_delta(-travel_adj),
        })

    net_home = eff_result.get("net_rating_a", 0.0)
    net_away = eff_result.get("net_rating_b", 0.0)
    net_gap = net_home - net_away
    if abs(net_gap) > 3.0:
        drivers.append({
            "type": "efficiency",
            "description": (f"Net rating gap: {home_name} {net_home:+.1f} "
                            f"vs {away_name} {net_away:+.1f}"),
            "impact_pct": round(net_gap * 1.5, 1),
        })

    drivers.sort(key=lambda d: abs(d.get("impact_pct", 0)), reverse=True)
    return drivers[:5]


def run():
    log.info("=== NBA Prediction Model Update Starting ===")
    now_utc = datetime.now(timezone.utc).isoformat()
    odds_api_key = os.environ.get("ODDS_API_KEY", "")

    season_year = datetime.now().year
    current_month = datetime.now().month
    # NBA season runs Oct–Jun; if before October treat as prior season
    if current_month < 9:
        season_year -= 1

    # ── 1. Fetch live data ───────────────────────────────────────────────────
    scoreboard_games = fetch_nba_scoreboard()
    standings = fetch_nba_standings(scoreboard_games)
    injuries = fetch_nba_injuries()
    odds_map = fetch_nba_betting_odds(odds_api_key)

    # ── 2. Compute injury impacts ────────────────────────────────────────────
    log.info("Computing NBA injury impact scores...")
    injury_impacts = compute_all_nba_team_impacts(injuries)

    is_offseason = len(scoreboard_games) == 0

    # ── 3. Bootstrap ELO from ESPN schedule history (3 seasons) ─────────────
    log.info("Bootstrapping NBA ELO from ESPN historical schedules...")
    all_games = []
    for yr in range(season_year - 2, season_year + 1):
        try:
            season_games = fetch_nba_season_games(yr)
            all_games.extend(season_games)
        except Exception as e:
            log.warning(f"Failed to fetch NBA season {yr}: {e}")

    if all_games:
        elo_dict, game_history = compute_nba_elo(all_games)
        log.info(f"NBA ELO computed from {len(all_games)} games")
    else:
        elo_dict = {t: 1500.0 for t in NBA_TEAMS}
        game_history = {t: [] for t in NBA_TEAMS}
        log.warning("No NBA historical games found; using default ELO=1500")

    for team in NBA_TEAMS:
        if team not in elo_dict:
            elo_dict[team] = 1500.0
        if team not in game_history:
            game_history[team] = []

    # ── 4. Build efficiency data ─────────────────────────────────────────────
    log.info("Computing NBA efficiency data...")
    # Estimate points_for / points_against from completed games
    pts_by_team = {t: {"points_for": 0, "points_against": 0, "games_played": 0}
                   for t in NBA_TEAMS}
    for g in all_games:
        ta, tb = g["team_a"], g["team_b"]
        sa, sb = g["score_a"], g["score_b"]
        if ta in pts_by_team:
            pts_by_team[ta]["points_for"] += sa
            pts_by_team[ta]["points_against"] += sb
            pts_by_team[ta]["games_played"] += 1
        if tb in pts_by_team:
            pts_by_team[tb]["points_for"] += sb
            pts_by_team[tb]["points_against"] += sa
            pts_by_team[tb]["games_played"] += 1
    # Fill from standings if available
    for team, s in standings.items():
        if team in pts_by_team and s.get("points_for", 0) > 0:
            pts_by_team[team].update(s)

    efficiency_data = compute_nba_efficiency(pts_by_team)

    # ── 5. Train logistic regression ─────────────────────────────────────────
    log.info("Training NBA logistic regression...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

    logistic_model = logistic_scaler = logistic_calibrator = None
    model_metrics = {"log_loss": None, "brier_score": None, "auc": None,
                     "calibration_buckets": [], "historical_accuracy": [],
                     "n_training_games": len(all_games)}

    if len(all_games) > 100:
        try:
            X, y = build_nba_features(all_games, elo_dict, game_history, efficiency_data)
            if len(X) > 50:
                logistic_scaler = StandardScaler()
                X_sc = logistic_scaler.fit_transform(X)
                clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                calibrated = CalibratedClassifierCV(clf, method="isotonic", cv=3)
                calibrated.fit(X_sc, y)
                logistic_model = calibrated
                logistic_calibrator = calibrated  # same object

                probs = calibrated.predict_proba(X_sc)[:, 1]
                model_metrics["log_loss"] = round(float(log_loss(y, probs)), 4)
                model_metrics["brier_score"] = round(float(brier_score_loss(y, probs)), 4)
                model_metrics["auc"] = round(float(roc_auc_score(y, probs)), 4)
                log.info(f"NBA logistic trained. Log loss: {model_metrics['log_loss']:.4f}")
        except Exception as e:
            log.error(f"NBA logistic training failed: {e}")

    # ── 6. Train XGBoost ─────────────────────────────────────────────────────
    log.info("Training NBA XGBoost...")
    xgb_model = xgb_scaler = None
    if len(all_games) > 100:
        try:
            X, y = build_nba_features(all_games, elo_dict, game_history, efficiency_data)
            if len(X) > 50:
                xgb_model, xgb_scaler = train_xgboost(X, y)
                if xgb_model:
                    model_metrics["xgboost_available"] = True
                    log.info("NBA XGBoost trained")
        except Exception as e:
            log.error(f"NBA XGBoost training failed: {e}")

    # ── 7. Bayesian ratings ──────────────────────────────────────────────────
    log.info("Computing NBA Bayesian ratings...")
    bayesian_ratings = nba_update_bayesian(all_games, elo_dict)

    # ── 8. Generate per-game predictions ─────────────────────────────────────
    log.info(f"Generating NBA predictions for {len(scoreboard_games)} games...")
    predictions_list = []

    for game in scoreboard_games:
        try:
            home = game["home_team"]
            away = game["away_team"]
            game_id = game["game_id"]
            neutral = bool(game.get("neutral", False))

            rest_home = 2  # default NBA rest
            rest_away = 2
            dist = nba_travel_distance(home, away, is_home_a=True)
            rest_adj_home = nba_rest_adjustment(rest_home, rest_away)
            rest_adj_away = nba_rest_adjustment(rest_away, rest_home)
            travel_adj_away = nba_travel_adjustment(dist)

            inj_adj_home, inj_adj_away = nba_injury_elo_adjustment(home, away, injury_impacts)
            home_inj = injury_impacts.get(home, {})
            away_inj = injury_impacts.get(away, {})

            elo_result = nba_predict_game(
                home, away, elo_dict, game_history,
                is_home_a=True, neutral=neutral,
                rest_adj_a=rest_adj_home + inj_adj_home,
                rest_adj_b=rest_adj_away + travel_adj_away + inj_adj_away,
            )

            bayes_result = nba_bayes_predict(home, away, bayesian_ratings,
                                              is_home_a=True, neutral=neutral)

            eff_result = nba_efficiency_predict(home, away, efficiency_data,
                                                 is_home_a=True, neutral=neutral)

            # Logistic
            log_prob = 0.5
            if logistic_model and logistic_scaler:
                try:
                    eff_h = efficiency_data.get(home, {})
                    eff_a = efficiency_data.get(away, {})
                    hfa_fl = 1.0 if not neutral else 0.0
                    feat = np.array([[
                        elo_result["elo_diff"],
                        hfa_fl,
                        eff_h.get("net_rating", 0) - eff_a.get("net_rating", 0),
                        eff_h.get("off_rating", 115) - eff_a.get("off_rating", 115),
                        eff_h.get("def_rating", 115) - eff_a.get("def_rating", 115),
                        eff_h.get("pyth", 0.5) - eff_a.get("pyth", 0.5),
                        0.0,
                    ]], dtype=np.float32)
                    X_sc = logistic_scaler.transform(feat)
                    log_prob = float(logistic_model.predict_proba(X_sc)[0, 1])
                except Exception:
                    pass

            # XGBoost
            xgb_prob = None
            if xgb_model and xgb_scaler:
                try:
                    X_sc = xgb_scaler.transform(feat)
                    xgb_prob = float(xgb_model.predict_proba(X_sc)[0, 1])
                except Exception:
                    pass

            ensemble_prob = nba_ensemble_predict(
                logistic_prob=log_prob,
                xgb_prob=xgb_prob,
                elo_prob=elo_result["prob"],
                pyth_prob=eff_result["pyth_prob"],
                eff_prob=eff_result["eff_prob"],
            )

            mu_home = bayesian_ratings.get(home, {}).get("mu", elo_dict.get(home, 1500.0))
            mu_away = bayesian_ratings.get(away, {}).get("mu", elo_dict.get(away, 1500.0))
            sig_home = bayesian_ratings.get(home, {}).get("sigma", 75.0)
            sig_away = bayesian_ratings.get(away, {}).get("sigma", 75.0)

            mc_result = nba_simulate_game(
                mu_home, mu_away, sig_home, sig_away,
                is_home_a=True, neutral=neutral, n=10000,
            )

            market_odds = match_odds_to_game(game, odds_map)
            market_home_prob = market_edge = kelly_pct = None
            if market_odds:
                market_home_prob = market_odds.get("home_prob")
                if market_home_prob:
                    market_edge = round(ensemble_prob - market_home_prob, 4)
                    kelly_pct = kelly_criterion(ensemble_prob, market_home_prob)

            adj_block = {
                "rest_home": rest_home,
                "rest_away": rest_away,
                "rest_diff": rest_home - rest_away,
                "travel_dist_miles": round(dist, 0),
                "travel_adj": travel_adj_away,
                "home_elo_bonus": 0 if neutral else 100,
            }

            predictions_list.append({
                "game_id": game_id,
                "game_time": game["game_time"],
                "status": game.get("status", ""),
                "home_team": home,
                "away_team": away,
                "home_name": game.get("home_name", home),
                "away_name": game.get("away_name", away),
                "home_logo": game.get("home_logo", ""),
                "away_logo": game.get("away_logo", ""),
                "home_wins": game.get("home_wins", 0),
                "home_losses": game.get("home_losses", 0),
                "away_wins": game.get("away_wins", 0),
                "away_losses": game.get("away_losses", 0),
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
                "adjustments": adj_block,
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
                    "home_elo_penalty": home_inj.get("elo_penalty", 0.0),
                    "away_elo_penalty": away_inj.get("elo_penalty", 0.0),
                    "home_impact_score": home_inj.get("impact_score", 0.0),
                    "away_impact_score": away_inj.get("impact_score", 0.0),
                    "home_key_players_out": home_inj.get("key_players_out", []),
                    "away_key_players_out": away_inj.get("key_players_out", []),
                },
                "prediction_drivers": compute_nba_prediction_drivers(
                    home, away, elo_result, eff_result, adj_block,
                    home_inj, away_inj,
                    game.get("home_name", home), game.get("away_name", away),
                ),
            })

        except Exception as e:
            log.error(f"Error processing NBA game {game.get('game_id', '?')}: {e}")

    # ── 9. Build leaderboard ─────────────────────────────────────────────────
    log.info("Building NBA leaderboard...")
    leaderboard_list = []

    for team in NBA_TEAMS:
        elo = elo_dict.get(team, 1500.0)
        bayes = bayesian_ratings.get(team, {"mu": elo, "sigma": 75.0})
        eff = efficiency_data.get(team, {})
        wins = standings.get(team, {}).get("wins", 0)
        losses = standings.get(team, {}).get("losses", 0)
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
            "net_rating": round(eff.get("net_rating", 0.0), 2),
            "off_rating": round(eff.get("off_rating", 115.0), 2),
            "def_rating": round(eff.get("def_rating", 115.0), 2),
            "pyth": round(eff.get("pyth", 0.5), 4),
            "wins": wins,
            "losses": losses,
            "playoff_prob": 0.5,  # placeholder — no season sim for NBA yet
            "champ_prob": 0.033,
            "trend": trend,
            "injury_elo_penalty": team_inj.get("elo_penalty", 0.0),
            "injury_impact_score": team_inj.get("impact_score", 0.0),
            "injury_players_count": team_inj.get("total_players", 0),
        })

    leaderboard_list.sort(key=lambda x: x["elo"], reverse=True)

    # ── 10. Write JSON files ─────────────────────────────────────────────────
    log.info("Writing NBA JSON files...")

    # Flat injuries list
    injuries_flat = []
    for team_abbrev, player_list in injuries.items():
        for p in player_list:
            injuries_flat.append({
                "team": team_abbrev,
                "player_name": p.get("player", ""),
                "position": p.get("position", ""),
                "status": p.get("status", ""),
                "injury_description": p.get("injury_description", ""),
            })

    predictions_json = {
        "updated": now_utc,
        "season": season_year,
        "is_offseason": is_offseason,
        "games": predictions_list,
    }
    leaderboard_json = {
        "updated": now_utc,
        "season": season_year,
        "teams": leaderboard_list,
    }
    metrics_json = {
        "updated": now_utc,
        "log_loss": model_metrics.get("log_loss"),
        "brier_score": model_metrics.get("brier_score"),
        "auc": model_metrics.get("auc"),
        "n_training_games": model_metrics.get("n_training_games", 0),
        "xgboost_available": xgb_model is not None,
        "calibration_buckets": model_metrics.get("calibration_buckets", []),
        "historical_accuracy": model_metrics.get("historical_accuracy", []),
    }

    (DATA_DIR / "nba_predictions.json").write_text(
        json.dumps(predictions_json, indent=2, default=str)
    )
    (DATA_DIR / "nba_leaderboard.json").write_text(
        json.dumps(leaderboard_json, indent=2, default=str)
    )
    (DATA_DIR / "nba_model_metrics.json").write_text(
        json.dumps(metrics_json, indent=2, default=str)
    )
    (DATA_DIR / "nba_injuries.json").write_text(
        json.dumps({"updated": now_utc, "injuries": injuries_flat}, indent=2, default=str)
    )

    log.info("=== NBA Update complete ===")
    log.info(f"  Games predicted: {len(predictions_list)}")
    log.info(f"  Teams in leaderboard: {len(leaderboard_list)}")
    log.info(f"  Training games: {len(all_games)}")


if __name__ == "__main__":
    run()
