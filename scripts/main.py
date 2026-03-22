"""
Main Orchestrator - Sport3 Refactor
Clean entry point that runs NFL and NBA prediction pipelines.
Delegates fetching to data_fetcher.py, math/ML to model_engine.py,
and JSON exports to output_writer.py.

Usage:
  python scripts/main.py              # Run both NFL and NBA pipelines
  python scripts/main.py --nfl-only   # Run only NFL pipeline
  python scripts/main.py --nba-only   # Run only NBA pipeline
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data_fetcher import (
    download_fte_data, fetch_espn_scoreboard, fetch_espn_future_games,
    fetch_espn_standings, fetch_espn_injuries, fetch_espn_depth_charts,
    fetch_espn_completed_games, fetch_betting_odds,
    fetch_nba_scoreboard, fetch_nba_future_games, fetch_nba_standings,
    fetch_nba_injuries, fetch_nba_depth_charts, fetch_nba_player_ppg,
    fetch_nba_season_games_espn,
    nfl_days_since_last_game, days_since_last_game,
    NFL_TEAMS, NFL_TEAM_NAMES, NBA_TEAMS, NBA_TEAM_NAMES,
    normalize_player_name,
)
from scripts.model_engine import (
    build_player_values, value_tier_label, extend_elo_with_espn,
    build_team_efficiency_data, generate_prediction_drivers,
    american_to_prob, remove_vig, kelly_criterion,
    build_nba_player_values, nba_value_tier_label, nba_travel_distance,
    build_nba_efficiency_data, compute_nba_pythagorean,
    build_nba_features, train_nba_logistic, predict_nba_logistic,
    train_nba_xgboost, evaluate_nba_model, nba_ensemble_predict,
    compute_h2h, compute_streak, generate_nba_prediction_drivers,
)
from scripts.output_writer import (
    write_nfl_predictions, write_nfl_elo_ratings, write_nfl_leaderboard,
    write_nfl_model_metrics, write_nfl_injuries,
    write_nba_predictions, write_nba_leaderboard, write_nba_model_metrics,
    write_nba_injuries, write_nba_elo_ratings, write_ensemble_weights,
    DATA_DIR,
)

from model.elo_model import compute_elo, predict_game as elo_predict_game, get_trend
from model.logistic_model import (build_features, train_logistic, evaluate_model,
                                   calibration_buckets, predict_matchups,
                                   historical_accuracy_by_year)
from model.bayesian_model import update_ratings, predict_game as bayes_predict
from model.efficiency_model import (compute_pythagorean, compute_efficiency,
                                     travel_distance, travel_elo_adjustment,
                                     rest_elo_adjustment,
                                     efficiency_predict_game, hierarchical_team_rating)
from model.monte_carlo import simulate_game, simulate_season
from model.ensemble_model import (build_xgb_features, train_xgboost, predict_xgboost,
                                   ensemble_predict)
from model.injury_model import compute_all_team_impacts, injury_elo_adjustment, compute_all_nba_team_impacts
from model.nba_elo import (
    compute_nba_elo, predict_nba_game,
    nba_recent_form, nba_get_trend, nba_expected_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_nfl():
    """Run the NFL prediction pipeline."""
    import pandas as pd
    log.info("=== NFL Prediction Model Update Starting ===")
    now_utc = datetime.now(timezone.utc).isoformat()
    odds_api_key = os.environ.get("ODDS_API_KEY", "")

    # ── 1. Download data ───────────────────────────────────────────────────────────────
    fte_df = download_fte_data()
    scoreboard_games, current_week = fetch_espn_scoreboard()
    standings = fetch_espn_standings()
    injuries = fetch_espn_injuries()
    odds_map = fetch_betting_odds(odds_api_key, sport="americanfootball_nfl")

    # Fallback standings from FTE data when ESPN returns empty (offseason)
    if not standings and not fte_df.empty:
        log.info("ESPN standings empty (offseason), computing W-L from FTE data...")
        last_season = fte_df[fte_df["season"] == fte_df["season"].max()]
        completed = last_season.dropna(subset=["score1", "score2"])
        from scripts.data_fetcher import abbrev_norm
        for _, row in completed.iterrows():
            for team_col, opp_col, score_col, opp_score_col in [
                ("team1", "team2", "score1", "score2"), ("team2", "team1", "score2", "score1"),
            ]:
                team = abbrev_norm(str(row[team_col]))
                if team not in standings:
                    standings[team] = {"wins": 0, "losses": 0, "ties": 0,
                                       "points_for": 0.0, "points_against": 0.0, "games_played": 0}
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

    depth_charts = fetch_espn_depth_charts()
    player_values = build_player_values(depth_charts)
    log.info("Computing injury impact scores...")
    injury_impacts = compute_all_team_impacts(injuries, player_values)
    write_nfl_injuries(injuries, player_values, value_tier_label, now_utc)

    season_year = datetime.now().year
    if datetime.now().month < 8:
        season_year -= 1

    is_offseason = len(scoreboard_games) == 0
    if not is_offseason and scoreboard_games and all(g.get("status") == "STATUS_FINAL" for g in scoreboard_games):
        try:
            recent_times = [g["game_time"] for g in scoreboard_games if g.get("game_time")]
            if recent_times:
                latest_dt = datetime.fromisoformat(max(recent_times).replace("Z", "+00:00"))
                if (datetime.now(timezone.utc) - latest_dt).days > 21:
                    is_offseason = True
                    log.info("Detected NFL offseason")
        except Exception as e:
            log.warning(f"Could not check offseason recency: {e}")

    future_games = fetch_espn_future_games(current_week, season_year, weeks_ahead=3)
    all_games_for_prediction = scoreboard_games + future_games

    # ── 2. Build ELO ratings ──────────────────────────────────────────────────────────────
    log.info("Computing ELO ratings from FTE data...")
    fte_cutoff_date = None
    if not fte_df.empty:
        elo_dict, game_history = compute_elo(fte_df)
        completed_fte = fte_df.dropna(subset=["score1", "score2"])
        if not completed_fte.empty:
            fte_cutoff_date = pd.to_datetime(completed_fte["date"]).max()
            log.info(f"FTE data ends at {fte_cutoff_date.date()}")
    else:
        elo_dict = {t: 1500.0 for t in NFL_TEAMS}
        game_history = {t: [] for t in NFL_TEAMS}

    for team in NFL_TEAMS:
        if team not in elo_dict:
            elo_dict[team] = 1500.0
        if team not in game_history:
            game_history[team] = []

    log.info("Extending ELO with live ESPN results...")
    espn_completed = fetch_espn_completed_games(season_year)
    if espn_completed:
        elo_dict, game_history = extend_elo_with_espn(
            elo_dict, game_history, espn_completed, fte_cutoff_date=fte_cutoff_date
        )
    log.info(f"ELO updated for {len(elo_dict)} teams after live extension")

    # ── 3. Efficiency & pythagorean ──────────────────────────────────────────────────────
    log.info("Computing efficiency and pythagorean ratings...")
    efficiency_data = build_team_efficiency_data(standings, fte_df, NFL_TEAMS)
    teams_pts_data = {
        team: {"points_for": standings.get(team, {}).get("points_for", 350),
               "points_against": standings.get(team, {}).get("points_against", 350)}
        for team in NFL_TEAMS
    }
    pythagorean_data = compute_pythagorean(teams_pts_data)

    # ── 4. Train logistic model ─────────────────────────────────────────────────────────
    log.info("Training logistic regression model...")
    logistic_model, logistic_scaler, logistic_calibrator = None, None, None
    model_metrics = {"log_loss": None, "brier_score": None, "auc": None, "calibration_buckets": [], "historical_accuracy": []}
    if not fte_df.empty and len(fte_df) > 100:
        try:
            X, y = build_features(fte_df, elo_dict, game_history, efficiency_data, pythagorean_data)
            if len(X) > 50:
                logistic_model, logistic_scaler, logistic_calibrator = train_logistic(X, y)
                metrics = evaluate_model(logistic_model, logistic_scaler, logistic_calibrator, X, y)
                model_metrics.update(metrics)
                model_metrics["calibration_buckets"] = calibration_buckets(
                    logistic_model, logistic_scaler, logistic_calibrator, X, y)
                model_metrics["historical_accuracy"] = historical_accuracy_by_year(
                    fte_df, logistic_model, logistic_scaler, logistic_calibrator,
                    elo_dict, game_history, efficiency_data, pythagorean_data)
                log.info(f"Logistic model trained. Log loss: {metrics['log_loss']:.4f}")
        except Exception as e:
            log.error(f"Logistic model training failed: {e}")

    # ── 5. Train XGBoost ────────────────────────────────────────────────────────────────
    log.info("Training XGBoost model...")
    xgb_model, xgb_scaler = None, None
    if not fte_df.empty and len(fte_df) > 100:
        try:
            X_xgb, y_xgb = build_xgb_features(fte_df, elo_dict, game_history, efficiency_data, pythagorean_data)
            if len(X_xgb) > 50:
                xgb_model, xgb_scaler = train_xgboost(X_xgb, y_xgb)
                if xgb_model:
                    log.info("XGBoost model trained successfully")
        except Exception as e:
            log.error(f"XGBoost training failed: {e}")

    # ── 6. Bayesian ratings ──────────────────────────────────────────────────────────────
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

    # ── 7. Season simulation ────────────────────────────────────────────────────────────
    log.info("Running season simulation...")
    remaining_schedule = [
        {"team_a": g["home_team"], "team_b": g["away_team"], "is_home_a": True,
         "neutral": bool(g.get("neutral", False)), "week": g.get("week", 0), "game_id": g.get("game_id", "")}
        for g in all_games_for_prediction
        if g.get("status", "") in ("STATUS_SCHEDULED", "STATUS_IN_PROGRESS")
    ]
    for team in NFL_TEAMS:
        if team in bayesian_ratings:
            bayesian_ratings[team]["wins"] = standings.get(team, {}).get("wins", 0)
    try:
        season_sim = simulate_season(NFL_TEAMS, remaining_schedule, bayesian_ratings, n_sims=5000)
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

    # ── 8. Generate per-game predictions ──────────────────────────────────────────────
    log.info(f"Generating predictions for {len(all_games_for_prediction)} games...")
    predictions_list = []

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

    for game in all_games_for_prediction:
        try:
            from datetime import date as _date_cls
            home = game["home_team"]
            away = game["away_team"]
            game_id = game["game_id"]
            neutral = bool(game.get("neutral", False))

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
            inj_adj_home, inj_adj_away = injury_elo_adjustment(home, away, injury_impacts)
            home_inj_impact = injury_impacts.get(home, {})
            away_inj_impact = injury_impacts.get(away, {})

            matchup_data = [{"game_id": game_id, "team_a": home, "team_b": away, "is_home_a": True,
                              "neutral": neutral, "rest_diff": rest_home - rest_away, "travel_diff": dist, "turnover_diff": 0}]

            elo_result = elo_predict_game(home, away, elo_dict, game_history, is_home_a=True, neutral=neutral,
                rest_adj_a=rest_adj_home + inj_adj_home, rest_adj_b=rest_adj_away + travel_adj_away + inj_adj_away)
            bayes_result = bayes_predict(home, away, bayesian_ratings, is_home_a=True, neutral=neutral)
            eff_result = efficiency_predict_game(home, away, efficiency_data, pythagorean_data, not neutral, neutral)

            log_prob = 0.5
            if logistic_model and logistic_scaler and logistic_calibrator:
                log_preds = predict_matchups(matchup_data, logistic_model, logistic_scaler, logistic_calibrator,
                    elo_dict, game_history, efficiency_data, pythagorean_data)
                if log_preds:
                    log_prob = log_preds[0]["logistic_prob"]

            xgb_prob = None
            if xgb_model and xgb_scaler:
                xgb_preds = predict_xgboost(matchup_data, xgb_model, xgb_scaler,
                    elo_dict, game_history, efficiency_data, pythagorean_data)
                if xgb_preds and xgb_preds[0]["xgb_prob"] is not None:
                    xgb_prob = xgb_preds[0]["xgb_prob"]

            ensemble_prob = ensemble_predict(logistic_prob=log_prob, xgb_prob=xgb_prob,
                elo_prob=elo_result["prob"], pyth_prob=eff_result["pyth_prob"], eff_prob=eff_result["eff_prob"])

            mu_home = bayesian_ratings.get(home, {}).get("mu", elo_dict.get(home, 1500.0))
            mu_away = bayesian_ratings.get(away, {}).get("mu", elo_dict.get(away, 1500.0))
            sig_home = bayesian_ratings.get(home, {}).get("sigma", 75.0)
            sig_away = bayesian_ratings.get(away, {}).get("sigma", 75.0)
            mc_result = simulate_game(mu_home, mu_away, sig_home, sig_away, is_home_a=True, neutral=neutral, n=10000)

            market_odds = match_odds_to_game(game, odds_map)
            market_home_prob = market_odds.get("home_prob") if market_odds else None
            market_edge = round(ensemble_prob - market_home_prob, 4) if market_home_prob else None
            kelly_pct = kelly_criterion(ensemble_prob, market_home_prob) if market_home_prob else None

            adj_dict = {
                "rest_home": rest_home, "rest_away": rest_away, "rest_diff": rest_home - rest_away,
                "travel_dist_miles": round(dist, 0), "travel_adj": travel_adj_away,
                "home_elo_bonus": 0 if neutral else 65,
            }
            pred_drivers = generate_prediction_drivers(game, home, away, elo_dict, efficiency_data, injury_impacts, adj_dict)

            winner = home if ensemble_prob >= 0.5 else away
            winner_prob = ensemble_prob if ensemble_prob >= 0.5 else 1 - ensemble_prob
            loser = away if ensemble_prob >= 0.5 else home
            elo_gap = abs(elo_dict.get(home, 1500) - elo_dict.get(away, 1500))
            confidence = "strong" if winner_prob > 0.70 else "moderate" if winner_prob > 0.60 else "slight"
            explanation = (
                f"The model gives {NFL_TEAM_NAMES.get(winner, winner)} a {winner_prob*100:.1f}% win probability —"
                f" a {confidence} favorite over {NFL_TEAM_NAMES.get(loser, loser)}."
                f" The ELO gap is {elo_gap:.0f} points"
                + (f" with home field adding approximately 65 ELO points." if not neutral else " at a neutral site.")
                + (f" Rest advantage favors {home if adj_dict['rest_diff'] > 0 else away}." if abs(adj_dict.get('rest_diff', 0)) >= 3 else "")
                + (f" Travel distance {round(dist)} miles factors into the away team rating." if dist > 1000 else "")
            )

            predictions_list.append({
                "game_id": game_id, "game_time": game["game_time"],
                "week": game.get("week", 0), "status": game.get("status", ""),
                "is_future": bool(game.get("is_future", False)),
                "home_team": home, "away_team": away,
                "home_name": game.get("home_name", home), "away_name": game.get("away_name", away),
                "home_logo": game.get("home_logo", ""), "away_logo": game.get("away_logo", ""),
                "neutral": neutral, "home_score": game.get("home_score", 0), "away_score": game.get("away_score", 0),
                "predictions": {"ensemble_prob": ensemble_prob, "logistic_prob": round(log_prob, 4),
                    "elo_prob": round(elo_result["prob"], 4), "xgb_prob": xgb_prob,
                    "pyth_prob": eff_result["pyth_prob"], "eff_prob": eff_result["eff_prob"],
                    "bayesian_prob": bayes_result["bayesian_prob"]},
                "market": {"home_prob": market_home_prob, "edge": market_edge, "kelly_pct": kelly_pct,
                    "home_american": market_odds.get("home_american") if market_odds else None,
                    "away_american": market_odds.get("away_american") if market_odds else None},
                "monte_carlo": mc_result, "adjustments": adj_dict,
                "elo": {"home": round(elo_dict.get(home, 1500.0), 1), "away": round(elo_dict.get(away, 1500.0), 1),
                    "diff": round(elo_result["elo_diff"], 1)},
                "bayesian": {"home_mu": bayes_result["mu_a"], "home_sigma": bayes_result["sigma_a"],
                    "away_mu": bayes_result["mu_b"], "away_sigma": bayes_result["sigma_b"],
                    "home_band": bayes_result["uncertainty_band_a"], "away_band": bayes_result["uncertainty_band_b"]},
                "injuries": {
                    "home": [{**p, "value_tier": value_tier_label(player_values.get(p.get("player", ""), 1.0))} for p in injuries.get(home, [])[:5]],
                    "away": [{**p, "value_tier": value_tier_label(player_values.get(p.get("player", ""), 1.0))} for p in injuries.get(away, [])[:5]],
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
                "prediction_drivers": pred_drivers, "explanation": explanation,
            })
        except Exception as e:
            log.error(f"Error processing game {game.get('game_id', '?')}: {e}")

    # ── 9. Build leaderboard ────────────────────────────────────────────────────────────
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
            "team": team, "team_name": NFL_TEAM_NAMES.get(team, team), "abbrev": team.lower(),
            "logo": f"https://a.espncdn.com/i/teamlogos/nfl/500/{team.lower()}.png",
            "elo": round(elo, 1), "sigma": round(bayes["sigma"], 1), "mu": round(bayes["mu"], 1),
            "lower_band": round(bayes["mu"] - bayes["sigma"], 1), "upper_band": round(bayes["mu"] + bayes["sigma"], 1),
            "pyth": round(pyth, 4), "net_eff": round(net_eff, 4),
            "off_eff": round(efficiency_data.get(team, {}).get("off_eff", 1.0), 4),
            "def_eff": round(efficiency_data.get(team, {}).get("def_eff", 1.0), 4),
            "wins": wins, "losses": losses, "ties": ties,
            "playoff_prob": round(playoff_prob, 4), "sb_prob": round(sb_prob, 4), "trend": trend,
            "offensive_rating": round(hier["offensive_rating"], 4),
            "defensive_rating": round(hier["defensive_rating"], 4),
            "injury_elo_penalty": team_inj.get("elo_penalty", 0.0),
            "injury_impact_score": team_inj.get("impact_score", 0.0),
            "injury_players_count": team_inj.get("total_players", 0),
        })
    elo_ratings_list.sort(key=lambda x: x["elo"], reverse=True)
    leaderboard_list = list(elo_ratings_list)

    # ── 10. Write JSON files ────────────────────────────────────────────────────────────
    log.info("Writing NFL JSON output files...")
    week_num = current_week if current_week else (scoreboard_games[0].get("week", 0) if scoreboard_games else 0)
    write_nfl_predictions(predictions_list, season_year, week_num, is_offseason)
    write_nfl_elo_ratings(elo_ratings_list, season_year)
    write_nfl_leaderboard(leaderboard_list, season_year)
    write_nfl_model_metrics(model_metrics, season_year, fte_df, xgb_model)
    log.info("=== NFL Update complete ===")
    log.info(f"  Games predicted: {len(predictions_list)}")
    log.info(f"  Teams in leaderboard: {len(leaderboard_list)}")
    log.info(f"  Model metrics: {model_metrics.get('log_loss', 'N/A')}")

