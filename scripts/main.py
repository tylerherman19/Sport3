"""
main.py — Clean orchestrator for Sport3 prediction pipeline.
Imports modular components from data_fetcher, model_engine, and output_writer
and runs both the NFL and NBA pipelines.

Usage:
    python scripts/main.py            # runs both NFL + NBA
    python scripts/main.py --nfl      # NFL only
    python scripts/main.py --nba      # NBA only

Module responsibilities:
    data_fetcher.py  — all HTTP requests (ESPN, cdn.nba.com, The Odds API, FTE)
    model_engine.py  — ELO math, ML training, ensemble, prediction drivers
    output_writer.py — JSON file I/O
"""

import os
import sys
import logging
import argparse
from datetime import datetime, date, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# ---- Data fetching ----
from scripts.data_fetcher import (
    # NFL
    fetch_fte_data, fetch_nfl_scoreboard, fetch_nfl_future_games,
    fetch_nfl_completed_games, fetch_nfl_standings, fetch_nfl_injuries,
    fetch_nfl_depth_charts, fetch_nfl_betting_odds, nfl_abbrev_norm,
    _nfl_normalize_name,
    # NBA
    fetch_nba_scoreboard, fetch_nba_future_games, fetch_nba_season_games_espn,
    fetch_nba_standings_cdn, fetch_nba_injuries_espn, fetch_nba_depth_charts_espn,
    fetch_nba_player_stats_cdn, fetch_nba_odds_api, nba_abbrev_norm,
)

# ---- Model engine ----
from scripts.model_engine import (
    NFL_TEAMS, NFL_TEAM_NAMES,
    nfl_days_since_last_game, extend_elo_with_espn,
    build_nfl_efficiency_data, match_odds_to_game, generate_nfl_prediction_drivers,
    NBA_TEAMS, NBA_TEAM_NAMES,
    build_nba_player_values, build_nba_efficiency_data, compute_nba_pythagorean,
    build_nba_features, train_nba_logistic, train_nba_xgboost, evaluate_nba_model,
    predict_nba_logistic, nba_ensemble_predict, compute_nba_h2h, compute_nba_streak,
    nba_days_since_last_game, generate_nba_prediction_drivers, nba_value_tier_label,
)

# ---- Output writers ----
from scripts.output_writer import (
    write_nfl_predictions, write_nfl_elo_ratings, write_nfl_leaderboard,
    write_nfl_model_metrics, write_nfl_injuries,
    write_nba_predictions, write_nba_leaderboard, write_nba_model_metrics,
    write_nba_injuries, write_nba_ensemble_weights, load_nba_ensemble_weights,
    guard_nba_empty_output, write_nba_features, write_nba_results, DATA_DIR,
)

# ---- Model sub-modules ----
from model.elo_model import compute_elo, predict_game as elo_predict_game, get_trend
from model.logistic_model import (build_features as build_nfl_logistic_features, train_logistic,
                                   evaluate_model, calibration_buckets,
                                   predict_matchups, historical_accuracy_by_year)
from model.bayesian_model import update_ratings, predict_game as bayes_predict
from model.efficiency_model import (compute_pythagorean, compute_efficiency,
                                     travel_distance, travel_elo_adjustment,
                                     rest_elo_adjustment, efficiency_predict_game,
                                     hierarchical_team_rating)
from model.monte_carlo import simulate_game, simulate_season
from model.ensemble_model import (build_xgb_features, train_xgboost, predict_xgboost,
                                   ensemble_predict, kelly_criterion,
                                   american_to_prob, remove_vig)
from model.injury_model import (compute_all_team_impacts, injury_elo_adjustment,
                                 compute_all_nba_team_impacts)
from model.nba_elo import (compute_nba_elo, predict_nba_game, nba_recent_form,
                            nba_get_trend, nba_expected_score)
try:
    from model.nba_elo_model import save_ratings as save_nba_elo_ratings
    HAS_NBA_ELO_MODEL = True
except ImportError:
    HAS_NBA_ELO_MODEL = False
    save_nba_elo_ratings = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DEPTH_VALUE_MAP = {1: 1.0, 2: 0.55, 3: 0.25}


def _nfl_depth_to_value(depth_pos):
    return _DEPTH_VALUE_MAP.get(depth_pos, 0.15)


def _build_nfl_player_values(depth_charts):
    return {name: _nfl_depth_to_value(pos) for name, pos in depth_charts.items()}


def _nfl_value_tier_label(mult):
    if mult >= 1.8: return "superstar"
    if mult >= 1.3: return "all-star"
    if mult >= 0.8: return "starter"
    if mult >= 0.4: return "backup"
    return "rotation"


# ============================================================
# NFL pipeline
# ============================================================

def run_nfl():
    log.info("=== NFL Prediction Model Update Starting ===")
    now_utc      = datetime.now(timezone.utc).isoformat()
    odds_api_key = os.environ.get("ODDS_API_KEY", "")

    fte_df                         = fetch_fte_data()
    scoreboard_games, current_week = fetch_nfl_scoreboard()
    standings                      = fetch_nfl_standings()
    injuries                       = fetch_nfl_injuries()
    odds_map                       = fetch_nfl_betting_odds(odds_api_key)

    if not standings and not fte_df.empty:
        log.info("ESPN standings empty — computing W-L from FTE data...")
        last_season = fte_df[fte_df["season"] == fte_df["season"].max()]
        completed   = last_season.dropna(subset=["score1", "score2"])
        for _, row in completed.iterrows():
            for tc, oc, sc, osc in [("team1","team2","score1","score2"),("team2","team1","score2","score1")]:
                team = nfl_abbrev_norm(str(row[tc]))
                standings.setdefault(team, {"wins":0,"losses":0,"ties":0,
                                             "points_for":0.0,"points_against":0.0,"games_played":0})
                won = float(row[sc]) > float(row[osc]); tied = float(row[sc]) == float(row[osc])
                if won:   standings[team]["wins"]   += 1
                elif tied: standings[team]["ties"]   += 1
                else:      standings[team]["losses"] += 1
                standings[team]["points_for"]     += float(row[sc])
                standings[team]["points_against"] += float(row[osc])
                standings[team]["games_played"]   += 1

    depth_charts  = fetch_nfl_depth_charts()
    player_values = _build_nfl_player_values(depth_charts)
    log.info("Computing NFL injury impacts...")
    injury_impacts = compute_all_team_impacts(injuries, player_values)

    if injuries:
        injuries_list = []
        for team, players in injuries.items():
            for p in players:
                pname = p.get("player", "")
                pmult = player_values.get(pname) or player_values.get(_nfl_normalize_name(pname), 1.0)
                injuries_list.append({
                    "player": pname, "team": team, "position": p.get("position",""),
                    "status": p.get("status",""),
                    "injury_description": p.get("injury_description", p.get("status","")),
                    "value_tier": _nfl_value_tier_label(pmult),
                })
        write_nfl_injuries(injuries_list, now_utc)
    else:
        log.warning("No NFL injury data — keeping existing nfl_injuries.json")

    season_year = datetime.now().year
    if datetime.now().month < 8:
        season_year -= 1

    is_offseason = len(scoreboard_games) == 0
    if not is_offseason and scoreboard_games and all(g.get("status")=="STATUS_FINAL" for g in scoreboard_games):
        try:
            rt = [g["game_time"] for g in scoreboard_games if g.get("game_time")]
            if rt:
                latest = datetime.fromisoformat(max(rt).replace("Z","+00:00"))
                if (datetime.now(timezone.utc) - latest).days > 21:
                    is_offseason = True; log.info("Detected NFL offseason")
        except Exception as e:
            log.warning(f"Could not check offseason recency: {e}")

    future_games = fetch_nfl_future_games(current_week, season_year, weeks_ahead=3)
    all_games    = scoreboard_games + future_games

    log.info("Computing ELO from FTE...")
    fte_cutoff_date = None
    if not fte_df.empty:
        elo_dict, game_history = compute_elo(fte_df)
        cft = fte_df.dropna(subset=["score1","score2"])
        if not cft.empty:
            fte_cutoff_date = pd.to_datetime(cft["date"]).max()
    else:
        elo_dict, game_history = {t:1500.0 for t in NFL_TEAMS}, {t:[] for t in NFL_TEAMS}
    for t in NFL_TEAMS:
        elo_dict.setdefault(t, 1500.0); game_history.setdefault(t, [])

    espn_completed = fetch_nfl_completed_games(season_year)
    if espn_completed:
        elo_dict, game_history = extend_elo_with_espn(
            elo_dict, game_history, espn_completed, fte_cutoff_date=fte_cutoff_date)

    efficiency_data = build_nfl_efficiency_data(standings, fte_df)
    teams_pts_data  = {
        t: {"points_for": standings.get(t,{}).get("points_for",350),
            "points_against": standings.get(t,{}).get("points_against",350)} for t in NFL_TEAMS}
    pythagorean_data = compute_pythagorean(teams_pts_data)

    logistic_model = logistic_scaler = logistic_calibrator = None
    model_metrics  = {"log_loss":None,"brier_score":None,"auc":None,"calibration_buckets":[],"historical_accuracy":[]}
    if not fte_df.empty and len(fte_df) > 100:
        try:
            X, y = build_nfl_logistic_features(fte_df, elo_dict, game_history, efficiency_data, pythagorean_data)
            if len(X) > 50:
                logistic_model, logistic_scaler, logistic_calibrator = train_logistic(X, y)
                model_metrics.update(evaluate_model(logistic_model, logistic_scaler, logistic_calibrator, X, y))
                model_metrics["calibration_buckets"] = calibration_buckets(logistic_model, logistic_scaler, logistic_calibrator, X, y)
                model_metrics["historical_accuracy"] = historical_accuracy_by_year(fte_df, logistic_model, logistic_scaler, logistic_calibrator, elo_dict, game_history, efficiency_data, pythagorean_data)
        except Exception as e:
            log.error(f"NFL logistic training failed: {e}")

    xgb_model = xgb_scaler = None
    if not fte_df.empty and len(fte_df) > 100:
        try:
            X_x, y_x = build_xgb_features(fte_df, elo_dict, game_history, efficiency_data, pythagorean_data)
            if len(X_x) > 50:
                xgb_model, xgb_scaler = train_xgboost(X_x, y_x)
        except Exception as e:
            log.error(f"NFL XGBoost training failed: {e}")

    games_list = []
    if not fte_df.empty:
        rec = fte_df[fte_df["season"] >= season_year-3].dropna(subset=["score1","score2"])
        games_list = rec[["team1","team2","score1","score2","date","neutral","season"]].to_dict("records")
    bayesian_ratings = update_ratings(games_list, elo_dict)
    for t in NFL_TEAMS: bayesian_ratings.setdefault(t, {"mu":elo_dict.get(t,1500.0),"sigma":75.0})

    remaining_schedule = [
        {"team_a":g["home_team"],"team_b":g["away_team"],"is_home_a":True,
         "neutral":bool(g.get("neutral",False)),"week":g.get("week",0),"game_id":g.get("game_id","")}
        for g in all_games if g.get("status","") in ("STATUS_SCHEDULED","STATUS_IN_PROGRESS")
    ]
    for t in NFL_TEAMS:
        if t in bayesian_ratings: bayesian_ratings[t]["wins"] = standings.get(t,{}).get("wins",0)
    try:
        season_sim = simulate_season(NFL_TEAMS, remaining_schedule, bayesian_ratings, n_sims=5000)
    except Exception as e:
        log.error(f"NFL season simulation failed: {e}")
        season_sim = {t:{"playoff_prob":0.5,"division_win_prob":0.3,"sb_prob":0.03,"wins_avg":8.0} for t in NFL_TEAMS}

    log.info(f"Generating NFL predictions for {len(all_games)} games...")
    predictions_list = []
    for game in all_games:
        try:
            home, away = game["home_team"], game["away_team"]
            game_id = game["game_id"]; neutral = bool(game.get("neutral",False))
            game_date_str = game.get("game_time","")[:10]
            try: game_date = datetime.strptime(game_date_str,"%Y-%m-%d").date()
            except (ValueError,TypeError): game_date = date.today()
            rh = nfl_days_since_last_game(game_history,home,game_date)
            ra = nfl_days_since_last_game(game_history,away,game_date)
            dist = travel_distance(home,away,is_home_a=True)
            raj  = rest_elo_adjustment(rh,ra)
            taj  = travel_elo_adjustment(dist)
            ijah, ijaa = injury_elo_adjustment(home,away,injury_impacts)
            hi = injury_impacts.get(home,{}); ai = injury_impacts.get(away,{})
            md = [{"game_id":game_id,"team_a":home,"team_b":away,"is_home_a":True,"neutral":neutral,
                   "rest_diff":rh-ra,"travel_diff":dist,"turnover_diff":0}]
            er  = elo_predict_game(home,away,elo_dict,game_history,is_home_a=True,neutral=neutral,
                                   rest_adj_a=raj+ijah,rest_adj_b=-raj+taj+ijaa)
            br  = bayes_predict(home,away,bayesian_ratings,is_home_a=True,neutral=neutral)
            effr = efficiency_predict_game(home,away,efficiency_data,pythagorean_data,not neutral,neutral)
            lp = 0.5
            if logistic_model and logistic_scaler and logistic_calibrator:
                lps = predict_matchups(md,logistic_model,logistic_scaler,logistic_calibrator,
                                       elo_dict,game_history,efficiency_data,pythagorean_data)
                if lps: lp = lps[0]["logistic_prob"]
            xp = None
            if xgb_model and xgb_scaler:
                xps = predict_xgboost(md,xgb_model,xgb_scaler,elo_dict,game_history,efficiency_data,pythagorean_data)
                if xps and xps[0]["xgb_prob"] is not None: xp = xps[0]["xgb_prob"]
            ep = ensemble_predict(logistic_prob=lp,xgb_prob=xp,elo_prob=er["prob"],
                                  pyth_prob=effr["pyth_prob"],eff_prob=effr["eff_prob"])
            mh = bayesian_ratings.get(home,{}).get("mu",elo_dict.get(home,1500.0))
            ma = bayesian_ratings.get(away,{}).get("mu",elo_dict.get(away,1500.0))
            sh = bayesian_ratings.get(home,{}).get("sigma",75.0)
            sa = bayesian_ratings.get(away,{}).get("sigma",75.0)
            mc = simulate_game(mh,ma,sh,sa,is_home_a=True,neutral=neutral,n=10000)
            mo = match_odds_to_game(game,odds_map)
            mhp = mo.get("home_prob") if mo else None
            me  = round(ep-mhp,4) if mhp else None
            kp  = kelly_criterion(ep,mhp) if mhp else None
            adj = {"rest_home":rh,"rest_away":ra,"rest_diff":rh-ra,
                   "travel_dist_miles":round(dist,0),"travel_adj":taj,"home_elo_bonus":0 if neutral else 65}
            pd2 = generate_nfl_prediction_drivers(game,home,away,elo_dict,efficiency_data,injury_impacts,adj)
            winner = home if ep>=0.5 else away; wp = ep if ep>=0.5 else 1-ep; loser = away if ep>=0.5 else home
            elo_gap = abs(elo_dict.get(home,1500)-elo_dict.get(away,1500))
            conf = "strong" if wp>0.70 else "moderate" if wp>0.60 else "slight"
            expl = (f"The model gives {NFL_TEAM_NAMES.get(winner,winner)} a {wp*100:.1f}% win probability —"
                    f" a {conf} favorite over {NFL_TEAM_NAMES.get(loser,loser)}."
                    f" ELO gap: {elo_gap:.0f} pts"
                    + (" with home field +65 ELO." if not neutral else " at a neutral site.")
                    + (f" Rest: {home if adj['rest_diff']>0 else away} has the edge." if abs(adj.get('rest_diff',0))>=3 else "")
                    + (f" Travel: {round(dist)} miles." if dist>1000 else ""))
            predictions_list.append({
                "game_id":game_id,"game_time":game["game_time"],"week":game.get("week",0),
                "status":game.get("status",""),"is_future":bool(game.get("is_future",False)),
                "home_team":home,"away_team":away,
                "home_name":game.get("home_name",home),"away_name":game.get("away_name",away),
                "home_logo":game.get("home_logo",""),"away_logo":game.get("away_logo",""),
                "neutral":neutral,"home_score":game.get("home_score",0),"away_score":game.get("away_score",0),
                "predictions":{"ensemble_prob":ep,"logistic_prob":round(lp,4),"elo_prob":round(er["prob"],4),
                               "xgb_prob":xp,"pyth_prob":effr["pyth_prob"],"eff_prob":effr["eff_prob"],
                               "bayesian_prob":br["bayesian_prob"]},
                "market":{"home_prob":mhp,"edge":me,"kelly_pct":kp,
                          "home_american":mo.get("home_american") if mo else None,
                          "away_american":mo.get("away_american") if mo else None},
                "monte_carlo":mc,"adjustments":adj,
                "elo":{"home":round(elo_dict.get(home,1500.0),1),"away":round(elo_dict.get(away,1500.0),1),"diff":round(er["elo_diff"],1)},
                "bayesian":{"home_mu":br["mu_a"],"home_sigma":br["sigma_a"],"away_mu":br["mu_b"],"away_sigma":br["sigma_b"],
                            "home_band":br["uncertainty_band_a"],"away_band":br["uncertainty_band_b"]},
                "injuries":{
                    "home":[{**p,"value_tier":_nfl_value_tier_label(player_values.get(p.get("player",""),1.0))} for p in injuries.get(home,[])[:5]],
                    "away":[{**p,"value_tier":_nfl_value_tier_label(player_values.get(p.get("player",""),1.0))} for p in injuries.get(away,[])[:5]],
                },
                "injury_impact":{"home_elo_penalty":hi.get("elo_penalty",0.0),"away_elo_penalty":ai.get("elo_penalty",0.0),
                                 "home_impact_score":hi.get("impact_score",0.0),"away_impact_score":ai.get("impact_score",0.0),
                                 "home_key_players_out":hi.get("key_players_out",[]),"away_key_players_out":ai.get("key_players_out",[]),
                                 "home_star_count":hi.get("star_count",0),"away_star_count":ai.get("star_count",0),
                                 "home_star_stack_multiplier":hi.get("star_stack_multiplier",1.0),
                                 "away_star_stack_multiplier":ai.get("star_stack_multiplier",1.0)},
                "prediction_drivers":pd2,"explanation":expl,
            })
        except Exception as e:
            log.error(f"Error processing NFL game {game.get('game_id','?')}: {e}")

    elo_ratings_list = []
    for t in NFL_TEAMS:
        elo = elo_dict.get(t,1500.0); bayes = bayesian_ratings.get(t,{"mu":elo,"sigma":75.0})
        pyth = pythagorean_data.get(t,{}).get("pyth",0.5); ti = injury_impacts.get(t,{})
        elo_ratings_list.append({
            "team":t,"team_name":NFL_TEAM_NAMES.get(t,t),"abbrev":t.lower(),
            "logo":f"https://a.espncdn.com/i/teamlogos/nfl/500/{t.lower()}.png",
            "elo":round(elo,1),"sigma":round(bayes["sigma"],1),"mu":round(bayes["mu"],1),
            "lower_band":round(bayes["mu"]-bayes["sigma"],1),"upper_band":round(bayes["mu"]+bayes["sigma"],1),
            "pyth":round(pyth,4),"net_eff":round(efficiency_data.get(t,{}).get("net_eff",0.0),4),
            "off_eff":round(efficiency_data.get(t,{}).get("off_eff",1.0),4),
            "def_eff":round(efficiency_data.get(t,{}).get("def_eff",1.0),4),
            "wins":standings.get(t,{}).get("wins",0),"losses":standings.get(t,{}).get("losses",0),
            "ties":standings.get(t,{}).get("ties",0),
            "playoff_prob":round(season_sim.get(t,{}).get("playoff_prob",0.5),4),
            "sb_prob":round(season_sim.get(t,{}).get("sb_prob",0.03),4),
            "trend":get_trend(game_history,t),
            "offensive_rating":round(hierarchical_team_rating(t,efficiency_data)["offensive_rating"],4),
            "defensive_rating":round(hierarchical_team_rating(t,efficiency_data)["defensive_rating"],4),
            "injury_elo_penalty":ti.get("elo_penalty",0.0),
            "injury_impact_score":ti.get("impact_score",0.0),
            "injury_players_count":ti.get("total_players",0),
        })
    elo_ratings_list.sort(key=lambda x: x["elo"], reverse=True)

    week_num = current_week or (scoreboard_games[0].get("week",0) if scoreboard_games else 0)
    write_nfl_predictions(predictions_list, season_year, week_num, is_offseason, now_utc)
    write_nfl_elo_ratings(elo_ratings_list, season_year, now_utc)
    write_nfl_leaderboard(elo_ratings_list, season_year, now_utc)
    write_nfl_model_metrics(model_metrics, len(fte_df) if not fte_df.empty else 0, xgb_model is not None, now_utc)
    log.info("=== NFL Update complete ===")
    log.info(f"  Games predicted: {len(predictions_list)}  |  Teams: {len(elo_ratings_list)}")


# ============================================================
# NBA pipeline
# ============================================================

NBA_COORDS = {
    "ATL":[33.749,-84.388],"BOS":[42.36,-71.059],"BKN":[40.683,-73.975],
    "CHA":[35.227,-80.843],"CHI":[41.878,-87.63],"CLE":[41.499,-81.694],
    "DAL":[32.777,-96.797],"DEN":[39.739,-104.99],"DET":[42.331,-83.046],
    "GSW":[37.768,-122.388],"HOU":[29.76,-95.37],"IND":[39.768,-86.158],
    "LAC":[34.043,-118.267],"LAL":[34.043,-118.267],"MEM":[35.15,-90.049],
    "MIA":[25.762,-80.192],"MIL":[43.044,-87.917],"MIN":[44.978,-93.265],
    "NOP":[29.951,-90.072],"NYK":[40.751,-73.993],"OKC":[35.463,-97.515],
    "ORL":[28.538,-81.379],"PHI":[39.901,-75.172],"PHX":[33.448,-112.074],
    "POR":[45.523,-122.677],"SAC":[38.582,-121.494],"SAS":[29.424,-98.494],
    "TOR":[43.653,-79.383],"UTA":[40.761,-111.891],"WAS":[38.907,-77.037],
}


def _nba_travel_dist(home, away):
    import math
    ch = NBA_COORDS.get(home); ca = NBA_COORDS.get(away)
    if not ch or not ca: return 0.0
    R=3959.0; lat1=math.radians(ch[0]); lat2=math.radians(ca[0])
    dlat=lat2-lat1; dlon=math.radians(ca[1]-ch[1])
    x=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R*2*math.atan2(math.sqrt(x),math.sqrt(1-x))


def run_nba():
    log.info("=== NBA Prediction Model Update Starting ===")
    now_utc      = datetime.now(timezone.utc).isoformat()
    odds_api_key = os.environ.get("ODDS_API_KEY", "")
    cm           = datetime.now(timezone.utc).month
    cy           = datetime.now(timezone.utc).year
    season_year  = cy + 1 if cm >= 10 else cy

    scoreboard_games = fetch_nba_scoreboard()
    standings        = fetch_nba_standings_cdn()
    injuries         = fetch_nba_injuries_espn()
    odds_map         = fetch_nba_odds_api(odds_api_key)
    is_offseason     = len(scoreboard_games) == 0

    future_games = fetch_nba_future_games(days_ahead=14)
    existing_ids = {g["game_id"] for g in scoreboard_games}
    future_games = [g for g in future_games if g["game_id"] not in existing_ids]
    all_games    = scoreboard_games + future_games

    if not scoreboard_games and not future_games:
        existing_path = DATA_DIR / "nba_predictions.json"
        if guard_nba_empty_output(existing_path):
            log.error("Both fetches 0 — aborting to preserve existing nba_predictions.json")
            return
        log.warning("Both fetches 0 games — proceeding with empty list")

    depth_charts     = fetch_nba_depth_charts_espn()
    player_ppg       = fetch_nba_player_stats_cdn(season_year)
    nba_player_values = build_nba_player_values(depth_charts, player_ppg)
    injury_impacts   = compute_all_nba_team_impacts(injuries, nba_player_values)

    if injuries:
        inj_list = []
        for team, players in injuries.items():
            for p in players:
                pname = p.get("player",""); pmult = nba_player_values.get(pname, 1.0)
                inj_list.append({"player":pname,"team":team,"position":p.get("position",""),
                                  "status":p.get("status",""),
                                  "injury_description":p.get("injury_description",p.get("status","")),
                                  "value_tier":nba_value_tier_label(pmult)})
        write_nba_injuries(inj_list, now_utc)

    historical_games = fetch_nba_season_games_espn(
        seasons=[season_year-4,season_year-3,season_year-2,season_year-1,season_year])
    if historical_games:
        elo_dict, game_history = compute_nba_elo(historical_games)
    else:
        elo_dict, game_history = {t:1500.0 for t in NBA_TEAMS}, {t:[] for t in NBA_TEAMS}
    for t in NBA_TEAMS:
        elo_dict.setdefault(t,1500.0); game_history.setdefault(t,[])

    # Abort guard: if historical fetch is too small the model falls back to
    # 50% neutral defaults and overwrites good predictions. Preserve existing output.
    if len(historical_games) < 200:
        log.critical(
            f"NBA historical fetch returned only {len(historical_games)} games "
            f"(need >= 200) — aborting NBA pipeline to prevent overwriting "
            f"predictions with 50% defaults. Check ESPN scoreboard API availability."
        )
        return

    # Write game results immediately so dashboard has fresh history.
    results_for_output = [
        {"date": g["date"], "season": g.get("season", season_year),
         "home_team": g["team1"], "away_team": g["team2"],
         "home_score": g["score1"], "away_score": g["score2"],
         "winner": g["team1"] if g["score1"] > g["score2"] else g["team2"]}
        for g in historical_games if g.get("score1") and g.get("score2")
    ]
    write_nba_results(results_for_output[-2000:], len(results_for_output), now_utc)

    if HAS_NBA_ELO_MODEL and save_nba_elo_ratings:
        try: save_nba_elo_ratings(elo_dict, game_history, season_year)
        except Exception as e: log.warning(f"save_nba_elo_ratings failed: {e}")

    efficiency_data = build_nba_efficiency_data(standings)
    pyth_data       = compute_nba_pythagorean(efficiency_data)

    logistic_model = logistic_scaler = logistic_calibrator = None
    model_metrics  = {"log_loss":None,"brier_score":None,"auc":None}
    if historical_games and len(historical_games) > 100:
        try:
            X, y = build_nba_features(historical_games,elo_dict,game_history,efficiency_data,pyth_data)
            if len(X) > 50:
                logistic_model, logistic_scaler, logistic_calibrator = train_nba_logistic(X, y)
                if logistic_model:
                    model_metrics.update(evaluate_nba_model(logistic_model,logistic_scaler,logistic_calibrator,X,y))
                # Write feature metadata — confirms model trained on real data, not empty arrays.
                seasons_used = sorted(set(g.get("season", season_year) for g in historical_games))
                feature_names = [
                    "elo_diff_with_hfa", "hfa", "rest_diff", "pyth_elo_diff",
                    "net_rating_diff", "off_rating_diff", "def_rating_diff",
                    "pace_diff", "turnover_rate_diff", "three_point_rate_diff",
                    "rebound_rate_diff", "free_throw_rate_diff", "recent_form_diff", "b2b",
                ]
                write_nba_features(
                    feature_vectors=[{"name": feature_names[i] if i < len(feature_names) else f"feature_{i}"}
                                     for i in range(X.shape[1])],
                    count=int(len(X)),
                    seasons=seasons_used,
                    now_utc=now_utc,
                )
        except Exception as e:
            log.error(f"NBA logistic training failed: {e}")

    xgb_model = xgb_scaler = None
    if historical_games and len(historical_games) > 100:
        try:
            X_x,y_x = build_nba_features(historical_games,elo_dict,game_history,efficiency_data,pyth_data)
            if len(X_x) > 50: xgb_model,xgb_scaler = train_nba_xgboost(X_x, y_x)
        except Exception as e:
            log.error(f"NBA XGBoost training failed: {e}")

    rec_games = [g for g in historical_games if g.get("season",0)>=season_year-1]
    bayesian_ratings = update_ratings(rec_games, elo_dict)
    for t in NBA_TEAMS: bayesian_ratings.setdefault(t,{"mu":elo_dict.get(t,1500.0),"sigma":75.0})

    remaining = [{"team_a":g["home_team"],"team_b":g["away_team"],"is_home_a":True,"neutral":bool(g.get("neutral",False))}
                 for g in all_games if g.get("status","") in ("STATUS_SCHEDULED","STATUS_IN_PROGRESS")]
    for t in NBA_TEAMS:
        if t in bayesian_ratings: bayesian_ratings[t]["wins"] = standings.get(t,{}).get("wins",0)
    try:
        season_sim = simulate_season(NBA_TEAMS, remaining, bayesian_ratings, n_sims=3000)
    except Exception as e:
        log.error(f"NBA season sim failed: {e}")
        season_sim = {t:{"playoff_prob":0.53,"sb_prob":0.033,"wins_avg":41.0} for t in NBA_TEAMS}

    nba_weights = load_nba_ensemble_weights()
    try:
        from model.ensemble_model import learn_ensemble_weights
        ch2 = [g for g in historical_games if g.get("score1") is not None and g.get("score2") is not None][-500:]
        if len(ch2) >= 50:
            sp, ac = [], []
            for _g in ch2:
                _h,_a = _g["team1"],_g["team2"]
                _eh = elo_dict.get(_h,1500.0)+(100.0 if not _g.get("neutral") else 0.0)
                _ea = elo_dict.get(_a,1500.0)
                _ph = pyth_data.get(_h,{}).get("pyth",0.5); _pa = pyth_data.get(_a,{}).get("pyth",0.5)
                _phj = _ph*(1.0+(100.0 if not _g.get("neutral") else 0.0)/1500.0); _pd = _phj+_pa
                _ep = efficiency_data.get(_h,{}).get("net_rating",0.0)
                _ea2= efficiency_data.get(_a,{}).get("net_rating",0.0)
                sp.append({"elo":nba_expected_score(_eh,_ea),
                           "pyth":_phj/_pd if _pd>0 else 0.5,
                           "eff":nba_expected_score(1500+float(_ep or 0)*10+(100.0 if not _g.get("neutral") else 0.0),1500+_ea2*10),
                           "log":0.5,"xgb":0.5})
                ac.append(1 if _g.get("score1",0)>_g.get("score2",0) else 0)
            learned = learn_ensemble_weights(sp, ac, weight_keys=["elo","pyth","eff"])
            if learned:
                nba_weights["elo"]=round(learned.get("elo",0.25)*0.6,4)
                nba_weights["pyth"]=round(learned.get("pyth",0.20)*0.6,4)
                nba_weights["eff"]=round(learned.get("eff",0.15)*0.6,4)
                _t = sum(nba_weights.values())
                nba_weights = {k:round(v/_t,4) for k,v in nba_weights.items()}
                write_nba_ensemble_weights(nba_weights)
    except Exception as e:
        log.warning(f"NBA ensemble weight learning skipped: {e}")

    log.info(f"Generating NBA predictions for {len(all_games)} games...")
    predictions_list = []
    h2h_cache = {}
    for _g in all_games:
        p = (_g["home_team"],_g["away_team"])
        if p not in h2h_cache: h2h_cache[p] = compute_nba_h2h(_g["home_team"],_g["away_team"],historical_games)

    for game in all_games:
        try:
            home,away = game["home_team"],game["away_team"]
            game_id   = game["game_id"]; neutral = bool(game.get("neutral",False))
            gds = game.get("game_time","")[:10]
            try: game_date = datetime.strptime(gds,"%Y-%m-%d").date()
            except (ValueError,TypeError): game_date = date.today()
            day_before = game_date - timedelta(days=1)
            rh = nba_days_since_last_game(game_history,home,game_date)
            ra = nba_days_since_last_game(game_history,away,game_date)
            dist = _nba_travel_dist(home,away)
            b2bh = any(g.get("date","")[:10]==str(day_before) for g in game_history.get(home,[])[-3:])
            b2ba = any(g.get("date","")[:10]==str(day_before) for g in game_history.get(away,[])[-3:])
            if b2bh: rh=1
            if b2ba: ra=1
            rd = rh-ra
            ijah = -injury_impacts.get(home,{}).get("elo_penalty",0.0)
            ijaa = -injury_impacts.get(away,{}).get("elo_penalty",0.0)
            helo = elo_dict.get(home,1500.0); aelo = elo_dict.get(away,1500.0)
            hfa  = 0.0 if neutral else 100.0
            er   = predict_nba_game(home,away,elo_dict,home_b2b=b2bh,away_b2b=b2ba,rest_diff=rd,travel_miles=dist,neutral=neutral)
            ep2  = er["prob"]
            if ijah!=0.0 or ijaa!=0.0:
                ep2 = nba_expected_score(er["home_adj_elo"]+ijah, er["away_adj_elo"]+ijaa)
            br  = bayes_predict(home,away,bayesian_ratings,is_home_a=True,neutral=neutral)
            bpr = br.get("bayesian_prob",0.5)
            phome = pyth_data.get(home,{}).get("pyth",0.5); paway = pyth_data.get(away,{}).get("pyth",0.5)
            phj  = phome*(1.0+hfa/1500.0); _pd = phj+paway
            pytp = phj/_pd if _pd>0 else 0.5
            heff = efficiency_data.get(home,{}); aeff = efficiency_data.get(away,{})
            effp = nba_expected_score(1500+heff.get("net_rating",0.0)*10+hfa, 1500+aeff.get("net_rating",0.0)*10)
            pehome = 1500+(phome-0.5)*400; peaway = 1500+(paway-0.5)*400
            feat = [(helo+hfa)-aelo,hfa,rd,pehome-peaway,
                    heff.get("net_rating",0)-aeff.get("net_rating",0),
                    heff.get("offensive_rating",110)-aeff.get("offensive_rating",110),
                    aeff.get("defensive_rating",110)-heff.get("defensive_rating",110),
                    heff.get("pace",100)-aeff.get("pace",100),
                    aeff.get("turnover_rate",0.5)-heff.get("turnover_rate",0.5),
                    heff.get("three_point_rate",0.35)-aeff.get("three_point_rate",0.35),
                    heff.get("rebound_rate",0.5)-aeff.get("rebound_rate",0.5),
                    heff.get("free_throw_rate",0.20)-aeff.get("free_throw_rate",0.20),
                    nba_recent_form(game_history,home)-nba_recent_form(game_history,away),
                    float(b2bh)-float(b2ba)]
            lp  = predict_nba_logistic(feat,logistic_model,logistic_scaler,logistic_calibrator)
            xp  = None
            if xgb_model and xgb_scaler:
                try: xp = float(xgb_model.predict_proba(xgb_scaler.transform([feat]))[0][1])
                except Exception as e: log.warning(f"NBA XGB pred failed: {e}")
            ensp = nba_ensemble_predict(elo_prob=ep2,pyth_prob=pytp,eff_prob=effp,
                                         log_prob=lp,xgb_prob=xp,weights=nba_weights)
            muh = bayesian_ratings.get(home,{}).get("mu",helo); mua = bayesian_ratings.get(away,{}).get("mu",aelo)
            sh  = bayesian_ratings.get(home,{}).get("sigma",75.0); sa2 = bayesian_ratings.get(away,{}).get("sigma",75.0)
            mc  = simulate_game(muh,mua,sh,sa2,is_home_a=True,neutral=neutral,n=10000)
            mo  = None
            for _,odds in odds_map.items():
                hm=odds.get("home_team_name",""); am=odds.get("away_team_name","")
                hg=game.get("home_name",""); ag=game.get("away_name","")
                if (hg.lower() in hm.lower() or hm.lower() in hg.lower()) and \
                   (ag.lower() in am.lower() or am.lower() in ag.lower()):
                    mo=odds; break
            mhp = mo.get("home_prob") if mo else None
            me2 = round(ensp-mhp,4) if mhp else None
            kp2 = kelly_criterion(ensp,mhp) if mhp else None
            adj = {"rest_home":rh,"rest_away":ra,"rest_diff":rd,"travel_dist_miles":round(dist,0),
                   "b2b_home":b2bh,"b2b_away":b2ba,"home_elo_bonus":0 if neutral else 100}
            pd3 = generate_nba_prediction_drivers(game,home,away,elo_dict,efficiency_data,injury_impacts,adj)
            winner=home if ensp>=0.5 else away; wp2=ensp if ensp>=0.5 else 1-ensp; loser=away if ensp>=0.5 else home
            conf="strong" if wp2>0.70 else "moderate" if wp2>0.60 else "slight"
            wn=NBA_TEAM_NAMES.get(winner,winner); ln=NBA_TEAM_NAMES.get(loser,loser)
            weff=efficiency_data.get(winner,{})
            expl=(f"The model gives {wn} a {wp2*100:.1f}% win probability — a {conf} favorite over {ln}."
                  f" {wn} net rating: {weff.get('net_rating',0.0):+.1f}."
                  +(f" {NBA_TEAM_NAMES.get(home,home)} on a back-to-back." if b2bh else "")
                  +(f" {NBA_TEAM_NAMES.get(away,away)} on a back-to-back." if b2ba else "")
                  +(f" Home court +100 ELO for {NBA_TEAM_NAMES.get(home,home)}." if not neutral else ""))
            hi2 = injury_impacts.get(home,{}); ai2 = injury_impacts.get(away,{})
            predictions_list.append({
                "game_id":game_id,"game_time":game["game_time"],"status":game.get("status",""),
                "is_future":bool(game.get("is_future",False)),
                "home_team":home,"away_team":away,
                "home_name":game.get("home_name",home),"away_name":game.get("away_name",away),
                "home_logo":game.get("home_logo",""),"away_logo":game.get("away_logo",""),
                "neutral":neutral,"home_score":game.get("home_score",0),"away_score":game.get("away_score",0),
                "predictions":{"ensemble_prob":round(ensp,4),"logistic_prob":round(lp,4),"elo_prob":round(ep2,4),
                               "xgb_prob":round(xp,4) if xp is not None else None,
                               "pyth_prob":round(pytp,4),"eff_prob":round(effp,4),"bayesian_prob":round(bpr,4)},
                "market":{"home_prob":mhp,"edge":me2,"kelly_pct":kp2,
                          "home_american":mo.get("home_american") if mo else None,
                          "away_american":mo.get("away_american") if mo else None},
                "monte_carlo":mc,"adjustments":adj,
                "h2h":h2h_cache.get((home,away),compute_nba_h2h(home,away,historical_games)),
                "elo":{"home":round(helo,1),"away":round(aelo,1),"diff":round(helo-aelo,1)},
                "bayesian":{"home_mu":br.get("mu_a",muh),"home_sigma":br.get("sigma_a",sh),
                            "away_mu":br.get("mu_b",mua),"away_sigma":br.get("sigma_b",sa2),
                            "home_band":br.get("uncertainty_band_a",sh),"away_band":br.get("uncertainty_band_b",sa2)},
                "efficiency":{"home_off_rating":heff.get("offensive_rating",110.0),
                              "home_def_rating":heff.get("defensive_rating",110.0),
                              "home_net_rating":heff.get("net_rating",0.0),
                              "away_off_rating":aeff.get("offensive_rating",110.0),
                              "away_def_rating":aeff.get("defensive_rating",110.0),
                              "away_net_rating":aeff.get("net_rating",0.0),
                              "home_pace":heff.get("pace",100.0),"away_pace":aeff.get("pace",100.0)},
                "injuries":{
                    "home":[{**p,"value_tier":nba_value_tier_label(nba_player_values.get(p.get("player",""),1.0))} for p in injuries.get(home,[])[:5]],
                    "away":[{**p,"value_tier":nba_value_tier_label(nba_player_values.get(p.get("player",""),1.0))} for p in injuries.get(away,[])[:5]],
                },
                "injury_impact":{"home_elo_penalty":hi2.get("elo_penalty",0.0),"away_elo_penalty":ai2.get("elo_penalty",0.0),
                                 "home_key_players_out":hi2.get("key_players_out",[]),"away_key_players_out":ai2.get("key_players_out",[]),
                                 "home_star_count":hi2.get("star_count",0),"away_star_count":ai2.get("star_count",0),
                                 "home_star_stack_multiplier":hi2.get("star_stack_multiplier",1.0),
                                 "away_star_stack_multiplier":ai2.get("star_stack_multiplier",1.0)},
                "prediction_drivers":pd3,"explanation":expl,
            })
        except Exception as e:
            log.error(f"Error processing NBA game {game.get('game_id','?')}: {e}")

    leaderboard = []
    for t in NBA_TEAMS:
        elo=elo_dict.get(t,1500.0); bayes=bayesian_ratings.get(t,{"mu":elo,"sigma":75.0})
        eff=efficiency_data.get(t,{}); pyth=pyth_data.get(t,{}).get("pyth",0.5)
        ti=injury_impacts.get(t,{}); streak=compute_nba_streak(game_history,t)
        leaderboard.append({
            "team":t,"team_name":NBA_TEAM_NAMES.get(t,t),"abbrev":t.lower(),
            "logo":f"https://a.espncdn.com/i/teamlogos/nba/500/{t.lower()}.png",
            "elo":round(elo,1),"sigma":round(bayes.get("sigma",75.0),1),"mu":round(bayes.get("mu",elo),1),
            "lower_band":round(bayes.get("mu",elo)-bayes.get("sigma",75.0),1),
            "upper_band":round(bayes.get("mu",elo)+bayes.get("sigma",75.0),1),
            "pyth":round(pyth,4),"net_eff":round(eff.get("net_rating",0.0),2),
            "off_eff":round(eff.get("offensive_rating",110.0),1),
            "def_eff":round(eff.get("defensive_rating",110.0),1),
            "net_rating":round(eff.get("net_rating",0.0),2),
            "offensive_rating":round(eff.get("offensive_rating",110.0),1),
            "defensive_rating":round(eff.get("defensive_rating",110.0),1),
            "pace":round(eff.get("pace",100.0),1),
            "wins":standings.get(t,{}).get("wins",0),"losses":standings.get(t,{}).get("losses",0),"ties":0,
            "playoff_prob":round(season_sim.get(t,{}).get("playoff_prob",0.53),4),
            "sb_prob":round(season_sim.get(t,{}).get("sb_prob",0.033),4),
            "champ_prob":round(season_sim.get(t,{}).get("sb_prob",0.033),4),
            "trend":nba_get_trend(game_history,t),
            "streak_type":streak["type"],"streak_count":streak["count"],
            "injury_elo_penalty":ti.get("elo_penalty",0.0),
            "injury_impact_score":ti.get("impact_score",0.0),
            "injury_players_count":ti.get("total_players",0),
        })
    leaderboard.sort(key=lambda x: x["elo"], reverse=True)

    write_nba_predictions(predictions_list, season_year, is_offseason, now_utc)
    write_nba_leaderboard(leaderboard, season_year, now_utc)
    write_nba_model_metrics(model_metrics, len(historical_games), xgb_model is not None, now_utc)
    log.info("=== NBA Update complete ===")
    log.info(f"  Games predicted: {len(predictions_list)}  |  Teams: {len(leaderboard)}")


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Sport3 prediction pipeline")
    parser.add_argument("--nfl", action="store_true")
    parser.add_argument("--nba", action="store_true")
    args = parser.parse_args()
    run_nfl_flag = args.nfl or (not args.nfl and not args.nba)
    run_nba_flag = args.nba or (not args.nfl and not args.nba)
    try:
        if run_nfl_flag: run_nfl()
    except Exception:
        import traceback; traceback.print_exc()
    try:
        if run_nba_flag: run_nba()
    except Exception:
        import traceback; traceback.print_exc(); sys.exit(1)


if __name__ == "__main__":
    main()
