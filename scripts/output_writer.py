"""
Output Writer Module - Sport3 Refactor
All JSON export functions for NFL and NBA prediction data.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def write_json(filepath: Path, data: dict, description: str = ""):
    """Write JSON data to a file with logging."""
    filepath.write_text(json.dumps(data, indent=2, default=str))
    size_kb = filepath.stat().st_size / 1024
    log.info(f"Wrote {description or filepath.name}: {size_kb:.1f} KB")


# ─── NFL Output Writers ────────────────────────────────────────────────────────

def write_nfl_predictions(predictions_list, season_year, week_num, is_offseason):
    """Write NFL predictions JSON files."""
    now_utc = datetime.now(timezone.utc).isoformat()
    predictions_json = {
        "updated": now_utc,
        "season": season_year,
        "week": week_num,
        "is_offseason": is_offseason,
        "games": predictions_list,
    }
    write_json(DATA_DIR / "predictions.json", predictions_json, "predictions.json")
    write_json(DATA_DIR / "nfl_predictions.json", predictions_json, "nfl_predictions.json")
    return predictions_json


def write_nfl_elo_ratings(elo_ratings_list, season_year):
    """Write NFL ELO ratings JSON file."""
    now_utc = datetime.now(timezone.utc).isoformat()
    elo_ratings_json = {
        "updated": now_utc,
        "season": season_year,
        "ratings": elo_ratings_list,
    }
    write_json(DATA_DIR / "elo_ratings.json", elo_ratings_json, "elo_ratings.json")
    return elo_ratings_json


def write_nfl_leaderboard(leaderboard_list, season_year):
    """Write NFL leaderboard JSON files."""
    now_utc = datetime.now(timezone.utc).isoformat()
    leaderboard_json = {
        "updated": now_utc,
        "season": season_year,
        "teams": leaderboard_list,
    }
    write_json(DATA_DIR / "leaderboard.json", leaderboard_json, "leaderboard.json")
    write_json(DATA_DIR / "nfl_leaderboard.json", leaderboard_json, "nfl_leaderboard.json")
    return leaderboard_json


def write_nfl_model_metrics(model_metrics, season_year, fte_df, xgb_model):
    """Write NFL model metrics JSON file."""
    now_utc = datetime.now(timezone.utc).isoformat()
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
    write_json(DATA_DIR / "model_metrics.json", model_metrics_json, "model_metrics.json")
    return model_metrics_json


def write_nfl_injuries(injuries, player_values, value_tier_label_fn, now_utc=None):
    """Write NFL injuries JSON file."""
    if not now_utc:
        now_utc = datetime.now(timezone.utc).isoformat()
    if not injuries:
        log.warning("No NFL injury data fetched — keeping existing nfl_injuries.json")
        return
    from scripts.data_fetcher import normalize_player_name
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
                "value_tier": value_tier_label_fn(pmult),
            })
    write_json(
        DATA_DIR / "nfl_injuries.json",
        {"updated": now_utc, "injuries": nfl_injuries_list},
        "nfl_injuries.json"
    )


# ─── NBA Output Writers ────────────────────────────────────────────────────────

def write_nba_predictions(predictions_list, season_year, is_offseason):
    """Write NBA predictions JSON file."""
    now_utc = datetime.now(timezone.utc).isoformat()
    predictions_json = {
        "updated": now_utc,
        "season": season_year,
        "league": "nba",
        "is_offseason": is_offseason,
        "games": predictions_list,
    }
    write_json(DATA_DIR / "nba_predictions.json", predictions_json, "nba_predictions.json")
    return predictions_json


def write_nba_leaderboard(leaderboard_list, season_year):
    """Write NBA leaderboard JSON file."""
    now_utc = datetime.now(timezone.utc).isoformat()
    leaderboard_json = {
        "updated": now_utc,
        "season": season_year,
        "league": "nba",
        "teams": leaderboard_list,
    }
    write_json(DATA_DIR / "nba_leaderboard.json", leaderboard_json, "nba_leaderboard.json")
    return leaderboard_json


def write_nba_model_metrics(model_metrics, season_year, historical_games, xgb_model):
    """Write NBA model metrics JSON file."""
    now_utc = datetime.now(timezone.utc).isoformat()
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
    write_json(DATA_DIR / "nba_model_metrics.json", model_metrics_json, "nba_model_metrics.json")
    return model_metrics_json


def write_nba_injuries(injuries, player_values, nba_value_tier_label_fn, now_utc=None):
    """Write NBA injuries JSON file."""
    if not now_utc:
        now_utc = datetime.now(timezone.utc).isoformat()
    if not injuries:
        log.warning("No NBA injury data fetched — keeping existing nba_injuries.json")
        return
    nba_injuries_list = []
    for team, players in injuries.items():
        for p in players:
            pname = p.get("player", "")
            pmult = player_values.get(pname, 1.0)
            nba_injuries_list.append({
                "player": pname,
                "team": team,
                "position": p.get("position", ""),
                "status": p.get("status", ""),
                "injury_description": p.get("injury_description", p.get("status", "")),
                "value_tier": nba_value_tier_label_fn(pmult),
            })
    write_json(
        DATA_DIR / "nba_injuries.json",
        {"updated": now_utc, "injuries": nba_injuries_list},
        "nba_injuries.json"
    )


def write_nba_elo_ratings(elo_dict, game_history, season_year):
    """Write NBA ELO ratings JSON file."""
    now_utc = datetime.now(timezone.utc).isoformat()
    ratings_list = [
        {
            "team": team,
            "elo": round(elo, 1),
            "games_played": len(game_history.get(team, [])),
        }
        for team, elo in sorted(elo_dict.items(), key=lambda x: x[1], reverse=True)
    ]
    elo_json = {"updated": now_utc, "season": season_year, "ratings": ratings_list}
    write_json(DATA_DIR / "nba_elo_ratings.json", elo_json, "nba_elo_ratings.json")
    return elo_json


def write_ensemble_weights(weights: dict, league: str = "nba"):
    """Write learned ensemble weights to file."""
    filepath = DATA_DIR / f"ensemble_weights_{league}.json"
    write_json(filepath, weights, f"ensemble_weights_{league}.json")
