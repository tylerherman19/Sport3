"""
output_writer.py — JSON output functions for Sport3.
All file I/O is isolated here; no fetch/model logic.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _write(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2, default=str))
    log.info(f"Wrote {path.name} ({path.stat().st_size // 1024} KB)")


# ---- NFL writers ----

def write_nfl_predictions(predictions_list: list, season_year: int,
                           week_num: int, is_offseason: bool, now_utc: str):
    obj = {"updated": now_utc, "season": season_year, "week": week_num,
           "is_offseason": is_offseason, "games": predictions_list}
    _write(DATA_DIR / "predictions.json", obj)
    _write(DATA_DIR / "nfl_predictions.json", obj)


def write_nfl_elo_ratings(elo_ratings_list: list, season_year: int, now_utc: str):
    _write(DATA_DIR / "elo_ratings.json",
           {"updated": now_utc, "season": season_year, "ratings": elo_ratings_list})


def write_nfl_leaderboard(leaderboard_list: list, season_year: int, now_utc: str):
    obj = {"updated": now_utc, "season": season_year, "teams": leaderboard_list}
    _write(DATA_DIR / "leaderboard.json", obj)
    _write(DATA_DIR / "nfl_leaderboard.json", obj)


def write_nfl_model_metrics(model_metrics: dict, n_training_games: int,
                              xgb_available: bool, now_utc: str):
    _write(DATA_DIR / "model_metrics.json", {
        "updated": now_utc,
        "log_loss": model_metrics.get("log_loss"),
        "brier_score": model_metrics.get("brier_score"),
        "auc": model_metrics.get("auc"),
        "calibration_buckets": model_metrics.get("calibration_buckets", []),
        "historical_accuracy": model_metrics.get("historical_accuracy", []),
        "n_training_games": n_training_games,
        "xgboost_available": xgb_available,
    })


def write_nfl_injuries(injuries_list: list, now_utc: str):
    _write(DATA_DIR / "nfl_injuries.json", {"updated": now_utc, "injuries": injuries_list})


# ---- NBA writers ----

def write_nba_predictions(predictions_list: list, season_year: int,
                           is_offseason: bool, now_utc: str):
    _write(DATA_DIR / "nba_predictions.json", {
        "updated": now_utc, "season": season_year, "league": "nba",
        "is_offseason": is_offseason, "games": predictions_list,
    })


def write_nba_leaderboard(leaderboard_list: list, season_year: int, now_utc: str):
    _write(DATA_DIR / "nba_leaderboard.json",
           {"updated": now_utc, "season": season_year, "league": "nba", "teams": leaderboard_list})


def write_nba_model_metrics(model_metrics: dict, n_training_games: int,
                              xgb_available: bool, now_utc: str):
    _write(DATA_DIR / "nba_model_metrics.json", {
        "updated": now_utc, "league": "nba",
        "log_loss": model_metrics.get("log_loss"),
        "brier_score": model_metrics.get("brier_score"),
        "auc": model_metrics.get("auc"),
        "n_training_games": n_training_games,
        "xgboost_available": xgb_available,
        "calibration_buckets": [],
        "historical_accuracy": [],
    })


def write_nba_injuries(injuries_list: list, now_utc: str):
    _write(DATA_DIR / "nba_injuries.json", {"updated": now_utc, "injuries": injuries_list})


def write_nba_ensemble_weights(weights: dict):
    _write(DATA_DIR / "ensemble_weights_nba.json", weights)


def load_nba_ensemble_weights() -> dict:
    path = DATA_DIR / "ensemble_weights_nba.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"elo": 0.25, "pyth": 0.20, "eff": 0.15, "log": 0.25, "xgb": 0.15}


def guard_nba_empty_output(existing_path: Path) -> bool:
    """Return True if we should abort to preserve existing data."""
    if existing_path.exists():
        try:
            return bool(json.loads(existing_path.read_text()).get("games"))
        except Exception:
            pass
    return False
