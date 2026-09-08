"""Refresh public NFL moneylines without retraining or rewriting forecasts.

Use when the model artifact is current but odds need a fast refresh.  ESPN's
public scoreboard supplies a single listed sportsbook's moneylines; this keeps
the existing prediction payload intact and updates only its ``market`` object.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data_fetcher import (  # noqa: E402
    fetch_nfl_scoreboard,
    fetch_nfl_future_games,
    fetch_nfl_espn_betting_odds,
)
from model.ensemble_model import kelly_criterion  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent / "data"


def _key(home, away):
    return f"{away}_at_{home}"


def refresh_payload(payload, odds_map):
    """Return payload with only matched game market objects refreshed."""
    updated = 0
    for game in payload.get("games", []):
        odds = odds_map.get(_key(game.get("home_name", ""), game.get("away_name", "")))
        if not odds:
            continue
        model_prob = game.get("predictions", {}).get("ensemble_prob")
        market_prob = odds["home_prob"]
        game["market"] = {
            "home_prob": market_prob,
            "away_prob": odds["away_prob"],
            "edge": round(float(model_prob) - market_prob, 4) if model_prob is not None else None,
            "kelly_pct": kelly_criterion(model_prob, odds["home_american"]) if model_prob is not None else None,
            "home_american": odds["home_american"],
            "away_american": odds["away_american"],
            "source": odds.get("source", "ESPN"),
        }
        updated += 1
    return updated


def main():
    scoreboard, week = fetch_nfl_scoreboard()
    season = next((g.get("season") for g in scoreboard if g.get("season")), None)
    future = fetch_nfl_future_games(week, season or 0, weeks_ahead=3)
    odds_map = fetch_nfl_espn_betting_odds(scoreboard + future)
    if not odds_map:
        raise RuntimeError("ESPN returned no parseable NFL moneylines; existing market data preserved")

    source_path = DATA_DIR / "nfl_predictions.json"
    if not source_path.exists():
        raise RuntimeError("nfl_predictions.json is missing")
    payload = json.loads(source_path.read_text())
    count = refresh_payload(payload, odds_map)
    if count == 0:
        raise RuntimeError("No ESPN odds matched published games; existing market data preserved")

    encoded = json.dumps(payload, indent=2)
    (DATA_DIR / "nfl_predictions.json").write_text(encoded)
    (DATA_DIR / "predictions.json").write_text(encoded)
    log.info("Refreshed market data for %s game(s) from %s ESPN odds row(s)", count, len(odds_map))


if __name__ == "__main__":
    main()
