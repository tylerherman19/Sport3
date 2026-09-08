"""Refresh the published NFL market-edge backtest from cached nflverse data.

This deliberately avoids the live prediction pipeline. It only evaluates the
already shipped walk-forward QB-Elo against historical archived moneylines.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import qb_model
from scripts.backtest_nfl import walkforward_metrics
from scripts.data_fetcher import nfl_abbrev_norm


CACHE = ROOT / "data" / "nflverse_cache" / "games.csv"
OUTPUT = ROOT / "data" / "model_metrics.json"


def cached_games():
    raw = pd.read_csv(CACHE)
    df = raw.dropna(subset=["home_score", "away_score"]).copy()
    df["team1"] = df["home_team"].apply(nfl_abbrev_norm)
    df["team2"] = df["away_team"].apply(nfl_abbrev_norm)
    df["score1"] = df["home_score"].astype(float)
    df["score2"] = df["away_score"].astype(float)
    df["date"] = df["gameday"]
    df["neutral"] = (df["location"] == "Neutral").astype(int)
    return df[["date", "season", "team1", "team2", "score1", "score2", "neutral", "week", "game_type", "game_id"]].sort_values("date").reset_index(drop=True)


def main():
    games = cached_games()
    qb_ctx = qb_model.build_qb_context(games, first_season=2014)
    metrics = walkforward_metrics(games, qb_ctx)
    if not metrics or not metrics.get("market_edge_backtest"):
        raise RuntimeError("No market-edge backtest produced from the local cache")
    payload = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    payload["market_edge_backtest"] = metrics["market_edge_backtest"]
    payload["updated"] = datetime.now(timezone.utc).isoformat()
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    primary = next(r for r in metrics["market_edge_backtest"]["cutoffs"] if r["min_edge"] == 0.05)
    print("market games={}; 5% edge bets={}; roi={:.2%}; net={:+.2f}u".format(
        metrics["market_edge_backtest"]["n_market_games"], primary["bets"], primary["roi"], primary["net_units"]))


if __name__ == "__main__":
    main()
