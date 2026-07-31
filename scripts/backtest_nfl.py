"""
backtest_nfl.py — Model vs. Vegas, on real historical closing lines.

Answers the question "is this actually accurate" with evidence instead of a
guess. Uses model.elo_model.annotate_pregame_elo(), which stamps each
historical game with the two teams' ELO ratings as they stood *before* that
game — i.e. walk-forward, no lookahead by construction — so this is a fair
backtest, not an in-sample fit. Only ELO is tested this way; the logistic/
XGBoost sub-models are trained once on the full history in production, so
using them here would leak future games into "past" predictions. A fair
backtest of those would require season-by-season walk-forward retraining,
which this script does not attempt.

Vegas' historical closing lines (moneyline / spread) ship in nflverse's own
games.csv — no separate odds API or key needed for this.

Usage: python scripts/backtest_nfl.py [--since 2015]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from model.elo_model import annotate_pregame_elo, expected_score
from scripts.data_fetcher import NFLVERSE_GAMES_URL, nfl_abbrev_norm

logging.basicConfig(level=logging.WARNING)


def american_to_prob(odds):
    if odds is None or pd.isna(odds):
        return None
    odds = float(odds)
    return 100.0 / (odds + 100.0) if odds > 0 else -odds / (-odds + 100.0)


def devig(p_home, p_away):
    if p_home is None or p_away is None:
        return None
    total = p_home + p_away
    return p_home / total if total > 0 else None


def spread_to_prob(spread_line, std=13.86):
    """spread_line is the home team's line (negative = home favored)."""
    if spread_line is None or pd.isna(spread_line):
        return None
    from scipy.stats import norm
    return float(norm.cdf(-spread_line / std))


def brier(probs, actuals):
    return float(np.mean((np.array(probs) - np.array(actuals)) ** 2))


def log_loss(probs, actuals):
    p = np.clip(np.array(probs), 1e-6, 1 - 1e-6)
    y = np.array(actuals)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(probs, actuals):
    pred = (np.array(probs) >= 0.5).astype(int)
    return float(np.mean(pred == np.array(actuals)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2010, help="Earliest season to include")
    args = ap.parse_args()

    print("Downloading nflverse game history (with Vegas lines)...")
    raw = pd.read_csv(NFLVERSE_GAMES_URL)
    raw = raw.dropna(subset=["home_score", "away_score"]).copy()
    raw = raw[raw["season"] >= args.since].reset_index(drop=True)

    df = pd.DataFrame({
        "date": raw["gameday"],
        "season": raw["season"],
        "team1": raw["home_team"].apply(nfl_abbrev_norm),
        "team2": raw["away_team"].apply(nfl_abbrev_norm),
        "score1": raw["home_score"].astype(float),
        "score2": raw["away_score"].astype(float),
        "neutral": (raw["location"] == "Neutral").astype(int),
        "home_moneyline": raw["home_moneyline"],
        "away_moneyline": raw["away_moneyline"],
        "spread_line": raw["spread_line"],
    }).sort_values("date").reset_index(drop=True)

    print(f"{len(df)} games, seasons {df['season'].min()}-{df['season'].max()}")
    print("Computing walk-forward pre-game ELO (no lookahead)...")
    annotated = annotate_pregame_elo(df)
    annotated["neutral"] = annotated["neutral"].values

    rows = []
    for _, r in annotated.iterrows():
        hfa = 0.0 if r["neutral"] else 65.0
        elo_prob = expected_score(r["elo1_pre"] + hfa, r["elo2_pre"])

        ml_home_p = american_to_prob(r["home_moneyline"])
        ml_away_p = american_to_prob(r["away_moneyline"])
        vegas_prob = devig(ml_home_p, ml_away_p)
        market_source = "moneyline"
        if vegas_prob is None:
            vegas_prob = spread_to_prob(r["spread_line"])
            market_source = "spread"
        if vegas_prob is None:
            continue  # no market data for this game at all — exclude from comparison

        actual = 1 if r["score1"] > r["score2"] else 0
        rows.append({
            "season": int(r["season"]), "elo_prob": elo_prob,
            "vegas_prob": vegas_prob, "actual": actual, "market_source": market_source,
        })

    bt = pd.DataFrame(rows)
    print(f"\n{len(bt)} games with usable market data ({(bt.market_source=='moneyline').sum()} moneyline, "
          f"{(bt.market_source=='spread').sum()} spread-only)\n")

    print(f"{'':12}{'Brier':>10}{'LogLoss':>10}{'Accuracy':>10}{'N':>8}")
    print("-" * 50)
    for label, probs in [("ELO", bt["elo_prob"]), ("Vegas", bt["vegas_prob"])]:
        print(f"{label:12}{brier(probs, bt['actual']):>10.4f}"
              f"{log_loss(probs, bt['actual']):>10.4f}"
              f"{accuracy(probs, bt['actual']):>10.3f}{len(bt):>8}")

    print("\nBy season (last 10):")
    print(f"{'Season':8}{'ELO Brier':>12}{'Vegas Brier':>12}{'ELO Acc':>10}{'Vegas Acc':>10}{'N':>6}")
    for season, grp in bt.groupby("season"):
        if season < bt["season"].max() - 9:
            continue
        print(f"{season:<8}{brier(grp['elo_prob'], grp['actual']):>12.4f}"
              f"{brier(grp['vegas_prob'], grp['actual']):>12.4f}"
              f"{accuracy(grp['elo_prob'], grp['actual']):>10.3f}"
              f"{accuracy(grp['vegas_prob'], grp['actual']):>10.3f}{len(grp):>6}")

    elo_b, vegas_b = brier(bt["elo_prob"], bt["actual"]), brier(bt["vegas_prob"], bt["actual"])
    print(f"\n{'ELO beats Vegas' if elo_b < vegas_b else 'Vegas beats ELO'} on Brier score "
          f"({elo_b:.4f} vs {vegas_b:.4f}) across {len(bt)} games since {args.since}.")


if __name__ == "__main__":
    main()
