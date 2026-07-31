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

from model.elo_model import annotate_pregame_elo, compute_elo, expected_score
from model.efficiency_model import compute_pythagorean, efficiency_predict_game
from model.logistic_model import build_features, train_logistic, predict_matchups
from model.ensemble_model import build_xgb_features, train_xgboost, predict_xgboost, ensemble_predict
from scripts.data_fetcher import NFLVERSE_GAMES_URL, nfl_abbrev_norm
from scripts.model_engine import NFL_TEAMS, build_nfl_efficiency_data

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


def _season_standings(train_df, prev_season):
    """Final points_for/points_against from the single most recent completed season."""
    season_df = train_df[train_df["season"] == prev_season]
    standings = {}
    for team in NFL_TEAMS:
        home = season_df[season_df["team1"] == team]
        away = season_df[season_df["team2"] == team]
        gp = len(home) + len(away)
        pf = float(home["score1"].sum() + away["score2"].sum()) if gp else 350.0
        pa = float(home["score2"].sum() + away["score1"].sum()) if gp else 350.0
        standings[team] = {"points_for": pf, "points_against": pa, "games_played": gp or 1}
    return standings


def run_full_ensemble_backtest(df, since, min_train_seasons=5):
    """
    Walk-forward retrain of the *actual production ensemble* (ELO + logistic +
    XGBoost + pythagorean + efficiency, blended via the real ensemble_predict())
    once per season, trained only on strictly-prior seasons, then scored against
    that season's real games and Vegas closing lines.

    Simplification, disclosed: ELO/game-history/pythagorean/efficiency are all
    frozen at a start-of-season snapshot (end of the prior season) for the whole
    test season, rather than updated game-by-game the way the live daily pipeline
    does. That means this understates how good the live system actually performs
    (it doesn't get to sharpen its ratings as the real season unfolds) — treat
    these numbers as a conservative lower bound, not the live system's true skill.
    player_form is not included (it isn't a trained/retrainable component, and
    doing it justice here would need its own walk-forward EPA computation).
    """
    seasons = sorted(df["season"].unique())
    rows = []
    for test_season in seasons:
        if test_season < since:
            continue
        train_df = df[df["season"] < test_season]
        if train_df["season"].nunique() < min_train_seasons:
            continue

        elo_dict, game_history = compute_elo(train_df)
        standings = _season_standings(train_df, train_df["season"].max())
        pythagorean_data = compute_pythagorean(
            {t: {"points_for": s["points_for"], "points_against": s["points_against"]} for t, s in standings.items()}
        )
        efficiency_data = build_nfl_efficiency_data(standings, train_df)

        train_df_annotated = annotate_pregame_elo(train_df)

        log_model = scaler = calib = None
        X, y = build_features(train_df_annotated, elo_dict, game_history, efficiency_data, pythagorean_data)
        if len(X) > 50:
            log_model, scaler, calib = train_logistic(X, y)

        xgb_model = xgb_scaler = None
        X_x, y_x = build_xgb_features(train_df_annotated, elo_dict, game_history, efficiency_data, pythagorean_data)
        if len(X_x) > 50:
            xgb_model, xgb_scaler = train_xgboost(X_x, y_x)

        test_df = df[df["season"] == test_season].reset_index(drop=True)
        matchups = [
            {"game_id": i, "team_a": r["team1"], "team_b": r["team2"], "is_home_a": True,
             "neutral": bool(r["neutral"]), "rest_diff": 0, "travel_diff": 0, "turnover_diff": 0}
            for i, r in test_df.iterrows()
        ]
        lps = predict_matchups(matchups, log_model, scaler, calib, elo_dict, game_history,
                                efficiency_data, pythagorean_data) if log_model else None
        xps = predict_xgboost(matchups, xgb_model, xgb_scaler, elo_dict, game_history,
                               efficiency_data, pythagorean_data) if xgb_model else None

        for i, r in test_df.iterrows():
            home, away, neutral = r["team1"], r["team2"], bool(r["neutral"])
            hfa = 0.0 if neutral else 65.0
            elo_prob = expected_score(elo_dict.get(home, 1500.0) + hfa, elo_dict.get(away, 1500.0))
            effr = efficiency_predict_game(home, away, efficiency_data, pythagorean_data, not neutral, neutral)
            lp = lps[i]["logistic_prob"] if lps else None
            xp = xps[i]["xgb_prob"] if xps and xps[i]["xgb_prob"] is not None else None
            ensemble_prob = ensemble_predict(logistic_prob=lp, xgb_prob=xp, elo_prob=elo_prob,
                                              pyth_prob=effr["pyth_prob"], eff_prob=effr["eff_prob"])

            ml_home_p = american_to_prob(r["home_moneyline"])
            ml_away_p = american_to_prob(r["away_moneyline"])
            vegas_prob = devig(ml_home_p, ml_away_p) or spread_to_prob(r["spread_line"])
            if vegas_prob is None:
                continue

            actual = 1 if r["score1"] > r["score2"] else 0
            rows.append({"season": int(test_season), "ensemble_prob": ensemble_prob,
                          "elo_prob": elo_prob, "vegas_prob": vegas_prob, "actual": actual})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2010, help="Earliest season to include")
    ap.add_argument("--full-ensemble", action="store_true",
                     help="Also walk-forward retrain + score the full production ensemble (slower)")
    args = ap.parse_args()

    print("Downloading nflverse game history (with Vegas lines)...")
    raw = pd.read_csv(NFLVERSE_GAMES_URL)
    raw = raw.dropna(subset=["home_score", "away_score"]).copy()
    # Keep the FULL history (not just >= --since) so ELO is properly warmed up and
    # --full-ensemble has real prior-season data to train on for its earliest test
    # year. --since only controls which seasons are included in the reported results.

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
        if r["season"] < args.since:
            continue
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

    if args.full_ensemble:
        print("\n" + "=" * 60)
        print("FULL ENSEMBLE (walk-forward retrained per season) — see")
        print("run_full_ensemble_backtest() docstring for disclosed simplifications.")
        print("=" * 60)
        fe = run_full_ensemble_backtest(df, args.since)
        print(f"\n{len(fe)} games scored\n")
        print(f"{'':12}{'Brier':>10}{'LogLoss':>10}{'Accuracy':>10}{'N':>8}")
        print("-" * 50)
        for label, probs in [("ELO (frozen)", fe["elo_prob"]), ("Full ensemble", fe["ensemble_prob"]), ("Vegas", fe["vegas_prob"])]:
            print(f"{label:14}{brier(probs, fe['actual']):>8.4f}"
                  f"{log_loss(probs, fe['actual']):>10.4f}"
                  f"{accuracy(probs, fe['actual']):>10.3f}{len(fe):>8}")
        ens_b, veg_b = brier(fe["ensemble_prob"], fe["actual"]), brier(fe["vegas_prob"], fe["actual"])
        print(f"\n{'Full ensemble beats Vegas' if ens_b < veg_b else 'Vegas beats full ensemble'} on Brier "
              f"({ens_b:.4f} vs {veg_b:.4f}).")


if __name__ == "__main__":
    main()
