"""
Honest walk-forward backtest of the SHIPPED NFL headline model (pure
QB-adjusted Elo). External review 2026-09-06, item 4: the metrics shown on
the site must come from a walk-forward evaluation - every game predicted
only from information available before kickoff - not from an in-sample fit
of the display logistic model (which trained and scored on the same rows).

Method: replay all completed games in chronological order through the
production code path (model.elo_model.compute_elo + model.qb_model game
adjustments), capturing each game's pre-game ratings via pregame_out, then
score the implied probabilities. Ratings only ever update from games that
already happened, so this is a true walk-forward: no lookahead anywhere.

Vegas closing moneylines (when present in the local nflverse games cache)
are reported as a benchmark reference only - they are never used as a
model input.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from model.elo_model import compute_elo, era_hfa, expected_score

GAMES_CACHE = Path(__file__).resolve().parent.parent / "data" / "nflverse_cache" / "games.csv"

FIRST_SCORED_SEASON = 2017  # QB stats warmup starts 2014; score the last 9 completed seasons


def _auc(probs, actuals):
    """Mann-Whitney AUC (no sklearn dependency)."""
    order = np.argsort(probs)
    ranks = np.empty(len(probs))
    ranks[order] = np.arange(1, len(probs) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(probs, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = sums[inv] / counts[inv]
    pos = ranks[actuals == 1].sum()
    n_pos = int((actuals == 1).sum()); n_neg = int((actuals == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    return float((pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _calibration(probs, actuals):
    buckets = []
    edges = list(range(0, 100, 10))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo / 100) & (probs < (hi / 100 if hi < 100 else 1.01))
        n = int(m.sum())
        if n == 0:
            buckets.append({"bucket": f"{lo}-{hi}%", "predicted": None, "actual": None, "count": 0})
            continue
        buckets.append({"bucket": f"{lo}-{hi}%",
                        "predicted": round(float(probs[m].mean()), 3),
                        "actual": round(float(actuals[m].mean()), 3),
                        "count": n})
    return buckets


def _score(probs, actuals):
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(actuals, dtype=float)
    return {
        "accuracy": round(float(np.mean((p >= 0.5) == y)), 4),
        "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
        "brier_score": round(float(np.mean((p - y) ** 2)), 4),
        "auc": (round(_auc(p, y), 4) if _auc(p, y) is not None else None),
    }


def _american_to_prob(o):
    try:
        o = float(o)
    except (TypeError, ValueError):
        return None
    if math.isnan(o):
        return None
    return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def _vegas_map():
    """{(date_str, home, away): devigged home prob} from the local nflverse cache, if present."""
    if not GAMES_CACHE.exists():
        return {}
    try:
        raw = pd.read_csv(GAMES_CACHE, usecols=["gameday", "home_team", "away_team",
                                                "home_moneyline", "away_moneyline"])
    except Exception:
        return {}
    out = {}
    for r in raw.itertuples():
        hp = _american_to_prob(r.home_moneyline); ap = _american_to_prob(r.away_moneyline)
        if hp is None or ap is None:
            continue
        out[(str(r.gameday), r.home_team, r.away_team)] = hp / (hp + ap)
    return out


def walkforward_metrics(fte_df, qb_ctx, first_scored_season=FIRST_SCORED_SEASON):
    """Score the shipped headline walk-forward. Returns a model_metrics dict or None."""
    if fte_df is None or fte_df.empty:
        return None
    qb_map = (qb_ctx or {}).get("game_adj", {})

    df = fte_df.dropna(subset=["score1", "score2"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Era-rolling HFA exactly as production computes it (all played games per season)
    home_win_rates = {}
    for s, sdf in df.groupby("season"):
        if len(sdf):
            home_win_rates[int(s)] = float((sdf["score1"] > sdf["score2"]).mean())

    pregame = {}
    compute_elo(df, qb_map=qb_map, pregame_out=pregame)

    vegas = _vegas_map()
    recs = []
    for row in df.itertuples():
        if int(row.season) < first_scored_season:
            continue
        key = (str(pd.to_datetime(row.date).date()), row.team1, row.team2)
        base = pregame.get(key)
        if base is None:
            continue
        e1, e2 = base
        qa1, qa2 = qb_map.get(key, (0.0, 0.0))
        hfa = 0.0 if getattr(row, "neutral", 0) else era_hfa(home_win_rates, int(row.season))
        prob = expected_score(e1 + hfa + qa1, e2 + qa2)
        actual = 1.0 if row.score1 > row.score2 else (0.5 if row.score1 == row.score2 else 0.0)
        recs.append({"season": int(row.season), "prob": prob, "actual": actual,
                     "vegas": vegas.get(key)})

    if len(recs) < 100:
        return None
    r = pd.DataFrame(recs)
    r = r[r["actual"] != 0.5]  # ties unscorable, same as the research harness
    probs = r["prob"].values
    actuals = r["actual"].values

    m = _score(probs, actuals)
    m["calibration_buckets"] = _calibration(probs, actuals)
    m["historical_accuracy"] = [
        {"year": int(s), "accuracy": round(float(((g["prob"] >= 0.5) == (g["actual"] == 1)).mean()), 4)}
        for s, g in r.groupby("season")
    ]
    m["evaluation"] = (f"walk-forward backtest of the shipped QB-adjusted Elo, "
                       f"{first_scored_season}-{int(r['season'].max())}, pre-game information only")
    m["n_scored_games"] = int(len(r))

    v = r.dropna(subset=["vegas"])
    if len(v) >= 100:
        m["vegas_benchmark"] = dict(_score(v["vegas"].values, v["actual"].values), n_games=int(len(v)))
    return m
