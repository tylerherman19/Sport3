"""
Walk-forward scorer for the SHIPPED NFL code path, used to validate the offseason
roster-value layer against the cdeaebc baseline.

Unlike research/wf_elo.py and research/wf_qb.py (standalone labs that re-implement
the mechanics), this harness imports the production modules and scores the exact
ratings the live pipeline ships: model.elo_model.compute_elo for the Elo chain,
model.qb_model.build_qb_context for the QB overlay, and the same era-rolling HFA.

Fairness: compute_elo processes games in date order and stamps every game with the
two teams' ratings as they stood BEFORE that game (pregame_out). Scoring those
pregame ratings is therefore walk-forward by construction - no result can influence
its own prediction. The QB overlay is likewise a pre-game adjustment.

Usage:
    python3 research/wf_roster.py                # baseline, no roster layer
    python3 research/wf_roster.py --roster       # with the roster layer at its shipped scale
    python3 research/wf_roster.py --sweep        # scale sweep table
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.elo_model import compute_elo, era_hfa, expected_score
from model.qb_model import build_qb_context

REPO = Path(__file__).resolve().parents[1]
GAMES = REPO / "data" / "nflverse_cache" / "games.csv"
SINCE = 2017          # scoring window used by research/RESULTS.md
LEDGER_FIRST = 2015   # first offseason the roster layer is applied to
FIRST_SEASON = 1999

ABBR = {"WSH": "WAS", "JAC": "JAX", "LVR": "LV", "LA": "LAR", "OAK": "LV",
        "SD": "LAC", "STL": "LAR", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE",
        "HST": "HOU", "SL": "LAR"}


def norm(t):
    return ABBR.get(str(t).upper(), str(t).upper())


def american_to_prob(o):
    if o is None or (isinstance(o, float) and math.isnan(o)):
        return None
    o = float(o)
    return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def devig(a, b):
    if a is None or b is None:
        return None
    return a / (a + b)


def load_games():
    raw = pd.read_csv(GAMES, low_memory=False)
    raw = raw.dropna(subset=["home_score", "away_score"]).copy()
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["gameday"]),
        "season": raw["season"].astype(int),
        "team1": raw["home_team"].map(norm),
        "team2": raw["away_team"].map(norm),
        "score1": raw["home_score"].astype(float),
        "score2": raw["away_score"].astype(float),
        "neutral": (raw["location"] == "Neutral").astype(int),
        "week": raw["week"].astype(int),
        "game_id": raw["game_id"],
        "vegas": [devig(american_to_prob(h), american_to_prob(a))
                  for h, a in zip(raw["home_moneyline"], raw["away_moneyline"])],
    }).sort_values("date").reset_index(drop=True)
    return df


def score_path(df, qb_map, roster_adjustments=None):
    """Run production compute_elo over the full frame, then score every game in the
    window from its own pregame ratings + the same HFA/QB adjustment compute_elo used."""
    pregame = {}
    compute_elo(df, qb_map=qb_map, pregame_out=pregame,
                roster_adjustments=roster_adjustments)

    # era-rolling HFA, recomputed exactly as compute_elo does internally
    home_win_rates = {}
    for s in sorted(df["season"].unique()):
        sdf = df[df["season"] == s]
        if len(sdf):
            home_win_rates[s] = float((sdf["score1"] > sdf["score2"]).mean())

    recs = []
    for g in df.itertuples():
        key = (str(pd.to_datetime(g.date).date()), g.team1, g.team2)
        if key not in pregame:
            continue
        e1, e2 = pregame[key]
        hfa = 0.0 if g.neutral else era_hfa(home_win_rates, g.season, default=48.0)
        a1, a2 = (qb_map or {}).get(key, (0.0, 0.0))
        prob = expected_score(e1 + hfa + a1, e2 + a2)
        if g.season >= SINCE:
            actual = 1.0 if g.score1 > g.score2 else (0.5 if g.score1 == g.score2 else 0.0)
            recs.append({"season": g.season, "week": g.week, "team1": g.team1,
                         "team2": g.team2, "prob": prob, "vegas": g.vegas,
                         "actual": actual})
    return pd.DataFrame(recs)


def metrics(recs, col="prob"):
    d = recs.dropna(subset=["vegas"])
    d = d[d["actual"] != 0.5]
    p = np.clip(d[col].values, 1e-6, 1 - 1e-6)
    y = d["actual"].values
    return (float(np.mean((p >= 0.5) == y)),
            float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            float(np.mean((p - y) ** 2)),
            len(d))


def per_season(recs, col="prob"):
    out = {}
    for s, sub in recs.groupby("season"):
        out[int(s)] = metrics(sub, col)
    return out


def report(label, recs):
    acc, ll, br, n = metrics(recs)
    vacc, vll, vbr, _ = metrics(recs, "vegas")
    print(f"{label:<28} N={n}  acc={acc:.4f} ll={ll:.4f} br={br:.4f}"
          f"   | vegas acc={vacc:.4f} ll={vll:.4f}")
    return acc, ll, br, n


def build_inputs():
    df = load_games()
    print(f"{len(df)} completed games, seasons {df.season.min()}-{df.season.max()}")
    qb = build_qb_context(df, first_season=2014)
    print(f"QB adjustments on {len(qb['game_adj'])} games")
    return df, qb["game_adj"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--seasons", action="store_true", help="print per-season table")
    ap.add_argument("--grid", type=str, default="",
                    help="comma-separated elo_per_av values to sweep")
    ap.add_argument("--compare", type=float, default=None,
                    help="per-season baseline-vs-roster table at this elo_per_av")
    args = ap.parse_args()

    df, qb_map = build_inputs()
    base = score_path(df, qb_map)
    b = report("baseline (cdeaebc)", base)

    if args.roster or args.sweep or args.grid:
        from model.roster_value import build_offseason_ledger, elo_adjustments
        # Build the ledger from LEDGER_FIRST so the ratings that enter the scored
        # window have already absorbed a few offseasons of the layer, rather than
        # the layer switching on mid-chain.
        seasons = sorted(int(s) for s in df["season"].unique())
        ledger = build_offseason_ledger([s for s in seasons if s >= LEDGER_FIRST])
        if args.grid:
            scales = [float(x) for x in args.grid.split(",")]
        elif args.sweep:
            scales = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        else:
            scales = [None]
        for sc in scales:
            adj = elo_adjustments(ledger) if sc is None else elo_adjustments(ledger, elo_per_av=sc)
            recs = score_path(df, qb_map, roster_adjustments=adj)
            r = report(f"roster elo_per_av={sc}", recs)
            print(f"    delta acc {r[0]-b[0]:+.4f}  ll {r[1]-b[1]:+.4f}")

    if args.compare is not None:
        from model.roster_value import build_offseason_ledger, elo_adjustments
        seasons = sorted(int(s) for s in df["season"].unique())
        ledger = build_offseason_ledger([s for s in seasons if s >= LEDGER_FIRST])
        recs = score_path(df, qb_map,
                          roster_adjustments=elo_adjustments(ledger, elo_per_av=args.compare))
        report(f"roster elo_per_av={args.compare}", recs)
        pb, pr = per_season(base), per_season(recs)
        better_ll = better_acc = 0
        print(f"\n  {'season':<7}{'n':<6}{'acc base':<10}{'acc new':<10}{'ll base':<10}{'ll new':<10}")
        for s_ in sorted(pb):
            ab, lb, _, n = pb[s_]
            an, ln, _, _ = pr[s_]
            better_ll += ln < lb
            better_acc += an > ab
            print(f"  {s_:<7}{n:<6}{ab:<10.4f}{an:<10.4f}{lb:<10.4f}{ln:<10.4f}")
        print(f"  log loss improves in {better_ll}/{len(pb)} seasons, "
              f"accuracy in {better_acc}/{len(pb)}")
        adj = elo_adjustments(ledger, elo_per_av=args.compare)
        allv = [v for d in adj.values() for v in d.values()]
        print(f"  applied Elo deltas: min {min(allv):+.1f} max {max(allv):+.1f} "
              f"sd {np.std(allv):.1f}; cap binds on "
              f"{sum(abs(v) >= 39.99 for v in allv)}/{len(allv)} team-seasons")

    if args.seasons:
        ps = per_season(base)
        for s in sorted(ps):
            acc, ll, br, n = ps[s]
            print(f"  {s}  n={n:<4} acc={acc:.4f} ll={ll:.4f}")
