"""
Walk-forward scorer for the SumerSports EPA boundary layer (model/sumer_epa.py),
measured against the shipped baseline (QB-adjusted Elo + roster-value layer).

Same fairness construction as research/wf_roster.py: production compute_elo
processes games in date order and stamps pre-game ratings, so scoring those
ratings is walk-forward by construction. The EPA deltas for season S read only
the completed S-1 SumerSports table, so no scored game can see its own future.

The layer can only affect seasons 2023+ (first prior table is 2022), so the
headline comparison is reported for the full 2017-2025 window AND the 2023-2025
sub-window where the layer is active; per-season rows show where any gain or
loss actually lives.

Usage:
    python3 research/wf_sumer.py --sweep                 # scale sweep table
    python3 research/wf_sumer.py --compare 250           # per-season at one scale
    python3 research/wf_sumer.py --grid 0,100,250,400    # explicit scales
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.wf_roster import build_inputs, score_path, report, per_season, metrics
from model import sumer_epa
from model.elo_model import compute_elo, era_hfa, expected_score
from model.roster_value import build_offseason_ledger, elo_adjustments
import pandas as pd

SINCE_ACTIVE = 2023  # first season whose boundary can carry an EPA delta


def score_path_anchor(df, qb_map, roster_adjustments, anchor, weight):
    """wf_roster.score_path with the boundary-anchor blend threaded through."""
    pregame = {}
    compute_elo(df, qb_map=qb_map, pregame_out=pregame,
                roster_adjustments=roster_adjustments,
                boundary_anchor=anchor, boundary_anchor_weight=weight)
    home_win_rates = {}
    for s, sdf in df.groupby("season"):
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
        if g.season >= 2017:
            actual = 1.0 if g.score1 > g.score2 else (0.5 if g.score1 == g.score2 else 0.0)
            recs.append({"season": g.season, "week": g.week, "team1": g.team1,
                         "team2": g.team2, "prob": prob, "vegas": g.vegas, "actual": actual})
    return pd.DataFrame(recs)


def active_window(recs):
    return recs[recs["season"] >= SINCE_ACTIVE]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--grid", type=str, default="")
    ap.add_argument("--compare", type=float, default=None)
    ap.add_argument("--cap", type=float, default=sumer_epa.DEFAULT_CAP)
    ap.add_argument("--anchor-sweep", action="store_true",
                    help="sweep the boundary-anchor blend (EPA as primary signal)")
    args = ap.parse_args()

    df, qb_map = build_inputs()
    seasons = sorted(int(s) for s in df["season"].unique())
    ledger = build_offseason_ledger([s for s in seasons if s >= 2015])
    roster = elo_adjustments(ledger)

    base = score_path(df, qb_map, roster_adjustments=roster)
    b_all = report("shipped baseline (roster)", base)
    b_act = report("shipped baseline 2023+", active_window(base))

    if args.grid:
        scales = [float(x) for x in args.grid.split(",")]
    elif args.sweep:
        scales = [0.0, 100.0, 200.0, 250.0, 300.0, 400.0, 600.0]
    elif args.compare is not None:
        scales = [args.compare]
    else:
        scales = []

    for sc in scales:
        sumer = sumer_epa.boundary_deltas_for_seasons(seasons, elo_per_epa=sc, cap=args.cap)
        merged = sumer_epa.merge_adjustments(roster, sumer)
        recs = score_path(df, qb_map, roster_adjustments=merged)
        r_all = report(f"roster+sumer epe={sc:g} (all)", recs)
        r_act = report(f"roster+sumer epe={sc:g} 2023+", active_window(recs))
        print(f"    active-window delta: acc {r_act[0]-b_act[0]:+.4f}  ll {r_act[1]-b_act[1]:+.4f}")

    if args.anchor_sweep:
        for k in (450.0, 700.0):
            anchor = sumer_epa.anchor_ratings_for_seasons(
                seasons, elo_per_epa=k, cap=sumer_epa.DEFAULT_ANCHOR_CAP)
            for w in (0.15, 0.33, 0.5, 0.75, 1.0):
                recs = score_path_anchor(df, qb_map, roster, anchor, w)
                r_all = report(f"anchor K={k:g} w={w:g} (all)", recs)
                r_act = report(f"anchor K={k:g} w={w:g} 2023+", active_window(recs))
                print(f"    active-window delta: acc {r_act[0]-b_act[0]:+.4f}  ll {r_act[1]-b_act[1]:+.4f}")

    if args.compare is not None:
        sumer = sumer_epa.boundary_deltas_for_seasons(seasons, elo_per_epa=args.compare, cap=args.cap)
        merged = sumer_epa.merge_adjustments(roster, sumer)
        recs = score_path(df, qb_map, roster_adjustments=merged)
        pb, pr = per_season(base), per_season(recs)
        print(f"\n  {'season':<7}{'n':<6}{'acc base':<10}{'acc new':<10}{'ll base':<10}{'ll new':<10}")
        for s_ in sorted(pb):
            ab, lb, _, n = pb[s_]
            an, ln, _, _ = pr[s_]
            print(f"  {s_:<7}{n:<6}{ab:<10.4f}{an:<10.4f}{lb:<10.4f}{ln:<10.4f}")
        sumer_d = sumer_epa.boundary_deltas(2026, elo_per_epa=args.compare, cap=args.cap)
        if sumer_d:
            vals = list(sumer_d.values())
            import numpy as np
            print(f"  2026 boundary deltas: min {min(vals):+.1f} max {max(vals):+.1f} "
                  f"sd {np.std(vals):.1f}; cap binds on "
                  f"{sum(abs(v) >= args.cap - 0.01 for v in vals)}/{len(vals)} teams")
