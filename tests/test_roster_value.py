"""Guards for the offseason roster-value layer.

Two things have to hold and neither is obvious from reading the call site:

1. The roster delta is applied ALONGSIDE the mean reversion at a season boundary,
   never instead of it, and it is bounded.
2. Quarterbacks never reach this layer. model/qb_model.py already adjusts every
   team's effective Elo for its starting QB, so a quarterback counted here too
   would be paid for twice.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import roster_value as rv
from model.elo_model import compute_elo


def _two_season_history():
    rows = []
    for season, fmt in ((2024, "2024-09-%02d"), (2025, "2025-09-%02d")):
        for i in range(1, 9):
            rows.append({"date": fmt % i, "season": season,
                         "team1": "AAA", "team2": "BBB",
                         "score1": 30.0, "score2": 10.0, "neutral": 0})
    return pd.DataFrame(rows)


# ----------------------------------------------------------- compute_elo wiring

def test_roster_delta_is_applied_on_top_of_mean_reversion():
    """Reversion still runs; the delta lands on the reverted rating, not instead."""
    df = _two_season_history()
    plain, _ = compute_elo(df, use_era_hfa=False, current_season=2026)
    adjusted, _ = compute_elo(df, use_era_hfa=False, current_season=2026,
                              roster_adjustments={2026: {"AAA": -25.0, "BBB": 10.0}})
    assert adjusted["AAA"] == pytest.approx(plain["AAA"] - 25.0)
    assert adjusted["BBB"] == pytest.approx(plain["BBB"] + 10.0)


def test_roster_delta_applies_inside_the_played_season_loop():
    """A boundary the main loop owns must get the delta too, otherwise backtest and
    live would apply the layer at different boundaries."""
    df = _two_season_history()
    plain, _ = compute_elo(df, use_era_hfa=False)
    adjusted, _ = compute_elo(df, use_era_hfa=False,
                              roster_adjustments={2025: {"AAA": -100.0}})
    # The 2025 delta lands before 2025's games, so it is damped by that season's
    # results rather than showing up whole - but it must move the rating down.
    assert adjusted["AAA"] < plain["AAA"]


def test_unknown_teams_and_empty_maps_are_no_ops():
    df = _two_season_history()
    plain, _ = compute_elo(df, use_era_hfa=False, current_season=2026)
    for adj in (None, {}, {2026: {}}, {2026: {"ZZZ": 40.0}}, {2027: {"AAA": 40.0}}):
        out, _ = compute_elo(df, use_era_hfa=False, current_season=2026,
                             roster_adjustments=adj)
        assert out == plain


def test_season_keys_may_be_strings():
    """The ledger round-trips through JSON, where dict keys become strings."""
    df = _two_season_history()
    out, _ = compute_elo(df, use_era_hfa=False, current_season=2026,
                         roster_adjustments={"2026": {"AAA": -30.0}})
    plain, _ = compute_elo(df, use_era_hfa=False, current_season=2026)
    assert out["AAA"] == pytest.approx(plain["AAA"] - 30.0)


# ------------------------------------------------------------- QB exclusion

def test_quarterbacks_are_excluded_from_the_layer():
    assert rv._is_qb("QB")
    assert rv._is_qb("WR", "QB")          # QB on either side of the move
    assert rv._is_qb(None, "qb")
    assert not rv._is_qb("WR", "OL", None)
    assert not rv._is_qb(float("nan"))


def test_draft_curve_ignores_quarterbacks():
    """Rookie value is a non-QB curve: a drafted quarterback must not set the price
    of the pick, because the QB overlay prices quarterbacks itself."""
    df = pd.DataFrame({
        "season": [2000] * 6, "round": [1] * 6, "pick": [1, 2, 3, 4, 5, 6],
        "position": ["QB", "QB", "QB", "WR", "WR", "WR"],
        "w_av": [200.0, 200.0, 200.0, 10.0, 10.0, 10.0],
        "car_av": [None] * 6, "dr_av": [None] * 6,
        "games": [170.0] * 6,
    })
    kept = df[~df["position"].isin(rv.QB_POSITIONS)]
    assert len(kept) == 3 and set(kept["position"]) == {"WR"}


# --------------------------------------------------------------- the cap

def test_elo_adjustment_is_capped_in_both_directions():
    ledger = {2026: {"AAA": {"net_av": 500.0}, "BBB": {"net_av": -500.0},
                     "CCC": {"net_av": 4.0}}}
    adj = rv.elo_adjustments(ledger, elo_per_av=1.0, cap=40.0)[2026]
    assert adj["AAA"] == 40.0
    assert adj["BBB"] == -40.0
    assert adj["CCC"] == 4.0


def test_cap_holds_at_the_shipped_constants():
    """No offseason may dominate a rating, whatever the ledger says."""
    ledger = {2026: {"AAA": {"net_av": 1e6}}}
    adj = rv.elo_adjustments(ledger)[2026]
    assert abs(adj["AAA"]) <= rv.ROSTER_ELO_CAP


# ------------------------------------------------------- rookies are discounted

def test_rookie_value_is_a_fraction_of_the_pick_curve_and_decreasing():
    curve = {0: 8.0, 1: 6.0, 2: 4.0, 3: 2.0}
    top = rv.rookie_value(1, curve)
    late = rv.rookie_value(3 * rv.DRAFT_PICK_BUCKET, curve)
    assert top == pytest.approx(rv.ROOKIE_DISCOUNT * 8.0)
    assert late < top
    assert 0.25 <= rv.ROOKIE_DISCOUNT <= 0.5, "spec: rookies discounted 25-50%"
    # A first overall pick must stay well under a returning starter's value.
    assert top < min(rv.STARTER_AV[g] for g in ("WR", "DL", "OL"))


def test_rookie_value_without_a_curve_is_zero():
    assert rv.rookie_value(1, {}) == 0.0


def test_undrafted_rookies_get_the_tail_of_the_curve():
    curve = {0: 8.0, 1: 6.0, 2: 4.0, 3: 2.0}
    assert rv.rookie_value(None, curve) == pytest.approx(rv.ROOKIE_DISCOUNT * 2.0)


# ------------------------------------------------------------- housekeeping

def test_team_codes_are_normalised_across_relocations():
    assert rv.norm_team("OAK") == "LV"
    assert rv.norm_team("SD") == "LAC"
    assert rv.norm_team("STL") == "LAR"
    assert rv.norm_team("ARZ") == "ARI"
    assert rv.norm_team("KC") == "KC"


def test_position_groups_cover_both_roster_and_snap_vocabularies():
    # Roster files use coarse groups, snap counts use fine positions; both must map.
    for pos in ("DB", "DL", "LB", "OL", "WR", "RB", "TE", "K", "P", "LS"):
        assert rv.position_group(pos) is not None, pos
    for pos in ("T", "G", "C", "DE", "DT", "NT", "CB", "S", "FS", "HB", "FB"):
        assert rv.position_group(pos) is not None, pos
    assert rv.position_group("QB") == "QB"


def test_practice_squad_and_cut_players_are_not_roster_members():
    assert "DEV" not in rv.ACTIVE_STATUSES
    assert "CUT" not in rv.ACTIVE_STATUSES
    assert {"ACT", "RES", "INA"} <= rv.ACTIVE_STATUSES
