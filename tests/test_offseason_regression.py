"""Regression tests for the 2026 Week 1 defect: ratings that never crossed the offseason.

The walk-forward harness always has the target season's games in frame, so it
regresses at every season boundary and never reproduced any of this. The live
pipeline builds ratings from COMPLETED games only, so before Week 1 it stopped at
the previous season and shipped raw end-of-last-season state.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.elo_model import compute_elo
from model.qb_model import TEAM_REVERT, VET_REVERT, _state_for_season
from scripts.model_engine import build_nfl_efficiency_data, nfl_season_stats_available


def _two_season_history():
    """A tiny league where one team wins everything and the other loses everything,
    so end-of-season Elo is guaranteed to be far from 1500."""
    rows = []
    for season, dates in ((2024, "2024-09-%02d"), (2025, "2025-09-%02d")):
        for i in range(1, 9):
            rows.append({"date": dates % i, "season": season,
                         "team1": "AAA", "team2": "BBB",
                         "score1": 30.0, "score2": 10.0, "neutral": 0})
    return pd.DataFrame(rows)


def test_current_season_applies_offseason_regression():
    df = _two_season_history()
    raw, _ = compute_elo(df, use_era_hfa=False)
    regressed, _ = compute_elo(df, use_era_hfa=False, current_season=2026)

    # Without current_season the caller gets end-of-2025 ratings untouched.
    assert raw["AAA"] > 1500.0 and raw["BBB"] < 1500.0
    for team in ("AAA", "BBB"):
        assert regressed[team] == pytest.approx(raw[team] * 0.67 + 1500.0 * 0.33)

    # The spread must actually compress -- this is the whole point.
    assert (regressed["AAA"] - regressed["BBB"]) < (raw["AAA"] - raw["BBB"])


def test_regression_is_applied_once_per_missing_season():
    df = _two_season_history()
    raw, _ = compute_elo(df, use_era_hfa=False)
    two, _ = compute_elo(df, use_era_hfa=False, current_season=2027)
    expected = raw["AAA"]
    for _ in range(2):
        expected = expected * 0.67 + 1500.0 * 0.33
    assert two["AAA"] == pytest.approx(expected)


def test_no_double_regression_once_the_season_has_results():
    """Self-healing: when the season's own games arrive the main loop owns the
    boundary and current_season must become a no-op."""
    df = _two_season_history()
    df_2026 = pd.concat([df, pd.DataFrame([{
        "date": "2026-09-13", "season": 2026, "team1": "AAA", "team2": "BBB",
        "score1": 24.0, "score2": 17.0, "neutral": 0}])], ignore_index=True)
    with_flag, _ = compute_elo(df_2026, use_era_hfa=False, current_season=2026)
    without, _ = compute_elo(df_2026, use_era_hfa=False)
    assert with_flag == without


def test_game_history_resets_across_the_offseason():
    df = _two_season_history()
    _, hist = compute_elo(df, use_era_hfa=False, current_season=2026)
    # Stale 2025 results must not feed recent-form or head-to-head in a new season.
    assert hist["AAA"] == []


def test_zero_game_standings_do_not_become_a_rating():
    """ESPN reports 0 points for / 0 against before a team plays -- present, not
    missing -- so the old `.get(key, 350)` defaults never fired and every team
    collapsed to off_eff = def_eff = 0, net_eff = -2.0."""
    standings = {t: {"points_for": 0.0, "points_against": 0.0, "games_played": 0,
                     "wins": 0, "losses": 0} for t in ("KC", "LV", "MIA", "SEA")}
    assert nfl_season_stats_available(standings) is False
    eff = build_nfl_efficiency_data(standings, None)
    for team in standings:
        assert eff[team]["net_eff"] == 0.0
        assert eff[team]["off_eff"] == 1.0
        assert eff[team]["def_eff"] == 1.0
        assert eff[team]["elo_equiv"] == 1500.0


def test_season_stats_become_available_after_one_game():
    standings = {"KC": {"points_for": 27.0, "points_against": 20.0, "games_played": 1},
                 "LV": {"points_for": 20.0, "points_against": 27.0, "games_played": 1}}
    assert nfl_season_stats_available(standings) is True
    eff = build_nfl_efficiency_data(standings, None)
    assert eff["KC"]["net_eff"] > eff["LV"]["net_eff"]


def test_qb_state_crosses_the_offseason_for_an_unplayed_season():
    state = {"qb_rating": {"p1": 100.0, "p2": -100.0}, "qb_starts": {"p1": 50, "p2": 50},
             "team_val": {"AAA": 60.0, "BBB": -60.0}, "last_season": 2025}
    advanced = _state_for_season(state, 2026)
    # league averages here are 0.0 for both maps
    assert advanced["qb_rating"]["p1"] == pytest.approx(100.0 * (1 - VET_REVERT))
    assert advanced["team_val"]["AAA"] == pytest.approx(60.0 * (1 - TEAM_REVERT))
    # original state untouched
    assert state["team_val"]["AAA"] == 60.0


def test_qb_state_unchanged_when_the_season_is_already_played():
    state = {"qb_rating": {"p1": 100.0}, "qb_starts": {"p1": 50},
             "team_val": {"AAA": 60.0}, "last_season": 2026}
    assert _state_for_season(state, 2026)["team_val"]["AAA"] == 60.0
