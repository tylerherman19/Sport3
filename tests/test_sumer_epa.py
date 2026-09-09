"""Guards for the SumerSports EPA feed + layer.

What has to hold and is not obvious from the call sites:

1. The parser fails LOUDLY on a page shape change or an unmapped team name.
   A silent partial league would feed the model fabricated gaps.
2. boundary_deltas(season S) reads ONLY the completed S-1 table - the leak
   guard is the whole point of the layer.
3. The layer is bounded: cap binds on outlier seasons, and a missing table
   degrades to no layer, never to zeros treated as data.
4. merge_adjustments sums deltas per team-season; both layers land at the
   same boundary point in compute_elo.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import sumer_epa
from scripts.sumersports_feed import parse_team_table, TEAM_NAME_TO_ABBR

FIXTURE_HTML = """
<html><body><div>Last Updated </div><div>09-02-2026</div>
<table><tr><th>Team</th><th>Season</th><th>EPA/Play</th><th>Total EPA</th>
<th>Success %</th><th>EPA/Pass</th><th>EPA/Rush</th><th>Pass Yards</th>
<th>Pass TD</th><th>Rush Yards</th><th>Rush TD</th><th>ADoT</th></tr>
<tr><td>1.Kansas City Chiefs</td><td>2025</td><td>0.14</td><td>152.81</td>
<td>50.55%</td><td>0.22</td><td>0.02</td><td>5,250</td><td>41</td>
<td>1,970</td><td>18</td><td>7.39</td></tr>
<tr><td>2.New York Jets</td><td>2025</td><td>-0.11</td><td>-95.2</td>
<td>41.20%</td><td>-0.08</td><td>-0.02</td><td>3,100</td><td>17</td>
<td>1,500</td><td>9</td><td>6.10</td></tr>
</table></body></html>
"""


def _write_table(cache_dir, season, nets):
    teams = {}
    for team, (off, dfn) in nets.items():
        teams[team] = {"off": {"epa_play": off}, "def": {"epa_play": dfn}}
    (cache_dir / f"team_tables_{season}.json").write_text(json.dumps(
        {"season": season, "teams": teams, "fetched_at": "2026-09-09T00:00:00+00:00"}))


def test_parser_extracts_rows_and_converts_types():
    teams, stamp = parse_team_table(FIXTURE_HTML, expected_season=2025)
    assert stamp == "09-02-2026"
    assert set(teams) == {"KC", "NYJ"}
    kc = teams["KC"]
    assert kc["epa_play"] == pytest.approx(0.14)
    assert kc["success_pct"] == pytest.approx(0.5055)
    assert kc["pass_yards"] == 5250
    assert teams["NYJ"]["epa_pass"] == pytest.approx(-0.08)


def test_parser_rejects_season_mismatch():
    with pytest.raises(ValueError, match="season mismatch"):
        parse_team_table(FIXTURE_HTML, expected_season=2024)


def test_parser_rejects_unmapped_team():
    bad = FIXTURE_HTML.replace("Kansas City Chiefs", "Berlin Thunder")
    with pytest.raises(ValueError, match="unmapped team"):
        parse_team_table(bad, expected_season=2025)


def test_parser_rejects_empty_page():
    with pytest.raises(ValueError, match="no data rows"):
        parse_team_table("<html><body><table></table></body></html>")


def test_all_32_current_names_mapped():
    assert len([v for k, v in TEAM_NAME_TO_ABBR.items()
                if not k.startswith("Washington Football")]) == 32
    assert len(set(TEAM_NAME_TO_ABBR.values())) == 32


def test_boundary_deltas_read_only_prior_season(tmp_path):
    _write_table(tmp_path, 2025, {"KC": (0.10, -0.02), "NYJ": (-0.20, 0.07)})
    deltas = sumer_epa.boundary_deltas(2026, cache_dir=tmp_path,
                                       elo_per_epa=250.0, cap=40.0)
    assert deltas["KC"] == pytest.approx(250.0 * 0.12)
    assert deltas["NYJ"] == pytest.approx(-40.0)  # cap binds
    # No table for 2024 -> season 2025 gets no layer
    assert sumer_epa.boundary_deltas(2025, cache_dir=tmp_path) == {}


def test_missing_table_degrades_to_no_layer(tmp_path):
    assert sumer_epa.boundary_deltas(2026, cache_dir=tmp_path) == {}
    assert sumer_epa.boundary_deltas(2022, cache_dir=tmp_path) == {}


def test_merge_adjustments_sums_per_team():
    base = {2026: {"KC": 10.0, "BUF": -5.0}}
    extra = {2026: {"KC": 7.5, "NYJ": -12.0}, 2025: {"KC": 3.0}}
    merged = sumer_epa.merge_adjustments(base, extra)
    assert merged[2026]["KC"] == pytest.approx(17.5)
    assert merged[2026]["BUF"] == pytest.approx(-5.0)
    assert merged[2026]["NYJ"] == pytest.approx(-12.0)
    assert merged[2025]["KC"] == pytest.approx(3.0)
    # inputs untouched
    assert base == {2026: {"KC": 10.0, "BUF": -5.0}}


def test_current_net_epa_picks_latest(tmp_path):
    _write_table(tmp_path, 2024, {"KC": (0.10, 0.05)})
    _write_table(tmp_path, 2025, {"KC": (0.20, -0.05)})
    season, nets = sumer_epa.current_net_epa(cache_dir=tmp_path)
    assert season == 2025
    assert nets["KC"] == pytest.approx(0.25)


# ----------------------------------------------------------- compute_elo anchor wiring

def _two_season_history():
    import pandas as pd
    rows = []
    for season, fmt in ((2024, "2024-09-%02d"), (2025, "2025-09-%02d")):
        for i in range(1, 9):
            rows.append({"date": fmt % i, "season": season,
                         "team1": "AAA", "team2": "BBB",
                         "score1": 30.0, "score2": 10.0, "neutral": 0})
    return pd.DataFrame(rows)


def test_anchor_weight_zero_is_shipped_behaviour():
    from model.elo_model import compute_elo
    df = _two_season_history()
    plain, _ = compute_elo(df, use_era_hfa=False, current_season=2026)
    anchored, _ = compute_elo(df, use_era_hfa=False, current_season=2026,
                              boundary_anchor={2025: {"AAA": 1700.0, "BBB": 1300.0},
                                               2026: {"AAA": 1700.0, "BBB": 1300.0}},
                              boundary_anchor_weight=0.0)
    for team in plain:
        assert plain[team] == pytest.approx(anchored[team])


def test_anchor_weight_one_lands_on_anchor_at_boundary():
    from model.elo_model import compute_elo
    df = _two_season_history()
    anchored, _ = compute_elo(df, use_era_hfa=False, current_season=2026,
                              boundary_anchor={2025: {"AAA": 1700.0, "BBB": 1300.0}},
                              boundary_anchor_weight=1.0)
    # First game of 2025 is played from the anchor, not the regressed rating.
    # AAA went 8-0 in 2024, so the plain chain would enter 2025 far above 1700
    # after reversion? No - reversion pulls toward 1500; either way the anchor
    # replaces it: pregame of the first 2025 game must be exactly the anchor.
    pregame = {}
    compute_elo(df, use_era_hfa=False, pregame_out=pregame,
                boundary_anchor={2025: {"AAA": 1700.0, "BBB": 1300.0}},
                boundary_anchor_weight=1.0)
    first_2025 = pregame[("2025-09-01", "AAA", "BBB")]
    assert first_2025 == (1700.0, 1300.0)


def test_anchor_blend_is_between_ratings():
    from model.elo_model import compute_elo
    df = _two_season_history()
    plain_pre, blend_pre = {}, {}
    compute_elo(df, use_era_hfa=False, pregame_out=plain_pre)
    compute_elo(df, use_era_hfa=False, pregame_out=blend_pre,
                boundary_anchor={2025: {"AAA": 1700.0, "BBB": 1300.0}},
                boundary_anchor_weight=0.5)
    p = plain_pre[("2025-09-01", "AAA", "BBB")]
    b = blend_pre[("2025-09-01", "AAA", "BBB")]
    assert b[0] == pytest.approx(0.5 * p[0] + 0.5 * 1700.0)
    assert b[1] == pytest.approx(0.5 * p[1] + 0.5 * 1300.0)
