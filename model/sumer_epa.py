"""
SumerSports EPA layer — season-boundary team-strength anchor from play-level EPA.

Source: scripts/sumersports_feed.py caches the public SumerSports team tables
(offensive + defensive EPA/Play family) to data/sumersports_cache/. Coverage is
2022 onward (verified 2026-09-09: 2021 and earlier render empty pages), and the
current-season table refreshes weekly in-season (observed site stamps).

What the layer does: at each season boundary, right after the Elo mean
reversion, nudge every team by

    delta = clip(elo_per_epa * net_epa_prior_season, -cap, +cap)

where net_epa = off EPA/Play - def EPA/Play allowed. Net EPA is league-zero-sum
by construction (every play is someone's offense vs someone's defense), so the
deltas are self-centering and cannot drift the whole league.

Why a separate layer instead of folding into roster_value: roster_value answers
"what changed" (value in minus value out); this answers "how good was the team
on a per-play basis last season". The 33% mean reversion keeps two thirds of a
W-L-driven rating; EPA is the play-level measure of the same latent strength,
so the nudge replaces noise with signal at exactly the moment the chain is most
uncertain (Week 1). It is deliberately small and capped, same discipline as the
roster layer.

Known overlap, kept honest: net EPA includes quarterback play, and QB value is
owned by model.qb_model's overlay (roster_value excludes QBs for this reason).
A team that lost its QB keeps some credit here until games prove otherwise. The
mitigation is size, not surgery: the cap keeps the layer well inside the QB
overlay's range, and the walk-forward harness (research/wf_sumer.py) measures
the net effect with the QB overlay ACTIVE, so a harmful overlap shows up as a
negative delta in the scored window instead of hiding.

Leakage rules (hard):
  * The delta for season S reads ONLY the completed S-1 table.
  * Full-season tables are never used to predict games inside that same
    season's walk-forward. The in-season weekly table feeds the LIVE pipeline
    only, where "games so far" is genuinely pre-game information.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "sumersports_cache"

FIRST_TABLE_SEASON = 2022  # SumerSports serves 2022+ (verified 2026-09-09)

# Shipped scale, set from the walk-forward sweep in research/RESULTS.md.
# elo_per_epa=250 maps the observed net-EPA spread (sd ~0.11) to ~28 Elo points
# of prior-season signal; the cap keeps single-season outliers (|net| > 0.16)
# from overwhelming the chain. Never raise these without re-running the sweep.
DEFAULT_ELO_PER_EPA = 250.0
DEFAULT_CAP = 40.0
# Anchor family: 700 matches the regressed chain's cross-season spread
# (regressed rating sd ~81; net EPA sd ~0.11), 120 keeps one wild season
# from owning a team's whole rating.
DEFAULT_ANCHOR_ELO_PER_EPA = 700.0
DEFAULT_ANCHOR_CAP = 120.0


def load_season_table(season, cache_dir=None):
    """Cached SumerSports table for one season, or None if not fetched/absent."""
    path = (cache_dir or CACHE_DIR) / f"team_tables_{int(season)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def net_epa(table):
    """{team: off EPA/play - def EPA/play allowed} from one season table."""
    out = {}
    for team, sides in (table or {}).get("teams", {}).items():
        off = (sides.get("off") or {}).get("epa_play")
        dfn = (sides.get("def") or {}).get("epa_play")
        if off is None or dfn is None:
            continue
        out[team] = float(off) - float(dfn)
    return out


def boundary_deltas(season, cache_dir=None, elo_per_epa=DEFAULT_ELO_PER_EPA,
                    cap=DEFAULT_CAP):
    """{team: elo_delta} to apply at the boundary INTO `season`, built solely
    from the completed `season - 1` table. Empty when that table is missing -
    a missing feed must degrade to no layer, never to zeros treated as data.
    """
    table = load_season_table(int(season) - 1, cache_dir=cache_dir)
    if table is None:
        return {}
    deltas = {}
    for team, net in net_epa(table).items():
        deltas[team] = float(max(-cap, min(cap, elo_per_epa * net)))
    return deltas


def boundary_deltas_for_seasons(seasons, cache_dir=None,
                                elo_per_epa=DEFAULT_ELO_PER_EPA, cap=DEFAULT_CAP):
    """{season: {team: delta}} for every requested season; seasons without a
    prior table (2022 itself, or anything before the feed's coverage) simply
    get no entry, matching the roster layer's partial-ledger behaviour."""
    return {int(s): boundary_deltas(s, cache_dir=cache_dir,
                                    elo_per_epa=elo_per_epa, cap=cap)
            for s in seasons}


def anchor_ratings(season, cache_dir=None, elo_per_epa=DEFAULT_ELO_PER_EPA,
                   cap=DEFAULT_ANCHOR_CAP):
    """{team: rating} boundary anchor for season S from the completed S-1 table:
    1500 + clip(elo_per_epa * net_epa, +/-cap). Used by compute_elo's
    boundary_anchor blend - the "EPA as primary cross-season signal" mechanism,
    as opposed to boundary_deltas' "EPA as extra credit". Wider cap than the
    delta layer: an anchor is supposed to carry the spread.
    """
    table = load_season_table(int(season) - 1, cache_dir=cache_dir)
    if table is None:
        return {}
    return {team: 1500.0 + float(max(-cap, min(cap, elo_per_epa * net)))
            for team, net in net_epa(table).items()}


def anchor_ratings_for_seasons(seasons, cache_dir=None,
                               elo_per_epa=DEFAULT_ELO_PER_EPA,
                               cap=DEFAULT_ANCHOR_CAP):
    return {int(s): anchor_ratings(s, cache_dir=cache_dir,
                                   elo_per_epa=elo_per_epa, cap=cap)
            for s in seasons}


def merge_adjustments(base, extra):
    """Sum two {season: {team: delta}} maps (e.g. roster_value + this layer)
    into one map for compute_elo's roster_adjustments hook. Both layers are
    applied at the same boundary point, so summing is the honest composition.
    """
    merged = {int(s): dict(d) for s, d in (base or {}).items()}
    for season, deltas in (extra or {}).items():
        slot = merged.setdefault(int(season), {})
        for team, delta in deltas.items():
            slot[team] = slot.get(team, 0.0) + float(delta)
    return merged


def current_net_epa(cache_dir=None):
    """Most recent season's net EPA table, for display/driver text."""
    cache = cache_dir or CACHE_DIR
    seasons = sorted(int(p.stem.split("_")[-1])
                     for p in cache.glob("team_tables_*.json"))
    for season in reversed(seasons):
        table = load_season_table(season, cache_dir=cache)
        if table:
            return season, net_epa(table)
    return None, {}
