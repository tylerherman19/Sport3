"""
Offseason roster-value layer.

The Elo chain is roster-blind by design: the only thing that happens between the
last game of one season and the first game of the next is mean reversion toward
1500 (see model/elo_model.compute_elo). Reversion is an honest statement that we
know less about a team in September than we did in January, but it carries zero
information about WHAT changed - a team that lost six starters and a team that
returned all of them get the same treatment.

This module supplies the missing term. For each team it measures the net player
value that walked in and out of the building between the end of last season and
the start of this one, and converts that to a bounded Elo delta which
compute_elo() applies at the season boundary, immediately after the reversion.

QUARTERBACKS ARE EXCLUDED FROM THIS LAYER ENTIRELY. QB movement is already owned
by model/qb_model.py, which adjusts each team's effective Elo per game by
QB_MULT x (starter VALUE - team rolling VALUE). Counting a quarterback here as
well would double-count the single largest roster effect in the sport. The
exclusion is enforced in _is_qb() and applied to both sides of every ledger.

## The value metric

Pro-Football-Reference's Approximate Value is the natural unit for "what was this
player worth last season", but PFR serves 403 to automated clients and nflverse
publishes only *career* AV, and only for drafted players. So this module computes
a per-player-season AV proxy ("sAV") from free nflverse feeds, built the way AV
itself is built: allocate a position group's value by playing time, then adjust
skill and defensive players by what they actually produced.

    sAV = STARTER_AV[group] x (TIME_WEIGHT x snap_share + PROD_WEIGHT x prod_factor)

- snap_share: fraction of the team's offensive or defensive snaps the player was
  on the field for across the season (nflverse snap_counts), so a 17-game starter
  scores ~1.0 and a rotational backup ~0.3.
- prod_factor: season production relative to the median full-time starter at the
  same position that season - PPR points for skill positions, a standard IDP line
  for the front seven and secondary. Groups with no box score (offensive line,
  specialists) run on playing time alone with prod_factor pinned to 1.0.
- STARTER_AV: the AV a full-season starter at that group typically earns on PFR's
  scale, so ledger numbers read in familiar units (a good starter ~6, a rotational
  piece ~2, a special-teamer well under 1).

sAV is measured on the season BEFORE the offseason in question, so nothing in the
layer can see the season it is used to predict.

## Rookies

Draft picks are valued off a per-pick expected-AV curve fitted to drafts that were
already over before the season being predicted (DRAFT_CURVE_LAG years back), then
cut to ROOKIE_DISCOUNT of that. Unproven players must not swing a rating: at the
shipped discount the first overall pick is worth roughly a third of a returning
starter, and a late-round pick is worth almost nothing.

## Outputs

build_offseason_ledger() returns the full per-team breakdown - who arrived, who
left, what each was worth - which scripts/main.py persists to
data/offseason_roster_value.json so the site can later answer "why did LV's
rating move". elo_adjustments() reduces that to the {season: {team: delta}} map
compute_elo() consumes.
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "nflverse_cache"
LEDGER_PATH = DATA_DIR / "offseason_roster_value.json"

RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
WEEKLY_ROSTER_URL = RELEASE + "/weekly_rosters/roster_weekly_{season}.parquet"
SEASON_ROSTER_URL = RELEASE + "/rosters/roster_{season}.csv"
SNAP_COUNTS_URL = RELEASE + "/snap_counts/snap_counts_{season}.csv"
PLAYERS_URL = RELEASE + "/players/players.csv"
STATS_URL = RELEASE + "/stats_player/stats_player_week_{season}.csv"
DRAFT_URL = RELEASE + "/draft_picks/draft_picks.csv"

# ---------------------------------------------------------------- tuning knobs

# Quarterbacks are owned by model/qb_model.py and must never be counted here.
QB_POSITIONS = {"QB"}

# AV a full-season starter at each group typically earns, on PFR's scale.
STARTER_AV = {
    "RB": 5.0, "WR": 6.0, "TE": 5.0, "OL": 7.0,
    "DL": 6.0, "LB": 5.5, "DB": 5.5, "ST": 2.0,
}
TIME_WEIGHT = 0.6        # share of a group's value allocated by playing time
PROD_WEIGHT = 0.4        # ...and by production, where a box score exists
PROD_FACTOR_CAP = 2.0    # a career year is worth at most 2x a median starter
ST_BONUS_AV = 0.8        # special-teams snaps are worth something to everyone
MIN_LEDGER_AV = 0.25     # players below this are pooled into "other" in the ledger
LEDGER_TOP_N = 15        # ...as is everyone past the top N movers on each side, so the
                         # persisted file stays a few hundred KB rather than megabytes
LEDGER_DETAIL_SEASONS = 2  # only the most recent seasons keep their player lists on
                           # disk; older seasons keep the numbers. The file ships to
                           # a static site, and nobody is asking why ARI moved in 2016

ROOKIE_DISCOUNT = 0.35   # unproven: keep 35% of the pick's expected per-season AV
DRAFT_CURVE_LAG = 8      # only use drafts this many years old, so career AV is
                         # settled and the curve carries no lookahead
DRAFT_CURVE_WINDOW = 15  # ...over this many drafts
DRAFT_PICK_BUCKET = 8    # picks are bucketed this wide before smoothing

# Elo points per net AV, and the hard cap on one offseason's swing.
#
# ROSTER_ELO_PER_AV is measured on the shipped-path walk-forward harness
# (research/wf_roster.py, 2017-2025, N=2485). Accuracy is flat-to-positive from
# 0.25 to 0.6 and turns negative from 0.75; log loss improves throughout and
# saturates near 0.6. 0.25 and 0.5 are indistinguishable on the backtest
# (log loss 0.6234 vs 0.6231, ~a thousandth of a nat), so the tie is broken on
# the one piece of evidence the backtest structurally cannot provide: Week 1 of a
# season with no games played, which is the case this whole layer exists for. At
# 0.5 the Week 1 2026 board moved AWAY from the market in aggregate
# (6.92 -> 7.04 pp mean gap); at 0.25 it stayed flat (6.95) while still closing
# the MIA@LV gap the layer was built to address. Roster value is a noisy estimate
# and this is its first outing, so the shrunk end of the plateau is the honest
# choice. See research/RESULTS.md for the full sweep.
ROSTER_ELO_PER_AV = 0.25
#
# The cap is a design constraint from the outset: no single offseason may dominate
# a rating. At the shipped scale it does not bind on any of the 352 team-seasons
# from 2015-2026 (largest applied delta -10.1) - it is a guard against a
# pathological ledger rather than an active constraint. That guard is not
# hypothetical: before roster_snapshot() learned to read each team's own first
# populated week, the 2017 Hurricane Irma cancellation left TB and MIA with no
# week-1 roster rows and produced -131 net AV out of nothing.
ROSTER_ELO_CAP = 40.0

# Earliest offseason the layer can be built for: nflverse publishes weekly rosters
# and snap counts from 2013, and the ledger for season S needs season S-1 snaps.
FIRST_LEDGER_SEASON = 2015

# Roster snapshots: the 53-man plus inactives plus injured reserve. Practice squad
# (DEV) and players already cut (CUT) are not roster members.
ACTIVE_STATUSES = {"ACT", "INA", "RES", "EXE"}

# nflverse team codes drift across eras; normalize onto the abbreviations the Elo
# chain uses so a franchise's history survives a relocation.
_ABBR = {"WSH": "WAS", "JAC": "JAX", "LVR": "LV", "LA": "LAR", "OAK": "LV",
         "SD": "LAC", "STL": "LAR", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE",
         "HST": "HOU", "SL": "LAR"}

_POSITION_GROUP = {
    "QB": "QB",
    "RB": "RB", "HB": "RB", "FB": "RB",
    "WR": "WR", "TE": "TE",
    "T": "OL", "G": "OL", "C": "OL", "OL": "OL", "OT": "OL", "OG": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "EDGE": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
    "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB",
    "K": "ST", "P": "ST", "LS": "ST", "PK": "ST",
}

# Groups whose value is allocated by playing time alone - there is no public box
# score for an offensive lineman or a long snapper.
_NO_PRODUCTION_GROUPS = {"OL", "ST"}


def norm_team(t):
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return None
    t = str(t).upper().strip()
    return _ABBR.get(t, t)


def position_group(pos):
    if pos is None or (isinstance(pos, float) and math.isnan(pos)):
        return None
    return _POSITION_GROUP.get(str(pos).upper().strip())


def _is_qb(*positions):
    """True if any of the supplied position labels is a quarterback.

    Called on BOTH the previous-season and current-season position of every player
    in the ledger. A player listed at QB on either side is dropped outright: the
    QB overlay in model/qb_model.py is the only place quarterback movement is
    allowed to move a rating.
    """
    for p in positions:
        if p is None or (isinstance(p, float) and math.isnan(p)):
            continue
        if str(p).upper().strip() in QB_POSITIONS:
            return True
    return False


# ------------------------------------------------------------------- downloads

def _download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return dest
    except Exception as e:
        log.warning(f"roster_value: download failed {url}: {e}")
        return None


def _read(path, **kwargs):
    if path is None:
        return None
    try:
        if str(path).endswith(".parquet"):
            return pd.read_parquet(path, **kwargs)
        return pd.read_csv(path, low_memory=False, **kwargs)
    except Exception as e:
        log.warning(f"roster_value: could not parse {path}: {e}")
        return None


_ROSTER_COLS = ["season", "team", "position", "status", "full_name", "gsis_id",
                "week", "game_type", "years_exp", "entry_year", "rookie_year",
                "draft_number"]


def load_weekly_roster(season):
    """Week-by-week roster for a season, or None if nflverse has no weekly file.

    The weekly release is what makes this layer honest in backtest: it gives a real
    week-1 snapshot, so a player signed in week 10 is not retroactively counted as
    an offseason arrival. The season-level roster file cannot do that - it is a
    single end-of-season row per player.
    """
    path = _download(WEEKLY_ROSTER_URL.format(season=season),
                     CACHE_DIR / f"roster_weekly_{season}.parquet")
    df = _read(path)
    if df is None or df.empty:
        # The parquet is ~18x smaller than the csv and is the preferred source,
        # but fall back to the csv if it is missing or pyarrow is unavailable.
        df = _read(_download(WEEKLY_ROSTER_URL.format(season=season).replace(".parquet", ".csv"),
                             CACHE_DIR / f"roster_weekly_{season}.csv"))
    if df is None or df.empty:
        return None
    keep = [c for c in _ROSTER_COLS if c in df.columns]
    return df[keep].copy()


def load_season_roster(season):
    """Season-level roster - the fallback when no weekly file exists yet, which is
    the normal case for the current season before its Week 1 is played."""
    path = _download(SEASON_ROSTER_URL.format(season=season),
                     CACHE_DIR / f"roster_{season}.csv")
    df = _read(path)
    if df is None or df.empty:
        return None
    keep = [c for c in _ROSTER_COLS if c in df.columns]
    return df[keep].copy()


def roster_snapshot(season, when="start"):
    """{gsis_id: {"team", "position", "entry_year", "draft_number"}} for one team-
    membership snapshot.

    when="start": the week-1 regular-season roster.
    when="end":   the final regular-season week's roster.

    Together those two bracket exactly the offseason: who was here in January,
    who is here in September.
    """
    df = load_weekly_roster(season)
    source = "weekly"
    if df is not None and "week" in df.columns and "game_type" in df.columns:
        reg = df[df["game_type"].astype(str).str.upper() == "REG"].copy()
        if reg.empty:
            reg = df.copy()
        reg["week"] = pd.to_numeric(reg["week"], errors="coerce")
        reg = reg.dropna(subset=["week"])
        if reg.empty:
            df = None
        else:
            # Pick each team's OWN first/last populated week rather than a league-
            # wide week number. Teams do not always have a row in week 1: when
            # Hurricane Irma cancelled TB-MIA in 2017 neither club appeared in the
            # week-1 file at all, and a league-wide filter silently read both
            # rosters as empty, i.e. every 2016 player departed and nobody
            # arrived. That produced the two largest ledger entries in the whole
            # history (-131 and -126 net AV) out of nothing.
            picker = "min" if when == "start" else "max"
            target = reg.groupby("team")["week"].transform(picker)
            df = reg[reg["week"] == target]
    else:
        df = None

    if df is None or df.empty:
        # No weekly file (current season pre-Week-1, or an era nflverse doesn't
        # cover). The season roster is a single snapshot; for "start" that is the
        # best available view and for "end" it is the correct one.
        df = load_season_roster(season)
        source = "season"
    if df is None or df.empty:
        log.warning(f"roster_value: no roster available for {season}")
        return {}

    df = df[df["status"].astype(str).str.upper().isin(ACTIVE_STATUSES)]
    out = {}
    for r in df.itertuples():
        pid = getattr(r, "gsis_id", None)
        if not isinstance(pid, str) or not pid:
            continue
        out[pid] = {
            "team": norm_team(getattr(r, "team", None)),
            "position": getattr(r, "position", None),
            "name": getattr(r, "full_name", None),
            "entry_year": _int_or_none(getattr(r, "entry_year", None)
                                       or getattr(r, "rookie_year", None)),
            "draft_number": _int_or_none(getattr(r, "draft_number", None)),
        }
    log.info(f"roster_value: {season} {when} roster ({source}): {len(out)} players")
    return out


def _int_or_none(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def load_id_bridge():
    """{pfr_player_id: gsis_id}. Snap counts key on PFR ids, rosters on GSIS ids,
    and the roster files' own pfr_id column is ~20-30% empty, so the players
    release is the bridge."""
    path = _download(PLAYERS_URL, CACHE_DIR / "players.csv")
    df = _read(path, usecols=lambda c: c in ("gsis_id", "pfr_id"))
    if df is None or df.empty:
        return {}
    df = df.dropna(subset=["gsis_id", "pfr_id"])
    return dict(zip(df["pfr_id"], df["gsis_id"]))


# --------------------------------------------------------------- value metric

_IDP = {  # a standard individual-defensive-player scoring line
    "def_sacks": 6.0, "def_interceptions": 6.0, "def_fumbles_forced": 3.0,
    "def_tackles_for_loss": 1.5, "def_pass_defended": 1.5,
    "def_tackles_solo": 0.7, "def_tackle_assists": 0.3, "def_tds": 6.0,
}


def _season_production(season):
    """{gsis_id: production points} for the season, PPR for offence and an IDP
    line for defence. Returns an empty dict if the stats file is unavailable."""
    path = _download(STATS_URL.format(season=season),
                     CACHE_DIR / f"stats_player_week_{season}.csv")
    wanted = {"player_id", "season_type", "fantasy_points_ppr", *_IDP}
    df = _read(path, usecols=lambda c: c in wanted)
    if df is None or df.empty or "player_id" not in df.columns:
        return {}
    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.upper() == "REG"]
    off = df.groupby("player_id")["fantasy_points_ppr"].sum() if "fantasy_points_ppr" in df else None
    idp = None
    have = [c for c in _IDP if c in df.columns]
    if have:
        idp = sum(df.groupby("player_id")[c].sum().fillna(0) * _IDP[c] for c in have)
    prod = {}
    for series in (off, idp):
        if series is None:
            continue
        for pid, v in series.items():
            if isinstance(pid, str) and v == v:
                prod[pid] = max(prod.get(pid, 0.0), float(v))
    return prod


def season_player_value(season):
    """{gsis_id: {"sav", "group", "position", "team", "snap_share"}} for one season.

    This is the AV proxy described in the module docstring. Quarterbacks are
    computed like anyone else here but are filtered out by the ledger; keeping
    them in the table costs nothing and makes the numbers inspectable.
    """
    path = _download(SNAP_COUNTS_URL.format(season=season),
                     CACHE_DIR / f"snap_counts_{season}.csv")
    snaps = _read(path)
    if snaps is None or snaps.empty:
        log.warning(f"roster_value: no snap counts for {season}")
        return {}
    snaps = snaps[snaps["game_type"].astype(str).str.upper() == "REG"] \
        if "game_type" in snaps.columns else snaps
    snaps["team"] = snaps["team"].map(norm_team)

    games_per_team = snaps.groupby("team")["game_id"].nunique().to_dict()

    agg = snaps.groupby("pfr_player_id").agg(
        off_pct=("offense_pct", "sum"),
        def_pct=("defense_pct", "sum"),
        st_pct=("st_pct", "sum"),
        off_snaps=("offense_snaps", "sum"),
        def_snaps=("defense_snaps", "sum"),
        team=("team", "last"),
        position=("position", "last"),
        player=("player", "last"),
    ).reset_index()

    bridge = load_id_bridge()
    agg["gsis_id"] = agg["pfr_player_id"].map(bridge)
    agg = agg.dropna(subset=["gsis_id"])

    gp = agg["team"].map(lambda t: games_per_team.get(t, 17)).clip(lower=1)
    # Share of the team's snaps on the player's own side of the ball. A 17-game
    # every-down starter lands at ~1.0.
    agg["snap_share"] = ((agg["off_pct"] + agg["def_pct"]) / gp).clip(0.0, 1.25)
    agg["st_share"] = (agg["st_pct"] / gp).clip(0.0, 1.25)
    agg["group"] = agg["position"].map(position_group)
    agg = agg.dropna(subset=["group"])

    prod = _season_production(season)
    agg["prod"] = agg["gsis_id"].map(lambda p: prod.get(p, 0.0))

    # Production is only meaningful against same-position full-time starters of the
    # same season, so the reference is recomputed per season and per group.
    refs = {}
    for grp, sub in agg.groupby("group"):
        if grp in _NO_PRODUCTION_GROUPS or grp == "QB":
            continue
        starters = sub[sub["snap_share"] >= 0.55]
        ref = float(starters["prod"].median()) if len(starters) >= 5 else 0.0
        refs[grp] = ref if ref > 0 else None

    out = {}
    for r in agg.itertuples():
        base = STARTER_AV.get(r.group, 4.0)
        if r.group == "ST":
            # Kickers, punters and long snappers take no offensive or defensive
            # snaps at all, so their playing time IS their special-teams share.
            sav = base * r.st_share
        elif r.group in _NO_PRODUCTION_GROUPS or refs.get(r.group) is None:
            sav = base * r.snap_share + ST_BONUS_AV * r.st_share
        else:
            prod_factor = min(r.prod / refs[r.group], PROD_FACTOR_CAP)
            sav = base * (TIME_WEIGHT * r.snap_share + PROD_WEIGHT * prod_factor)
            sav += ST_BONUS_AV * r.st_share
        out[r.gsis_id] = {
            "sav": round(float(sav), 3),
            "group": r.group,
            "position": r.position,
            "team": r.team,
            "name": r.player,
            "snap_share": round(float(r.snap_share), 3),
        }
    log.info(f"roster_value: {season} player values for {len(out)} players")
    return out


# ------------------------------------------------------------- draft-pick curve

def _load_draft():
    path = _download(DRAFT_URL, CACHE_DIR / "draft_picks.csv")
    cols = ("season", "round", "pick", "gsis_id", "position",
            "w_av", "car_av", "dr_av", "games")
    return _read(path, usecols=lambda c: c in cols)


def _career_av(df):
    """Career AV per pick. nflverse populates w_av (PFR's weighted career AV) and
    leaves car_av entirely null, so w_av is the primary with the others as
    fallbacks in case the release schema shifts again."""
    out = pd.Series(0.0, index=df.index)
    for col in ("w_av", "car_av", "dr_av"):
        if col in df.columns:
            out = out.where(out > 0, pd.to_numeric(df[col], errors="coerce").fillna(0.0))
    return out.clip(lower=0.0)


_draft_curve_cache = {}


def draft_pick_curve(before_season):
    """Expected AV PER SEASON for a rookie taken at each pick, as a dict
    {pick: av}, fitted only on drafts old enough that the careers are settled.

    Career AV in the nflverse draft file is a snapshot of today, so fitting on
    recent drafts would leak the future into a backtest. DRAFT_CURVE_LAG keeps the
    window strictly behind the season being predicted. The career total is divided
    by the pick's expected career length so the number is comparable to a
    veteran's single-season sAV, and the caller then applies ROOKIE_DISCOUNT.
    """
    if before_season in _draft_curve_cache:
        return _draft_curve_cache[before_season]
    df = _load_draft()
    curve = {}
    if df is not None and not df.empty:
        hi = int(before_season) - DRAFT_CURVE_LAG
        lo = hi - DRAFT_CURVE_WINDOW
        d = df[(df["season"] > lo) & (df["season"] <= hi)].copy()
        d = d[~d["position"].astype(str).str.upper().isin(QB_POSITIONS)]
        d["av"] = _career_av(d)
        d["games"] = pd.to_numeric(d["games"], errors="coerce").fillna(0.0)
        d["pick"] = pd.to_numeric(d["pick"], errors="coerce")
        d = d.dropna(subset=["pick"])
        if len(d) > 200:
            d["bucket"] = (d["pick"] // DRAFT_PICK_BUCKET).astype(int)
            for bucket, sub in d.groupby("bucket"):
                seasons = max(float(sub["games"].mean()) / 17.0, 1.0)
                curve[bucket] = max(float(sub["av"].mean()) / seasons, 0.0)
            # Enforce monotonicity - the raw per-bucket means are noisy at the tail
            # and a later pick must never project above an earlier one.
            running = None
            for bucket in sorted(curve):
                if running is None or curve[bucket] < running:
                    running = curve[bucket]
                curve[bucket] = running
    _draft_curve_cache[before_season] = curve
    return curve


def rookie_value(draft_number, curve):
    """Discounted expected value of a rookie, in the same AV units as a veteran's
    sAV. Undrafted rookies and anything past the curve get the tail value."""
    if not curve:
        return 0.0
    if draft_number is None:
        bucket = max(curve)
    else:
        bucket = min(int(draft_number) // DRAFT_PICK_BUCKET, max(curve))
    return ROOKIE_DISCOUNT * curve.get(bucket, 0.0)


# ------------------------------------------------------------------- the ledger

def build_offseason_ledger(seasons, teams=None):
    """Per-team offseason value breakdown for each season in `seasons`.

    Returns {season: {team: {"net_av", "gained_av", "lost_av", "rookie_av",
                             "gained": [...], "lost": [...], "rookies": [...]}}}

    For season S the window is the end of S-1 to the start of S, values are the
    sAV each player earned in S-1, and quarterbacks are excluded on both sides.
    net_av is centred on the league so it reads as a relative move: roster churn
    is close to zero-sum between teams, and the draft injects value league-wide
    that says nothing about who got better than whom.
    """
    ledger = {}
    for season in sorted(int(s) for s in seasons):
        prev = roster_snapshot(season - 1, "end")
        cur = roster_snapshot(season, "start")
        if not prev or not cur:
            log.warning(f"roster_value: skipping {season} (missing roster snapshot)")
            continue
        values = season_player_value(season - 1)
        curve = draft_pick_curve(season)

        rows = {}

        def _row(team):
            return rows.setdefault(team, {"gained": [], "lost": [], "rookies": [],
                                          "gained_av": 0.0, "lost_av": 0.0,
                                          "rookie_av": 0.0})

        for pid, now in cur.items():
            was = prev.get(pid)
            team = now.get("team")
            if not team:
                continue
            if _is_qb(now.get("position"), (was or {}).get("position"),
                      (values.get(pid) or {}).get("position")):
                continue
            if was and was.get("team") == team:
                continue  # returning player, not an offseason move
            val = values.get(pid)
            if val is not None:
                entry = {"player": now.get("name") or val.get("name"),
                         "position": now.get("position") or val.get("position"),
                         "av": round(float(val["sav"]), 2),
                         "from": (was or {}).get("team")}
                r = _row(team)
                r["gained"].append(entry)
                r["gained_av"] += entry["av"]
            elif now.get("entry_year") == season:
                av = round(rookie_value(now.get("draft_number"), curve), 2)
                if av <= 0:
                    continue
                entry = {"player": now.get("name"), "position": now.get("position"),
                         "av": av, "pick": now.get("draft_number")}
                r = _row(team)
                r["rookies"].append(entry)
                r["rookie_av"] += av
            # else: a player with no prior-season snaps and no draft year - a
            # returnee from injury or the practice squad. Worth 0 by construction.

        for pid, was in prev.items():
            team = was.get("team")
            if not team:
                continue
            now = cur.get(pid)
            if now and now.get("team") == team:
                continue
            if _is_qb(was.get("position"), (now or {}).get("position"),
                      (values.get(pid) or {}).get("position")):
                continue
            val = values.get(pid)
            if val is None:
                continue
            entry = {"player": was.get("name") or val.get("name"),
                     "position": was.get("position") or val.get("position"),
                     "av": round(float(val["sav"]), 2),
                     "to": (now or {}).get("team")}
            r = _row(team)
            r["lost"].append(entry)
            r["lost_av"] += entry["av"]

        if teams:
            rows = {t: v for t, v in rows.items() if t in teams}
        if not rows:
            continue

        for team, r in rows.items():
            r["gained_av"] = round(r["gained_av"], 2)
            r["lost_av"] = round(r["lost_av"], 2)
            r["rookie_av"] = round(r["rookie_av"], 2)
            r["raw_net_av"] = round(r["gained_av"] + r["rookie_av"] - r["lost_av"], 2)
            for key in ("gained", "lost", "rookies"):
                ordered = sorted(r[key], key=lambda e: -e["av"])
                shown = [e for e in ordered[:LEDGER_TOP_N] if e["av"] >= MIN_LEDGER_AV]
                rest = ordered[len(shown):]
                r[key] = shown
                if rest:
                    r[f"{key}_other_count"] = len(rest)
                    r[f"{key}_other_av"] = round(sum(e["av"] for e in rest), 2)

        mean_net = float(np.mean([r["raw_net_av"] for r in rows.values()]))
        for r in rows.values():
            r["league_mean_net_av"] = round(mean_net, 2)
            r["net_av"] = round(r["raw_net_av"] - mean_net, 2)

        ledger[season] = rows
        log.info(f"roster_value: {season} ledger for {len(rows)} teams "
                 f"(net_av range {min(r['net_av'] for r in rows.values()):.1f} "
                 f"to {max(r['net_av'] for r in rows.values()):.1f})")
    return ledger


def elo_adjustments(ledger, elo_per_av=ROSTER_ELO_PER_AV, cap=ROSTER_ELO_CAP):
    """{season: {team: elo_delta}} for model.elo_model.compute_elo.

    The cap is what keeps one offseason from dominating a rating: no team can move
    more than `cap` Elo points on roster value alone, whatever the ledger says.
    """
    out = {}
    for season, rows in (ledger or {}).items():
        out[int(season)] = {
            team: round(float(np.clip(r["net_av"] * elo_per_av, -cap, cap)), 2)
            for team, r in rows.items()
        }
    return out


def annotate_ledger_with_elo(ledger, elo_per_av=ROSTER_ELO_PER_AV, cap=ROSTER_ELO_CAP):
    """Stamp each team's applied Elo delta into the ledger, so the persisted file
    explains the rating move rather than only the value that drove it."""
    adj = elo_adjustments(ledger, elo_per_av=elo_per_av, cap=cap)
    for season, rows in (ledger or {}).items():
        for team, r in rows.items():
            r["elo_adj"] = adj[int(season)][team]
            r["elo_capped"] = abs(r["net_av"] * elo_per_av) > cap + 1e-9
    return ledger


def write_ledger(ledger, path=LEDGER_PATH, elo_per_av=ROSTER_ELO_PER_AV,
                 cap=ROSTER_ELO_CAP, generated_at=None):
    """Persist the ledger as data/offseason_roster_value.json.

    This is data, not UI: it exists so a later change can show "why did LV's rating
    move" without recomputing anything.
    """
    annotate_ledger_with_elo(ledger, elo_per_av=elo_per_av, cap=cap)
    seasons = sorted((ledger or {}).keys())
    detailed = set(seasons[-LEDGER_DETAIL_SEASONS:])
    out = {}
    for season in seasons:
        rows = {}
        for team, r in ledger[season].items():
            if season in detailed:
                rows[team] = r
            else:
                rows[team] = {k: v for k, v in r.items() if not isinstance(v, list)}
        out[season] = rows
    payload = {
        "generated_at": generated_at,
        "metric": "sAV - snap-weighted Approximate Value proxy (see model/roster_value.py)",
        "quarterbacks_excluded": True,
        "elo_per_av": elo_per_av,
        "elo_cap": cap,
        "rookie_discount": ROOKIE_DISCOUNT,
        "detail_seasons": sorted(str(s) for s in detailed),
        "seasons": {str(s): rows for s, rows in out.items()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    log.info(f"roster_value: wrote {path}")
    return path
