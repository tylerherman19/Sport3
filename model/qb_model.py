"""
QB Adjustment Model - 538-style quarterback Elo overlay.

Implements the system described in FiveThirtyEight's "How Our NFL Predictions Work":
every QB gets a rolling VALUE rating from box-score performance (defense-adjusted),
teams get a slower rolling VALUE rating, and a team's effective Elo is adjusted
before each game by QB_MULT x (starter VALUE - team rolling VALUE).

Measured walk-forward (2017-2025, research/RESULTS.md): QB_MULT = 2.0 tested best
(538's own 3.3 ran hot in this implementation). Ratings are recomputed from history
each run - no persisted state needed, all inputs are free nflverse downloads.
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "nflverse_cache"

PLAYER_WEEK_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
DRAFT_URL = "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv"
DEPTH_CHART_URL = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{season}.csv"

QB_MULT = 2.0            # VALUE -> Elo conversion; measured best (research/RESULTS.md)
QB_ROLL = 0.1            # individual QB rolling weight per game (538: update every 10 games)
TEAM_ROLL = 0.05         # team rolling weight per game (538: every 20 games)
DEF_ROLL = 0.1           # defensive VALUE-allowed rolling weight
VET_REVERT = 0.25        # preseason reversion for veterans with 10-100 career starts

# Rookie initial VALUE by draft round (Elo points / 3.3, from 538's published scale)
DRAFT_INIT_ELO = {1: 113.0, 2: 40.0, 3: 25.0}

STATS_COLS = ["player_id", "player_display_name", "season", "week", "season_type",
              "game_id", "team", "opponent_team", "completions", "attempts",
              "passing_yards", "passing_tds", "passing_interceptions",
              "sacks_suffered", "carries", "rushing_yards", "rushing_tds"]


def _download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return dest
    except Exception as e:
        log.warning(f"QB model: download failed {url}: {e}")
        return None


def _game_value(frame):
    """538 VALUE formula per player-game."""
    return (-2.2 * frame["attempts"].fillna(0) + 3.7 * frame["completions"].fillna(0)
            + frame["passing_yards"].fillna(0) / 5 + 11.3 * frame["passing_tds"].fillna(0)
            - 14.1 * frame["passing_interceptions"].fillna(0) - 8 * frame["sacks_suffered"].fillna(0)
            - 1.1 * frame["carries"].fillna(0) + 0.6 * frame["rushing_yards"].fillna(0)
            + 15.9 * frame["rushing_tds"].fillna(0))


def load_qb_stats(first_season, last_season):
    """All QB player-week rows across seasons, with game VALUE attached."""
    frames = []
    for y in range(first_season, last_season + 1):
        path = _download(PLAYER_WEEK_URL.format(season=y), CACHE_DIR / f"stats_player_week_{y}.csv")
        if not path:
            continue
        try:
            f = pd.read_csv(path, usecols=lambda c: c in STATS_COLS, low_memory=False)
            frames.append(f)
        except Exception as e:
            log.warning(f"QB model: could not parse {path}: {e}")
    if not frames:
        return pd.DataFrame()
    qb = pd.concat(frames, ignore_index=True)
    qb = qb[qb["attempts"].fillna(0) + qb["carries"].fillna(0) > 0]
    qb["value"] = _game_value(qb)
    return qb


def load_draft_map():
    path = _download(DRAFT_URL, CACHE_DIR / "draft_picks.csv")
    if not path:
        return {}
    d = pd.read_csv(path, usecols=["season", "round", "pick", "gsis_id"])
    return {r.gsis_id: (int(r.season), int(r.round)) for r in d.itertuples() if isinstance(r.gsis_id, str)}


def _draft_init(draft_map, player_id, season):
    info = draft_map.get(player_id)
    if not info:
        return 0.0
    dseason, rnd = info
    if season > dseason + 3:
        return 0.0
    return DRAFT_INIT_ELO.get(rnd, 0.0) / 3.3


def _prep_starters(qb):
    """Per (game_id, team): starter (most attempts) + team total QB VALUE + opponent."""
    g = qb.sort_values("attempts", ascending=False).groupby(["game_id", "team"])
    starter = g.first()[["player_id", "player_display_name", "value"]].reset_index()
    teamval = qb.groupby(["game_id", "team"])["value"].sum().reset_index().rename(columns={"value": "team_value"})
    opp = qb.groupby(["game_id", "team"])["opponent_team"].first().reset_index()
    return starter.merge(teamval, on=["game_id", "team"]).merge(opp, on=["game_id", "team"])


def build_qb_context(games_df, first_season=2014):
    """
    Walk every completed game in games_df (columns: date, season, team1, team2, score1, score2, game_id)
    in order, maintaining rolling QB state. Returns:
      game_adj: {(date_str, team1, team2): (adj_home_elo, adj_away_elo)} - pre-game adjustments
      state:    {team_val, qb_rating, last_starter, last_starter_name} - end-of-history state
                used to project adjustments for future games
    """
    empty = {"game_adj": {}, "state": {"team_val": {}, "qb_rating": {}, "last_starter": {}, "last_starter_name": {}}}
    if games_df is None or games_df.empty:
        return empty
    seasons = games_df["season"].dropna().astype(int)
    if seasons.empty:
        return empty
    qb = load_qb_stats(min(first_season, int(seasons.min())), int(seasons.max()))
    if qb.empty:
        log.warning("QB model: no weekly stats available; adjustments disabled")
        return empty
    draft_map = load_draft_map()
    starter = _prep_starters(qb)
    skey = {(r.game_id, r.team): r for r in starter.itertuples()}

    qb_rating, qb_starts, team_val, def_allowed = {}, {}, {}, {}
    league_allowed = []
    game_adj = {}
    cur_season = None

    df = games_df.dropna(subset=["score1", "score2"]).sort_values("date")
    for g in df.itertuples():
        if cur_season != g.season:
            avg_qb = float(np.mean(list(qb_rating.values()))) if qb_rating else 0.0
            for p in list(qb_rating):
                if 10 <= qb_starts.get(p, 0) <= 100:
                    qb_rating[p] = qb_rating[p] * (1 - VET_REVERT) + avg_qb * VET_REVERT
            cur_season = g.season
        gid = getattr(g, "game_id", None)
        if gid is None or (isinstance(gid, float) and math.isnan(gid)):
            continue
        r1 = skey.get((gid, g.team1)); r2 = skey.get((gid, g.team2))
        if r1 is None or r2 is None:
            continue
        avg_qb = float(np.mean(list(qb_rating.values()))) if qb_rating else 0.0
        q1 = qb_rating.get(r1.player_id, _draft_init(draft_map, r1.player_id, g.season))
        q2 = qb_rating.get(r2.player_id, _draft_init(draft_map, r2.player_id, g.season))
        adj1 = QB_MULT * (q1 - team_val.get(g.team1, 0.0))
        adj2 = QB_MULT * (q2 - team_val.get(g.team2, 0.0))
        date_str = str(pd.to_datetime(g.date).date())
        game_adj[(date_str, g.team1, g.team2)] = (adj1, adj2)
        # post-game updates
        lg_avg = float(np.mean(league_allowed[-500:])) if league_allowed else 0.0
        for team, srow, opp in ((g.team1, r1, g.team2), (g.team2, r2, g.team1)):
            opp_allow = def_allowed.get(opp, lg_avg)
            team_val[team] = (1 - TEAM_ROLL) * team_val.get(team, 0.0) + TEAM_ROLL * (srow.team_value - (opp_allow - lg_avg))
            def_allowed[team] = (1 - DEF_ROLL) * def_allowed.get(team, 0.0) + DEF_ROLL * srow.team_value
            league_allowed.append(srow.team_value)
            pid = srow.player_id
            qv = srow.value - (opp_allow - lg_avg)
            qb_rating[pid] = (1 - QB_ROLL) * qb_rating.get(pid, _draft_init(draft_map, pid, g.season)) + QB_ROLL * qv
            qb_starts[pid] = qb_starts.get(pid, 0) + 1

    # last known starter per team (most recent game row in starter table)
    last_starter, last_starter_name = {}, {}
    for r in starter.itertuples():
        last_starter[r.team] = r.player_id
        last_starter_name[r.team] = r.player_display_name

    return {"game_adj": game_adj,
            "state": {"team_val": team_val, "qb_rating": qb_rating,
                      "last_starter": last_starter, "last_starter_name": last_starter_name,
                      "draft_map": draft_map}}


def projected_adjustment(state, team, season, depth_chart_qb=None):
    """
    Pre-game Elo adjustment for a FUTURE game: QB_MULT x (projected starter VALUE - team rolling VALUE).
    depth_chart_qb: optional {team: (player_id, name)} from the current season's depth chart;
    falls back to the team's last known starter. Unrated QBs get their draft-based initial value.
    """
    team_val = state.get("team_val", {})
    qb_rating = state.get("qb_rating", {})
    draft_map = state.get("draft_map", {})
    pid = None
    if depth_chart_qb and team in depth_chart_qb:
        pid = depth_chart_qb[team][0]
    if pid is None:
        pid = state.get("last_starter", {}).get(team)
    if pid is None:
        return 0.0
    avg_qb = float(np.mean(list(qb_rating.values()))) if qb_rating else 0.0
    q = qb_rating.get(pid, _draft_init(draft_map, pid, season))
    return QB_MULT * (q - team_val.get(team, 0.0))


def load_depth_chart_qbs(season):
    """Current-season QB1 per team from nflverse depth charts: {team: (player_id, name)}."""
    path = _download(DEPTH_CHART_URL.format(season=season), CACHE_DIR / f"depth_charts_{season}.csv")
    if not path:
        return {}
    try:
        d = pd.read_csv(path, low_memory=False)
        # nflverse schema drift: older files use position/depth_team/week,
        # current files use pos_abb/pos_rank/dt. Handle both.
        if "pos_abb" in d.columns:
            pos_col, rank_col = "pos_abb", "pos_rank"
        else:
            pos_col, rank_col = "position", "depth_team"
        d = d[(d[pos_col] == "QB") & (d[rank_col] == 1)]
        sort_col = "dt" if "dt" in d.columns else ("week" if "week" in d.columns else None)
        if sort_col:
            d = d.sort_values(sort_col)
        out = {}
        for r in d.itertuples():
            if "player_name" in d.columns:
                name = r.player_name
            else:
                name = r.first_name + " " + r.last_name
            out[r.team] = (r.gsis_id, name)
        return out
    except Exception as e:
        log.warning(f"QB model: depth chart parse failed: {e}")
        return {}
