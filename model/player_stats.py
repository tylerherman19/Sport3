"""
Player Stats Model — System 13 input.

Turns nflverse per-player weekly stat lines into team-level signals:
  - rolling offensive EPA/game (trailing form, all skill positions combined)
  - rolling defensive EPA/game allowed (opponent's offensive output against them)
  - starting QB identification + rolling EPA/play
  - player-vs-specific-opponent career splits ("how do they do against this team")

These feed generate_nfl_prediction_drivers() for the plain-English explanation
and player_form_prob() below, which is blended into the ensemble as a
standalone signal (weights.player_form in model/ensemble_model.py).
"""

import numpy as np
import pandas as pd


def _team_game_epa(player_week_df, team_col, n=8):
    """
    Per-(season, week) summed EPA (passing+rushing+receiving) for rows grouped
    by `team_col` — used for both a team's own offensive output (team_col="team")
    and what it allowed on defense (team_col="opponent_team").
    """
    epa_cols = [c for c in ("passing_epa", "rushing_epa", "receiving_epa") if c in player_week_df.columns]
    grouped = player_week_df.groupby([team_col, "season", "week"])[epa_cols].sum(min_count=1).fillna(0.0)
    grouped["total_epa"] = grouped[epa_cols].sum(axis=1)
    return grouped["total_epa"]


def build_team_epa_signals(player_week_df, teams, n=8):
    """
    Returns {team: {"off_epa_l{n}": float, "def_epa_allowed_l{n}": float}} for every
    team in `teams`, using each team's last n games of player-week data on file.

    Both signals are opponent-strength adjusted: a game's raw EPA is shifted by how
    much tougher/easier that specific opponent was than league average, so beating
    up on a bad defense no longer looks identical to the same output against an
    elite one. Without this, a team's rolling offensive EPA is really just "how
    good was my recent schedule," not "how good is this offense" — the single
    biggest known gap in this signal (unadjusted rolling EPA has no strength-of-
    schedule correction).
    """
    if player_week_df is None or player_week_df.empty:
        return {t: {f"off_epa_l{n}": 0.0, f"def_epa_allowed_l{n}": 0.0} for t in teams}

    off_series = _team_game_epa(player_week_df, "team", n)
    def_series = _team_game_epa(player_week_df, "opponent_team", n)

    # Opponent-strength priors: each team's average offensive / defensive EPA
    # across all games on file, used as a fixed "how tough is this opponent"
    # reference. Not a true iterative SOS (a la Massey/SRS) — just a single-pass
    # adjustment — but it's a live-inference signal, not a trained/backtested
    # one, so there's no lookahead-leakage concern in using the fullest data on hand.
    off_strength = off_series.groupby(level=0).mean()
    def_strength = def_series.groupby(level=0).mean()
    league_avg_off = float(off_strength.mean()) if len(off_strength) else 0.0
    league_avg_def = float(def_strength.mean()) if len(def_strength) else 0.0

    # For a given team's offensive game log, who was the opponent (whose defense
    # they faced)? And symmetrically for the defensive game log, who was the
    # offense they faced?
    opp_for_offense = player_week_df.groupby(["team", "season", "week"])["opponent_team"].first()
    opp_for_defense = player_week_df.groupby(["opponent_team", "season", "week"])["team"].first()

    signals = {}
    for t in teams:
        if t in off_series.index.get_level_values(0):
            off_recent = off_series.loc[t].sort_index().tail(n)
            adj = []
            for (season, week), raw_epa in off_recent.items():
                opp = opp_for_offense.get((t, season, week))
                opp_def = def_strength.get(opp, league_avg_def) if opp is not None else league_avg_def
                adj.append(raw_epa - (opp_def - league_avg_def))
            off_adj_avg = float(np.mean(adj)) if adj else 0.0
        else:
            off_adj_avg = 0.0

        if t in def_series.index.get_level_values(0):
            def_recent = def_series.loc[t].sort_index().tail(n)
            adj = []
            for (season, week), raw_epa in def_recent.items():
                opp = opp_for_defense.get((t, season, week))
                opp_off = off_strength.get(opp, league_avg_off) if opp is not None else league_avg_off
                adj.append(raw_epa - (opp_off - league_avg_off))
            def_adj_avg = float(np.mean(adj)) if adj else 0.0
        else:
            def_adj_avg = 0.0

        signals[t] = {
            f"off_epa_l{n}": off_adj_avg,
            f"def_epa_allowed_l{n}": def_adj_avg,
        }
    return signals


def identify_starting_qb(player_week_df, team):
    """Most recent game's leading passer (by attempts) for `team` — proxy for current QB1."""
    if player_week_df is None or player_week_df.empty:
        return None
    sub = player_week_df[(player_week_df["team"] == team) & (player_week_df["position"] == "QB")]
    if sub.empty:
        return None
    latest_sw = sub[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).iloc[-1]
    latest_game = sub[(sub["season"] == latest_sw["season"]) & (sub["week"] == latest_sw["week"])]
    if latest_game.empty or latest_game["attempts"].sum() == 0:
        return None
    starter = latest_game.loc[latest_game["attempts"].idxmax()]
    return {"player_id": starter["player_id"], "player_name": starter["player_display_name"]}


def _pass_defense_strength(player_week_df):
    """
    {team: avg passing EPA allowed per game} — how generous a pass defense has
    been, on average, across the games on file. Used to opponent-adjust QB EPA.
    """
    sub = player_week_df[player_week_df["attempts"] >= 5]
    per_game = sub.groupby(["opponent_team", "season", "week"])["passing_epa"].sum(min_count=1).fillna(0.0)
    return per_game.groupby(level=0).mean()


def qb_epa_per_play(player_week_df, player_id, n=8):
    """
    Trailing passing EPA per dropback for a specific player (min 5 attempts/game),
    opponent-adjusted for each game's pass-defense quality — same rationale as the
    team-level SOS adjustment in build_team_epa_signals().
    """
    if player_week_df is None or player_week_df.empty or not player_id:
        return 0.0
    sub = player_week_df[(player_week_df["player_id"] == player_id) & (player_week_df["attempts"] >= 5)]
    sub = sub.sort_values(["season", "week"]).tail(n)
    if sub.empty or sub["attempts"].sum() == 0:
        return 0.0

    pass_def_strength = _pass_defense_strength(player_week_df)
    league_avg_pass_epa = float(pass_def_strength.mean()) if len(pass_def_strength) else 0.0

    # Both passing_epa (per game) and pass_def_strength (per game) are total-EPA
    # units, so the adjustment is a straight per-game offset, same as the
    # team-level version — no extra scaling needed.
    adj_epa_total = 0.0
    for _, row in sub.iterrows():
        opp_strength = pass_def_strength.get(row["opponent_team"], league_avg_pass_epa)
        adj_epa_total += row["passing_epa"] - (opp_strength - league_avg_pass_epa)
    return float(adj_epa_total / sub["attempts"].sum())


def player_vs_opponent(player_week_df, player_id, opponent_team):
    """
    Career averages for a specific player against a specific opponent —
    "how do they do against this team" answered directly from game logs.
    Returns None if they've never played that opponent in the data on file.
    """
    if player_week_df is None or player_week_df.empty or not player_id:
        return None
    sub = player_week_df[
        (player_week_df["player_id"] == player_id) &
        (player_week_df["opponent_team"] == opponent_team)
    ]
    if sub.empty:
        return None
    out = {"games": int(len(sub))}
    if sub["attempts"].sum() > 0:
        out["avg_passing_yards"]  = round(float(sub["passing_yards"].mean()), 1)
        out["avg_passing_tds"]    = round(float(sub["passing_tds"].mean()), 2)
        out["avg_interceptions"]  = round(float(sub["passing_interceptions"].mean()), 2)
        out["avg_passing_epa"]    = round(float(sub["passing_epa"].mean()), 2)
    return out


def player_form_prob(home_off_epa, home_def_epa_allowed, away_off_epa, away_def_epa_allowed,
                      home_qb_epa_pp=0.0, away_qb_epa_pp=0.0, scale=9.0):
    """
    Rule-based win probability from rolling player/team EPA form — home offense
    vs away defense, away offense vs home defense, plus a QB efficiency tilt.
    A lightweight complement to the score-differential-based ELO/Pythagorean/
    efficiency systems, squashed through a logistic to keep it in [0, 1].
    """
    net = (home_off_epa - away_def_epa_allowed) - (away_off_epa - home_def_epa_allowed)
    net += 6.0 * (home_qb_epa_pp - away_qb_epa_pp)
    return float(1.0 / (1.0 + np.exp(-net / scale)))
