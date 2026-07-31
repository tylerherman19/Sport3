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
    """
    if player_week_df is None or player_week_df.empty:
        return {t: {f"off_epa_l{n}": 0.0, f"def_epa_allowed_l{n}": 0.0} for t in teams}

    off_series = _team_game_epa(player_week_df, "team", n)
    def_series = _team_game_epa(player_week_df, "opponent_team", n)

    signals = {}
    for t in teams:
        off_vals = off_series.loc[t] if t in off_series.index.get_level_values(0) else pd.Series(dtype=float)
        def_vals = def_series.loc[t] if t in def_series.index.get_level_values(0) else pd.Series(dtype=float)
        off_recent = off_vals.sort_index().tail(n)
        def_recent = def_vals.sort_index().tail(n)
        signals[t] = {
            f"off_epa_l{n}": float(off_recent.mean()) if len(off_recent) else 0.0,
            f"def_epa_allowed_l{n}": float(def_recent.mean()) if len(def_recent) else 0.0,
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


def qb_epa_per_play(player_week_df, player_id, n=8):
    """Trailing passing EPA per dropback for a specific player (min 5 attempts/game)."""
    if player_week_df is None or player_week_df.empty or not player_id:
        return 0.0
    sub = player_week_df[(player_week_df["player_id"] == player_id) & (player_week_df["attempts"] >= 5)]
    sub = sub.sort_values(["season", "week"]).tail(n)
    if sub.empty or sub["attempts"].sum() == 0:
        return 0.0
    return float(sub["passing_epa"].sum() / sub["attempts"].sum())


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
