"""
ELO Model — System 2
Computes ELO ratings from historical data and predicts game outcomes.

FIX (Issue 4): NFL K-factor now decays from k_base (20) to k_base/2 (10)
over the course of each regular season (17 games), mirroring the dynamic
K-factor already used in model/nba_elo.py. Early-season games (high
uncertainty) get full K; late-season games (stable ratings) get reduced K.
K reverts to k_base at the start of each new season.
"""

import numpy as np
import pandas as pd
from math import log, exp


def expected_score(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def mov_multiplier(point_diff, elo_diff):
    return log(abs(point_diff) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))


def mov_multiplier_538(point_diff, winner_elo_diff):
    """
    FiveThirtyEight's margin-of-victory multiplier with the autocorrelation
    adjustment computed from the WINNER's pregame Elo perspective: upsets
    (negative winner diff) inflate the multiplier, chalk wins deflate it.
    Measured better than the abs-diff variant on the walk-forward harness
    (research/RESULTS.md).
    """
    return log(abs(point_diff) + 1) * (2.2 / (winner_elo_diff * 0.001 + 2.2))


def era_hfa(home_win_rates, current_season, default=48.0, window=10):
    """
    Rolling home-field advantage in Elo points from the trailing `window`
    completed seasons' home win rate, per 538's rolling 10-year average
    (~48 pts in the modern era). home_win_rates: {season: home_win_pct}.
    """
    vals = [home_win_rates[s] for s in range(current_season - window, current_season)
            if s in home_win_rates]
    if len(vals) < 3:
        return default
    from math import log10
    p = sum(vals) / len(vals)
    p = min(max(p, 0.5), 0.75)
    return 400.0 * log10(p / (1.0 - p))


def compute_elo(historical_df, k_base=20.0, hfa=48.0, initial_elo=1500.0, regress_pct=0.33,
                qb_map=None, use_era_hfa=True, pregame_out=None, current_season=None):
    """
    Process historical games and compute current ELO ratings.
    Returns dict: {team: elo}
    Also returns recent game history for form calculation.

    2026-09 accuracy pass (research/RESULTS.md, all walk-forward verified):
    - HFA defaults to 48 (modern-era average) and, with use_era_hfa, is recomputed
      per season from the trailing 10 completed seasons' home win rate.
    - K is flat (the within-season K decay was measured to hurt accuracy).
    - Margin-of-victory uses 538's winner-perspective autocorrelation adjustment.
    - qb_map: optional {(date_str, team1, team2): (adj_home, adj_away)} from
      model.qb_model - applied to both prediction expectation and post-game update.
    - current_season: the season the caller is about to PREDICT. historical_df
      holds completed games only, so before Week 1 of a new season the per-season
      loop below stops at the previous season and never applies that season's
      offseason regression. Passing current_season closes the gap (see the block
      after the loop). Omit it to get the raw end-of-history ratings.
    """
    df = historical_df.copy()
    df = df.dropna(subset=["score1", "score2"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    elo_dict = {}
    last_season = {}
    game_history = {}  # {team: [{"result":1/0, "elo_diff":x}, ...]}
    if pregame_out is not None:
        pregame_out.clear()  # {(date_str, team1, team2): (e1_base, e2_base)}

    seasons = sorted(df["season"].unique())

    # Trailing home win rates drive the era-rolling HFA
    home_win_rates = {}
    for s in seasons:
        sdf = df[df["season"] == s]
        played = sdf.dropna(subset=["score1", "score2"])
        if len(played):
            home_win_rates[s] = float((played["score1"] > played["score2"]).mean())

    for season in seasons:
        season_df = df[df["season"] == season]
        season_hfa = era_hfa(home_win_rates, season, default=hfa) if use_era_hfa else hfa

        # Regress ELOs toward mean at start of each season
        for team in list(elo_dict.keys()):
            elo_dict[team] = elo_dict[team] * (1 - regress_pct) + initial_elo * regress_pct
            game_history[team] = []

        for _, row in season_df.iterrows():
            team1 = row["team1"]
            team2 = row["team2"]
            score1 = row["score1"]
            score2 = row["score2"]
            neutral = row.get("neutral", 0)

            if team1 not in elo_dict:
                elo_dict[team1] = initial_elo
                game_history[team1] = []
            if team2 not in elo_dict:
                elo_dict[team2] = initial_elo
                game_history[team2] = []

            e1 = elo_dict[team1]
            e2 = elo_dict[team2]

            # Home field adjustment (era-rolling per season)
            hfa_adj = 0 if neutral else season_hfa

            if pregame_out is not None:
                pregame_out[(str(pd.to_datetime(row["date"]).date()), team1, team2)] = (e1, e2)

            # QB adjustments (538-style), keyed by game date
            qb_adj1 = qb_adj2 = 0.0
            if qb_map:
                date_str = str(pd.to_datetime(row["date"]).date())
                qb_adj1, qb_adj2 = qb_map.get((date_str, team1, team2), (0.0, 0.0))

            adj_e1 = e1 + hfa_adj + qb_adj1
            adj_e2 = e2 + qb_adj2

            exp1 = expected_score(adj_e1, adj_e2)
            exp2 = 1.0 - exp1

            actual1 = 1.0 if score1 > score2 else (0.5 if score1 == score2 else 0.0)
            actual2 = 1.0 - actual1

            point_diff = abs(score1 - score2)

            if point_diff > 0:
                # 538 winner-perspective autocorrelation adjustment
                winner_diff = (adj_e1 - adj_e2) if actual1 == 1.0 else (adj_e2 - adj_e1)
                mov = mov_multiplier_538(point_diff, winner_diff)
            else:
                mov = 1.0

            # Flat K (measured: within-season K decay costs accuracy)
            k = k_base

            elo_dict[team1] = e1 + k * mov * (actual1 - exp1)
            elo_dict[team2] = e2 + k * mov * (actual2 - exp2)

            game_history[team1].append({
                "result": actual1,
                "elo_diff": adj_e1 - e2,
                "date": row["date"].isoformat(),
                "opponent": team2,
            })
            game_history[team2].append({
                "result": actual2,
                "elo_diff": e2 - adj_e1,
                "date": row["date"].isoformat(),
                "opponent": team1,
            })

    # Offseason regression for season(s) that have no completed games yet.
    #
    # The loop above regresses at the start of every season it iterates, but it
    # only iterates seasons PRESENT in historical_df -- and historical_df is
    # completed games only. Before Week 1 of a new season the loop therefore
    # stops at the previous season, and the ratings handed to the live pipeline
    # are raw end-of-last-season values: no mean reversion, so no accounting for
    # roster turnover. That over-disperses the whole league (measured 2026-09-08:
    # rating sd 120 vs 81 regressed) and manufactures large phantom edges against
    # the market. It is a live-only defect -- the walk-forward harness always has
    # the target season's games in frame, so it regresses and never reproduces it.
    #
    # Regress once per season boundary crossed. Self-healing: the moment the new
    # season's first result lands in historical_df the loop owns that boundary
    # and the range below is empty, so ratings are never regressed twice.
    if current_season is not None and seasons:
        for _ in range(max(0, int(current_season) - int(max(seasons)))):
            for team in list(elo_dict.keys()):
                elo_dict[team] = elo_dict[team] * (1 - regress_pct) + initial_elo * regress_pct
                game_history[team] = []

    return elo_dict, game_history


def annotate_pregame_elo(df, k_base=20.0, hfa=48.0, initial_elo=1500.0, regress_pct=0.33,
                         qb_map=None):
    """
    Stamps each completed game row with the two teams' base ELO ratings as they
    stood immediately before that game (elo1_pre/elo2_pre), computed by the exact
    same recurrence as compute_elo() (era-rolling HFA, flat K, winner-perspective
    MOV, optional QB adjustments) so training features stay self-consistent with
    the shipped ratings.
    """
    df = df.copy()
    df = df.dropna(subset=["score1", "score2"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    pregame = {}
    compute_elo(df, k_base=k_base, hfa=hfa, initial_elo=initial_elo,
                regress_pct=regress_pct, qb_map=qb_map, pregame_out=pregame)

    elo1_pre_col, elo2_pre_col = [], []
    for row in df.itertuples():
        key = (str(pd.to_datetime(row.date).date()), row.team1, row.team2)
        e1, e2 = pregame.get(key, (1500.0, 1500.0))
        elo1_pre_col.append(e1)
        elo2_pre_col.append(e2)

    df["elo1_pre"] = elo1_pre_col
    df["elo2_pre"] = elo2_pre_col
    return df


def recent_form(game_history, team, n=5):
    """Win rate over last n games."""
    hist = game_history.get(team, [])
    if not hist:
        return 0.5
    recent = hist[-n:]
    return sum(g["result"] for g in recent) / len(recent)


def head_to_head_modifier(game_history_full, team_a, team_b, n=10):
    """Returns ELO modifier based on head-to-head record."""
    h2h = [g for g in game_history_full.get(team_a, []) if g.get("opponent") == team_b]
    if len(h2h) < 3:
        return 0.0
    h2h = h2h[-n:]
    win_rate = sum(g["result"] for g in h2h) / len(h2h)
    return (win_rate - 0.5) * 20.0


def predict_game(team_a, team_b, elo_dict, game_history,
                 is_home_a=True, neutral=False,
                 rest_adj_a=0.0, rest_adj_b=0.0,
                 injury_adj_a=0.0, injury_adj_b=0.0,
                 hfa=65.0, form_blend=0.3):
    """
    Predict win probability for team_a vs team_b.
    Returns dict with prob, elo values, adjustments.
    """
    base_elo_a = elo_dict.get(team_a, 1500.0)
    base_elo_b = elo_dict.get(team_b, 1500.0)

    # Recent form adjustment
    form_a = recent_form(game_history, team_a)
    form_b = recent_form(game_history, team_b)
    form_adj_a = (form_a - 0.5) * form_blend * 100.0
    form_adj_b = (form_b - 0.5) * form_blend * 100.0

    h2h_a = head_to_head_modifier(game_history, team_a, team_b)
    h2h_b = head_to_head_modifier(game_history, team_b, team_a)

    adj_elo_a = base_elo_a + form_adj_a + rest_adj_a + injury_adj_a + h2h_a
    adj_elo_b = base_elo_b + form_adj_b + rest_adj_b + injury_adj_b + h2h_b

    if not neutral and is_home_a:
        adj_elo_a += hfa
    elif not neutral and not is_home_a:
        adj_elo_b += hfa

    prob_a = expected_score(adj_elo_a, adj_elo_b)

    return {
        "prob": float(prob_a),
        "elo_a": float(base_elo_a),
        "elo_b": float(base_elo_b),
        "adj_elo_a": float(adj_elo_a),
        "adj_elo_b": float(adj_elo_b),
        "form_a": float(form_a),
        "form_b": float(form_b),
        "elo_diff": float(adj_elo_a - adj_elo_b),
        "h2h_adj_a": float(h2h_a),
        "h2h_adj_b": float(h2h_b),
    }


def get_trend(game_history, team, window=5):
    """Returns 'up', 'down', or 'neutral' based on recent results."""
    hist = game_history.get(team, [])
    if len(hist) < window * 2:
        return "neutral"
    recent = sum(g["result"] for g in hist[-window:]) / window
    older = sum(g["result"] for g in hist[-window*2:-window]) / window
    if recent > older + 0.15:
        return "up"
    elif recent < older - 0.15:
        return "down"
    return "neutral"
