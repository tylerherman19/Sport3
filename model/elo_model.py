"""
ELO Model — System 2
Computes ELO ratings from historical data and predicts game outcomes.
"""

import numpy as np
import pandas as pd
from math import log, exp


def expected_score(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def mov_multiplier(point_diff, elo_diff):
    """Margin of Victory multiplier for ELO K-factor scaling.
    Uses abs(elo_diff) to prevent sign issues; capped at 1.5 per 538 convention
    to prevent blowouts from over-updating ratings.
    """
    raw = log(abs(point_diff) + 1) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))
    return min(raw, 1.5)


def compute_elo(historical_df, k_base=20.0, hfa=65.0, initial_elo=1500.0, regress_pct=0.33):
    """
    Process FiveThirtyEight CSV and compute current ELO ratings.
    Returns dict: {team: elo}
    Also returns recent game history for form calculation.
    """
    df = historical_df.copy()
    df = df.dropna(subset=["score1", "score2"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    elo_dict = {}
    last_season = {}
    game_history = {}  # {team: [{"result":1/0, "elo_diff":x}, ...]}

    seasons = sorted(df["season"].unique())

    for season in seasons:
        season_df = df[df["season"] == season]

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

            # Home field adjustment
            hfa_adj = 0 if neutral else hfa
            adj_e1 = e1 + hfa_adj

            exp1 = expected_score(adj_e1, e2)
            exp2 = 1.0 - exp1

            actual1 = 1.0 if score1 > score2 else (0.5 if score1 == score2 else 0.0)
            actual2 = 1.0 - actual1

            point_diff = abs(score1 - score2)
            elo_diff_abs = abs(adj_e1 - e2)

            if point_diff > 0:
                mov = mov_multiplier(point_diff, elo_diff_abs)
            else:
                mov = 1.0

            k = k_base
            elo_dict[team1] = e1 + k * mov * (actual1 - exp1)
            elo_dict[team2] = e2 + k * mov * (actual2 - exp2)

            game_history[team1].append({
                "result": actual1,
                "elo_diff": adj_e1 - e2,
                "date": row["date"].isoformat()
            })
            game_history[team2].append({
                "result": actual2,
                "elo_diff": e2 - adj_e1,
                "date": row["date"].isoformat()
            })

    return elo_dict, game_history


def recent_form(game_history, team, n=5):
    """Win rate over last n games."""
    hist = game_history.get(team, [])
    if not hist:
        return 0.5
    recent = hist[-n:]
    return sum(g["result"] for g in recent) / len(recent)



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

    adj_elo_a = base_elo_a + form_adj_a + rest_adj_a + injury_adj_a
    adj_elo_b = base_elo_b + form_adj_b + rest_adj_b + injury_adj_b

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
        "elo_diff": float(adj_elo_a - adj_elo_b)
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
