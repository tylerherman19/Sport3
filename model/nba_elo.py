"""
NBA ELO Rating Model — dedicated module for NBA-specific ELO computation.

NBA parameters differ from NFL:
  - Higher home court advantage (100 vs 65 ELO points)
  - Back-to-back penalty: -40 ELO (teams degrade sharply on zero days rest)
  - Rest bonus: +15 ELO per extra rest day vs opponent (capped at 2 days)
  - Travel penalty: -0.02 ELO per mile for away team (capped at -50)
  - Season reversion: 25% toward 1500 (less than NFL's 33%)
"""

import math

# ── Default parameters ────────────────────────────────────────────────────────

K_BASE = 25.0   # higher base; decays dynamically toward 12.5 by end of season
HFA = 100.0           # home court advantage in ELO points
INITIAL_ELO = 1500.0
REGRESS_PCT = 0.25    # 25% end-of-season reversion toward mean

# Game-day adjustments applied at prediction time (not during ELO update)
B2B_PENALTY = -40.0           # ELO penalty if team played last night
REST_BONUS_PER_DAY = 15.0     # ELO bonus per extra rest day advantage
REST_BONUS_MAX_DAYS = 2       # cap: max 2 days of rest advantage counted
TRAVEL_PENALTY_PER_MILE = -0.02  # ELO penalty per mile traveled (away only)
TRAVEL_PENALTY_MAX = -50.0    # floor on travel penalty


# ── Core ELO math ─────────────────────────────────────────────────────────────

def nba_expected_score(elo_a, elo_b):
    """Win probability for team A given adjusted ELO ratings."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def nba_mov_multiplier(point_diff, elo_diff):
    """
    NBA-calibrated margin-of-victory multiplier for ELO update.
    Uses 1.5 constant (vs NFL 2.2) reflecting tighter NBA margin distributions
    (NBA mean ~10 pts vs NFL ~13 pts, higher game total dampens per-point signal).
    """
    return math.log(abs(point_diff) + 1) * (1.5 / (abs(elo_diff) * 0.001 + 1.5))


# ── Historical ELO computation ────────────────────────────────────────────────

def compute_nba_elo(games, k_base=K_BASE, hfa=HFA,
                    initial_elo=INITIAL_ELO, regress_pct=REGRESS_PCT):
    """
    Compute NBA ELO ratings from a list of completed historical games.

    Parameters
    ----------
    games : list[dict]
        Each dict must have: date (str YYYY-MM-DD), season (int), team1 (str),
        team2 (str), score1 (int), score2 (int), neutral (0|1).
        team1 is treated as the home team unless neutral=1.
    k_base : float
        Base K-factor (default 20).
    hfa : float
        Home court advantage in ELO points (default 100).
    initial_elo : float
        Starting ELO for new teams (default 1500).
    regress_pct : float
        Fraction of season-end ELO deviation from mean that is regressed
        toward 1500 at the start of the next season (default 0.25).

    Returns
    -------
    elo_dict : dict[str, float]
        Current ELO rating for each team.
    game_history : dict[str, list[dict]]
        Per-team game log with keys: result (0/1), elo_diff, date.
    """
    import pandas as pd

    if not games:
        return {}, {}

    df = pd.DataFrame(games)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    elo_dict = {}
    game_history = {}

    seasons = sorted(df["season"].unique()) if "season" in df.columns else [df["date"].dt.year.max()]

    for season in seasons:
        season_df = df[df["season"] == season] if "season" in df.columns else df

        # End-of-season regression toward mean
        for team in list(elo_dict.keys()):
            elo_dict[team] = elo_dict[team] * (1 - regress_pct) + initial_elo * regress_pct
            game_history[team] = []

        # Track per-team games played this season for dynamic K-factor
        team_game_counts: dict = {}
        NBA_SEASON_GAMES = 82  # regular season games per team

        for _, row in season_df.iterrows():
            team1 = row["team1"]
            team2 = row["team2"]
            score1 = row.get("score1", 0)
            score2 = row.get("score2", 0)
            neutral = row.get("neutral", 0)

            if pd.isna(score1) or pd.isna(score2):
                continue

            score1 = int(score1)
            score2 = int(score2)

            if team1 not in elo_dict:
                elo_dict[team1] = initial_elo
                game_history[team1] = []
            if team2 not in elo_dict:
                elo_dict[team2] = initial_elo
                game_history[team2] = []

            e1 = elo_dict[team1]
            e2 = elo_dict[team2]

            hfa_adj = 0.0 if neutral else hfa
            adj_e1 = e1 + hfa_adj

            exp1 = nba_expected_score(adj_e1, e2)
            exp2 = 1.0 - exp1

            actual1 = 1.0 if score1 > score2 else (0.5 if score1 == score2 else 0.0)
            actual2 = 1.0 - actual1

            point_diff = abs(score1 - score2)
            elo_diff_abs = abs(adj_e1 - e2)
            mov = nba_mov_multiplier(point_diff, elo_diff_abs) if point_diff > 0 else 1.0

            # Dynamic K-factor: decays from k_base to k_base/2 over the season
            progress = min(1.0, team_game_counts.get(team1, 0) / NBA_SEASON_GAMES)
            k = k_base * (1.0 - 0.5 * progress)

            elo_dict[team1] = e1 + k * mov * (actual1 - exp1)
            elo_dict[team2] = e2 + k * mov * (actual2 - exp2)

            team_game_counts[team1] = team_game_counts.get(team1, 0) + 1
            team_game_counts[team2] = team_game_counts.get(team2, 0) + 1

            date_str = row["date"].isoformat()
            game_history.setdefault(team1, []).append({
                "result": actual1,
                "elo_diff": adj_e1 - e2,
                "date": date_str,
            })
            game_history.setdefault(team2, []).append({
                "result": actual2,
                "elo_diff": e2 - adj_e1,
                "date": date_str,
            })

    return elo_dict, game_history


# ── Trend / form helpers ──────────────────────────────────────────────────────

def nba_recent_form(game_history, team, n=5):
    """Win rate for a team over their last N games (default 5)."""
    hist = game_history.get(team, [])
    if not hist:
        return 0.5
    recent = hist[-n:]
    return sum(g["result"] for g in recent) / len(recent)


def nba_get_trend(game_history, team, window=5):
    """
    Return 'up', 'down', or 'neutral' based on recent vs older win rate.
    Threshold: >0.15 difference = directional trend.
    """
    hist = game_history.get(team, [])
    if len(hist) < window * 2:
        return "neutral"
    recent = sum(g["result"] for g in hist[-window:]) / window
    older = sum(g["result"] for g in hist[-window * 2:-window]) / window
    if recent > older + 0.15:
        return "up"
    elif recent < older - 0.15:
        return "down"
    return "neutral"


# ── Game-day prediction (with adjustments) ────────────────────────────────────

def predict_nba_game(home, away, elo_dict,
                     home_b2b=False, away_b2b=False,
                     rest_diff=0, travel_miles=0.0,
                     neutral=False,
                     hfa=HFA):
    """
    Predict NBA home-team win probability with full game-day adjustments.

    Parameters
    ----------
    home, away : str
        Team abbreviations (e.g. 'LAL', 'GSW').
    elo_dict : dict[str, float]
        Current ELO ratings from compute_nba_elo().
    home_b2b : bool
        True if home team played last night (back-to-back).
    away_b2b : bool
        True if away team played last night (back-to-back).
    rest_diff : int
        home_rest_days − away_rest_days. Positive = home more rested.
    travel_miles : float
        Miles the away team traveled to reach this game.
    neutral : bool
        True if game is at a neutral site (no home court advantage).
    hfa : float
        Home court advantage (override default 100).

    Returns
    -------
    dict with keys:
      prob          — home team win probability (0–1)
      home_adj_elo  — home team adjusted ELO used for prediction
      away_adj_elo  — away team adjusted ELO used for prediction
      home_raw_elo  — base ELO before adjustments
      away_raw_elo  — base ELO before adjustments
    """
    home_raw = elo_dict.get(home, INITIAL_ELO)
    away_raw = elo_dict.get(away, INITIAL_ELO)

    home_adj = home_raw
    away_adj = away_raw

    # Home court advantage
    if not neutral:
        home_adj += hfa

    # Back-to-back penalties
    if home_b2b:
        home_adj += B2B_PENALTY
    if away_b2b:
        away_adj += B2B_PENALTY

    # Rest bonus/penalty: clamp rest_diff to ±REST_BONUS_MAX_DAYS days
    clamped_rest = max(-REST_BONUS_MAX_DAYS, min(REST_BONUS_MAX_DAYS, rest_diff))
    rest_adj = clamped_rest * REST_BONUS_PER_DAY
    home_adj += rest_adj

    # Travel penalty applies to away team only
    if travel_miles > 0:
        travel_adj = max(TRAVEL_PENALTY_MAX, travel_miles * TRAVEL_PENALTY_PER_MILE)
        away_adj += travel_adj

    prob = nba_expected_score(home_adj, away_adj)

    return {
        "prob": prob,
        "home_adj_elo": round(home_adj, 1),
        "away_adj_elo": round(away_adj, 1),
        "home_raw_elo": round(home_raw, 1),
        "away_raw_elo": round(away_raw, 1),
    }
