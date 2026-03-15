"""
Simple Rating System (SRS) — System 13
Massey-inspired linear algebra rating that self-consistently accounts for
strength of schedule.

SRS satisfies: SRS_team = avg_margin + avg_opponent_SRS
Solved iteratively (or via linear system): this is equivalent to the
Colley Matrix method used by BCS computers and Pro Football Reference.

Inspired by:
- Pro Football Reference SRS: https://www.pro-football-reference.com/blog/index37a8.html
- Massey ratings: https://masseyratings.com/theory/massey97.pdf
- Kenneth Massey, "Statistical Models Applied to the Rating of Sports Teams"
"""

import numpy as np


def compute_srs(standings, schedule_results, n_iterations=50):
    """
    Compute Simple Rating System ratings for all teams.

    Args:
        standings: {team: {wins, losses, points_for, points_against, games_played}}
        schedule_results: list of {team1, team2, score1, score2} (completed games only)

    Returns:
        {team: {srs, sos, avg_margin, elo_equivalent}}
    """
    teams = sorted(standings.keys())
    if not teams or not schedule_results:
        return {t: {"srs": 0.0, "sos": 0.0, "avg_margin": 0.0, "elo_equivalent": 1500.0}
                for t in teams}

    n = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    # Compute raw average margin per team (points_for - points_against) / games
    avg_margin = {}
    for t in teams:
        st = standings.get(t, {})
        gp = max(st.get("games_played", 1), 1)
        pf = st.get("points_for", 0)
        pa = st.get("points_against", 0)
        avg_margin[t] = (pf - pa) / gp

    # Count opponents and accumulate opponent ratings for SOS
    # Iterative solution: SRS[t] = avg_margin[t] + avg(SRS[opponents of t])
    srs = {t: avg_margin[t] for t in teams}

    # Build opponent list from schedule
    opponents = {t: [] for t in teams}
    margins_vs = {t: [] for t in teams}  # margin from each game (team's perspective)
    for game in schedule_results:
        t1 = game.get("team1") or game.get("home_team")
        t2 = game.get("team2") or game.get("away_team")
        s1 = game.get("score1") or game.get("home_score", 0)
        s2 = game.get("score2") or game.get("away_score", 0)
        if t1 in team_idx and t2 in team_idx:
            opponents[t1].append(t2)
            opponents[t2].append(t1)
            margins_vs[t1].append(s1 - s2)
            margins_vs[t2].append(s2 - s1)

    # Iterative convergence (50 iterations is more than enough for 32 teams)
    for _ in range(n_iterations):
        new_srs = {}
        for t in teams:
            opps = opponents[t]
            if not opps:
                new_srs[t] = avg_margin[t]
                continue
            # Per-game margin (actual) + opponent SRS
            game_margins = margins_vs[t]
            opp_srs = [srs.get(o, 0.0) for o in opps]
            new_srs[t] = np.mean([m + o for m, o in zip(game_margins, opp_srs)])
        srs = new_srs

    # Normalize to zero mean (relative ratings)
    mean_srs = np.mean(list(srs.values()))
    srs = {t: srs[t] - mean_srs for t in teams}

    # Compute SOS = SRS - avg_margin
    sos = {t: srs[t] - avg_margin.get(t, 0.0) for t in teams}

    # Convert SRS to ELO-equivalent (scale factor: SRS ≈ ELO/25)
    # i.e., 1 point of margin advantage ≈ 25 ELO points
    result = {}
    for t in teams:
        result[t] = {
            "srs": round(float(srs[t]), 2),
            "sos": round(float(sos[t]), 2),
            "avg_margin": round(float(avg_margin.get(t, 0.0)), 2),
            "elo_equivalent": round(1500.0 + srs[t] * 25.0, 1),
        }

    return result


def predict_game_srs(team_a, team_b, srs_ratings, is_home_a=True, hfa=65.0, neutral=False):
    """
    Predict win probability using SRS ELO-equivalent ratings.

    Args:
        team_a, team_b: team abbreviations
        srs_ratings: output of compute_srs()
        is_home_a: True if team_a is home
        hfa: home field advantage in ELO points
        neutral: neutral site flag

    Returns:
        {"prob_a": float, "srs_a": float, "srs_b": float}
    """
    elo_a = srs_ratings.get(team_a, {}).get("elo_equivalent", 1500.0)
    elo_b = srs_ratings.get(team_b, {}).get("elo_equivalent", 1500.0)

    if not neutral and is_home_a:
        elo_a += hfa
    elif not neutral and not is_home_a:
        elo_b += hfa

    diff = elo_a - elo_b
    prob_a = 1.0 / (1.0 + 10 ** (-diff / 400.0))

    return {
        "prob_a": round(float(prob_a), 4),
        "srs_a": srs_ratings.get(team_a, {}).get("srs", 0.0),
        "srs_b": srs_ratings.get(team_b, {}).get("srs", 0.0),
        "sos_a": srs_ratings.get(team_a, {}).get("sos", 0.0),
        "sos_b": srs_ratings.get(team_b, {}).get("sos", 0.0),
    }
