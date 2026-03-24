"""
Monte Carlo Simulation — System 10
Simulates game outcomes and full seasons using Bayesian rating distributions.
"""

import numpy as np
from scipy.special import expit


def simulate_game(mu_a, mu_b, sigma_a=75.0, sigma_b=75.0,
                  is_home_a=True, neutral=False, hfa=65.0,
                  n=10000, random_state=None, game_noise_std=14.0):
    """
    Simulate a single game N times by drawing team strengths from
    their Bayesian distributions.

    Returns:
        win_prob: float
        exp_margin: expected point margin (positive = team_a wins)
        prob_7plus: P(|margin| >= 7 AND team_a wins)
        prob_14plus: P(|margin| >= 14 AND team_a wins)
        prob_21plus: P(|margin| >= 21 AND team_a wins)
        margin_dist: list of percentile values for histogram display
    """
    rng = np.random.default_rng(random_state)

    # Draw team strengths from Bayesian distributions
    strength_a = rng.normal(mu_a, sigma_a, n)
    strength_b = rng.normal(mu_b, sigma_b, n)

    hfa_val = 0.0
    if not neutral:
        hfa_val = hfa if is_home_a else -hfa

    diff = (strength_a + hfa_val) - strength_b

    # Convert ELO diff to expected margin (roughly 25 ELO pts ≈ 1 point)
    expected_margin = diff / 25.0

    # Add game-level noise (NFL default ~14 pts, NBA ~11.5 pts — pass game_noise_std to override)
    game_noise = rng.normal(0, game_noise_std, n)
    simulated_margins = expected_margin + game_noise

    wins_a = simulated_margins > 0
    win_prob = float(wins_a.mean())

    exp_margin = float(simulated_margins.mean())

    prob_7plus = float((wins_a & (simulated_margins >= 7)).mean())
    prob_14plus = float((wins_a & (simulated_margins >= 14)).mean())
    prob_21plus = float((wins_a & (simulated_margins >= 21)).mean())

    # Distribution for histogram (percentiles)
    percentiles = np.percentile(simulated_margins, [5, 10, 25, 50, 75, 90, 95]).tolist()

    return {
        "win_prob": round(win_prob, 4),
        "exp_margin": round(exp_margin, 2),
        "prob_7plus": round(prob_7plus, 4),
        "prob_14plus": round(prob_14plus, 4),
        "prob_21plus": round(prob_21plus, 4),
        "margin_percentiles": [round(p, 1) for p in percentiles]
    }


def simulate_season(teams, schedule, ratings, n_sims=10000, random_state=42):
    """
    Simulate full season N times to compute playoff probabilities.

    teams: list of team abbreviations (32 teams)
    schedule: list of {team_a, team_b, is_home_a, neutral, week, game_id}
              Only includes remaining games
    ratings: {team: {mu, sigma}} from Bayesian model

    Returns: {team: {playoff_prob, division_win_prob, sb_prob, wins_avg}}
    """
    rng = np.random.default_rng(random_state)
    n_teams = len(teams)

    # Current wins (before simulation) — default 0 if not provided
    current_wins = {t: ratings.get(t, {}).get("wins", 0) for t in teams}

    # Vectorized season simulation
    # win_totals[sim, team_idx] = total wins after season
    team_idx = {t: i for i, t in enumerate(teams)}

    if not schedule:
        # No remaining games — use current standings
        result = {}
        sorted_teams = sorted(teams, key=lambda t: current_wins.get(t, 0), reverse=True)
        for i, t in enumerate(sorted_teams):
            result[t] = {
                "playoff_prob": round(1.0 if i < 14 else 0.0, 4),
                "division_win_prob": round(1.0 if i < 8 else 0.0, 4),
                "sb_prob": round(1.0 if i == 0 else 0.0, 4),
                "wins_avg": float(current_wins.get(t, 0))
            }
        return result

    win_totals = np.zeros((n_sims, n_teams))

    # Add current wins
    for t in teams:
        if t in team_idx:
            win_totals[:, team_idx[t]] = current_wins.get(t, 0)

    # Pre-compute game probabilities
    game_probs = []
    for game in schedule:
        ta = game["team_a"]
        tb = game["team_b"]
        mu_a = ratings.get(ta, {}).get("mu", 1500.0)
        mu_b = ratings.get(tb, {}).get("mu", 1500.0)
        sigma_a = ratings.get(ta, {}).get("sigma", 75.0)
        sigma_b = ratings.get(tb, {}).get("sigma", 75.0)
        hfa = 100.0 if (game.get("is_home_a", True) and not game.get("neutral", False)) else 0.0

        # Expected difference for probability
        diff = (mu_a + hfa) - mu_b
        base_prob = 1.0 / (1.0 + 10 ** (-diff / 400.0))

        game_probs.append({
            "idx_a": team_idx.get(ta, -1),
            "idx_b": team_idx.get(tb, -1),
            "mu_a": mu_a, "mu_b": mu_b,
            "sigma_a": sigma_a, "sigma_b": sigma_b,
            "hfa": hfa, "base_prob": base_prob
        })

    # Simulate all games across all sims
    for gp in game_probs:
        idx_a = gp["idx_a"]
        idx_b = gp["idx_b"]
        if idx_a < 0 or idx_b < 0:
            continue

        # Draw outcome probabilities with rating uncertainty
        str_a = rng.normal(gp["mu_a"], gp["sigma_a"] * 0.3, n_sims)
        str_b = rng.normal(gp["mu_b"], gp["sigma_b"] * 0.3, n_sims)
        diff = (str_a + gp["hfa"]) - str_b
        probs = 1.0 / (1.0 + np.exp(-diff / 400.0 * np.log(10)))

        outcomes = rng.random(n_sims) < probs
        win_totals[:, idx_a] += outcomes.astype(float)
        win_totals[:, idx_b] += (~outcomes).astype(float)

    # Determine playoff teams (top 7 per conference — simplified: top 14 overall)
    # In reality would need conference/division structure
    total_games = 17
    playoff_counts = np.zeros(n_teams)
    div_win_counts = np.zeros(n_teams)
    sb_counts = np.zeros(n_teams)

    for sim in range(n_sims):
        sim_wins = win_totals[sim]
        # Top 14 make playoffs
        playoff_threshold = sorted(sim_wins, reverse=True)[13] if n_teams >= 14 else 0
        for i in range(n_teams):
            if sim_wins[i] >= playoff_threshold:
                playoff_counts[i] += 1
            if i < 8:  # Simplification: treat as division winners
                pass

        # Super Bowl winner — highest win total
        sb_winner = np.argmax(sim_wins)
        sb_counts[sb_winner] += 1

    result = {}
    for t in teams:
        i = team_idx.get(t, -1)
        if i < 0:
            result[t] = {"playoff_prob": 0.0, "division_win_prob": 0.0, "sb_prob": 0.0, "wins_avg": 0.0}
            continue
        result[t] = {
            "playoff_prob": round(float(playoff_counts[i] / n_sims), 4),
            "division_win_prob": round(float(playoff_counts[i] / n_sims * 0.4), 4),
            "sb_prob": round(float(sb_counts[i] / n_sims), 4),
            "wins_avg": round(float(win_totals[:, i].mean()), 2)
        }

    return result


def margin_distribution_bins(mc_result, n_bins=20):
    """
    Returns histogram bins from margin percentiles for display.
    Approximates a normal distribution from percentile data.
    """
    percentiles = mc_result.get("margin_percentiles", [0, 0, 0, 0, 0, 0, 0])
    if len(percentiles) < 7:
        return []

    p5, p10, p25, p50, p75, p90, p95 = percentiles
    mean = p50
    std = (p75 - p25) / 1.35  # IQR approximation

    if std <= 0:
        std = 10.0

    x = np.linspace(mean - 3.5 * std, mean + 3.5 * std, n_bins)
    density = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    density = density / density.max()

    return [{"x": round(float(xi), 1), "y": round(float(di), 3)} for xi, di in zip(x, density)]
