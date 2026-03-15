"""
Monte Carlo Simulation — System 10
Simulates game outcomes and full seasons using Bayesian rating distributions.
Includes conference-aware playoff simulation (538-style bracket).
"""

import numpy as np
from scipy.special import expit


# NFL conference/division structure
NFL_DIVISIONS = {
    # AFC
    "BUF": ("AFC", "AFC East"),  "MIA": ("AFC", "AFC East"),
    "NE":  ("AFC", "AFC East"),  "NYJ": ("AFC", "AFC East"),
    "BAL": ("AFC", "AFC North"), "CIN": ("AFC", "AFC North"),
    "CLE": ("AFC", "AFC North"), "PIT": ("AFC", "AFC North"),
    "HOU": ("AFC", "AFC South"), "IND": ("AFC", "AFC South"),
    "JAX": ("AFC", "AFC South"), "TEN": ("AFC", "AFC South"),
    "KC":  ("AFC", "AFC West"),  "LV":  ("AFC", "AFC West"),
    "LAC": ("AFC", "AFC West"),  "DEN": ("AFC", "AFC West"),
    # NFC
    "DAL": ("NFC", "NFC East"),  "NYG": ("NFC", "NFC East"),
    "PHI": ("NFC", "NFC East"),  "WAS": ("NFC", "NFC East"),
    "CHI": ("NFC", "NFC North"), "DET": ("NFC", "NFC North"),
    "GB":  ("NFC", "NFC North"), "MIN": ("NFC", "NFC North"),
    "ATL": ("NFC", "NFC South"), "CAR": ("NFC", "NFC South"),
    "NO":  ("NFC", "NFC South"), "TB":  ("NFC", "NFC South"),
    "ARI": ("NFC", "NFC West"),  "LAR": ("NFC", "NFC West"),
    "SF":  ("NFC", "NFC West"),  "SEA": ("NFC", "NFC West"),
}


def simulate_game(mu_a, mu_b, sigma_a=75.0, sigma_b=75.0,
                  is_home_a=True, neutral=False, hfa=65.0,
                  n=10000, random_state=None):
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

    # Add game-level noise (NFL game std dev ~14 points)
    game_noise = rng.normal(0, 14.0, n)
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


def _game_prob(team_a, team_b, ratings, home_a=True, hfa=65.0):
    """ELO-based win probability for team_a."""
    mu_a = ratings.get(team_a, {}).get("mu", 1500.0)
    mu_b = ratings.get(team_b, {}).get("mu", 1500.0)
    adj = hfa if home_a else 0.0
    diff = (mu_a + adj) - mu_b
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def _play_game(team_a, team_b, ratings, rng, home_a=True):
    """Simulate a single playoff game; returns winner."""
    p = _game_prob(team_a, team_b, ratings, home_a=home_a)
    return team_a if rng.random() < p else team_b


def simulate_playoffs(afc_seeds, nfc_seeds, ratings, rng):
    """
    Simulate NFL playoffs: Wild Card → Divisional → Conference → Super Bowl.
    afc_seeds / nfc_seeds: lists of 7 team names in seed order (seed 1 = index 0).
    Higher seed plays at home through conference rounds.
    Returns the Super Bowl winner's abbreviation.
    """
    def simulate_conference(seeds):
        if len(seeds) < 7:
            return seeds[0] if seeds else None

        # Wild Card round: 2v7, 3v6, 4v5 (seed 1 gets a bye)
        wc_winners = [
            seeds[0],  # bye
            _play_game(seeds[1], seeds[6], ratings, rng, home_a=True),
            _play_game(seeds[2], seeds[5], ratings, rng, home_a=True),
            _play_game(seeds[3], seeds[4], ratings, rng, home_a=True),
        ]

        # Divisional round: 1 vs lowest, 2-winner vs other
        # Re-seed: sort wild card winners by their original seed index
        def seed_rank(t):
            return seeds.index(t) if t in seeds else 99

        wc_sorted = sorted(wc_winners, key=seed_rank)
        dv1 = _play_game(wc_sorted[0], wc_sorted[3], ratings, rng, home_a=True)
        dv2 = _play_game(wc_sorted[1], wc_sorted[2], ratings, rng, home_a=True)

        # Conference Championship: higher seed hosts
        home_a = seed_rank(dv1) < seed_rank(dv2)
        conf_champ = _play_game(dv1, dv2, ratings, rng, home_a=home_a)
        return conf_champ

    afc_champ = simulate_conference(afc_seeds)
    nfc_champ = simulate_conference(nfc_seeds)

    if afc_champ is None:
        return nfc_champ
    if nfc_champ is None:
        return afc_champ

    # Super Bowl — neutral site
    mu_afc = ratings.get(afc_champ, {}).get("mu", 1500.0)
    mu_nfc = ratings.get(nfc_champ, {}).get("mu", 1500.0)
    diff = mu_afc - mu_nfc
    p_afc = 1.0 / (1.0 + 10 ** (-diff / 400.0))
    return afc_champ if rng.random() < p_afc else nfc_champ


def simulate_season(teams, schedule, ratings, n_sims=10000, random_state=42):
    """
    Simulate full season N times to compute playoff/division/SB probabilities.

    teams: list of team abbreviations (32 teams)
    schedule: list of {team_a, team_b, is_home_a, neutral, week, game_id}
              Only includes remaining games
    ratings: {team: {mu, sigma, wins}} from Bayesian model

    Returns: {team: {playoff_prob, division_win_prob, sb_prob, wins_avg}}
    """
    rng = np.random.default_rng(random_state)
    n_teams = len(teams)

    current_wins = {t: ratings.get(t, {}).get("wins", 0) for t in teams}
    team_idx = {t: i for i, t in enumerate(teams)}

    if not schedule:
        # No remaining games — determine final standings from current wins
        # Group by division and conference to get proper seedings
        divisions = {}
        for t in teams:
            _, div = NFL_DIVISIONS.get(t, ("?", f"Other_{t}"))
            divisions.setdefault(div, []).append(t)

        div_winners = set()
        for div_teams in divisions.values():
            if div_teams:
                winner = max(div_teams, key=lambda t: current_wins.get(t, 0))
                div_winners.add(winner)

        afc_teams = [(t, current_wins.get(t, 0)) for t in teams
                     if NFL_DIVISIONS.get(t, ("?",))[0] == "AFC"]
        nfc_teams = [(t, current_wins.get(t, 0)) for t in teams
                     if NFL_DIVISIONS.get(t, ("?",))[0] == "NFC"]

        def get_seeds(conf_list, div_winners_set):
            srt = sorted(conf_list, key=lambda x: -x[1])
            divw = [t for t, _ in srt if t in div_winners_set][:4]
            wild = [t for t, _ in srt if t not in div_winners_set][:3]
            return divw + wild

        afc_seeds = get_seeds(afc_teams, div_winners)
        nfc_seeds = get_seeds(nfc_teams, div_winners)
        playoff_set = set(afc_seeds) | set(nfc_seeds)

        result = {}
        for t in teams:
            in_playoffs = t in playoff_set
            in_div = t in div_winners
            result[t] = {
                "playoff_prob": round(1.0 if in_playoffs else 0.0, 4),
                "division_win_prob": round(1.0 if in_div else 0.0, 4),
                "sb_prob": round(1.0 if t == (afc_seeds + nfc_seeds)[0] else 0.0, 4),
                "wins_avg": float(current_wins.get(t, 0))
            }
        return result

    win_totals = np.zeros((n_sims, n_teams))
    for t in teams:
        if t in team_idx:
            win_totals[:, team_idx[t]] = current_wins.get(t, 0)

    # Pre-compute base game probabilities for vectorized simulation
    game_probs = []
    for game in schedule:
        ta = game["team_a"]
        tb = game["team_b"]
        mu_a = ratings.get(ta, {}).get("mu", 1500.0)
        mu_b = ratings.get(tb, {}).get("mu", 1500.0)
        sigma_a = ratings.get(ta, {}).get("sigma", 75.0)
        sigma_b = ratings.get(tb, {}).get("sigma", 75.0)
        hfa = 65.0 if (game.get("is_home_a", True) and not game.get("neutral", False)) else 0.0
        diff = (mu_a + hfa) - mu_b
        base_prob = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        game_probs.append({
            "idx_a": team_idx.get(ta, -1),
            "idx_b": team_idx.get(tb, -1),
            "mu_a": mu_a, "mu_b": mu_b,
            "sigma_a": sigma_a, "sigma_b": sigma_b,
            "hfa": hfa, "base_prob": base_prob
        })

    for gp in game_probs:
        idx_a = gp["idx_a"]
        idx_b = gp["idx_b"]
        if idx_a < 0 or idx_b < 0:
            continue
        str_a = rng.normal(gp["mu_a"], gp["sigma_a"] * 0.3, n_sims)
        str_b = rng.normal(gp["mu_b"], gp["sigma_b"] * 0.3, n_sims)
        diff = (str_a + gp["hfa"]) - str_b
        probs = 1.0 / (1.0 + np.exp(-diff / 400.0 * np.log(10)))
        outcomes = rng.random(n_sims) < probs
        win_totals[:, idx_a] += outcomes.astype(float)
        win_totals[:, idx_b] += (~outcomes).astype(float)

    playoff_counts = np.zeros(n_teams)
    div_win_counts = np.zeros(n_teams)
    sb_counts = np.zeros(n_teams)

    # Group teams by division once (used in every sim)
    div_groups = {}
    for t in teams:
        _, div = NFL_DIVISIONS.get(t, ("?", f"Other_{t}"))
        div_groups.setdefault(div, []).append(t)

    for sim in range(n_sims):
        sim_wins = win_totals[sim]

        # Determine division winners
        div_winners = set()
        for div_teams in div_groups.values():
            if div_teams:
                winner = max(div_teams, key=lambda t: sim_wins[team_idx[t]])
                div_winners.add(winner)
                div_win_counts[team_idx[winner]] += 1

        # Build conference seedings
        afc_list = [(t, sim_wins[team_idx[t]]) for t in teams
                    if NFL_DIVISIONS.get(t, ("?",))[0] == "AFC"]
        nfc_list = [(t, sim_wins[team_idx[t]]) for t in teams
                    if NFL_DIVISIONS.get(t, ("?",))[0] == "NFC"]

        div_afc = {t for t in div_winners if NFL_DIVISIONS.get(t, ("?",))[0] == "AFC"}
        div_nfc = {t for t in div_winners if NFL_DIVISIONS.get(t, ("?",))[0] == "NFC"}

        def get_seeds(conf_list, div_set):
            srt = sorted(conf_list, key=lambda x: -x[1])
            divw = [t for t, _ in srt if t in div_set][:4]
            wild = [t for t, _ in srt if t not in div_set][:3]
            return divw + wild

        afc_seeds = get_seeds(afc_list, div_afc)
        nfc_seeds = get_seeds(nfc_list, div_nfc)

        # Playoff counts
        for t in set(afc_seeds) | set(nfc_seeds):
            if t in team_idx:
                playoff_counts[team_idx[t]] += 1

        # Super Bowl — simulate full bracket
        if len(afc_seeds) >= 7 and len(nfc_seeds) >= 7:
            sb_winner = simulate_playoffs(afc_seeds, nfc_seeds, ratings, rng)
        else:
            # Fallback: highest-rated team in playoff field wins
            all_seeds = afc_seeds + nfc_seeds
            sb_winner = max(all_seeds, key=lambda t: ratings.get(t, {}).get("mu", 1500.0)) if all_seeds else None

        if sb_winner and sb_winner in team_idx:
            sb_counts[team_idx[sb_winner]] += 1

    result = {}
    for t in teams:
        i = team_idx.get(t, -1)
        if i < 0:
            result[t] = {"playoff_prob": 0.0, "division_win_prob": 0.0, "sb_prob": 0.0, "wins_avg": 0.0}
            continue
        result[t] = {
            "playoff_prob": round(float(playoff_counts[i] / n_sims), 4),
            "division_win_prob": round(float(div_win_counts[i] / n_sims), 4),
            "sb_prob": round(float(sb_counts[i] / n_sims), 4),
            "wins_avg": round(float(win_totals[:, i].mean()), 2)
        }

    return result


def margin_distribution_bins(mc_result, n_bins=20):
    """
    Returns histogram bins from Monte Carlo margin percentile data.
    Uses non-parametric CDF interpolation from actual percentiles rather than
    assuming a normal distribution (NFL margins are bimodal/fat-tailed).
    """
    percentiles = mc_result.get("margin_percentiles", [0, 0, 0, 0, 0, 0, 0])
    if len(percentiles) < 7:
        return []

    p5, p10, p25, p50, p75, p90, p95 = percentiles

    # Known (x, CDF) anchor points from our percentile set
    # Extend tails with ~linear extrapolation for p1/p99 approximation
    tail_lo = p5 - abs(p10 - p5)
    tail_hi = p95 + abs(p95 - p90)

    x_known = np.array([tail_lo, p5, p10, p25, p50, p75, p90, p95, tail_hi])
    pct_probs = np.array([0.01,  0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

    # Sort to ensure monotonicity (should already be sorted, but protect against bad data)
    sort_idx = np.argsort(x_known)
    x_known = x_known[sort_idx]
    pct_probs = pct_probs[sort_idx]

    # Interpolate CDF at evenly-spaced bin centers
    x = np.linspace(tail_lo, tail_hi, n_bins)
    cdf = np.interp(x, x_known, pct_probs, left=0.0, right=1.0)

    # Differentiate CDF → PDF (density)
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    density = np.diff(cdf) / dx
    density = np.append(density, density[-1] if len(density) else 0.0)
    density = np.clip(density, 0.0, None)

    if density.max() > 0:
        density = density / density.max()

    return [{"x": round(float(xi), 1), "y": round(float(di), 3)} for xi, di in zip(x, density)]
