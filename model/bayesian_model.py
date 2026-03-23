"""
Bayesian Rating Model — System 3
Updates team ratings using Bayesian inference (Normal-Normal conjugate update).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def normal_update(prior_mu, prior_sigma, observation, obs_sigma=100.0):
    """
    Conjugate Normal-Normal update.
    prior: N(mu, sigma^2)
    likelihood: N(observation, obs_sigma^2)
    Returns posterior (mu, sigma).
    """
    precision_prior = 1.0 / (prior_sigma ** 2)
    precision_obs = 1.0 / (obs_sigma ** 2)
    precision_post = precision_prior + precision_obs

    mu_post = (precision_prior * prior_mu + precision_obs * observation) / precision_post
    sigma_post = np.sqrt(1.0 / precision_post)

    return float(mu_post), float(sigma_post)


def initialize_ratings(elo_dict, initial_sigma=75.0):
    """
    Initialize Bayesian ratings from ELO values.
    Returns {team: {mu, sigma}}
    """
    ratings = {}
    for team, elo in elo_dict.items():
        ratings[team] = {"mu": float(elo), "sigma": float(initial_sigma)}
    return ratings


def update_ratings(games, elo_dict, initial_sigma=75.0, obs_noise=100.0, regress_pct=0.33, hfa=65.0, margin_multiplier=25.0):
    """
    Process games chronologically, updating Bayesian ratings after each game.
    games: list of {team1, team2, score1, score2, date, neutral}
    Returns {team: {mu, sigma}}
    """
    ratings = initialize_ratings(elo_dict, initial_sigma)
    initial_elo = 1500.0

    if not games:
        return ratings

    df = pd.DataFrame(games)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

    # Group by season to regress at season start
    seasons_seen = set()

    for _, row in df.iterrows():
        team1 = str(row["team1"])
        team2 = str(row["team2"])
        season = row.get("season", 0)

        if season and season not in seasons_seen:
            seasons_seen.add(season)
            for team in ratings:
                old_mu = ratings[team]["mu"]
                ratings[team]["mu"] = old_mu * (1 - regress_pct) + initial_elo * regress_pct
                ratings[team]["sigma"] = min(
                    float(initial_sigma),
                    ratings[team]["sigma"] * 1.1
                )

        if team1 not in ratings:
            ratings[team1] = {"mu": initial_elo, "sigma": initial_sigma}
        if team2 not in ratings:
            ratings[team2] = {"mu": initial_elo, "sigma": initial_sigma}

        score1 = float(row.get("score1", 0))
        score2 = float(row.get("score2", 0))
        neutral = int(row.get("neutral", 0))
        hfa_adj = 0.0 if neutral else hfa

        mu1 = ratings[team1]["mu"] + hfa_adj
        mu2 = ratings[team2]["mu"]

        # Performance signal: ELO-equivalent of margin
        margin = score1 - score2
        perf_signal_1 = mu2 + margin * margin_multiplier
        perf_signal_2 = mu1 - margin * margin_multiplier

        # Clamp signals
        perf_signal_1 = np.clip(perf_signal_1, 800, 2200)
        perf_signal_2 = np.clip(perf_signal_2, 800, 2200)

        new_mu1, new_sig1 = normal_update(
            ratings[team1]["mu"], ratings[team1]["sigma"],
            perf_signal_1, obs_noise
        )
        new_mu2, new_sig2 = normal_update(
            ratings[team2]["mu"], ratings[team2]["sigma"],
            perf_signal_2, obs_noise
        )

        # Floor sigma at 20
        ratings[team1] = {"mu": new_mu1, "sigma": max(20.0, new_sig1)}
        ratings[team2] = {"mu": new_mu2, "sigma": max(20.0, new_sig2)}

    return ratings


def predict_game(team_a, team_b, ratings, is_home_a=True, neutral=False, hfa=65.0):
    """
    Predict P(team_a wins) using Bayesian ratings.
    Uses difference of Normal distributions.
    Returns {prob, mu_a, sigma_a, mu_b, sigma_b}
    """
    mu_a = ratings.get(team_a, {}).get("mu", 1500.0)
    sigma_a = ratings.get(team_a, {}).get("sigma", 75.0)
    mu_b = ratings.get(team_b, {}).get("mu", 1500.0)
    sigma_b = ratings.get(team_b, {}).get("sigma", 75.0)

    hfa_val = 0.0
    if not neutral:
        hfa_val = hfa if is_home_a else -hfa

    # P(A > B) = P(A - B > 0), A-B ~ N(mu_a - mu_b, sigma_a^2 + sigma_b^2)
    diff_mu = (mu_a + hfa_val) - mu_b
    diff_sigma = np.sqrt(sigma_a ** 2 + sigma_b ** 2)

    # P(A > B) = P(diff > 0) where diff ~ N(diff_mu, diff_sigma^2)
    # = Normal CDF at diff_mu / diff_sigma
    prob = float(norm.cdf(diff_mu / diff_sigma))

    return {
        "bayesian_prob": round(prob, 4),
        "mu_a": round(mu_a, 1),
        "sigma_a": round(sigma_a, 1),
        "mu_b": round(mu_b, 1),
        "sigma_b": round(sigma_b, 1),
        "uncertainty_band_a": [round(mu_a - sigma_a, 1), round(mu_a + sigma_a, 1)],
        "uncertainty_band_b": [round(mu_b - sigma_b, 1), round(mu_b + sigma_b, 1)]
    }


def get_all_ratings(ratings):
    """Returns sorted list of team ratings with uncertainty bands."""
    result = []
    for team, r in ratings.items():
        result.append({
            "team": team,
            "mu": round(r["mu"], 1),
            "sigma": round(r["sigma"], 1),
            "lower": round(r["mu"] - r["sigma"], 1),
            "upper": round(r["mu"] + r["sigma"], 1)
        })
    return sorted(result, key=lambda x: x["mu"], reverse=True)
