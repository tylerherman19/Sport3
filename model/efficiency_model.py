"""
Efficiency Model — Systems 4, 5, 6, 7, 8
Pythagorean Win Expectation, Offensive/Defensive Efficiency,
Turnover Regression, Rest/Travel Adjustments, Time Decay.
"""

import numpy as np
import pandas as pd
from math import log, exp
from geopy.distance import geodesic


# NFL team home city coordinates for travel distance calculation
TEAM_COORDS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7553, -84.4006),
    "BAL": (39.2780, -76.6227), "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160), "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945), "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456), "GB":  (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373), "KC":  (39.0490, -94.4839),
    "LAC": (33.8644, -118.2611), "LAR": (33.9534, -118.3392),
    "LV":  (36.0909, -115.1833), "MIA": (25.9580, -80.2389),
    "MIN": (44.9737, -93.2575), "NE":  (42.0909, -71.2643),
    "NO":  (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158), "SEA": (47.5952, -122.3316),
    "SF":  (37.4032, -121.9698), "TB":  (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9078, -76.8645),
}


def travel_distance(team_a, team_b, is_home_a):
    """Returns distance in miles team_b (away) traveled."""
    if is_home_a:
        away_team = team_b
        home_team = team_a
    else:
        away_team = team_a
        home_team = team_b

    coords_away = TEAM_COORDS.get(away_team)
    coords_home = TEAM_COORDS.get(home_team)

    if coords_away is None or coords_home is None:
        return 0.0

    try:
        dist = geodesic(coords_away, coords_home).miles
        return float(dist)
    except Exception:
        return 0.0


def travel_elo_adjustment(distance_miles):
    """ELO penalty for travel distance."""
    if distance_miles < 500:
        return 0.0
    elif distance_miles < 1500:
        return -5.0
    elif distance_miles < 2500:
        return -10.0
    else:
        return -15.0


def rest_elo_adjustment(rest_days, opponent_rest_days):
    """
    Rest/travel adjustments to ELO.
    Short week penalty, bye week bonus.
    """
    adj = 0.0
    # Team rest
    if rest_days <= 5:  # Thursday game (short week)
        adj -= 15.0
    elif rest_days >= 13:  # After bye week
        adj += 25.0

    # Relative rest advantage
    rest_diff = rest_days - opponent_rest_days
    adj += rest_diff * 1.5

    return float(adj)


def compute_pythagorean(teams_data, exponent=2.37):
    """
    System 4: Pythagorean Win Expectation.
    teams_data: {team: {points_for, points_against}}
    Returns {team: {pyth, elo_equiv}}
    """
    result = {}
    for team, data in teams_data.items():
        pf = max(float(data.get("points_for", 350)), 1.0)
        pa = max(float(data.get("points_against", 350)), 1.0)

        pyth = (pf ** exponent) / (pf ** exponent + pa ** exponent)
        elo_equiv = 1500.0 + (pyth - 0.5) * 400.0

        result[team] = {
            "pyth": round(pyth, 4),
            "elo_equiv": round(elo_equiv, 1),
            "points_for": pf,
            "points_against": pa
        }
    return result


def compute_efficiency(teams_data, league_avg_ypp=5.5, league_avg_ypp_allowed=5.5):
    """
    System 5: Offensive and Defensive Efficiency.
    teams_data: {team: {ypp_offense, ypp_allowed}}
    Returns {team: {off_eff, def_eff, net_eff, elo_equiv}}
    """
    result = {}
    ypp_values = [d.get("ypp_offense", league_avg_ypp) for d in teams_data.values()]
    ypp_allowed_values = [d.get("ypp_allowed", league_avg_ypp_allowed) for d in teams_data.values()]

    real_league_avg_off = np.mean(ypp_values) if ypp_values else league_avg_ypp
    real_league_avg_def = np.mean(ypp_allowed_values) if ypp_allowed_values else league_avg_ypp_allowed

    for team, data in teams_data.items():
        ypp_off = float(data.get("ypp_offense", real_league_avg_off))
        ypp_def = float(data.get("ypp_allowed", real_league_avg_def))

        off_eff = ypp_off / max(real_league_avg_off, 0.1)
        def_eff = real_league_avg_def / max(ypp_def, 0.1)
        net_eff = off_eff - def_eff

        elo_equiv = 1500.0 + net_eff * 200.0

        result[team] = {
            "off_eff": round(off_eff, 4),
            "def_eff": round(def_eff, 4),
            "net_eff": round(net_eff, 4),
            "elo_equiv": round(elo_equiv, 1)
        }
    return result


def turnover_regression_adjustment(team, to_data, games_played):
    """
    System 6: Turnover Regression.
    Adjusts ELO based on turnover luck regression.
    regression_weight = 1 - (games_played / 17)
    """
    regression_weight = max(0.0, 1.0 - (games_played / 17.0))

    actual_to_diff = float(to_data.get(team, {}).get("turnover_diff", 0))
    expected_to_diff = 0.0  # Regress toward mean

    adjusted_to_diff = actual_to_diff * (1 - regression_weight) + expected_to_diff * regression_weight
    adj = adjusted_to_diff * 3.0  # ~3 ELO points per turnover

    return float(adj), float(adjusted_to_diff)


def time_decay_weights(dates, lambda_decay=0.1):
    """
    System 8: Time Decay Weighting.
    w = exp(-lambda * t) where t is days since game.
    dates: array-like of datetime objects or strings.
    Returns numpy array of weights.
    """
    dates = pd.to_datetime(dates)
    now = pd.Timestamp.now()
    days_ago = (now - dates).dt.days.values.astype(float)
    days_ago = np.clip(days_ago, 0, None)
    weights = np.exp(-lambda_decay * days_ago / 30.0)  # t in months
    weights = weights / weights.sum()
    return weights


def efficiency_predict_game(team_a, team_b, efficiency_data, pythagorean_data,
                             is_home_a=True, neutral=False):
    """
    Predict win probability from efficiency + pythagorean data.
    Returns {eff_prob, pyth_prob}
    """
    eff_a = efficiency_data.get(team_a, {}).get("elo_equiv", 1500.0)
    eff_b = efficiency_data.get(team_b, {}).get("elo_equiv", 1500.0)
    pyth_a = pythagorean_data.get(team_a, {}).get("elo_equiv", 1500.0)
    pyth_b = pythagorean_data.get(team_b, {}).get("elo_equiv", 1500.0)

    hfa = 65.0 if (is_home_a and not neutral) else 0.0

    eff_diff = (eff_a + hfa) - eff_b
    eff_prob = 1.0 / (1.0 + 10 ** (-eff_diff / 400.0))

    pyth_diff = (pyth_a + hfa) - pyth_b
    pyth_prob = 1.0 / (1.0 + 10 ** (-pyth_diff / 400.0))

    return {
        "eff_prob": round(float(eff_prob), 4),
        "pyth_prob": round(float(pyth_prob), 4)
    }


def hierarchical_team_rating(team, efficiency_data):
    """
    System 11: Hierarchical Team Model.
    Offensive_rating = passing_eff * 0.6 + rushing_eff * 0.4
    Defensive_rating = pass_def_eff * 0.6 + rush_def_eff * 0.4
    """
    data = efficiency_data.get(team, {})
    passing_eff = float(data.get("passing_eff", 1.0))
    rushing_eff = float(data.get("rushing_eff", 1.0))
    pass_def_eff = float(data.get("pass_def_eff", 1.0))
    rush_def_eff = float(data.get("rush_def_eff", 1.0))

    offensive_rating = passing_eff * 0.6 + rushing_eff * 0.4
    defensive_rating = pass_def_eff * 0.6 + rush_def_eff * 0.4

    return {
        "offensive_rating": round(offensive_rating, 4),
        "defensive_rating": round(defensive_rating, 4),
        "team_strength": round((offensive_rating + defensive_rating) / 2.0, 4)
    }
