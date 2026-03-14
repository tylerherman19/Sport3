"""
NBA Model — ELO, Bayesian, Pythagorean, Efficiency, Injury, Ensemble
Mirrors the NFL model architecture with basketball-tuned constants.
"""

import numpy as np
import logging
from math import log, exp
from geopy.distance import geodesic

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

NBA_HFA = 100.0        # ~3.5 pt home advantage → ~100 ELO pts
NBA_K   = 20.0
NBA_INITIAL_ELO = 1500.0
NBA_PYTH_EXP = 13.91   # Basketball Pythagorean exponent
NBA_MC_NOISE = 12.0    # Points σ for Monte Carlo (basketball noise ~12 pts)

NBA_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN",
    "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA",
    "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX",
    "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

NBA_TEAM_NAMES = {
    "ATL": "Atlanta Hawks",      "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",      "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",      "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",   "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",        "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",  "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans","NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder","ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers","SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",  "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",          "WAS": "Washington Wizards",
}

# ESPN numeric team IDs for schedule fetching
ESPN_NBA_TEAM_IDS = {
    "ATL": 1,  "BOS": 2,  "BKN": 17, "CHA": 30, "CHI": 4,
    "CLE": 5,  "DAL": 6,  "DEN": 7,  "DET": 8,  "GSW": 9,
    "HOU": 10, "IND": 11, "LAC": 12, "LAL": 13, "MEM": 29,
    "MIA": 14, "MIL": 15, "MIN": 16, "NOP": 3,  "NYK": 18,
    "OKC": 25, "ORL": 19, "PHI": 20, "PHX": 21, "POR": 22,
    "SAC": 23, "SAS": 24, "TOR": 28, "UTA": 26, "WAS": 27,
}

# ESPN abbreviation normalization for NBA
ESPN_NBA_TO_ABBREV = {
    "GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS",
    "OKC": "OKC", "PHX": "PHX",
}

# Arena coordinates for travel distance
NBA_TEAM_COORDS = {
    "ATL": (33.7573, -84.3963), "BOS": (42.3662, -71.0621),
    "BKN": (40.6826, -73.9754), "CHA": (35.2251, -80.8392),
    "CHI": (41.8807, -87.6742), "CLE": (41.4965, -81.6882),
    "DAL": (32.7905, -96.8103), "DEN": (39.7487, -105.0077),
    "DET": (42.3411, -83.0554), "GSW": (37.7680, -122.3877),
    "HOU": (29.7508, -95.3621), "IND": (39.7639, -86.1555),
    "LAC": (34.0430, -118.2673),"LAL": (34.0430, -118.2673),
    "MEM": (35.1382, -90.0505), "MIA": (25.7814, -80.1870),
    "MIL": (43.0451, -87.9171), "MIN": (44.9795, -93.2762),
    "NOP": (29.9490, -90.0821), "NYK": (40.7505, -73.9934),
    "OKC": (35.4634, -97.5151), "ORL": (28.5392, -81.3839),
    "PHI": (39.9012, -75.1720), "PHX": (33.4457, -112.0712),
    "POR": (45.5316, -122.6668),"SAC": (38.5803, -121.4995),
    "SAS": (29.4270, -98.4375), "TOR": (43.6435, -79.3791),
    "UTA": (40.7683, -111.9011),"WAS": (38.8981, -77.0209),
}

# ── Injury tiers ──────────────────────────────────────────────────────────────

# Known superstars/all-stars for tier classification
NBA_SUPERSTAR_PLAYERS = {
    "LeBron James", "Stephen Curry", "Kevin Durant", "Nikola Jokic",
    "Giannis Antetokounmpo", "Luka Doncic", "Joel Embiid", "Jayson Tatum",
    "Damian Lillard", "Anthony Davis", "Kawhi Leonard", "Devin Booker",
    "Donovan Mitchell", "Ja Morant", "Trae Young", "Zion Williamson",
    "Paul George", "Jimmy Butler", "Tyrese Haliburton", "Shai Gilgeous-Alexander",
    "Victor Wembanyama", "Anthony Edwards", "Paolo Banchero",
}

NBA_ALLSTAR_PLAYERS = {
    "Bam Adebayo", "Draymond Green", "Klay Thompson", "Chris Paul",
    "Khris Middleton", "Brandon Ingram", "Karl-Anthony Towns", "Rudy Gobert",
    "De'Aaron Fox", "Jaylen Brown", "Julius Randle", "Pascal Siakam",
    "Darius Garland", "Jalen Brunson", "Cade Cunningham", "Evan Mobley",
    "Franz Wagner", "Mikal Bridges", "OG Anunoby", "Bradley Beal",
    "Zach LaVine", "CJ McCollum", "DeMar DeRozan", "Tobias Harris",
}

NBA_TIER_PENALTIES = {
    "superstar": 50.0,
    "allstar":   30.0,
    "starter":   15.0,
    "rotation":   8.0,
}
NBA_BACK_TO_BACK_PENALTY = 5.0
MAX_NBA_INJURY_PENALTY = 80.0


# ── ELO functions ─────────────────────────────────────────────────────────────

def _expected_score(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _mov_multiplier(point_diff, elo_diff_abs):
    return log(abs(point_diff) + 1) * (2.2 / (elo_diff_abs * 0.001 + 2.2))


def compute_nba_elo(games_list, k=NBA_K, hfa=NBA_HFA, regress_pct=0.33):
    """
    Compute NBA ELO ratings from a list of completed games.
    games_list: [{team_a, team_b, score_a, score_b, is_home_a, neutral, season, date}]
    Returns (elo_dict, game_history).
    """
    elo_dict = {}
    game_history = {}
    last_season = None

    for game in sorted(games_list, key=lambda g: g.get("date", "")):
        season = game.get("season", 0)
        if season != last_season:
            # Season regression
            for team in list(elo_dict.keys()):
                elo_dict[team] = elo_dict[team] * (1 - regress_pct) + NBA_INITIAL_ELO * regress_pct
                game_history[team] = []
            last_season = season

        team_a = game["team_a"]
        team_b = game["team_b"]
        score_a = float(game.get("score_a", 0))
        score_b = float(game.get("score_b", 0))
        is_home_a = game.get("is_home_a", True)
        neutral = game.get("neutral", False)

        for t in (team_a, team_b):
            if t not in elo_dict:
                elo_dict[t] = NBA_INITIAL_ELO
                game_history[t] = []

        e_a = elo_dict[team_a]
        e_b = elo_dict[team_b]

        hfa_adj = 0.0
        if not neutral:
            hfa_adj = hfa if is_home_a else -hfa

        adj_e_a = e_a + hfa_adj
        exp_a = _expected_score(adj_e_a, e_b)
        exp_b = 1.0 - exp_a

        actual_a = 1.0 if score_a > score_b else (0.5 if score_a == score_b else 0.0)
        actual_b = 1.0 - actual_a

        point_diff = abs(score_a - score_b)
        mov = _mov_multiplier(point_diff, abs(adj_e_a - e_b)) if point_diff > 0 else 1.0

        elo_dict[team_a] = e_a + k * mov * (actual_a - exp_a)
        elo_dict[team_b] = e_b + k * mov * (actual_b - exp_b)

        game_history[team_a].append({"result": actual_a, "elo_diff": adj_e_a - e_b, "date": game.get("date", "")})
        game_history[team_b].append({"result": actual_b, "elo_diff": e_b - adj_e_a, "date": game.get("date", "")})

    return elo_dict, game_history


def nba_predict_game(team_a, team_b, elo_dict, game_history,
                     is_home_a=True, neutral=False,
                     rest_adj_a=0.0, rest_adj_b=0.0,
                     injury_adj_a=0.0, injury_adj_b=0.0,
                     hfa=NBA_HFA, form_blend=0.20):
    """Predict NBA win probability for team_a."""
    base_a = elo_dict.get(team_a, NBA_INITIAL_ELO)
    base_b = elo_dict.get(team_b, NBA_INITIAL_ELO)

    hist_a = game_history.get(team_a, [])
    hist_b = game_history.get(team_b, [])
    form_a = (sum(g["result"] for g in hist_a[-5:]) / len(hist_a[-5:])) if hist_a else 0.5
    form_b = (sum(g["result"] for g in hist_b[-5:]) / len(hist_b[-5:])) if hist_b else 0.5

    adj_a = base_a + (form_a - 0.5) * form_blend * 100 + rest_adj_a + injury_adj_a
    adj_b = base_b + (form_b - 0.5) * form_blend * 100 + rest_adj_b + injury_adj_b

    if not neutral:
        if is_home_a:
            adj_a += hfa
        else:
            adj_b += hfa

    prob = _expected_score(adj_a, adj_b)
    return {
        "prob": float(prob),
        "elo_a": float(base_a),
        "elo_b": float(base_b),
        "adj_elo_a": float(adj_a),
        "adj_elo_b": float(adj_b),
        "form_a": float(form_a),
        "form_b": float(form_b),
        "elo_diff": float(adj_a - adj_b),
    }


def nba_get_trend(game_history, team, window=5):
    hist = game_history.get(team, [])
    if len(hist) < window * 2:
        return "neutral"
    recent = sum(g["result"] for g in hist[-window:]) / window
    older  = sum(g["result"] for g in hist[-window*2:-window]) / window
    if recent > older + 0.15:
        return "up"
    elif recent < older - 0.15:
        return "down"
    return "neutral"


# ── Pythagorean / Efficiency ──────────────────────────────────────────────────

def nba_pythagorean(pts_for, pts_against, exp=NBA_PYTH_EXP):
    """Basketball Pythagorean win expectation."""
    if pts_for + pts_against == 0:
        return 0.5
    pf_exp = pts_for ** exp
    pa_exp = pts_against ** exp
    return pf_exp / (pf_exp + pa_exp)


def compute_nba_efficiency(standings):
    """
    Estimate NBA offensive/defensive rating from points data.
    standings: {team: {points_for, points_against, games_played}}
    Returns {team: {off_rating, def_rating, net_rating, pace, pyth}}
    """
    efficiency = {}
    league_avg_pts = 115.0  # ~115 ppg NBA average

    for team, s in standings.items():
        pf = s.get("points_for", league_avg_pts * 82)
        pa = s.get("points_against", league_avg_pts * 82)
        gp = max(s.get("games_played", 1), 1)

        ppg_off = pf / gp
        ppg_def = pa / gp

        off_rating = ppg_off
        def_rating = ppg_def
        net_rating = off_rating - def_rating

        efficiency[team] = {
            "off_rating": round(off_rating, 2),
            "def_rating": round(def_rating, 2),
            "net_rating": round(net_rating, 2),
            "pace": 100.0,  # default; ESPN doesn't expose pace directly here
            "pyth": round(nba_pythagorean(pf, pa), 4),
        }

    return efficiency


def nba_efficiency_predict(team_a, team_b, efficiency_data, is_home_a=True, neutral=False):
    """Predict based on net rating difference."""
    eff_a = efficiency_data.get(team_a, {})
    eff_b = efficiency_data.get(team_b, {})
    net_a = eff_a.get("net_rating", 0.0)
    net_b = eff_b.get("net_rating", 0.0)
    pyth_a = eff_a.get("pyth", 0.5)
    pyth_b = eff_b.get("pyth", 0.5)

    hfa_elo = 0.0 if neutral else (NBA_HFA if is_home_a else -NBA_HFA)
    elo_from_net = 1500 + net_a * 10 + hfa_elo
    elo_opp = 1500 + net_b * 10
    eff_prob = _expected_score(elo_from_net, elo_opp)

    pyth_elo_a = 1500 + (pyth_a - 0.5) * 400 + hfa_elo
    pyth_elo_b = 1500 + (pyth_b - 0.5) * 400
    pyth_prob = _expected_score(pyth_elo_a, pyth_elo_b)

    return {
        "eff_prob": round(float(np.clip(eff_prob, 0.01, 0.99)), 4),
        "pyth_prob": round(float(np.clip(pyth_prob, 0.01, 0.99)), 4),
        "net_rating_a": net_a,
        "net_rating_b": net_b,
    }


# ── Rest / Travel ─────────────────────────────────────────────────────────────

def nba_travel_distance(team_a, team_b, is_home_a=True):
    away = team_b if is_home_a else team_a
    home = team_a if is_home_a else team_b
    c_away = NBA_TEAM_COORDS.get(away)
    c_home = NBA_TEAM_COORDS.get(home)
    if not c_away or not c_home:
        return 0.0
    try:
        return float(geodesic(c_away, c_home).miles)
    except Exception:
        return 0.0


def nba_rest_adjustment(rest_days, opp_rest_days):
    """
    NBA rest adjustment. Back-to-back = 1 rest day.
    Returns ELO adjustment for team (positive = advantage).
    """
    adj = 0.0
    if rest_days <= 1:     # back-to-back
        adj -= 20.0
    elif rest_days >= 3:   # 3+ days rest
        adj += 10.0
    # Relative
    diff = rest_days - opp_rest_days
    adj += diff * 2.0
    return adj


def nba_travel_adjustment(distance_miles):
    if distance_miles < 500:
        return 0.0
    elif distance_miles < 1500:
        return -5.0
    elif distance_miles < 2500:
        return -10.0
    else:
        return -15.0


# ── Bayesian ratings ──────────────────────────────────────────────────────────

def nba_update_bayesian(games_list, elo_dict, initial_sigma=75.0):
    """Simple Bayesian Normal-Normal update for NBA team strengths."""
    ratings = {t: {"mu": elo_dict.get(t, NBA_INITIAL_ELO), "sigma": initial_sigma}
               for t in NBA_TEAMS}

    for game in sorted(games_list, key=lambda g: g.get("date", "")):
        team_a = game.get("team_a")
        team_b = game.get("team_b")
        if not team_a or not team_b:
            continue
        for t in (team_a, team_b):
            if t not in ratings:
                ratings[t] = {"mu": NBA_INITIAL_ELO, "sigma": initial_sigma}

        score_a = float(game.get("score_a", 0))
        score_b = float(game.get("score_b", 0))
        margin = score_a - score_b  # positive = team_a won

        mu_a = ratings[team_a]["mu"]
        mu_b = ratings[team_b]["mu"]
        sig_a = ratings[team_a]["sigma"]
        sig_b = ratings[team_b]["sigma"]

        expected_margin = (mu_a - mu_b) / 25.0
        surprise = margin - expected_margin

        ratings[team_a]["mu"] = mu_a + 0.1 * surprise
        ratings[team_b]["mu"] = mu_b - 0.1 * surprise
        ratings[team_a]["sigma"] = max(sig_a * 0.99, 30.0)
        ratings[team_b]["sigma"] = max(sig_b * 0.99, 30.0)

    # Ensure all teams present
    for t in NBA_TEAMS:
        if t not in ratings:
            ratings[t] = {"mu": NBA_INITIAL_ELO, "sigma": initial_sigma}
        ratings[t]["uncertainty_band_a"] = ratings[t]["sigma"]
        ratings[t]["uncertainty_band_b"] = ratings[t]["sigma"]

    return ratings


def nba_bayes_predict(team_a, team_b, bayesian_ratings, is_home_a=True, neutral=False):
    r_a = bayesian_ratings.get(team_a, {"mu": NBA_INITIAL_ELO, "sigma": 75.0})
    r_b = bayesian_ratings.get(team_b, {"mu": NBA_INITIAL_ELO, "sigma": 75.0})
    hfa = 0.0 if neutral else (NBA_HFA if is_home_a else -NBA_HFA)
    elo_diff = (r_a["mu"] + hfa) - r_b["mu"]
    prob = _expected_score(r_a["mu"] + hfa, r_b["mu"])
    return {
        "bayesian_prob": round(float(prob), 4),
        "mu_a": r_a["mu"], "sigma_a": r_a["sigma"],
        "mu_b": r_b["mu"], "sigma_b": r_b["sigma"],
        "uncertainty_band_a": r_a["sigma"],
        "uncertainty_band_b": r_b["sigma"],
    }


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def nba_simulate_game(mu_home, mu_away, sig_home, sig_away,
                      is_home_a=True, neutral=False, n=10000):
    """Monte Carlo for NBA — σ ~12 pts noise."""
    rng = np.random.default_rng()
    hfa = 0.0 if neutral else (NBA_HFA if is_home_a else -NBA_HFA)

    str_home = rng.normal(mu_home + hfa, sig_home + 10, n)
    str_away = rng.normal(mu_away, sig_away + 10, n)
    noise = rng.normal(0, NBA_MC_NOISE, n)
    margin = (str_home - str_away) / 25.0 + noise

    win_prob = float(np.mean(margin > 0))
    exp_margin = float(np.mean(margin))
    prob_5plus  = float(np.mean(margin > 5))
    prob_10plus = float(np.mean(margin > 10))
    prob_15plus = float(np.mean(margin > 15))

    pcts = np.percentile(margin, [5, 15, 25, 50, 75, 85, 95]).tolist()

    return {
        "win_prob": round(win_prob, 4),
        "exp_margin": round(exp_margin, 2),
        "prob_5plus":  round(prob_5plus, 4),
        "prob_10plus": round(prob_10plus, 4),
        "prob_15plus": round(prob_15plus, 4),
        "margin_percentiles": [round(p, 1) for p in pcts],
    }


# ── Injury model ──────────────────────────────────────────────────────────────

def _player_tier(player_name):
    if player_name in NBA_SUPERSTAR_PLAYERS:
        return "superstar"
    if player_name in NBA_ALLSTAR_PLAYERS:
        return "allstar"
    return "starter"  # default


NBA_STATUS_MULTIPLIERS = {
    "out": 1.00, "injured reserve": 1.00, "ir": 1.00,
    "doubtful": 0.75, "questionable": 0.40, "limited": 0.25,
    "probable": 0.10, "day-to-day": 0.30,
}


def nba_injury_impact(team_injuries):
    """Compute NBA injury impact for a single team."""
    total_penalty = 0.0
    key_players_out = []

    for p in team_injuries:
        name = p.get("player", "Unknown")
        status = p.get("status", "").lower().strip()
        mult = next((v for k, v in NBA_STATUS_MULTIPLIERS.items() if k in status), 0.0)
        tier = _player_tier(name)
        base_penalty = NBA_TIER_PENALTIES.get(tier, 15.0)
        contribution = base_penalty * mult
        total_penalty += contribution

        if mult >= 0.75:
            key_players_out.append({
                "player": name,
                "tier": tier,
                "status": p.get("status", ""),
                "elo_impact": round(contribution, 1),
            })

    capped = min(total_penalty, MAX_NBA_INJURY_PENALTY)
    return {
        "elo_penalty": round(capped, 2),
        "impact_score": round(capped / MAX_NBA_INJURY_PENALTY, 4),
        "key_players_out": key_players_out,
        "total_players": len(team_injuries),
    }


def compute_all_nba_team_impacts(injuries):
    return {team: nba_injury_impact(pl) for team, pl in injuries.items()}


def nba_injury_elo_adjustment(home_team, away_team, injury_impacts):
    home_pen = injury_impacts.get(home_team, {}).get("elo_penalty", 0.0)
    away_pen = injury_impacts.get(away_team, {}).get("elo_penalty", 0.0)
    return -home_pen, -away_pen


# ── Ensemble ──────────────────────────────────────────────────────────────────

NBA_DEFAULT_WEIGHTS = {
    "logistic": 0.30, "xgboost": 0.25, "elo": 0.20,
    "pythagorean": 0.15, "efficiency": 0.10,
}


def nba_ensemble_predict(logistic_prob, xgb_prob, elo_prob,
                          pyth_prob, eff_prob, weights=None):
    if weights is None:
        weights = NBA_DEFAULT_WEIGHTS

    w_log  = weights.get("logistic", 0.30)
    w_xgb  = weights.get("xgboost", 0.25)
    w_elo  = weights.get("elo", 0.20)
    w_pyth = weights.get("pythagorean", 0.15)
    w_eff  = weights.get("efficiency", 0.10)

    if xgb_prob is None:
        total = w_log + w_elo + w_pyth + w_eff
        if total == 0:
            return 0.5
        prob = (w_log * logistic_prob + w_elo * elo_prob +
                w_pyth * pyth_prob + w_eff * eff_prob) / total
    else:
        total = w_log + w_xgb + w_elo + w_pyth + w_eff
        if total == 0:
            return 0.5
        prob = (w_log * logistic_prob + w_xgb * xgb_prob +
                w_elo * elo_prob + w_pyth * pyth_prob +
                w_eff * eff_prob) / total

    return round(float(np.clip(prob, 0.01, 0.99)), 4)


# ── Logistic regression features ─────────────────────────────────────────────

def build_nba_features(games_list, elo_dict, game_history, efficiency_data):
    """
    Build feature matrix for NBA logistic regression / XGBoost.
    Returns (X, y) numpy arrays.
    """
    rows, labels = [], []
    for game in sorted(games_list, key=lambda g: g.get("date", "")):
        team_a = game.get("team_a")
        team_b = game.get("team_b")
        score_a = float(game.get("score_a", 0))
        score_b = float(game.get("score_b", 0))
        is_home_a = game.get("is_home_a", True)
        neutral = float(game.get("neutral", False))

        if score_a == 0 and score_b == 0:
            continue

        elo_a = elo_dict.get(team_a, NBA_INITIAL_ELO)
        elo_b = elo_dict.get(team_b, NBA_INITIAL_ELO)
        hfa = NBA_HFA if (is_home_a and not neutral) else 0.0
        elo_diff = (elo_a + hfa) - elo_b

        eff_a = efficiency_data.get(team_a, {})
        eff_b = efficiency_data.get(team_b, {})
        net_a = eff_a.get("net_rating", 0.0)
        net_b = eff_b.get("net_rating", 0.0)
        off_a = eff_a.get("off_rating", 115.0)
        off_b = eff_b.get("off_rating", 115.0)
        def_a = eff_a.get("def_rating", 115.0)
        def_b = eff_b.get("def_rating", 115.0)
        pyth_a = eff_a.get("pyth", 0.5)
        pyth_b = eff_b.get("pyth", 0.5)

        hist_a = game_history.get(team_a, [])
        hist_b = game_history.get(team_b, [])
        form_a = (sum(g["result"] for g in hist_a[-5:]) / len(hist_a[-5:])) if hist_a else 0.5
        form_b = (sum(g["result"] for g in hist_b[-5:]) / len(hist_b[-5:])) if hist_b else 0.5

        row = [
            elo_diff,
            float(is_home_a and not neutral),
            net_a - net_b,
            off_a - off_b,
            def_a - def_b,
            pyth_a - pyth_b,
            form_a - form_b,
        ]
        rows.append(row)
        labels.append(1 if score_a > score_b else 0)

    if not rows:
        return np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=np.int32)

    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int32)


def nba_abbrev_norm(abbrev):
    return ESPN_NBA_TO_ABBREV.get(abbrev.upper(), abbrev.upper())
