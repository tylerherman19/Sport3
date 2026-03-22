"""
model_engine.py — Math, ML models, and prediction logic for Sport3.
Contains: ELO helpers, efficiency/pythagorean, feature builders,
logistic/XGBoost training, Bayesian prediction, ensemble, and
prediction-driver generation. No HTTP fetching; no JSON I/O.
"""

import logging
from datetime import datetime, timedelta

import numpy as np

log = logging.getLogger(__name__)

# ---- NFL constants ----

NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB",  "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV",  "MIA", "MIN", "NE",  "NO",  "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF",  "TB",  "TEN", "WAS",
]

NFL_TEAM_NAMES = {
    "ARI": "Arizona Cardinals",    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",     "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",   "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",       "DEN": "Denver Broncos",
    "DET": "Detroit Lions",        "GB":  "Green Bay Packers",
    "HOU": "Houston Texans",       "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "KC":  "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams",
    "LV":  "Las Vegas Raiders",    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",    "NE":  "New England Patriots",
    "NO":  "New Orleans Saints",   "NYG": "New York Giants",
    "NYJ": "New York Jets",        "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",  "SEA": "Seattle Seahawks",
    "SF":  "San Francisco 49ers",  "TB":  "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",     "WAS": "Washington Commanders",
}


def nfl_days_since_last_game(game_history: dict, team: str, before_date) -> int:
    dates = [g["date"] for g in game_history.get(team, []) if g.get("date", "") < str(before_date)]
    if not dates:
        return 7
    try:
        last = max(dates)
        return max(0, (before_date - datetime.strptime(last, "%Y-%m-%d").date()).days)
    except ValueError:
        return 7


def extend_elo_with_espn(elo_dict: dict, game_history: dict, espn_games: list,
                          fte_cutoff_date=None, k_base: float = 20.0, hfa: float = 65.0):
    import pandas as pd
    if not espn_games:
        return elo_dict, game_history
    cutoff = pd.to_datetime(fte_cutoff_date) if fte_cutoff_date else pd.Timestamp("2024-02-12")
    from model.elo_model import expected_score, mov_multiplier
    new_games = sorted(
        [g for g in espn_games if pd.to_datetime(g["date"]) > cutoff],
        key=lambda g: g["date"],
    )
    log.info(f"Extending ELO with {len(new_games)} ESPN games after {cutoff.date()}")
    for game in new_games:
        t1, t2 = game["team1"], game["team2"]
        s1, s2 = game["score1"], game["score2"]
        neutral = game.get("neutral", 0)
        for t in (t1, t2):
            elo_dict.setdefault(t, 1500.0)
            game_history.setdefault(t, [])
        e1, e2 = elo_dict[t1], elo_dict[t2]
        hfa_adj = 0 if neutral else hfa
        adj_e1 = e1 + hfa_adj
        exp1 = expected_score(adj_e1, e2)
        actual1 = 1.0 if s1 > s2 else (0.5 if s1 == s2 else 0.0)
        point_diff = abs(s1 - s2)
        mov = mov_multiplier(point_diff, abs(adj_e1 - e2)) if point_diff > 0 else 1.0
        elo_dict[t1] = e1 + k_base * mov * (actual1 - exp1)
        elo_dict[t2] = e2 + k_base * mov * ((1 - actual1) - (1 - exp1))
        game_history[t1].append({"result": actual1, "elo_diff": adj_e1 - e2, "date": game["date"]})
        game_history[t2].append({"result": 1 - actual1, "elo_diff": e2 - adj_e1, "date": game["date"]})
    return elo_dict, game_history


def build_nfl_efficiency_data(standings: dict, fte_df) -> dict:
    from model.efficiency_model import compute_efficiency
    league_ppg = 22.0
    efficiency = {}
    for team, s in standings.items():
        pf = s.get("points_for", 350)
        pa = s.get("points_against", 350)
        gp = max(s.get("games_played", 1), 1)
        ppg_off, ppg_def = pf / gp, pa / gp
        ypp_off = 5.5 * (ppg_off / league_ppg)
        ypp_def = 5.5 * (ppg_def / league_ppg)
        efficiency[team] = {
            "ypp_offense": ypp_off, "ypp_allowed": ypp_def,
            "off_eff": 0.0, "def_eff": 0.0, "net_eff": 0.0,
            "passing_eff": 1.0 + (ppg_off - league_ppg) / league_ppg * 0.6,
            "rushing_eff": 1.0 + (ppg_off - league_ppg) / league_ppg * 0.4,
            "pass_def_eff": 1.0 - (ppg_def - league_ppg) / league_ppg * 0.6,
            "rush_def_eff": 1.0 - (ppg_def - league_ppg) / league_ppg * 0.4,
        }
    computed = compute_efficiency(
        {t: {"ypp_offense": d["ypp_offense"], "ypp_allowed": d["ypp_allowed"]} for t, d in efficiency.items()}
    )
    for team in efficiency:
        if team in computed:
            efficiency[team].update(computed[team])
    for team in NFL_TEAMS:
        efficiency.setdefault(team, {
            "ypp_offense": 5.5, "ypp_allowed": 5.5,
            "off_eff": 1.0, "def_eff": 1.0, "net_eff": 0.0, "elo_equiv": 1500.0,
            "passing_eff": 1.0, "rushing_eff": 1.0, "pass_def_eff": 1.0, "rush_def_eff": 1.0,
        })
    return efficiency


def match_odds_to_game(game: dict, odds_map: dict):
    home, away = game.get("home_name", ""), game.get("away_name", "")
    for odds in odds_map.values():
        h = odds.get("home_team_name", "")
        a = odds.get("away_team_name", "")
        if (home.lower() in h.lower() or h.lower() in home.lower()) and \
           (away.lower() in a.lower() or a.lower() in away.lower()):
            return odds
    return None


def generate_nfl_prediction_drivers(game_info, home, away, elo_dict, efficiency_data, injury_impacts, adj):
    drivers = []
    elo_diff = abs(elo_dict.get(home, 1500.0) - elo_dict.get(away, 1500.0))
    if elo_diff >= 50:
        leader = home if elo_dict.get(home, 1500.0) > elo_dict.get(away, 1500.0) else away
        drivers.append(f"ELO advantage: {leader} +{elo_diff:.0f} rating points")
    for team in (home, away):
        impact = injury_impacts.get(team, {})
        for p in impact.get("key_players_out", [])[:3]:
            drivers.append(f"{p['player']} ({team}) [{p.get('value_tier','starter').title()}] {p['status'].upper()} — −{p['elo_impact']:.0f} ELO")
        if impact.get("star_count", 0) >= 2 and impact.get("elo_penalty", 0) > 0:
            drivers.append(f"{team} star stack: {impact['star_count']}× elite players out → ×{impact['star_stack_multiplier']:.2f} (−{impact['elo_penalty']:.0f} ELO)")
    net_h = efficiency_data.get(home, {}).get("net_eff", 0.0)
    net_a = efficiency_data.get(away, {}).get("net_eff", 0.0)
    if abs(net_h - net_a) >= 0.10:
        drivers.append(f"Efficiency gap: {home} net {net_h:+.3f} vs {away} net {net_a:+.3f}")
    rest_diff = adj.get("rest_diff", 0)
    if abs(rest_diff) >= 3:
        drivers.append(f"Rest advantage: {home if rest_diff > 0 else away} has {abs(rest_diff)} extra days")
    if adj.get("travel_dist_miles", 0) >= 1500:
        drivers.append(f"Travel penalty: away travels {adj['travel_dist_miles']:.0f} miles")
    if not game_info.get("neutral", False):
        drivers.append(f"Home field: {home} +65 ELO home advantage")
    return drivers


# ---- NBA constants ----

NBA_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

NBA_TEAM_NAMES = {
    "ATL": "Atlanta Hawks",        "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",        "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",        "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",     "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",      "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",      "IND": "Indiana Pacers",
    "LAC": "LA Clippers",          "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",      "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder","ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",   "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers","SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",            "WAS": "Washington Wizards",
}

NBA_PLAYER_TIERS = {
    "LeBron James": "superstar", "Stephen Curry": "superstar", "Kevin Durant": "superstar",
    "Giannis Antetokounmpo": "superstar", "Nikola Jokic": "superstar", "Luka Doncic": "superstar",
    "Joel Embiid": "superstar", "Jayson Tatum": "superstar", "Shai Gilgeous-Alexander": "superstar",
    "Anthony Edwards": "superstar", "Victor Wembanyama": "superstar",
    "Damian Lillard": "all-star", "Devin Booker": "all-star", "Anthony Davis": "all-star",
    "Bam Adebayo": "all-star", "Tyrese Haliburton": "all-star", "Donovan Mitchell": "all-star",
    "Jalen Brunson": "all-star", "De'Aaron Fox": "all-star", "Trae Young": "all-star",
    "Darius Garland": "all-star", "Karl-Anthony Towns": "all-star", "Cade Cunningham": "all-star",
    "Zach LaVine": "all-star", "Kyrie Irving": "all-star", "Jimmy Butler": "all-star",
    "Pascal Siakam": "all-star", "Scottie Barnes": "all-star", "Jaylen Brown": "all-star",
    "Paolo Banchero": "all-star", "Franz Wagner": "all-star", "Domantas Sabonis": "all-star",
    "James Harden": "all-star", "Paul George": "all-star", "Kawhi Leonard": "all-star",
}

_NBA_TIER_VALUE = {"superstar": 2.0, "all-star": 1.5, "starter": 1.0, "rotation": 0.5}
_NBA_DEPTH_VALUE_MAP = {1: 2.0, 2: 1.2, 3: 0.7}
_NBA_PPG_LEAGUE_AVG = 11.0


def _nba_depth_to_value(depth_pos: int) -> float:
    return _NBA_DEPTH_VALUE_MAP.get(depth_pos, 0.4)


def _ppg_to_value_mult(ppg: float) -> float:
    return min(ppg / _NBA_PPG_LEAGUE_AVG, 2.5) if ppg > 0 else 0.3


def nba_value_tier_label(mult: float) -> str:
    if mult >= 1.8: return "superstar"
    if mult >= 1.3: return "all-star"
    if mult >= 0.8: return "starter"
    if mult >= 0.4: return "backup"
    return "rotation"


def build_nba_player_values(depth_charts: dict, player_ppg: dict) -> dict:
    """Priority: PPG > depth chart > static tiers."""
    values = {name: _NBA_TIER_VALUE.get(tier, 1.0) for name, tier in NBA_PLAYER_TIERS.items()}
    for name, depth_pos in depth_charts.items():
        if name not in values or depth_pos == 1:
            values[name] = _nba_depth_to_value(depth_pos)
    for name, ppg in player_ppg.items():
        values[name] = _ppg_to_value_mult(ppg)
    return values


def build_nba_efficiency_data(standings: dict) -> dict:
    teams_with_data = [s for s in standings.values() if s.get("games_played", 0) > 0]
    league_off  = np.mean([s.get("offensive_rating", 110.0) for s in teams_with_data]) if teams_with_data else 110.0
    league_def  = np.mean([s.get("defensive_rating", 110.0) for s in teams_with_data]) if teams_with_data else 110.0
    league_pace = np.mean([s.get("pace", 100.0) for s in teams_with_data]) if teams_with_data else 100.0
    efficiency  = {}
    for team in NBA_TEAMS:
        s = standings.get(team, {})
        off_rtg = s.get("offensive_rating", league_off)
        def_rtg = s.get("defensive_rating", league_def)
        pf = s.get("points_for", 0); pa = s.get("points_against", 0)
        gp = max(s.get("games_played", 1), 1)
        efficiency[team] = {
            "offensive_rating": off_rtg, "defensive_rating": def_rtg,
            "net_rating": off_rtg - def_rtg, "pace": s.get("pace", league_pace),
            "off_eff": off_rtg / max(league_off, 1), "def_eff": league_def / max(def_rtg, 1),
            "net_eff": off_rtg - def_rtg,
            "turnover_rate": 1.0 / max(s.get("assist_turnover_ratio", 1.8), 0.1),
            "three_point_rate": s.get("three_point_rate", 0.35),
            "rebound_rate": s.get("rebound_rate", 0.5),
            "free_throw_rate": s.get("free_throw_rate", 0.20),
            "ppg_for": pf / gp, "ppg_against": pa / gp,
        }
    return efficiency


def compute_nba_pythagorean(efficiency_data: dict) -> dict:
    pyth_data = {}
    for team, eff in efficiency_data.items():
        pf = max(eff.get("ppg_for", 110.0), 1.0)
        pa = max(eff.get("ppg_against", 110.0), 1.0)
        exp = 13.91  # NBA canonical Morey exponent (was 16.5, NFL-calibrated)
        pyth_data[team] = {"pyth": round((pf ** exp) / ((pf ** exp) + (pa ** exp)), 4)}
    return pyth_data


def nba_days_since_last_game(game_history: dict, team: str, today) -> int:
    hist  = game_history.get(team, [])
    dates = [g["date"] for g in hist if g.get("date") and g["date"] <= str(today)]
    if not dates:
        return 7
    try:
        last = datetime.strptime(max(dates), "%Y-%m-%d").date()
        return max(0, (today - last).days)
    except ValueError:
        return 7


def build_nba_features(games, elo_dict, game_history, efficiency_data, pyth_data):
    from model.nba_elo import nba_recent_form
    features, targets = [], []
    for game in games:
        t1, t2 = game.get("team1", ""), game.get("team2", "")
        s1, s2 = game.get("score1", 0), game.get("score2", 0)
        neutral = game.get("neutral", 0)
        if not t1 or not t2 or s1 == s2:
            continue
        e1, e2 = elo_dict.get(t1, 1500.0), elo_dict.get(t2, 1500.0)
        hfa = 0 if neutral else 100.0
        eff1, eff2 = efficiency_data.get(t1, {}), efficiency_data.get(t2, {})
        pyth1 = pyth_data.get(t1, {}).get("pyth", 0.5)
        pyth2 = pyth_data.get(t2, {}).get("pyth", 0.5)
        game_date_str = game.get("date", "")
        try:
            game_date = datetime.strptime(game_date_str, "%Y-%m-%d").date()
            rest1 = nba_days_since_last_game(game_history, t1, game_date)
            rest2 = nba_days_since_last_game(game_history, t2, game_date)
            rest_diff = float(rest1 - rest2)
            b2b1 = 1.0 if rest1 <= 1 else 0.0
            b2b2 = 1.0 if rest2 <= 1 else 0.0
        except (ValueError, TypeError):
            rest_diff, b2b1, b2b2 = 0.0, 0.0, 0.0
        features.append([
            (e1 + hfa) - e2, hfa, rest_diff,
            (1500 + (pyth1 - 0.5) * 400) - (1500 + (pyth2 - 0.5) * 400),
            eff1.get("net_rating", 0.0) - eff2.get("net_rating", 0.0),
            eff1.get("offensive_rating", 110.0) - eff2.get("offensive_rating", 110.0),
            eff2.get("defensive_rating", 110.0) - eff1.get("defensive_rating", 110.0),
            eff1.get("pace", 100.0) - eff2.get("pace", 100.0),
            eff2.get("turnover_rate", 0.5) - eff1.get("turnover_rate", 0.5),
            eff1.get("three_point_rate", 0.35) - eff2.get("three_point_rate", 0.35),
            eff1.get("rebound_rate", 0.5) - eff2.get("rebound_rate", 0.5),
            eff1.get("free_throw_rate", 0.20) - eff2.get("free_throw_rate", 0.20),
            nba_recent_form(game_history, t1) - nba_recent_form(game_history, t2),
            b2b1 - b2b2,  # difference, not home-only (matches inference)
        ])
        targets.append(1 if s1 > s2 else 0)
    return (np.array(features) if features else np.array([]).reshape(0, 14), np.array(targets))


def train_nba_logistic(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import TimeSeriesSplit
    if len(X) < 50:
        return None, None, None
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    model  = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(X_s, y)
    probs = np.zeros(len(y))
    for tr, va in TimeSeriesSplit(n_splits=5).split(X_s):
        m = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        m.fit(X_s[tr], y[tr])
        probs[va] = m.predict_proba(X_s[va])[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(probs, y)
    return model, scaler, cal


def predict_nba_logistic(home_feat, model, scaler, calibrator):
    if model is None or scaler is None:
        return None
    try:
        raw = model.predict_proba(scaler.transform([home_feat]))[0][1]
        return float(calibrator.transform([raw])[0]) if calibrator else raw
    except Exception as e:
        log.warning(f"NBA logistic prediction failed: {e}")
        return None


def train_nba_xgboost(X, y):
    try:
        import xgboost as xgb
        from sklearn.preprocessing import StandardScaler
        if len(X) < 50:
            return None, None
        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)
        model  = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss", random_state=42,
        )
        model.fit(X_s, y)
        return model, scaler
    except ImportError:
        return None, None
    except Exception as e:
        log.error(f"NBA XGBoost training failed: {e}")
        return None, None


def evaluate_nba_model(model, scaler, calibrator, X, y) -> dict:
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
    if model is None or len(X) == 0:
        return {"log_loss": None, "brier_score": None, "auc": None}
    try:
        probs = model.predict_proba(scaler.transform(X))[:, 1]
        if calibrator:
            probs = calibrator.transform(probs)
        return {"log_loss": round(log_loss(y, probs), 4),
                "brier_score": round(brier_score_loss(y, probs), 4),
                "auc": round(roc_auc_score(y, probs), 4)}
    except Exception as e:
        log.warning(f"NBA model evaluation failed: {e}")
        return {"log_loss": None, "brier_score": None, "auc": None}


def nba_ensemble_predict(elo_prob, pyth_prob, eff_prob,
                          log_prob=None, xgb_prob=None, weights=None) -> float:
    if weights is None:
        weights = {"elo": 0.25, "pyth": 0.20, "eff": 0.15, "log": 0.25, "xgb": 0.15}
    # Handle None models by excluding their weights (avoids 50/50 contamination)
    if xgb_prob is None and log_prob is None:
        total_w = weights["elo"] + weights["pyth"] + weights["eff"]
        val = (weights["elo"]*elo_prob + weights["pyth"]*pyth_prob +
               weights["eff"]*eff_prob) / max(total_w, 1e-9)
    elif xgb_prob is None:
        total_w = weights["elo"] + weights["pyth"] + weights["eff"] + weights["log"]
        val = (weights["elo"]*elo_prob + weights["pyth"]*pyth_prob +
               weights["eff"]*eff_prob + weights["log"]*log_prob) / max(total_w, 1e-9)
    elif log_prob is None:
        total_w = weights["elo"] + weights["pyth"] + weights["eff"] + weights["xgb"]
        val = (weights["elo"]*elo_prob + weights["pyth"]*pyth_prob +
               weights["eff"]*eff_prob + weights["xgb"]*xgb_prob) / max(total_w, 1e-9)
    else:
        total_w = sum(weights.values())
        val = (weights["elo"]*elo_prob + weights["pyth"]*pyth_prob + weights["eff"]*eff_prob
               + weights["log"]*log_prob + weights["xgb"]*xgb_prob) / max(total_w, 1e-9)
    return float(max(0.01, min(0.99, val)))


def compute_nba_h2h(home, away, historical_games, last_n=10) -> dict:
    ms = [g for g in historical_games
          if (g["team1"] == home and g["team2"] == away) or
             (g["team1"] == away and g["team2"] == home)]
    ms = sorted(ms, key=lambda g: g.get("date", ""))
    home_wins = sum(1 for g in ms if (g["team1"] == home and g["score1"] > g["score2"]) or
                                      (g["team2"] == home and g["score2"] > g["score1"]))
    recent = ms[-last_n:]
    last_n_list = [{"date": g.get("date", ""),
                    "winner": home if (g["team1"]==home and g["score1"]>g["score2"]) or
                                      (g["team2"]==home and g["score2"]>g["score1"]) else away,
                    "margin": abs(g["score1"] - g["score2"])} for g in recent]
    return {"home_wins": home_wins, "away_wins": len(ms)-home_wins,
            "total_meetings": len(ms), "last_n": last_n_list}


def compute_nba_streak(game_history, team) -> dict:
    hist = game_history.get(team, [])
    if not hist:
        return {"type": "N", "count": 0}
    streak_type = "W" if hist[-1].get("result", 0) == 1 else "L"
    count = 0
    for g in reversed(hist):
        if (g.get("result", 0) == 1 and streak_type == "W") or \
           (g.get("result", 0) == 0 and streak_type == "L"):
            count += 1
        else:
            break
    return {"type": streak_type, "count": count}


def generate_nba_prediction_drivers(game_info, home, away, elo_dict, efficiency_data, injury_impacts, adj):
    drivers = []
    elo_diff = abs(elo_dict.get(home, 1500.0) - elo_dict.get(away, 1500.0))
    if elo_diff >= 75:
        leader = home if elo_dict.get(home, 1500.0) > elo_dict.get(away, 1500.0) else away
        drivers.append(f"ELO advantage: {leader} +{elo_diff:.0f} rating points")
    for team in (home, away):
        impact = injury_impacts.get(team, {})
        for p in impact.get("key_players_out", [])[:3]:
            drivers.append(f"{p['player']} ({team}) [{p.get('value_tier','starter').title()}] {p['status'].upper()} — −{p['elo_impact']:.0f} ELO")
        if impact.get("star_count", 0) >= 2 and impact.get("elo_penalty", 0) > 0:
            drivers.append(f"{team} star stack: {impact['star_count']}× elite out → ×{impact['star_stack_multiplier']:.2f} (−{impact['elo_penalty']:.0f} ELO)")
    home_eff, away_eff = efficiency_data.get(home, {}), efficiency_data.get(away, {})
    net_h, net_a = home_eff.get("net_rating", 0.0), away_eff.get("net_rating", 0.0)
    if abs(net_h - net_a) >= 3.0:
        drivers.append(f"Net rating gap: {home} {net_h:+.1f} vs {away} {net_a:+.1f}")
    off_diff = home_eff.get("offensive_rating", 110) - away_eff.get("offensive_rating", 110)
    if abs(off_diff) >= 3.0:
        drivers.append(f"Offensive rating: {home if off_diff > 0 else away} leads by {abs(off_diff):.1f} pts/100")
    if adj.get("b2b_home"):
        drivers.append(f"Back-to-back: {home} on zero days rest")
    if adj.get("b2b_away"):
        drivers.append(f"Back-to-back: {away} on zero days rest")
    rest_diff = adj.get("rest_diff", 0)
    if abs(rest_diff) >= 2 and not adj.get("b2b_home") and not adj.get("b2b_away"):
        drivers.append(f"Rest advantage: {home if rest_diff > 0 else away} has {abs(rest_diff)} extra days")
    if adj.get("travel_dist_miles", 0) >= 1500:
        drivers.append(f"Travel: away travels {adj['travel_dist_miles']:.0f} miles")
    if not game_info.get("neutral", False):
        drivers.append(f"Home court: {home} +100 ELO advantage")
    return drivers
