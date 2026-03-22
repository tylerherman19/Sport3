"""
Model Engine Module - Sport3 Refactor
Mathematical logic, ML model training, and prediction computation for NFL and NBA.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date

log = logging.getLogger(__name__)


# ─── Re-exports from ensemble_model (for betting odds) ──────────────────────────────────

def american_to_prob(american_odds: float) -> float:
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return (-american_odds) / (-american_odds + 100)


def remove_vig(home_prob: float, away_prob: float):
    total = home_prob + away_prob
    return home_prob / total, away_prob / total


def kelly_criterion(win_prob: float, market_prob: float) -> float:
    if market_prob <= 0 or market_prob >= 1:
        return 0.0
    b = (1 / market_prob) - 1
    q = 1 - win_prob
    kelly = (b * win_prob - q) / b
    return round(max(0.0, min(kelly, 1.0)), 4)


# ─── NFL Model Engine ────────────────────────────────────────────────────────────────────

_DEPTH_VALUE_MAP = {1: 1.0, 2: 0.55, 3: 0.25}


def depth_to_value_mult(depth_pos: int) -> float:
    return _DEPTH_VALUE_MAP.get(depth_pos, 0.15)


def build_player_values(depth_charts: dict) -> dict:
    return {name: depth_to_value_mult(pos) for name, pos in depth_charts.items()}


def value_tier_label(mult: float) -> str:
    if mult >= 1.8:
        return "superstar"
    if mult >= 1.3:
        return "all-star"
    if mult >= 0.8:
        return "starter"
    if mult >= 0.4:
        return "backup"
    return "rotation"


def extend_elo_with_espn(elo_dict, game_history, espn_games,
                          fte_cutoff_date=None, k_base=20.0, hfa=65.0):
    """
    Continue ELO computation with ESPN live results after FTE dataset ends.
    Skips games already covered by FTE (before cutoff date).
    """
    if not espn_games:
        return elo_dict, game_history

    if fte_cutoff_date:
        cutoff = pd.to_datetime(fte_cutoff_date)
    else:
        cutoff = pd.Timestamp("2024-02-12")

    from model.elo_model import expected_score, mov_multiplier

    new_games = sorted(
        [g for g in espn_games if pd.to_datetime(g["date"]) > cutoff],
        key=lambda g: g["date"]
    )

    log.info(f"Extending ELO with {len(new_games)} ESPN games after {cutoff.date()}")

    for game in new_games:
        team1 = game["team1"]
        team2 = game["team2"]
        score1 = game["score1"]
        score2 = game["score2"]
        neutral = game.get("neutral", 0)

        if team1 not in elo_dict:
            elo_dict[team1] = 1500.0
            game_history[team1] = []
        if team2 not in elo_dict:
            elo_dict[team2] = 1500.0
            game_history[team2] = []

        e1 = elo_dict[team1]
        e2 = elo_dict[team2]
        hfa_adj = 0 if neutral else hfa
        adj_e1 = e1 + hfa_adj
        exp1 = expected_score(adj_e1, e2)
        exp2 = 1.0 - exp1
        actual1 = 1.0 if score1 > score2 else (0.5 if score1 == score2 else 0.0)
        actual2 = 1.0 - actual1
        point_diff = abs(score1 - score2)
        elo_diff_abs = abs(adj_e1 - e2)
        mov = mov_multiplier(point_diff, elo_diff_abs) if point_diff > 0 else 1.0
        elo_dict[team1] = e1 + k_base * mov * (actual1 - exp1)
        elo_dict[team2] = e2 + k_base * mov * (actual2 - exp2)
        game_history.setdefault(team1, []).append({"result": actual1, "elo_diff": adj_e1 - e2, "date": game["date"]})
        game_history.setdefault(team2, []).append({"result": actual2, "elo_diff": e2 - adj_e1, "date": game["date"]})

    return elo_dict, game_history


def build_team_efficiency_data(standings, fte_df, NFL_TEAMS):
    """Build efficiency data from standings (using points) and FTE for YPP."""
    from model.efficiency_model import compute_efficiency
    efficiency_data = {}
    for team, s in standings.items():
        pf = s.get("points_for", 350)
        pa = s.get("points_against", 350)
        gp = max(s.get("games_played", 1), 1)
        ppg_off = pf / gp
        ppg_def = pa / gp
        league_ppg = 22.0
        ypp_off = 5.5 * (ppg_off / league_ppg)
        ypp_def = 5.5 * (ppg_def / league_ppg)
        efficiency_data[team] = {
            "ypp_offense": ypp_off, "ypp_allowed": ypp_def,
            "off_eff": 0.0, "def_eff": 0.0, "net_eff": 0.0,
            "passing_eff": 1.0 + (ppg_off - league_ppg) / league_ppg * 0.6,
            "rushing_eff": 1.0 + (ppg_off - league_ppg) / league_ppg * 0.4,
            "pass_def_eff": 1.0 - (ppg_def - league_ppg) / league_ppg * 0.6,
            "rush_def_eff": 1.0 - (ppg_def - league_ppg) / league_ppg * 0.4,
        }
    computed = compute_efficiency(
        {t: {"ypp_offense": d["ypp_offense"], "ypp_allowed": d["ypp_allowed"]}
         for t, d in efficiency_data.items()}
    )
    for team in efficiency_data:
        if team in computed:
            efficiency_data[team].update(computed[team])
    for team in NFL_TEAMS:
        if team not in efficiency_data:
            efficiency_data[team] = {
                "ypp_offense": 5.5, "ypp_allowed": 5.5,
                "off_eff": 1.0, "def_eff": 1.0, "net_eff": 0.0, "elo_equiv": 1500.0,
                "passing_eff": 1.0, "rushing_eff": 1.0, "pass_def_eff": 1.0, "rush_def_eff": 1.0,
            }
    return efficiency_data


def generate_prediction_drivers(game_info, home, away, elo_dict, efficiency_data, injury_impacts, adj):
    """Generate a list of plain-English prediction driver strings for a game."""
    drivers = []
    home_elo = elo_dict.get(home, 1500.0)
    away_elo = elo_dict.get(away, 1500.0)
    elo_diff = abs(home_elo - away_elo)
    if elo_diff >= 50:
        leader = home if home_elo > away_elo else away
        drivers.append(f"ELO advantage: {leader} +{elo_diff:.0f} rating points")
    for team in [home, away]:
        impact = injury_impacts.get(team, {})
        for p in impact.get("key_players_out", [])[:3]:
            tier_label = p.get("value_tier", "starter").title()
            drivers.append(f"{p['player']} ({team}) [{tier_label}] {p['status'].upper()} — −{p['elo_impact']:.0f} ELO")
        star_count = impact.get("star_count", 0)
        stack_mult = impact.get("star_stack_multiplier", 1.0)
        total_penalty = impact.get("elo_penalty", 0.0)
        if star_count >= 2 and total_penalty > 0:
            drivers.append(f"{team} star stack: {star_count}× elite players out → ×{stack_mult:.2f} multiplier (combined −{total_penalty:.0f} ELO)")
    net_home = efficiency_data.get(home, {}).get("net_eff", 0.0)
    net_away = efficiency_data.get(away, {}).get("net_eff", 0.0)
    if abs(net_home - net_away) >= 0.10:
        drivers.append(f"Efficiency gap: {home} net {net_home:+.3f} vs {away} net {net_away:+.3f}")
    rest_diff = adj.get("rest_diff", 0)
    if abs(rest_diff) >= 3:
        rested = home if rest_diff > 0 else away
        drivers.append(f"Rest advantage: {rested} has {abs(rest_diff)} extra days rest")
    travel_mi = adj.get("travel_dist_miles", 0)
    if travel_mi >= 1500:
        drivers.append(f"Travel penalty: away team travels {travel_mi:.0f} miles")
    if not game_info.get("neutral", False):
        drivers.append(f"Home field: {home} +65 ELO home advantage")
    return drivers


# ─── NBA Model Engine ────────────────────────────────────────────────────────────────────

_NBA_TIER_VALUE = {"superstar": 2.0, "all-star": 1.5, "starter": 1.0, "rotation": 0.5}

NBA_PLAYER_TIERS = {
    "LeBron James": "superstar", "Stephen Curry": "superstar", "Kevin Durant": "superstar",
    "Giannis Antetokounmpo": "superstar", "Nikola Jokic": "superstar", "Luka Doncic": "superstar",
    "Joel Embiid": "superstar", "Jayson Tatum": "superstar", "Shai Gilgeous-Alexander": "superstar",
    "Anthony Edwards": "superstar", "Victor Wembanyama": "superstar",
    "Damian Lillard": "all-star", "Devin Booker": "all-star", "Anthony Davis": "all-star",
    "Bam Adebayo": "all-star", "Tyrese Haliburton": "all-star", "Donovan Mitchell": "all-star",
    "Jalen Brunson": "all-star", "De'Aaron Fox": "all-star", "Trae Young": "all-star",
    "Darius Garland": "all-star", "Alperen Sengun": "all-star", "Evan Mobley": "all-star",
    "Karl-Anthony Towns": "all-star", "Cade Cunningham": "all-star", "Zach LaVine": "all-star",
    "Kyrie Irving": "all-star", "Jimmy Butler": "all-star", "Pascal Siakam": "all-star",
    "Scottie Barnes": "all-star", "Jaren Jackson Jr.": "all-star", "Jaylen Brown": "all-star",
    "Paolo Banchero": "all-star", "Franz Wagner": "all-star", "Domantas Sabonis": "all-star",
    "James Harden": "all-star", "Paul George": "all-star", "Kawhi Leonard": "all-star",
}

_NBA_DEPTH_VALUE_MAP = {1: 2.0, 2: 1.2, 3: 0.7}
_NBA_PPG_LEAGUE_AVG = 11.0


def nba_depth_to_value(depth_pos: int) -> float:
    return _NBA_DEPTH_VALUE_MAP.get(depth_pos, 0.4)


def ppg_to_value_mult(ppg: float) -> float:
    if ppg <= 0:
        return 0.3
    return min(ppg / _NBA_PPG_LEAGUE_AVG, 2.5)


def build_nba_player_values(live_depth: dict, player_ppg: dict) -> dict:
    """Build NBA player value multipliers using PPG > depth chart > static tiers."""
    values = {name: _NBA_TIER_VALUE.get(tier, 1.0) for name, tier in NBA_PLAYER_TIERS.items()}
    for name, depth_pos in live_depth.items():
        live_val = nba_depth_to_value(depth_pos)
        if name not in values or depth_pos == 1:
            values[name] = live_val
    for name, ppg in player_ppg.items():
        values[name] = ppg_to_value_mult(ppg)
    ppg_count = len(player_ppg)
    if ppg_count > 0:
        log.info(f"NBA player values: {ppg_count} players by PPG, {len(live_depth)} by depth chart, {len(NBA_PLAYER_TIERS)} by static tier")
    else:
        log.warning("NBA PPG fetch returned 0 players — falling back to depth chart + static tiers")
    return values


def nba_value_tier_label(mult: float) -> str:
    if mult >= 1.8:
        return "superstar"
    if mult >= 1.3:
        return "all-star"
    if mult >= 0.8:
        return "starter"
    if mult >= 0.4:
        return "backup"
    return "rotation"


def haversine(a, b):
    import math
    R = 3959.0
    dLat = (b[0] - a[0]) * math.pi / 180
    dLon = (b[1] - a[1]) * math.pi / 180
    x = (math.sin(dLat / 2)) ** 2 + math.cos(a[0] * math.pi / 180) * math.cos(b[0] * math.pi / 180) * (math.sin(dLon / 2)) ** 2
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))


NBA_COORDS = {
    "ATL": [33.7490, -84.3880], "BOS": [42.3601, -71.0589], "BKN": [40.6826, -73.9754],
    "CHA": [35.2271, -80.8431], "CHI": [41.8781, -87.6298], "CLE": [41.4993, -81.6944],
    "DAL": [32.7767, -96.7970], "DEN": [39.7392, -104.9903], "DET": [42.3314, -83.0458],
    "GSW": [37.7680, -122.3877], "HOU": [29.7604, -95.3698], "IND": [39.7684, -86.1581],
    "LAC": [34.0430, -118.2673], "LAL": [34.0430, -118.2673], "MEM": [35.1495, -90.0490],
    "MIA": [25.7617, -80.1918], "MIL": [43.0436, -87.9166], "MIN": [44.9778, -93.2650],
    "NOP": [29.9511, -90.0715], "NYK": [40.7505, -73.9934], "OKC": [35.4634, -97.5151],
    "ORL": [28.5383, -81.3792], "PHI": [39.9012, -75.1720], "PHX": [33.4484, -112.0740],
    "POR": [45.5231, -122.6765], "SAC": [38.5816, -121.4944], "SAS": [29.4241, -98.4936],
    "TOR": [43.6532, -79.3832], "UTA": [40.7608, -111.8910], "WAS": [38.9072, -77.0369],
}


def nba_travel_distance(home, away):
    ch = NBA_COORDS.get(home)
    ca = NBA_COORDS.get(away)
    if ch and ca:
        return haversine(ca, ch)
    return 0.0


def days_since_last_game(game_history, team, today):
    from datetime import date as date_cls
    hist = game_history.get(team, [])
    dates = [g["date"] for g in hist if g.get("date") and g["date"] <= str(today)]
    if not dates:
        return 7
    last_date_str = max(dates)
    try:
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
        return max(0, (today - last_date).days)
    except ValueError:
        return 7


def build_nba_efficiency_data(standings, NBA_TEAMS):
    efficiency = {}
    teams_with_data = [s for s in standings.values() if s.get("games_played", 0) > 0]
    if teams_with_data:
        league_off = np.mean([s.get("offensive_rating", 110.0) for s in teams_with_data])
        league_def = np.mean([s.get("defensive_rating", 110.0) for s in teams_with_data])
        league_pace = np.mean([s.get("pace", 100.0) for s in teams_with_data])
    else:
        league_off, league_def, league_pace = 110.0, 110.0, 100.0
    for team in NBA_TEAMS:
        s = standings.get(team, {})
        off_rtg = s.get("offensive_rating", league_off)
        def_rtg = s.get("defensive_rating", league_def)
        net_rtg = off_rtg - def_rtg
        pace = s.get("pace", league_pace)
        pf = s.get("points_for", 0)
        pa = s.get("points_against", 0)
        gp = max(s.get("games_played", 1), 1)
        efficiency[team] = {
            "offensive_rating": off_rtg, "defensive_rating": def_rtg, "net_rating": net_rtg,
            "pace": pace, "off_eff": off_rtg / max(league_off, 1), "def_eff": league_def / max(def_rtg, 1),
            "net_eff": net_rtg, "turnover_rate": 1.0 / max(s.get("assist_turnover_ratio", 1.8), 0.1),
            "three_point_rate": s.get("three_point_rate", 0.35), "rebound_rate": s.get("rebound_rate", 0.5),
            "free_throw_rate": s.get("free_throw_rate", 0.20), "ppg_for": pf / gp, "ppg_against": pa / gp,
        }
    return efficiency


def compute_nba_pythagorean(efficiency_data):
    pyth_data = {}
    for team, eff in efficiency_data.items():
        pf = max(eff.get("ppg_for", 110.0), 1.0)
        pa = max(eff.get("ppg_against", 110.0), 1.0)
        exp = 16.5
        pyth = (pf ** exp) / ((pf ** exp) + (pa ** exp))
        pyth_data[team] = {"pyth": round(pyth, 4)}
    return pyth_data


def build_nba_features(games, elo_dict, game_history, efficiency_data, pyth_data):
    from model.nba_elo import nba_recent_form
    features, targets = [], []
    for game in games:
        t1 = game.get("team1", "")
        t2 = game.get("team2", "")
        s1 = game.get("score1", 0)
        s2 = game.get("score2", 0)
        neutral = game.get("neutral", 0)
        if not t1 or not t2 or s1 == s2:
            continue
        e1 = elo_dict.get(t1, 1500.0)
        e2 = elo_dict.get(t2, 1500.0)
        hfa = 0 if neutral else 100.0
        elo_diff = (e1 + hfa) - e2
        eff1 = efficiency_data.get(t1, {})
        eff2 = efficiency_data.get(t2, {})
        off_diff = eff1.get("offensive_rating", 110.0) - eff2.get("offensive_rating", 110.0)
        def_diff = eff2.get("defensive_rating", 110.0) - eff1.get("defensive_rating", 110.0)
        net_diff = eff1.get("net_rating", 0.0) - eff2.get("net_rating", 0.0)
        pace_diff = eff1.get("pace", 100.0) - eff2.get("pace", 100.0)
        to_diff = eff2.get("turnover_rate", 0.5) - eff1.get("turnover_rate", 0.5)
        three_diff = eff1.get("three_point_rate", 0.35) - eff2.get("three_point_rate", 0.35)
        reb_diff = eff1.get("rebound_rate", 0.5) - eff2.get("rebound_rate", 0.5)
        ft_diff = eff1.get("free_throw_rate", 0.20) - eff2.get("free_throw_rate", 0.20)
        pyth1 = pyth_data.get(t1, {}).get("pyth", 0.5)
        pyth2 = pyth_data.get(t2, {}).get("pyth", 0.5)
        pyth_diff = (1500 + (pyth1 - 0.5) * 400) - (1500 + (pyth2 - 0.5) * 400)
        form1 = nba_recent_form(game_history, t1)
        form2 = nba_recent_form(game_history, t2)
        form_diff = form1 - form2
        game_date_str = game.get("date", "")
        try:
            game_date = datetime.strptime(game_date_str, "%Y-%m-%d").date()
            rest1 = days_since_last_game(game_history, t1, game_date)
            rest2 = days_since_last_game(game_history, t2, game_date)
            rest_diff = float(rest1 - rest2)
            b2b1 = 1.0 if rest1 <= 1 else 0.0
        except (ValueError, TypeError):
            rest_diff = 0.0
            b2b1 = 0.0
        features.append([elo_diff, hfa, rest_diff, pyth_diff, net_diff, off_diff, def_diff,
                          pace_diff, to_diff, three_diff, reb_diff, ft_diff, form_diff, b2b1])
        targets.append(1 if s1 > s2 else 0)
    return np.array(features) if features else np.array([]).reshape(0, 14), np.array(targets)


def train_nba_logistic(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import TimeSeriesSplit
    if len(X) < 50:
        return None, None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(X_scaled, y)
    tscv = TimeSeriesSplit(n_splits=5)
    probs_oos = np.zeros(len(y))
    for train_idx, val_idx in tscv.split(X_scaled):
        m = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        m.fit(X_scaled[train_idx], y[train_idx])
        probs_oos[val_idx] = m.predict_proba(X_scaled[val_idx])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probs_oos, y)
    return model, scaler, calibrator


def predict_nba_logistic(home_feat, model, scaler, calibrator):
    if model is None or scaler is None:
        return 0.5
    try:
        x = scaler.transform([home_feat])
        raw_prob = model.predict_proba(x)[0][1]
        return float(calibrator.transform([raw_prob])[0]) if calibrator else raw_prob
    except Exception as e:
        log.warning(f"NBA logistic prediction failed: {e}")
        return 0.5


def train_nba_xgboost(X, y):
    try:
        import xgboost as xgb
        from sklearn.preprocessing import StandardScaler
        if len(X) < 50:
            return None, None
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=42,
        )
        model.fit(X_scaled, y)
        return model, scaler
    except ImportError:
        log.warning("XGBoost not available for NBA model")
        return None, None
    except Exception as e:
        log.error(f"NBA XGBoost training failed: {e}")
        return None, None


def evaluate_nba_model(model, scaler, calibrator, X, y):
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
    if model is None or len(X) == 0:
        return {"log_loss": None, "brier_score": None, "auc": None}
    try:
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]
        if calibrator:
            probs = calibrator.transform(probs)
        return {
            "log_loss": round(log_loss(y, probs), 4),
            "brier_score": round(brier_score_loss(y, probs), 4),
            "auc": round(roc_auc_score(y, probs), 4),
        }
    except Exception as e:
        log.warning(f"NBA model evaluation failed: {e}")
        return {"log_loss": None, "brier_score": None, "auc": None}


def nba_ensemble_predict(elo_prob, pyth_prob, eff_prob, log_prob=None, xgb_prob=None, weights=None):
    if weights is None:
        weights = {"elo": 0.25, "pyth": 0.20, "eff": 0.15, "log": 0.25, "xgb": 0.15}
    xgb_val = xgb_prob if xgb_prob is not None else log_prob if log_prob is not None else elo_prob
    log_val = log_prob if log_prob is not None else elo_prob
    total_w = sum(weights.values())
    val = (weights["elo"] * elo_prob + weights["pyth"] * pyth_prob + weights["eff"] * eff_prob +
           weights["log"] * log_val + weights["xgb"] * xgb_val) / total_w
    return float(max(0.01, min(0.99, val)))


def compute_h2h(home, away, historical_games, last_n=10):
    meetings = [
        g for g in historical_games
        if (g["team1"] == home and g["team2"] == away) or (g["team1"] == away and g["team2"] == home)
    ]
    meetings = sorted(meetings, key=lambda g: g.get("date", ""))
    home_wins = sum(
        1 for g in meetings
        if (g["team1"] == home and g["score1"] > g["score2"]) or
           (g["team2"] == home and g["score2"] > g["score1"])
    )
    away_wins = len(meetings) - home_wins
    recent = meetings[-last_n:]
    last_n_list = []
    for g in recent:
        h_won = (g["team1"] == home and g["score1"] > g["score2"]) or \
                (g["team2"] == home and g["score2"] > g["score1"])
        last_n_list.append({"date": g.get("date", ""), "winner": home if h_won else away, "margin": abs(g["score1"] - g["score2"])})
    return {"home_wins": home_wins, "away_wins": away_wins, "total_meetings": len(meetings), "last_n": last_n_list}


def compute_streak(game_history, team):
    hist = game_history.get(team, [])
    if not hist:
        return {"type": "N", "count": 0}
    streak_type = "W" if hist[-1].get("result", 0) == 1 else "L"
    count = 0
    for g in reversed(hist):
        result = g.get("result", 0)
        if (result == 1 and streak_type == "W") or (result == 0 and streak_type == "L"):
            count += 1
        else:
            break
    return {"type": streak_type, "count": count}


def generate_nba_prediction_drivers(game_info, home, away, elo_dict, efficiency_data, injury_impacts, adj):
    drivers = []
    home_elo = elo_dict.get(home, 1500.0)
    away_elo = elo_dict.get(away, 1500.0)
    elo_diff = abs(home_elo - away_elo)
    if elo_diff >= 75:
        leader = home if home_elo > away_elo else away
        drivers.append(f"ELO advantage: {leader} +{elo_diff:.0f} rating points")
    for team in [home, away]:
        impact = injury_impacts.get(team, {})
        for p in impact.get("key_players_out", [])[:3]:
            tier_label = p.get("value_tier", "starter").title()
            drivers.append(f"{p['player']} ({team}) [{tier_label}] {p['status'].upper()} — −{p['elo_impact']:.0f} ELO")
        star_count = impact.get("star_count", 0)
        stack_mult = impact.get("star_stack_multiplier", 1.0)
        total_penalty = impact.get("elo_penalty", 0.0)
        if star_count >= 2 and total_penalty > 0:
            drivers.append(f"{team} star stack: {star_count}× elite players out → ×{stack_mult:.2f} multiplier (combined −{total_penalty:.0f} ELO)")
    home_eff = efficiency_data.get(home, {})
    away_eff = efficiency_data.get(away, {})
    net_home = home_eff.get("net_rating", 0.0)
    net_away = away_eff.get("net_rating", 0.0)
    if abs(net_home - net_away) >= 3.0:
        drivers.append(f"Net rating gap: {home} {net_home:+.1f} vs {away} {net_away:+.1f}")
    off_diff = home_eff.get("offensive_rating", 110) - away_eff.get("offensive_rating", 110)
    if abs(off_diff) >= 3.0:
        leader = home if off_diff > 0 else away
        drivers.append(f"Offensive rating advantage: {leader} ({abs(off_diff):.1f} pts/100)")
    b2b_home = adj.get("b2b_home", False)
    b2b_away = adj.get("b2b_away", False)
    if b2b_home:
        drivers.append(f"Back-to-back: {home} playing on zero days rest (−5 rating)")
    if b2b_away:
        drivers.append(f"Back-to-back: {away} playing on zero days rest (−5 rating)")
    rest_diff = adj.get("rest_diff", 0)
    if abs(rest_diff) >= 2 and not b2b_home and not b2b_away:
        rested = home if rest_diff > 0 else away
        drivers.append(f"Rest advantage: {rested} has {abs(rest_diff)} extra days rest")
    travel_mi = adj.get("travel_dist_miles", 0)
    if travel_mi >= 1500:
        drivers.append(f"Travel: away team travels {travel_mi:.0f} miles")
    if not game_info.get("neutral", False):
        drivers.append(f"Home court: {home} +100 ELO advantage")
    return drivers
