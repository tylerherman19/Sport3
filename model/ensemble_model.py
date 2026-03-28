"""
Ensemble Model — Systems 9, 11, 12
XGBoost classifier, Hierarchical team model, weighted ensemble.

FIX (Issue 6): DEFAULT_WEIGHTS updated to reduce combined ELO+Pythagorean
weight from 0.35 to 0.25. Both models are derived from score differentials
and are correlated; giving them 35% combined weight double-counted that
signal on top of logistic/XGBoost which already use both as features.
New defaults: logistic=0.35, xgboost=0.25, elo=0.15, pythagorean=0.10,
efficiency=0.15. These defaults are used as fallback only — learn_ensemble_weights()
(called in update_data.py) will override them with empirically optimized values.
"""

import numpy as np
import pandas as pd
from scipy.special import expit

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


XGB_FEATURE_COLS = [
    "elo_diff",
    "home_field",
    "rest_days_diff",
    "pythagorean_diff",
    "travel_diff",
    "last5_win_rate_diff",
    "offensive_rating_diff",
    "defensive_rating_diff",
    "pace_diff",
    "third_down_conv_diff",
    "red_zone_eff_diff",
    "penalty_yards_diff",
    "time_of_possession_diff",
    "passing_eff_diff",
    "rushing_eff_diff",
    "pass_def_eff_diff",
    "rush_def_eff_diff",
]

# Reduced combined ELO+Pythagorean weight from 0.35 -> 0.25 to avoid
# double-counting score-differential signal already captured by logistic/XGBoost.
# logistic weight raised from 0.30 -> 0.35 and efficiency from 0.10 -> 0.15
# to compensate. These are fallback defaults; learn_ensemble_weights() produces
# empirically optimized values that override these at runtime.
DEFAULT_WEIGHTS = {
    "logistic": 0.35,
    "xgboost": 0.25,
    "elo": 0.15,
    "pythagorean": 0.10,
    "efficiency": 0.15,
}


def build_xgb_features(df, elo_dict, game_history, efficiency_data, pythagorean_data, advanced_stats=None):
    """Build extended feature matrix for XGBoost.

    FIX: rest_days_diff and travel_diff are now computed from actual game dates
    and team coordinates rather than hardcoded 0.0.
    """
    from model.elo_model import recent_form
    from math import sin, cos, sqrt, atan2, radians

    def _haversine(c1, c2):
        R = 3958.8
        lat1, lon1 = radians(c1[0]), radians(c1[1])
        lat2, lon2 = radians(c2[0]), radians(c2[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    try:
        from model.efficiency_model import TEAM_COORDS
    except ImportError:
        TEAM_COORDS = {}

    rows = []
    labels = []
    df = df.dropna(subset=["score1", "score2", "elo1_pre", "elo2_pre"])
    df = df.sort_values("date").reset_index(drop=True)

    if advanced_stats is None:
        advanced_stats = {}

    # Track last game date per team for rest calculation
    team_last_date: dict = {}

    for _, row in df.iterrows():
        team1 = str(row["team1"])
        team2 = str(row["team2"])
        neutral = int(row.get("neutral", 0))
        game_date_str = str(row.get("date", ""))

        elo1 = float(row.get("elo1_pre", 1500))
        elo2 = float(row.get("elo2_pre", 1500))
        hfa = 65.0 if not neutral else 0.0
        elo_diff = (elo1 + hfa) - elo2

        pyth1 = pythagorean_data.get(team1, {}).get("pyth", 0.5)
        pyth2 = pythagorean_data.get(team2, {}).get("pyth", 0.5)

        eff1 = efficiency_data.get(team1, {}).get("net_eff", 0.0)
        eff2 = efficiency_data.get(team2, {}).get("net_eff", 0.0)

        off1 = efficiency_data.get(team1, {}).get("off_eff", 1.0)
        off2 = efficiency_data.get(team2, {}).get("off_eff", 1.0)
        def1 = efficiency_data.get(team1, {}).get("def_eff", 1.0)
        def2 = efficiency_data.get(team2, {}).get("def_eff", 1.0)

        lr1 = recent_form(game_history, team1)
        lr2 = recent_form(game_history, team2)

        adv1 = advanced_stats.get(team1, {})
        adv2 = advanced_stats.get(team2, {})

        # Compute actual rest days from game dates
        try:
            from datetime import datetime as _dt
            gd = _dt.strptime(game_date_str[:10], "%Y-%m-%d").date()
            last1 = team_last_date.get(team1)
            last2 = team_last_date.get(team2)
            rest1 = (gd - last1).days if last1 else 7
            rest2 = (gd - last2).days if last2 else 7
            rest_days_diff = float(rest1 - rest2)
            team_last_date[team1] = gd
            team_last_date[team2] = gd
        except (ValueError, TypeError):
            rest_days_diff = 0.0

        # Compute travel distance: away team (team2) from home city to team1's city
        coord1 = TEAM_COORDS.get(team1)
        coord2 = TEAM_COORDS.get(team2)
        if coord1 and coord2 and not neutral:
            travel_diff = _haversine(coord2, coord1)
        else:
            travel_diff = 0.0

        pass1 = efficiency_data.get(team1, {}).get("passing_eff", 1.0)
        pass2 = efficiency_data.get(team2, {}).get("passing_eff", 1.0)
        rush1 = efficiency_data.get(team1, {}).get("rushing_eff", 1.0)
        rush2 = efficiency_data.get(team2, {}).get("rushing_eff", 1.0)
        pdef1 = efficiency_data.get(team1, {}).get("pass_def_eff", 1.0)
        pdef2 = efficiency_data.get(team2, {}).get("pass_def_eff", 1.0)
        rdef1 = efficiency_data.get(team1, {}).get("rush_def_eff", 1.0)
        rdef2 = efficiency_data.get(team2, {}).get("rush_def_eff", 1.0)

        # turnover_diff and turnover_rate_diff removed — always 0.0 (Issue 3 fix)
        feature = [
            elo_diff,
            float(not neutral),
            rest_days_diff,   # computed from actual game dates
            pyth1 - pyth2,
            travel_diff,      # computed from team coordinates
            lr1 - lr2,
            off1 - off2,
            def1 - def2,
            float(adv1.get("pace", 0)) - float(adv2.get("pace", 0)),
            float(adv1.get("third_down_pct", 0)) - float(adv2.get("third_down_pct", 0)),
            float(adv1.get("red_zone_pct", 0)) - float(adv2.get("red_zone_pct", 0)),
            float(adv1.get("penalty_yards", 0)) - float(adv2.get("penalty_yards", 0)),
            float(adv1.get("time_of_possession", 0)) - float(adv2.get("time_of_possession", 0)),
            pass1 - pass2,
            rush1 - rush2,
            pdef1 - pdef2,
            rdef1 - rdef2,
        ]
        assert len(feature) == len(XGB_FEATURE_COLS), \
            f"build_xgb_features: Feature length mismatch: {len(feature)} vs {len(XGB_FEATURE_COLS)}"

        score1 = float(row["score1"])
        score2 = float(row["score2"])
        outcome = 1 if score1 > score2 else 0

        rows.append(feature)
        labels.append(outcome)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


def train_xgboost(X, y):
    """Train XGBoost classifier with early stopping on a held-out validation set."""
    if not HAS_XGB:
        return None, None
    if len(X) < 20 or len(np.unique(y)) < 2:
        # Not enough samples/classes for a stable boosted model.
        return None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reserve last 20% of (time-ordered) data for early-stopping validation
    split = max(1, int(len(X_scaled) * 0.8))
    if split >= len(X_scaled):
        return None, None
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    y_train, y_val = y[:split], y[split:]
    if len(y_val) == 0 or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return None, None

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, scaler


def predict_xgboost(matchups, model, scaler, elo_dict, game_history,
                    efficiency_data, pythagorean_data, advanced_stats=None):
    """Predict XGBoost probabilities for a list of matchups."""
    if model is None or not HAS_XGB:
        return [{
            "game_id": m.get("game_id", ""),
            "xgb_prob": None
        } for m in matchups]

    if advanced_stats is None:
        advanced_stats = {}

    from model.elo_model import recent_form
    results = []

    for m in matchups:
        team_a = m["team_a"]
        team_b = m["team_b"]
        is_home = m.get("is_home_a", True)
        neutral = m.get("neutral", False)

        elo_a = elo_dict.get(team_a, 1500.0)
        elo_b = elo_dict.get(team_b, 1500.0)
        hfa = 65.0 if (is_home and not neutral) else 0.0
        elo_diff = (elo_a + hfa) - elo_b

        pyth_a = pythagorean_data.get(team_a, {}).get("pyth", 0.5)
        pyth_b = pythagorean_data.get(team_b, {}).get("pyth", 0.5)

        eff_a = efficiency_data.get(team_a, {}).get("net_eff", 0.0)
        eff_b = efficiency_data.get(team_b, {}).get("net_eff", 0.0)

        off_a = efficiency_data.get(team_a, {}).get("off_eff", 1.0)
        off_b = efficiency_data.get(team_b, {}).get("off_eff", 1.0)
        def_a = efficiency_data.get(team_a, {}).get("def_eff", 1.0)
        def_b = efficiency_data.get(team_b, {}).get("def_eff", 1.0)

        lr_a = recent_form(game_history, team_a)
        lr_b = recent_form(game_history, team_b)

        adv_a = advanced_stats.get(team_a, {})
        adv_b = advanced_stats.get(team_b, {})

        pass_a = efficiency_data.get(team_a, {}).get("passing_eff", 1.0)
        pass_b = efficiency_data.get(team_b, {}).get("passing_eff", 1.0)
        rush_a = efficiency_data.get(team_a, {}).get("rushing_eff", 1.0)
        rush_b = efficiency_data.get(team_b, {}).get("rushing_eff", 1.0)
        pdef_a = efficiency_data.get(team_a, {}).get("pass_def_eff", 1.0)
        pdef_b = efficiency_data.get(team_b, {}).get("pass_def_eff", 1.0)
        rdef_a = efficiency_data.get(team_a, {}).get("rush_def_eff", 1.0)
        rdef_b = efficiency_data.get(team_b, {}).get("rush_def_eff", 1.0)

        # turnover_diff and turnover_rate_diff removed — always 0.0 (Issue 3 fix)
        feature = np.array([[
            elo_diff,
            float(is_home and not neutral),
            float(m.get("rest_diff", 0)),
            pyth_a - pyth_b,
            float(m.get("travel_diff", 0)),
            lr_a - lr_b,
            off_a - off_b,
            def_a - def_b,
            float(adv_a.get("pace", 0)) - float(adv_b.get("pace", 0)),
            float(adv_a.get("third_down_pct", 0)) - float(adv_b.get("third_down_pct", 0)),
            float(adv_a.get("red_zone_pct", 0)) - float(adv_b.get("red_zone_pct", 0)),
            float(adv_a.get("penalty_yards", 0)) - float(adv_b.get("penalty_yards", 0)),
            float(adv_a.get("time_of_possession", 0)) - float(adv_b.get("time_of_possession", 0)),
            pass_a - pass_b,
            rush_a - rush_b,
            pdef_a - pdef_b,
            rdef_a - rdef_b,
        ]], dtype=np.float32)
        assert feature.shape[1] == len(XGB_FEATURE_COLS), \
            f"predict_xgboost: Feature length mismatch: {feature.shape[1]} vs {len(XGB_FEATURE_COLS)}"

        X_scaled = scaler.transform(feature)
        prob = float(model.predict_proba(X_scaled)[0, 1])

        results.append({
            "game_id": m.get("game_id", f"{team_a}_vs_{team_b}"),
            "xgb_prob": round(prob, 4)
        })

    return results


def ensemble_predict(logistic_prob, xgb_prob, elo_prob, pyth_prob, eff_prob, weights=None):
    """
    System 12: Weighted ensemble of all models.
    Returns final ensemble probability.
    Handles None for both logistic_prob and xgb_prob independently (4-case structure).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    w_log  = weights.get("logistic", 0.35)
    w_xgb  = weights.get("xgboost", 0.25)
    w_elo  = weights.get("elo", 0.15)
    w_pyth = weights.get("pythagorean", 0.10)
    w_eff  = weights.get("efficiency", 0.15)

    if logistic_prob is None and xgb_prob is None:
        # Neither ML model available — use physics models only
        total = w_elo + w_pyth + w_eff
        if total < 1e-9:
            return 0.5
        prob = (w_elo * elo_prob + w_pyth * pyth_prob + w_eff * eff_prob) / total

    elif logistic_prob is None:
        # XGBoost available, logistic not
        total = w_xgb + w_elo + w_pyth + w_eff
        if total < 1e-9:
            return 0.5
        prob = (w_xgb * xgb_prob + w_elo * elo_prob +
                w_pyth * pyth_prob + w_eff * eff_prob) / total

    elif xgb_prob is None:
        # Logistic available, XGBoost not
        total = w_log + w_elo + w_pyth + w_eff
        if total < 1e-9:
            return 0.5
        prob = (w_log * logistic_prob + w_elo * elo_prob +
                w_pyth * pyth_prob + w_eff * eff_prob) / total

    else:
        # Both available — full ensemble
        total = w_log + w_xgb + w_elo + w_pyth + w_eff
        if total < 1e-9:
            return 0.5
        prob = (w_log * logistic_prob + w_xgb * xgb_prob + w_elo * elo_prob +
                w_pyth * pyth_prob + w_eff * eff_prob) / total

    return round(float(np.clip(prob, 0.01, 0.99)), 4)


def learn_ensemble_weights(sub_probs, actuals, weight_keys=None):
    """
    Learn optimal ensemble weights empirically by minimising log-loss on
    historical predictions using scipy.optimize.

    Parameters
    ----------
    sub_probs : list[dict]
        Each dict maps model name -> probability for the home team.
        Keys should be a subset of: 'elo', 'pyth', 'eff', 'log', 'xgb'.
    actuals : list[int]
        1 if home team won, 0 otherwise.
    weight_keys : list[str], optional
        Ordered list of model keys to include. Defaults to all present in sub_probs[0].

    Returns
    -------
    dict[str, float]  -- normalised weights summing to 1.0
    """
    from scipy.optimize import minimize

    if not sub_probs or not actuals:
        return {}

    if weight_keys is None:
        weight_keys = [k for k in sub_probs[0].keys() if sub_probs[0][k] is not None]

    # Build matrix: rows=games, cols=models
    P = np.array([
        [float(row.get(k, 0.5) or 0.5) for k in weight_keys]
        for row in sub_probs
    ], dtype=np.float64)
    y = np.array(actuals, dtype=np.float64)

    def neg_log_loss(w):
        w = np.clip(w, 1e-6, None)
        w = w / w.sum()
        probs = np.clip(P @ w, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))

    n = len(weight_keys)
    w0 = np.ones(n) / n
    result = minimize(neg_log_loss, w0, method="L-BFGS-B",
                      bounds=[(0.0, 1.0)] * n)
    w_opt = np.clip(result.x, 0.0, None)
    w_norm = w_opt / w_opt.sum() if w_opt.sum() > 0 else w0
    return {k: round(float(v), 4) for k, v in zip(weight_keys, w_norm)}


def kelly_criterion(model_prob, market_prob, bankroll_fraction=1.0):
    """
    Kelly Criterion bet sizing.
    f = (bp - q) / b where b = odds - 1, p = model_prob, q = 1 - model_prob
    market_prob already has vig removed.
    """
    if market_prob <= 0 or market_prob >= 1:
        return 0.0

    # Decimal odds from market prob
    decimal_odds = 1.0 / market_prob
    b = decimal_odds - 1.0
    p = model_prob
    q = 1.0 - p

    kelly = (b * p - q) / b if b > 0 else 0.0
    kelly = max(0.0, kelly)

    # Quarter Kelly for conservative sizing
    return round(float(kelly * 0.25 * bankroll_fraction), 4)


def american_to_prob(american_odds):
    """Convert American odds to implied probability."""
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100.0)
    else:
        return 100.0 / (american_odds + 100.0)


def remove_vig(prob_a, prob_b):
    """Remove vig by normalizing probabilities."""
    total = prob_a + prob_b
    if total == 0:
        return 0.5, 0.5
    return prob_a / total, prob_b / total
