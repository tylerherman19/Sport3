"""
Ensemble Model — Systems 9, 11, 12
XGBoost classifier, Hierarchical team model, weighted ensemble.
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
    "net_efficiency_diff",
    "turnover_diff",
    "travel_diff",
    "last5_win_rate_diff",
    "offensive_rating_diff",
    "defensive_rating_diff",
    "pace_diff",
    "turnover_rate_diff",
    "third_down_conv_diff",
    "red_zone_eff_diff",
    "penalty_yards_diff",
    "time_of_possession_diff",
]

DEFAULT_WEIGHTS = {
    "logistic": 0.27,
    "xgboost": 0.22,
    "elo": 0.20,
    "pythagorean": 0.12,
    "efficiency": 0.09,
    "srs": 0.10,  # Simple Rating System (Massey/PFR-inspired)
}


def build_xgb_features(df, elo_dict, game_history, efficiency_data, pythagorean_data, advanced_stats=None):
    """Build extended feature matrix for XGBoost.

    Features must exactly match predict_xgboost() to avoid training/inference mismatch.
    Rest days are computed from game date deltas; travel dist from team locations.
    Turnover diff is 0.0 in both training and prediction (no historical data).
    """
    from model.elo_model import recent_form
    from model.efficiency_model import travel_distance

    rows = []
    labels = []
    df = df.dropna(subset=["score1", "score2", "elo1_pre", "elo2_pre"])
    df = df.sort_values("date").reset_index(drop=True)

    if advanced_stats is None:
        advanced_stats = {}

    last_game_date = {}  # {team: pd.Timestamp} — track last game per team

    for _, row in df.iterrows():
        team1 = str(row["team1"])
        team2 = str(row["team2"])
        neutral = int(row.get("neutral", 0))

        # Compute rest days from date delta (default 7 if first game)
        game_date = pd.to_datetime(row["date"]) if row.get("date") is not None else None
        rest1 = 7.0
        rest2 = 7.0
        if game_date is not None:
            if team1 in last_game_date:
                rest1 = max(1.0, float((game_date - last_game_date[team1]).days))
            if team2 in last_game_date:
                rest2 = max(1.0, float((game_date - last_game_date[team2]).days))
        rest_diff = rest1 - rest2

        # Update last game date for both teams
        if game_date is not None:
            last_game_date[team1] = game_date
            last_game_date[team2] = game_date

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

        # Travel distance: team1 is home in FTE data
        try:
            dist = travel_distance(team1, team2, is_home_a=True)
        except Exception:
            dist = 0.0

        feature = [
            elo_diff,
            float(not neutral),
            rest_diff,   # computed from game date deltas (was hardcoded 0.0)
            pyth1 - pyth2,
            eff1 - eff2,
            0.0,         # turnover_diff — not in FTE historical data; 0.0 in prediction too
            dist,        # computed from team locations (was hardcoded 0.0)
            lr1 - lr2,
            off1 - off2,
            def1 - def2,
            float(adv1.get("pace", 0)) - float(adv2.get("pace", 0)),
            float(adv1.get("turnover_rate", 0)) - float(adv2.get("turnover_rate", 0)),
            float(adv1.get("third_down_pct", 0)) - float(adv2.get("third_down_pct", 0)),
            float(adv1.get("red_zone_pct", 0)) - float(adv2.get("red_zone_pct", 0)),
            float(adv1.get("penalty_yards", 0)) - float(adv2.get("penalty_yards", 0)),
            float(adv1.get("time_of_possession", 0)) - float(adv2.get("time_of_possession", 0)),
        ]

        assert len(feature) == len(XGB_FEATURE_COLS), \
            f"Feature length mismatch: {len(feature)} vs {len(XGB_FEATURE_COLS)}"

        score1 = float(row["score1"])
        score2 = float(row["score2"])
        outcome = 1 if score1 > score2 else 0

        rows.append(feature)
        labels.append(outcome)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


def train_xgboost(X, y):
    """Train XGBoost classifier."""
    if not HAS_XGB:
        return None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_scaled, y)
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

        feature = np.array([[
            elo_diff,
            float(is_home and not neutral),
            float(m.get("rest_diff", 0)),
            pyth_a - pyth_b,
            eff_a - eff_b,
            float(m.get("turnover_diff", 0)),
            float(m.get("travel_diff", 0)),
            lr_a - lr_b,
            off_a - off_b,
            def_a - def_b,
            float(adv_a.get("pace", 0)) - float(adv_b.get("pace", 0)),
            float(adv_a.get("turnover_rate", 0)) - float(adv_b.get("turnover_rate", 0)),
            float(adv_a.get("third_down_pct", 0)) - float(adv_b.get("third_down_pct", 0)),
            float(adv_a.get("red_zone_pct", 0)) - float(adv_b.get("red_zone_pct", 0)),
            float(adv_a.get("penalty_yards", 0)) - float(adv_b.get("penalty_yards", 0)),
            float(adv_a.get("time_of_possession", 0)) - float(adv_b.get("time_of_possession", 0)),
        ]], dtype=np.float32)

        X_scaled = scaler.transform(feature)
        prob = float(model.predict_proba(X_scaled)[0, 1])

        results.append({
            "game_id": m.get("game_id", f"{team_a}_vs_{team_b}"),
            "xgb_prob": round(prob, 4)
        })

    return results


def ensemble_predict(logistic_prob, xgb_prob, elo_prob, pyth_prob, eff_prob,
                     srs_prob=None, weights=None):
    """
    System 12: Weighted ensemble of all models.
    Returns final ensemble probability.
    Models with None probability are excluded and weights renormalized.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Build list of (weight_key, prob) pairs, skip None probs
    candidates = [
        ("logistic",    logistic_prob),
        ("xgboost",     xgb_prob),
        ("elo",         elo_prob),
        ("pythagorean", pyth_prob),
        ("efficiency",  eff_prob),
        ("srs",         srs_prob),
    ]

    total_w = 0.0
    weighted_sum = 0.0
    for key, prob in candidates:
        if prob is None:
            continue
        w = weights.get(key, 0.0)
        total_w += w
        weighted_sum += w * prob

    if total_w == 0:
        return 0.5

    prob = weighted_sum / total_w
    return round(float(np.clip(prob, 0.01, 0.99)), 4)


def compute_adaptive_weights(model_brier_scores, base_weights=None):
    """
    Compute ensemble weights inversely proportional to each model's Brier score.
    Lower Brier score = better calibration = higher weight.
    Falls back to base_weights for any model with no Brier score available.
    Inspired by FiveThirtyEight's performance-weighted ensemble methodology.
    """
    if base_weights is None:
        base_weights = DEFAULT_WEIGHTS.copy()

    valid = {k: v for k, v in model_brier_scores.items() if v is not None and v > 0}
    if not valid:
        return base_weights.copy()

    # Invert Brier scores: 1/score so better models get higher raw weight
    inv_scores = {k: 1.0 / v for k, v in valid.items()}
    total_inv = sum(inv_scores.values())

    # Start from base weights; replace weights for models with Brier data
    weights = base_weights.copy()
    for k, inv_v in inv_scores.items():
        if k in weights:
            # Blend: 50% adaptive, 50% base to avoid extreme swings on small samples
            adaptive_w = inv_v / total_inv
            weights[k] = 0.5 * adaptive_w + 0.5 * base_weights.get(k, adaptive_w)

    # Renormalize to sum to 1.0
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: round(v / total_w, 4) for k, v in weights.items()}

    return weights


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
