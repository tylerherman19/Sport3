"""
Logistic Regression Model — System 1
Trains on FiveThirtyEight historical data with isotonic calibration.

FIX (Issue 2): Multicollinearity between ELO and Pythagorean addressed by:
  1. Removing "pythagorean_diff" from FEATURE_COLS — ELO already encodes the
     same points-differential signal. Pythagorean is kept as a standalone
     ensemble model input but should not be double-counted inside logistic.
  2. Reducing regularization strength from C=1.0 to C=0.1 (stronger L2 penalty)
     to shrink any remaining correlated coefficient inflation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


# pythagorean_diff removed — it is highly correlated with elo_diff (both
# derived from score differentials) and caused coefficient inflation.
# It remains a standalone ensemble component in ensemble_model.py.
FEATURE_COLS = [
    "elo_diff",
    "home_field_advantage",
    "rest_days_diff",
    "turnover_diff_adjusted",
    "travel_distance_diff",
    "last5_win_rate_diff",
    "offensive_rating_diff",
    "defensive_rating_diff",
]


def build_features(df, elo_dict, game_history, efficiency_data, pythagorean_data):
    """
    Build feature matrix from historical data.
    df: FiveThirtyEight game-level dataframe
    Returns X (features), y (outcomes)
    """
    rows = []
    labels = []

    df = df.dropna(subset=["score1", "score2", "elo1_pre", "elo2_pre"])
    df = df.sort_values("date").reset_index(drop=True)

    for _, row in df.iterrows():
        team1 = row["team1"]
        team2 = row["team2"]
        neutral = int(row.get("neutral", 0))

        elo1 = float(row.get("elo1_pre", 1500))
        elo2 = float(row.get("elo2_pre", 1500))
        elo_diff = elo1 - elo2 + (65 if not neutral else 0)

        eff1 = efficiency_data.get(team1, {}).get("net_eff", 0.0)
        eff2 = efficiency_data.get(team2, {}).get("net_eff", 0.0)

        off1 = efficiency_data.get(team1, {}).get("off_eff", 1.0)
        off2 = efficiency_data.get(team2, {}).get("off_eff", 1.0)
        def1 = efficiency_data.get(team1, {}).get("def_eff", 1.0)
        def2 = efficiency_data.get(team2, {}).get("def_eff", 1.0)

        rest_diff = 0.0
        travel_diff = 0.0
        turnover_diff = 0.0

        from model.elo_model import recent_form
        lr1 = recent_form(game_history, team1)
        lr2 = recent_form(game_history, team2)
        last5_diff = lr1 - lr2

        # pythagorean_diff intentionally omitted — see module docstring
        feature = [
            elo_diff,
            1.0 if not neutral else 0.0,
            rest_diff,
            turnover_diff,
            travel_diff,
            last5_diff,
            off1 - off2,
            def1 - def2,
        ]

        score1 = row["score1"]
        score2 = row["score2"]
        outcome = 1 if score1 > score2 else 0

        rows.append(feature)
        labels.append(outcome)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


def train_logistic(X, y):
    """Train logistic regression with isotonic calibration.

    C=0.1 (stronger L2 regularization vs previous C=1.0) to mitigate
    coefficient inflation from any residual correlation among features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        C=0.1,          # stronger regularization (was C=1.0)
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    )
    model.fit(X_scaled, y)

    tscv = TimeSeriesSplit(n_splits=5)
    oof_probs = np.zeros(len(y))
    for train_idx, val_idx in tscv.split(X_scaled):
        m = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs", random_state=42)
        m.fit(X_scaled[train_idx], y[train_idx])
        oof_probs[val_idx] = m.predict_proba(X_scaled[val_idx])[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_probs, y)

    return model, scaler, calibrator


def evaluate_model(model, scaler, calibrator, X, y):
    """Compute log loss, brier score, AUC on full data."""
    X_scaled = scaler.transform(X)
    raw_probs = model.predict_proba(X_scaled)[:, 1]
    cal_probs = calibrator.transform(raw_probs)

    return {
        "log_loss": float(log_loss(y, cal_probs)),
        "brier_score": float(brier_score_loss(y, cal_probs)),
        "auc": float(roc_auc_score(y, cal_probs))
    }


def calibration_buckets(model, scaler, calibrator, X, y, n_buckets=10):
    """Generate calibration curve data."""
    X_scaled = scaler.transform(X)
    raw_probs = model.predict_proba(X_scaled)[:, 1]
    cal_probs = calibrator.transform(raw_probs)

    buckets = []
    edges = np.linspace(0, 1, n_buckets + 1)
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (cal_probs >= lo) & (cal_probs < hi)
        if mask.sum() > 0:
            pred_mean = float(cal_probs[mask].mean())
            act_mean = float(y[mask].mean())
            count = int(mask.sum())
        else:
            pred_mean = float((lo + hi) / 2)
            act_mean = float((lo + hi) / 2)
            count = 0
        buckets.append({
            "bucket": f"{int(lo*100)}-{int(hi*100)}%",
            "predicted": round(pred_mean, 3),
            "actual": round(act_mean, 3),
            "count": count
        })
    return buckets


def predict_matchups(matchups, model, scaler, calibrator,
                     elo_dict, game_history, efficiency_data, pythagorean_data):
    """
    matchups: list of dicts with team_a, team_b, is_home_a, rest_diff, travel_diff, turnover_diff
    Returns list of {game_id, logistic_prob}
    """
    results = []
    from model.elo_model import recent_form

    for m in matchups:
        team_a = m["team_a"]
        team_b = m["team_b"]
        is_home = m.get("is_home_a", True)
        neutral = m.get("neutral", False)

        elo_a = elo_dict.get(team_a, 1500.0)
        elo_b = elo_dict.get(team_b, 1500.0)
        elo_diff = elo_a - elo_b + (65 if is_home and not neutral else 0)

        off_a = efficiency_data.get(team_a, {}).get("off_eff", 1.0)
        off_b = efficiency_data.get(team_b, {}).get("off_eff", 1.0)
        def_a = efficiency_data.get(team_a, {}).get("def_eff", 1.0)
        def_b = efficiency_data.get(team_b, {}).get("def_eff", 1.0)

        lr_a = recent_form(game_history, team_a)
        lr_b = recent_form(game_history, team_b)

        # pythagorean_diff intentionally omitted — see module docstring
        feature = np.array([[
            elo_diff,
            1.0 if is_home and not neutral else 0.0,
            float(m.get("rest_diff", 0)),
            float(m.get("turnover_diff", 0)),
            float(m.get("travel_diff", 0)),
            lr_a - lr_b,
            off_a - off_b,
            def_a - def_b,
        ]], dtype=np.float32)

        X_scaled = scaler.transform(feature)
        raw_prob = model.predict_proba(X_scaled)[0, 1]
        cal_prob = float(calibrator.transform([raw_prob])[0])

        results.append({
            "game_id": m.get("game_id", f"{team_a}_vs_{team_b}"),
            "logistic_prob": round(cal_prob, 4)
        })

    return results


def historical_accuracy_by_year(df, model, scaler, calibrator,
                                elo_dict, game_history, efficiency_data, pythagorean_data):
    """Returns per-year accuracy for model performance section."""
    results = []
    years = sorted(df["season"].unique())[-10:]

    for year in years:
        year_df = df[df["season"] == year].dropna(subset=["score1", "score2"])
        if len(year_df) < 5:
            continue

        X_year, y_year = build_features(
            year_df, elo_dict, game_history, efficiency_data, pythagorean_data
        )
        if len(X_year) == 0:
            continue

        try:
            X_scaled = scaler.transform(X_year)
            raw = model.predict_proba(X_scaled)[:, 1]
            cal = calibrator.transform(raw)
            preds = (cal >= 0.5).astype(int)
            acc = float((preds == y_year).mean())
        except Exception:
            acc = 0.0

        results.append({"year": int(year), "accuracy": round(acc, 4)})

    return results
