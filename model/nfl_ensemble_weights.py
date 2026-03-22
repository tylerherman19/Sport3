"""
NFL Ensemble Weight Learner — called from scripts/update_data.py step 7b.

FIX (Issue 5): learn_ensemble_weights() was never called in the NFL pipeline.
This module is called once per pipeline run to find the optimal ensemble weights
by minimizing log-loss on the last 500 historical FTE games. Weights are saved
to data/ensemble_weights_nfl.json and reloaded on subsequent runs.
"""

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def learn_nfl_weights(fte_df, pythagorean_data, efficiency_data, default_weights):
    """
    Learn optimal NFL ensemble weights from historical FTE data.

    Parameters
    ----------
    fte_df : pd.DataFrame
        FiveThirtyEight NFL ELO dataframe (must have elo1_pre, elo2_pre, score1, score2).
    pythagorean_data : dict
        {team: {pyth: float}} from compute_pythagorean().
    efficiency_data : dict
        {team: {elo_equiv: float}} from compute_efficiency().
    default_weights : dict
        Fallback weights if learning fails or insufficient data.

    Returns
    -------
    dict[str, float]  — weight dict summing to 1.0
    """
    weights_path = DATA_DIR / "ensemble_weights_nfl.json"

    # Load previously saved weights if available (avoids full recompute each run)
    nfl_weights = dict(default_weights)
    if weights_path.exists():
        try:
            nfl_weights = json.loads(weights_path.read_text())
            log.info(f"Loaded persisted NFL ensemble weights: {nfl_weights}")
            return nfl_weights
        except Exception:
            pass

    try:
        from model.ensemble_model import learn_ensemble_weights
        from model.elo_model import expected_score as _elo_es

        completed_fte = fte_df.dropna(subset=["score1", "score2", "elo1_pre", "elo2_pre"])
        completed_fte = completed_fte.sort_values("date").tail(500)

        if len(completed_fte) < 50:
            log.warning("Not enough completed games for NFL weight learning — using defaults")
            return nfl_weights

        sub_probs_hist = []
        actuals_hist = []

        for _, _row in completed_fte.iterrows():
            _neutral = int(_row.get("neutral", 0))
            _hfa = 65.0 if not _neutral else 0.0

            _elo_h = float(_row.get("elo1_pre", 1500)) + _hfa
            _elo_a = float(_row.get("elo2_pre", 1500))
            _elo_prob = _elo_es(_elo_h, _elo_a)

            _home = str(_row["team1"])
            _away = str(_row["team2"])
            _pyth_h = pythagorean_data.get(_home, {}).get("pyth", 0.5)
            _pyth_a = pythagorean_data.get(_away, {}).get("pyth", 0.5)
            _pyth_elo_h = 1500.0 + (_pyth_h - 0.5) * 400.0
            _pyth_elo_a = 1500.0 + (_pyth_a - 0.5) * 400.0
            _pyth_prob = _elo_es(_pyth_elo_h + _hfa, _pyth_elo_a)

            _eff_h = efficiency_data.get(_home, {}).get("elo_equiv", 1500.0)
            _eff_a = efficiency_data.get(_away, {}).get("elo_equiv", 1500.0)
            _eff_prob = _elo_es(_eff_h + _hfa, _eff_a)

            sub_probs_hist.append({
                "elo": _elo_prob, "pyth": _pyth_prob, "eff": _eff_prob,
                "log": 0.5,  # logistic not available per historical row
                "xgb": 0.5,
            })
            actuals_hist.append(1 if float(_row["score1"]) > float(_row["score2"]) else 0)

        # Learn weights on the three rule-based models; keep log/xgb at fixed share
        learned = learn_ensemble_weights(
            sub_probs_hist, actuals_hist, weight_keys=["elo", "pyth", "eff"]
        )
        if learned:
            nfl_weights["elo"]         = round(learned.get("elo",  0.15) * 0.6, 4)
            nfl_weights["pythagorean"] = round(learned.get("pyth", 0.10) * 0.6, 4)
            nfl_weights["efficiency"]  = round(learned.get("eff",  0.15) * 0.6, 4)
            # Normalise all five weights to sum to 1
            _total = sum(nfl_weights.values())
            nfl_weights = {k: round(v / _total, 4) for k, v in nfl_weights.items()}
            weights_path.write_text(json.dumps(nfl_weights, indent=2))
            log.info(f"Learned NFL ensemble weights: {nfl_weights}")

    except Exception as e:
        log.warning(f"NFL ensemble weight learning skipped: {e}")

    return nfl_weights
