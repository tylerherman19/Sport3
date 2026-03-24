"""
NFL Ensemble Weight Learner — called from scripts/update_data.py step 7b.

FIX (Issue 5): learn_ensemble_weights() was never called in the NFL pipeline.
This module is called once per pipeline run to find the optimal ensemble weights
by minimizing log-loss on the last 500 historical FTE games. Weights are saved
to data/ensemble_weights_nfl.json and reloaded on subsequent runs.
"""

import json
import logging
from datetime import datetime
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

    # Load previously saved weights if cache is fresh (< 30 days old); relearn otherwise
    nfl_weights = dict(default_weights)
    if weights_path.exists():
        try:
            file_age_days = (
                datetime.now() -
                datetime.fromtimestamp(weights_path.stat().st_mtime)
            ).days
            if file_age_days < 30:
                nfl_weights = json.loads(weights_path.read_text())
                log.info(
                    f"Loaded cached NFL weights ({file_age_days}d old): "
                    f"{nfl_weights}"
                )
                return nfl_weights
            else:
                log.info(
                    f"NFL weights cache is {file_age_days}d old — relearning"
                )
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

        # Build per-season running totals over full FTE history to avoid future
        # information leakage: each historical game is evaluated with pythagorean
        # and efficiency ratings computed only from games played *before* it.
        _all_done = fte_df.dropna(subset=["score1", "score2"]).sort_values(["season", "date"])
        _running = {}   # {(season, team): {pf, pa, gp}}
        _snap = {}      # {(date, team1, team2): {team: {pf, pa, gp}}}
        for _, _r in _all_done.iterrows():
            _s = _r.get("season", 0)
            _rh, _ra = str(_r["team1"]), str(_r["team2"])
            _rk = (str(_r["date"]), _rh, _ra)
            _snap[_rk] = {
                _rh: dict(_running.get((_s, _rh), {"pf": 0.0, "pa": 0.0, "gp": 0})),
                _ra: dict(_running.get((_s, _ra), {"pf": 0.0, "pa": 0.0, "gp": 0})),
            }
            for _tm, _pfc, _pac in [(_rh, "score1", "score2"), (_ra, "score2", "score1")]:
                _tk = (_s, _tm)
                if _tk not in _running:
                    _running[_tk] = {"pf": 0.0, "pa": 0.0, "gp": 0}
                _running[_tk]["pf"] += float(_r[_pfc])
                _running[_tk]["pa"] += float(_r[_pac])
                _running[_tk]["gp"] += 1

        def _pyth_from_snap(ts):
            """Pythagorean win % from pre-game running season totals."""
            if ts.get("gp", 0) == 0:
                return 0.5
            pf = max(ts.get("pf", 1.0), 1.0)
            pa = max(ts.get("pa", 1.0), 1.0)
            return (pf ** 2.37) / (pf ** 2.37 + pa ** 2.37)

        def _eff_elo_from_snap(ts, _lgppg=22.0):
            """Efficiency ELO equivalent from pre-game running season totals."""
            gp = ts.get("gp", 0)
            if gp == 0:
                return 1500.0
            ppg_off = ts.get("pf", 0.0) / gp
            ppg_def = ts.get("pa", 0.0) / gp
            off_eff = ppg_off / _lgppg
            def_eff = _lgppg / max(ppg_def, 0.1)
            return 1500.0 + (off_eff - def_eff) * 200.0

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

            # Look up the pre-game running totals for this specific historical matchup
            _row_snap = _snap.get((str(_row.get("date", "")), _home, _away), {})
            _pyth_h = _pyth_from_snap(_row_snap.get(_home, {}))
            _pyth_a = _pyth_from_snap(_row_snap.get(_away, {}))
            _pyth_elo_h = 1500.0 + (_pyth_h - 0.5) * 400.0
            _pyth_elo_a = 1500.0 + (_pyth_a - 0.5) * 400.0
            _pyth_prob = _elo_es(_pyth_elo_h + _hfa, _pyth_elo_a)

            _eff_h = _eff_elo_from_snap(_row_snap.get(_home, {}))
            _eff_a = _eff_elo_from_snap(_row_snap.get(_away, {}))
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
