# NFL accuracy program — research log

Harness: `wf_elo.py` + `wf_qb.py` (walk-forward, no lookahead; every game predicted before its
result touches any rating). Data: nflverse games.csv (1999-2025) + stats_player_week_2014-2025 +
draft_picks. Scored on games with usable Vegas closing moneylines, ties excluded.

## Round 1 (2026-09-06)

Fair baseline (production params, ELO-only), 2016-2025, N=2750:
acc 0.6324 / ll 0.6393 / brier 0.2243. Vegas: acc 0.6647 / ll 0.6086 / br 0.2106.
(For reference, the repo's own frozen-ensemble backtest scored 0.602 — conservative because
ratings freeze at season start; this harness updates week-by-week.)

One-at-a-time (2016-2025):
- hfa 65 -> 48:  +0.76 acc (0.6400). hfa 40/55 similar to 48.
- k_decay off:   +0.5  (0.6375)
- k 24/28:       +0.3
- regress 0.25:  +0.5; 0.45/0.55 hurt
- playoff x1.2:  no change -> rejected
- bye_rest +25:  -0.3 -> rejected (538's bye edge does not replicate 2016-25)

Best combo: hfa48 + no k-decay + k20 + regress 0.33 => 0.6415 (2016-2025)

QB adjustment (538 VALUE formula, rolling 0.1 QB / 0.05 team, defense-adjusted,
draft-position rookie init, veteran 25% reversion, scored 2017-2025, N=2485, Vegas 0.6660):
- no-QB same window: 0.6414 / ll 0.6363
- QB on, mult 2.0:   0.6459 / ll 0.6258 / br 0.2183  (mult grid: 1.7 acc 0.6479, 2.0 best ll, 3.3 (538's) 0.6414, 4.0 worse)
- travel +4/1000mi:  no help -> rejected
- 538 winner-perspective MOV multiplier: +0.1 acc (0.6471), ll flat
- era-rolling HFA (trailing 10-season home win rate): 0.6483 / ll 0.6255 / br 0.2181
- 2014-15 warmup files added

Best stack end of round 1 (ELO-only): hfa era-rolling (~48) + k20 flat + regress .33
+ QB adj mult 2.0 + 538 MOV = **acc 0.6483, ll 0.6255, brier 0.2181** vs Vegas 0.6660.
Gain over fair baseline: +1.6 pts accuracy. 2025 season: model 0.662 vs Vegas 0.658 (beat the market).

## Queue for round 2
- Wire winners into production (elo_model.py + update_data.py; QB state needs a persisted
  qb_ratings.json and weekly-stats ingestion for the live season incl. 2026 file)
- Full ensemble fair harness: LR/XGB retrained per season on prior seasons only, predicting
  with week-by-week current state; re-optimize ensemble weights against walk-forward outputs
  (current DEFAULT_WEIGHTS were tuned on leaky evaluation; frozen ensemble backtested WORSE
  than ELO alone, 0.602 vs 0.632)
- Candidate: add qb_diff as LR/XGB feature
- Candidate: isotonic recalibration of final prob on walk-forward residuals

## Round 1.5 — ML layer check (2017-2025, N=2485)
Stacked models trained per season on strictly-prior seasons, features from the improved
walk state (elo_diff, qb_diff, hfa, rest_diff, last5_diff):
- LR stack: 0.6455 / ll 0.6252 — does NOT beat improved ELO alone (0.6483 / 0.6255)
- XGB stack: 0.6398 — worse
- 0.7 ELO + 0.3 XGB blend: 0.6491 acc (noise-level +2 games), ll worse
Conclusion: once ELO carries QB adjustments + era HFA, ML stacking adds no signal.
Production's ensemble architecture (weights tuned on leaky evals) is likely diluting,
not helping. Round 2 = make production's core the improved ELO and re-derive any blend
weight walk-forward, or drop the blend.
