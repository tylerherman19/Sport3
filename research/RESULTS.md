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

---

## Production ship (2026-09-06) — QB-adjusted Elo is now the model

The winning stack shipped to the live pipeline (`scripts/main.py --nfl`, the
"Update NFL Model" Actions workflow):

- `model/qb_model.py` (new): full 538-style QB overlay - per-start QB VALUE,
  rolling ratings (0.1 QB / 0.05 team / 0.1 defense-allowed), vet 25% reversion
  (10-100 starts), draft-initialized rookies, QB_MULT 2.0. Data: nflverse
  stats_player_week + draft_picks + depth_charts, cached in data/nflverse_cache.
- `model/elo_model.py`: flat K=20 (within-season decay removed - measured to
  hurt), era-rolling HFA (trailing 10 completed seasons' home win rate, ~34 pts
  for 2026), winner-perspective MOV autocorrelation adjustment, qb_map hook.
  annotate_pregame_elo now wraps compute_elo so training features can't drift
  from shipped ratings again.
- `scripts/model_engine.py` + `scripts/main.py`: same mechanics in the ESPN
  live extension; headline probability is now the pure QB-adjusted Elo
  expected score. The 12-model ensemble is demoted to display context
  (walk-forward: ensemble 60-64% vs QB-Elo 65.2%).
- `scripts/data_fetcher.py`: nflverse games loader passes game_id through
  (needed by the QB model).

Fair walk-forward verification of the SHIPPED code path (production modules,
not the lab): **0.6523 acc / 0.6251 log-loss / 0.2180 Brier**, 2017-2025,
N=2485, vs Vegas closing 0.6660 / 0.6255 / 0.2164. Final production
compute_elo ratings matched the walk-forward scorer exactly (max diff 0.0).
Sanity: SEA-NE Week 1 2026 headline 0.7068 reproduced by hand to 0.7069.

Also fixed: depth-chart schema drift (nflverse now uses pos_abb/pos_rank/dt);
explanation text no longer hardcodes +65 home-field ELO.

## Vegas line-setting logic (2026-09-07)

Research directive: absorb the LOGIC of how sportsbooks originate and move
lines, then prototype whatever transfers to free (non-odds) data. Sources:

- Yahoo Sports, veteran Vegas oddsmakers (Chris Andrews / South Point,
  Tristan Davis / BetMGM): openers come from power ratings plus context;
  sharp early action is treated as error-correction on the opener.
  https://sports.yahoo.com/how-are-point-spreads-made-for-nfl-games-veteran-vegas-oddsmakers-explain-140051148.html
- Sports Betting Primer, opening-line construction: books blend MULTIPLE
  models (scoring margin, off/def efficiency, pace, SOS, recent form, roster
  metrics) into one power rating; spread = rating diff + systematic context
  adjustments (HFA now ~1.5-2.5 pts, venue-dependent - Seattle/Denver higher;
  short weeks; cross-timezone travel; QB absence worth 3-7 pts; wind >15mph).
  Moneylines derive from spread via win-prob conversion tables (key numbers
  3 and 7). Market-making books (Circa, Pinnacle, BetCRIS) originate; retail
  books copy and shade toward their customer base's biases.
  https://sportsbettingprime.com/how-sportsbooks-set-opening-lines.html
- WagerLex, midweek line movement: lines open Sunday night; Tue-Wed is the
  sharp window (low limits, professional flow); Thu-Sun is recreational.
  A sustained half-point move propagating across books in 30-90 min = sharp
  positioning; quick reversals carry no signal. Closing line = the market's
  best estimate with maximum information; CLV is the only proven long-run
  profitability indicator.
  https://wagerlex.com/the-stack/nfl-week-line-movement-primer/

What transfers to our stack: their architecture IS our architecture - a
blended power rating (our QB-adjusted Elo), era-compressed HFA (ours is
rolling, 33.8 pts for 2026 ~= the cited 1.5-2.5 pts), a QB injury/change
adjustment (our 538-style QB overlay ~= their 3-7 pt QB swing), and
rest/travel context. Their residual edge is information timing - final
injury confirmations, weather, sharp flow between open and close - which
free feeds do not carry pre-game.

Prototypes tested on the fair walk-forward harness (QB stack, production
constants, 2017-2025, N=2485; baseline 0.6483 acc / 0.6255 ll / 0.2181 br):

- Venue-specific HFA (per-team trailing-10-season home win rate):
  0.6394 / 0.6298 / 0.2202 - REJECTED (global era HFA wins by 0.9 pts)
- Short-week penalty (<6 days rest, -10/-20/-30 Elo): 0.6483-0.6487,
  no change - REJECTED (rest differential already carries what little
  signal exists)
- West-to-east early-window penalty (PT/MT away team, <=1pm ET kickoff,
  -10/-20/-30): monotonically worse (0.6483 -> 0.6455) - REJECTED
- (Round 1 already rejected: bye-week rest bonus, per-1000-mile travel)

Verdict: none of the documented situational angles add measurable signal on
top of the shipped QB-Elo stack over 2017-2025. The ~1.4 pt accuracy gap to
the closing market (65.2 vs 66.6) is information, not methodology. Odds-data
features (review item 10) remain excluded per the user's standing decision.

---

## Week 1 2026 defect: ratings that never crossed the offseason (2026-09-08)

Reported from the live board: Miami at Las Vegas showed MIA 61.0% against a
market price of MIA 37.7% — a 23.3 pp edge on a game the market made LV a
-185 home favourite. It was not one bad game. Across all 64 priced Week 1
matchups the mean absolute model-vs-market gap was **9.25 pp**, with 14 games
past 15 pp, and every outlier ran the same direction: the model more extreme
than the market at both ends.

### Root cause

`compute_elo()` regresses ratings toward 1500 at the start of each season it
iterates. It iterates seasons **present in the dataframe**, and the loader
(`fetch_nfl_historical_games`) drops rows with no score — completed games only.
Before Week 1 of a new season there are no 2026 rows, so the loop stopped at
2025 and returned raw end-of-2025 ratings. `extend_elo_with_espn()` then had no
2026 results to apply. The shipped `data/elo_ratings.json` was literally the
post-Super-Bowl-LX table.

Nothing caught it because **the walk-forward harness cannot reproduce it**. The
harness iterates a fixed frame that always contains the season being scored, so
it crosses every boundary and regresses every time. Backtest and production
diverge only in the one week of the year when the target season has no games —
which is exactly the week this shipped.

Measured effect on the rating distribution (2026 pre-Week-1):

| | sd | spread | LV | MIA |
|---|---|---|---|---|
| shipped (no regression) | 120.4 | 484.0 | 1272 | 1434 |
| regressed 0.33 | 80.7 | 324.3 | 1348 | 1456 |

538-family NFL Elo runs sd ≈ 80. At sd 120 the whole league is over-dispersed,
which is what manufactured the phantom edges. Replaying the 64 Week 1 games with
the regression applied and the shipped QB adjustments held fixed:

- mean |model − market|: 9.25 pp → **7.49 pp**
- games past 15 pp: 14 → **6**
- MIA@LV: model LV 39.0% → **48.6%** (market 62.3%)

The same defect had two more heads, both fixed:

- **QB overlay.** `build_qb_context()` reverts veteran QB ratings at a season
  rollover, but the rollover only fires when a game from the new season arrives —
  so `projected_adjustment()` for Week 1 ran off unreverted end-of-2025 state.
- **Points-based sub-models.** ESPN reports `pointsFor`/`pointsAgainst` as `0`
  (present, not missing) before a team plays, so `.get(key, 350)` never fired.
  Every team's Pythagorean came out exactly 0.500 and `compute_efficiency`
  returned `off_eff = def_eff = 0`, `net_eff = -2.0` league-wide. PYTH and EFF
  were therefore the *same* home-field constant for every game (both 59.25% for
  every home team), displayed on the card as two independent systems and drawing
  full ensemble weight. They now report unavailable until a game is played.

### QB team-baseline offseason reversion (new: `TEAM_REVERT`)

The QB adjustment is `QB_MULT × (starter VALUE − team rolling VALUE)`. Veteran QB
ratings reverted each offseason; the team baseline never did. A team that
rebuilt its offense kept last season's baseline, and a stale-low baseline
inflates the swing credited to an incoming starter — LV's inferred Week 1
adjustment was +95 Elo on top of an already-unregressed rating.

Swept on the walk-forward harness (production constants, era HFA, k20,
regress .33, QB_MULT 2.0, 2017-2025, N=2485):

| team_revert | acc | log loss | brier |
|---|---|---|---|
| 0.00 (shipped) | 0.6483 | 0.6255 | 0.2181 |
| 0.33 | 0.6519 | 0.6244 | 0.2176 |
| **0.50** | **0.6531** | **0.6240** | **0.2174** |
| 0.70 | 0.6531 | 0.6237 | 0.2173 |
| 0.90 | 0.6543 | 0.6235 | 0.2172 |
| 1.00 | 0.6535 | 0.6235 | 0.2171 |

Per-season at 0.50: log loss improves in **8 of 9** seasons, accuracy in 7 of 9.
The curve is flat from 0.5 to 1.0 (~12 games of 2485 separate them — noise), so
0.50 is taken as the point where the gain is realised rather than the grid
argmax. Vegas closing over the same window: 0.6660 / 0.6072 / 0.2099.

### Guard

`tests/test_offseason_regression.py` covers all three heads, including that the
regression is applied once per missing season and becomes a no-op the moment the
season's own results land (so ratings can never be regressed twice).

---

## Offseason roster value (2026-09-08)

Built on top of cdeaebc. That commit made the Week 1 ratings *regress*; they still
carried no information about **what** changed over the offseason. Mean reversion
says "we know less than we did in January" and stops there, so a team that lost
six starters and a team that returned all of them got identical treatment.

`model/roster_value.py` supplies the missing term: net player value in and out
between the end of last season and the start of this one, converted to a bounded
Elo delta that `compute_elo()` applies at each season boundary **immediately after
the reversion, not instead of it**.

### The value metric

Pro-Football-Reference's Approximate Value is the natural unit, but PFR serves 403
to automated clients and nflverse publishes only *career* AV, and only for drafted
players — no free per-player-per-season AV feed exists that a daily Action can
depend on. So the layer computes an AV proxy ("sAV") the way AV itself is built,
from free nflverse feeds:

    sAV = STARTER_AV[group] x (0.6 x snap_share + 0.4 x prod_factor)

- `snap_share`: fraction of the team's offensive/defensive snaps across the season
  (nflverse `snap_counts`). A 17-game every-down starter ≈ 1.0.
- `prod_factor`: season production against the median full-time starter at the same
  position that season — PPR points for skill positions, a standard IDP line for
  the front seven and secondary. Offensive line and specialists have no public box
  score and run on playing time alone.
- `STARTER_AV`: what a full-season starter at that group earns on PFR's scale, so
  ledger numbers read in familiar units. Resulting 2025 medians: OL 2.9, DL 1.9,
  DB 1.7, LB 1.6, WR 1.5, TE 1.4, RB 0.9; top of the league ≈ 8.

sAV is measured on the season *before* the offseason in question, so nothing in the
layer can see the season it is used to predict.

**Quarterbacks are excluded entirely**, on both sides of every move. `qb_model.py`
already adjusts each team's effective Elo by `QB_MULT x (starter VALUE − team
VALUE)` per game; counting a quarterback here as well would pay for him twice.

**Rookies** are priced off a per-pick expected-AV curve (`w_av` / expected career
length) fitted only to drafts at least 8 years old, so the curve carries no
lookahead, then cut to **35%** — the middle of the 25-50% band. A first overall
pick is worth ~2.5 AV, less than a returning median starter at any position; a
seventh-rounder ~0.8.

Membership comes from the nflverse **weekly** roster release: the previous season's
final regular-season week vs. this season's week 1, which brackets exactly the
offseason and means a week-10 signing is not retroactively counted as an offseason
arrival. Practice squad and already-cut players are not roster members.

Net AV is centred on the league, because roster churn is near zero-sum between
teams and the draft injects value league-wide that says nothing about who got
better than whom.

### Scale sweep (shipped-path walk-forward, 2017-2025, N=2485)

`research/wf_roster.py` scores the production modules rather than a lab
re-implementation: `compute_elo` for the chain, `build_qb_context` for the overlay,
the same era-rolling HFA. Baseline reproduces cdeaebc to within one game
(acc 0.6547 / ll 0.6237 here vs 0.6549 / 0.6235 reported).

| elo_per_av | acc | log loss | brier |
|---|---|---|---|
| 0 (cdeaebc baseline) | 0.6547 | 0.6237 | 0.2173 |
| **0.25 (shipped)** | **0.6551** | **0.6234** | **0.2172** |
| 0.50 | 0.6551 | 0.6233 | 0.2171 |
| 0.60 | 0.6555 | 0.6232 | 0.2171 |
| 0.75 | 0.6543 | 0.6231 | 0.2171 |
| 1.00 | 0.6507 | 0.6231 | 0.2171 |
| 1.50 | 0.6507 | 0.6231 | 0.2171 |

Accuracy is flat-to-positive from 0.25 to 0.6 and turns negative from 0.75; log
loss improves monotonically and saturates near 0.6. Per-season at 0.25, log loss
improves in 6 of 9 seasons (unchanged in 1, worse in 2021 and 2024); accuracy nets
+1 game. Largest applied delta over 352 team-seasons: −10.1 / +8.5, sd 2.9. The
±40 cap never binds at this scale.

0.25 and 0.5 are indistinguishable on the backtest, so the tie was broken on the
one thing the backtest structurally cannot see — Week 1 of a season with no games
played, which is the case this layer exists for (the same blind spot that let the
cdeaebc defect ship). Week 1 2026, 64 priced games:

| | mean \|model − market\| | games past 15pp | MIA@LV gap |
|---|---|---|---|
| cdeaebc baseline | 6.92 pp | 5 | 18.5 pp |
| roster 0.25 (shipped) | 6.95 pp | 5 | **17.4 pp** |
| roster 0.50 | 7.04 pp | 6 | 16.3 pp |

At 0.5 the board moves away from the market in aggregate; at 0.25 it stays flat
while still closing the MIA@LV gap. Roster value is a noisy estimate on its first
outing, so the shrunk end of the plateau was taken.

**MIA@LV specifically**: model LV 43.7% → **44.8%** against a market price of
62.3%, gap 18.5 → 17.4 pp. MIA had the league's most negative 2026 offseason
(−32 net AV: Waddle, Fitzpatrick, Chubb, Douglas, Jones all out) and takes −8 Elo;
LV's ledger nets to zero, so the whole move is MIA's.

Honest caveat: the aggregate is flat, not better. The layer narrows all four MIA
games but pushes NYJ — the largest *positive* ledger, +32 net AV — further from the
market in all four of its games. The model and the market disagree about the Jets;
the backtest is the arbiter and it is mildly positive.

### Published metric

`scripts/backtest_nfl.walkforward_metrics` (what the site shows) now takes the same
`roster_adjustments` map the live ratings were built with — scoring a configuration
we no longer ship would be publishing a different model's number.
Shipped path: **acc 0.6549 → 0.6553, log loss 0.6235 → 0.6233** (N=2486).

### Not done, deliberately

The Bayesian/sim layer (`model/bayesian_model.py`) does **not** see roster value.
It is an independent rating built from scores alone, and feeding it the same
offseason signal the Elo chain already carries would correlate two of the
ensemble's supposedly-independent inputs. Making the sim roster-aware is a separate
decision with its own validation, not a free rider on this one.

`data/offseason_roster_value.json` persists the per-team ledger — who arrived, who
left, what each was worth, the applied Elo delta. Nothing reads it yet; no UI in
this change.

## SumerSports EPA layer (2026-09-09)

Source: `scripts/sumersports_feed.py` mirrors the public, no-auth SumerSports
team tables (offensive + defensive EPA/Play family, success rate, pass/rush
splits; seasons 2022-2025 verified live; 2021 and earlier render empty) into
`data/sumersports_cache/`. CI refreshes the current season daily; completed
seasons are static. This is a raw-stats feed, eligible under the no-market-data
rule.

User-directed evaluation: SumerSports EPA as a **primary** cross-season signal,
weight it higher than the existing inputs. Two mechanism families, both
walk-forward (production `compute_elo` + QB overlay + roster layer, scored
2017-2025, N=2485; layer can only affect the 2023-2025 boundaries, so the
2023+ sub-window, N=854, is the honest read):

**Family 1 — additive boundary deltas** (`boundary_deltas`, Elo per net-EPA/play
sweep, cap ±40): monotonically harmful. epe=100: -0.12 acc / +0.0017 ll on
2023+. epe=250: -1.05 acc. epe=600: -0.70 acc. Stacking EPA on top of the
retained two-thirds rating double-counts last season.

**Family 2 — boundary anchor blend** (`boundary_anchor`, rating =
(1-w)*regressed + w*(1500 + K*net_EPA)): also monotonically harmful in w.
K=450: w=0.15 -0.12 acc (ll a hair better), w=0.33 -0.23, w=0.5 -0.23,
w=0.75 -0.94, w=1.0 **-1.41 acc / +0.0022 ll**. K=700 similar (w=1.0: -0.70).
Full EPA-primary costs 0.7-1.4 points of accuracy.

**Cross-sectional check** (why both families fail): full-season net EPA
correlates with same-season win% at 0.865-0.876 — *below* the crude PPG proxy
the efficiency slot already uses (0.881-0.913 all three years). Prior-season →
next-season wins: EPA beats the PPG proxy once of three (2022→23: 0.454 vs
0.450; 2023→24: 0.274 vs 0.314; 2024→25: 0.253 vs 0.294). The Elo chain
already prices this information; the QB overlay owns the part of EPA that is
the quarterback.

**Verdict: rejected at every weight, including primary.** Same disposition as
round 1's playoff multiplier and bye-rest bonus. What shipped instead:
the feed + cache in CI, `model/sumer_epa.py` + the `compute_elo`
`boundary_anchor` mechanism (defaults are the shipped behaviour, fully tested),
`research/wf_sumer.py` for re-runs, per-team `sumer_net_epa` published in
`nfl_leaderboard.json` as display context, and a zero-weight `sumer_prob`
logged per game in `nfl_predictions.json` so the signal grades live going
forward. If SumerSports ever earns real weight it will be through that live
grading or through weekly snapshots, not through full-season tables.
