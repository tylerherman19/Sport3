/* main_fixes.js — QC patches for prediction log postmortem logic.
 * Loaded after main.js in index.html. Overrides 3 functions with fixes.
 * See git log for full change description.
 *
 * Fixes applied:
 * 1. resolveActualWinners: ESPNFIX LVR->LV (ESPN sends LVR, we store LV)
 *    + UTC date methods (prevent TZ date shift for evening games)
 * 2. adjustWeightsFromLog: bayes_pick removed from modelMap
 *    (Bayesian model has no ensemble weight slot)
 * 3. generatePostMortemExplanation: efficiency gap<-3 branch now
 *    guarded by !modelWasCorrect (was always emitting "model went wrong")
 *
 * Kalshi sandbox: Kalshi tab removed from index.html + renderAll.
 * Full Kalshi implementation preserved in sandbox/kalshi/.
 */

// Override 1: resolveActualWinners — ESPNFIX direction fix + UTC date fix
async function resolveActualWinners(league) {
  const key = `sport3_log_${league}`;
  let entries = [];
  try { entries = JSON.parse(localStorage.getItem(key) || '[]'); } catch { return; }

  const unresolved = entries.filter(e => !e.actual_winner);
  if (!unresolved.length) return;

  // Group by date
  const dateGroups = {};
  unresolved.forEach(e => {
    const d = e.game_time ? (() => {
      const dt = new Date(e.game_time);
      // QC fix: use UTC methods — game_time is a UTC ISO string; local getDate()
      // can shift the date by 1 day for evening games in US time zones (UTC-5/6).
      const y = dt.getUTCFullYear();
      const m = String(dt.getUTCMonth() + 1).padStart(2, '0');
      const day = String(dt.getUTCDate()).padStart(2, '0');
      return `${y}${m}${day}`;
    })() : null;
    if (!d) return;
    if (!dateGroups[d]) dateGroups[d] = [];
    dateGroups[d].push(e);
  });

  const sport = league === 'nba' ? 'basketball/nba' : 'football/nfl';

  // ESPN often returns short abbreviations that differ from what we store
  // QC fix: LVR->LV corrected (ESPN sends 'LVR'; we store 'LV' from abbrev_norm)
  const ESPNFIX = {
    'GS': 'GSW', 'NY': 'NYK', 'NO': 'NOP', 'SA': 'SAS', 'WSH': 'WAS',
    'UTH': 'UTA', 'JAC': 'JAX', 'ARZ': 'ARI', 'CLV': 'CLE', 'HST': 'HOU', 'LVR': 'LV'
  };
  const normAbbr = a => { const u = (a || '').toUpperCase(); return ESPNFIX[u] || u; };

  for (const [date, dayEntries] of Object.entries(dateGroups)) {
    try {
      const url = `https://site.api.espn.com/apis/site/v2/sports/${sport}/scoreboard?dates=${date}`;
      const data = await fetch(url).then(r => r.json());
      const espnEvents = data.events || [];
      espnEvents.forEach(ev => {
        const comp = (ev.competitions || [])[0];
        if (!comp) return;
        const competitors = comp.competitors || [];
        if (competitors.length < 2) return;
        const isFinalEvent = ev.status?.type?.name === 'STATUS_FINAL';
        let winner = competitors.find(c => c.winner);
        if (!winner && isFinalEvent) {
          // Fallback: determine winner by score when winner flag isn't set
          const homeComp = competitors.find(c => c.homeAway === 'home');
          const awayComp = competitors.find(c => c.homeAway === 'away');
          if (homeComp && awayComp) {
            const hs = parseInt(homeComp.score, 10);
            const as_ = parseInt(awayComp.score, 10);
            if (!isNaN(hs) && !isNaN(as_) && hs !== as_) {
              winner = hs > as_ ? homeComp : awayComp;
            }
          }
        }
        if (!winner) return;
        const winnerAbbrev = normAbbr(winner.team?.abbreviation);

        // Match to our log entry by game_id or team names
        const match = dayEntries.find(e =>
          e.game_id === ev.id ||
          (e.home_team === normAbbr(competitors.find(c => c.homeAway === 'home')?.team?.abbreviation) &&
           e.away_team === normAbbr(competitors.find(c => c.homeAway === 'away')?.team?.abbreviation))
        );
        if (match && winnerAbbrev) {
          const entry = entries.find(e => e.game_id === match.game_id);
          if (entry) {
            entry.actual_winner = winnerAbbrev;
            // Store actual scores for post-mortem analysis
            const homeComp = competitors.find(c => c.homeAway === 'home');
            const awayComp = competitors.find(c => c.homeAway === 'away');
            if (homeComp?.score != null) entry.home_score = parseInt(homeComp.score, 10);
            if (awayComp?.score != null) entry.away_score = parseInt(awayComp.score, 10);
          }
        }
      });
    } catch { /* network failure, skip */ }
  }

  try { localStorage.setItem(key, JSON.stringify(entries)); } catch {}
  adjustWeightsFromLog(league);
}

// Override 2: adjustWeightsFromLog — bayes_pick removed from modelMap
function adjustWeightsFromLog(league) {
  const key = `sport3_log_${league}`;
  let entries = [];
  try { entries = JSON.parse(localStorage.getItem(key) || '[]'); } catch { return; }

  const resolved = entries.filter(e => e.actual_winner != null);
  if (resolved.length < 10) return; // need enough data before nudging

  // Use the most recent 20 resolved games
  const recent = resolved.slice(-20);

  // Map log pick fields to ensemble weight keys.
  // QC fix: bayes_pick removed — the Bayesian model is not part of the ensemble
  // weight set (logistic/xgboost/elo/pyth/eff), so feeding its accuracy back
  // into the xgboost weight slot was incorrect cross-model contamination.
  const modelMap = [
    { pickKey: 'elo_pick',  weightKey: 'elo'      },
    { pickKey: 'lr_pick',   weightKey: 'logistic' },
    { pickKey: 'pyth_pick', weightKey: 'pyth'     },
    { pickKey: 'eff_pick',  weightKey: 'eff'      },
  ];

  // Compute accuracy per model
  const accuracies = modelMap.map(({ pickKey, weightKey }) => {
    const valid = recent.filter(e => e[pickKey] != null);
    if (!valid.length) return { weightKey, acc: 0.5 };
    const correct = valid.filter(e => e[pickKey] === e.actual_winner).length;
    return { weightKey, acc: correct / valid.length };
  });

  const avgAcc = accuracies.reduce((s, m) => s + m.acc, 0) / accuracies.length;

  // Nudge weights: better-than-average models gain, worse ones lose
  accuracies.forEach(({ weightKey, acc }) => {
    const delta = Math.max(-0.02, Math.min(0.02, (acc - avgAcc) * 0.1));
    state.weights[weightKey] = Math.max(0.01, state.weights[weightKey] + delta);
  });

  // Re-normalise so weights sum to 1.0
  const total = Object.values(state.weights).reduce((s, v) => s + v, 0);
  for (const k of Object.keys(state.weights)) state.weights[k] /= total;

  saveWeights();
  syncWeightSliders();
}

// Override 3: generatePostMortemExplanation — gap<-3 guarded by !modelWasCorrect
function generatePostMortemExplanation(e) {
  const winnerName = e.actual_winner === e.home_team ? e.home_name : e.away_name;
  const loserName  = e.actual_winner === e.home_team ? e.away_name : e.home_name;
  const winnerTeam = e.actual_winner;
  const loserTeam  = e.actual_winner === e.home_team ? e.away_team : e.home_team;
  const probPct = Math.round(e.predicted_prob * 100);
  const winnerWasHome = e.actual_winner === e.home_team;
  const modelWasCorrect = e.predicted_winner === e.actual_winner;
  const predictedName   = e.predicted_winner === e.home_team ? e.home_name : e.away_name;

  // ── Score header ─────────────────────────────────────────────
  let scoreHeaderHtml = '';
  let narrativeLines = [];
  let margin = null;
  if (e.home_score != null && e.away_score != null) {
    margin = Math.abs(e.home_score - e.away_score);
    const winScore = winnerWasHome ? e.home_score : e.away_score;
    const lossScore = winnerWasHome ? e.away_score : e.home_score;
    scoreHeaderHtml = `<div class="pm-score-header">
      Final: <strong>${winnerName} ${winScore} – ${lossScore} ${loserName}</strong>
      <span class="pm-margin">(${margin}-pt margin)</span>
    </div>`;

    // Sentence 1 — margin context
    if (margin <= 5) {
      if (modelWasCorrect) {
        narrativeLines.push(`The model correctly backed ${predictedName} at ${probPct}% — a well-called coin-flip game.`);
      } else {
        narrativeLines.push(`The model picked ${predictedName} at ${probPct}% — a reasonable call in a coin-flip game.`);
      }
    } else if (margin <= 12) {
      if (modelWasCorrect) {
        narrativeLines.push(`${winnerName} won by ${margin} pts — the model's ${probPct}% on ${predictedName} proved correct.`);
      } else {
        narrativeLines.push(`${winnerName} won by ${margin} pts. The model favoured ${predictedName} at ${probPct}%.`);
      }
    } else {
      if (modelWasCorrect) {
        narrativeLines.push(`${winnerName} dominated, winning by ${margin} pts — the model's confidence in ${predictedName} was well placed.`);
      } else {
        narrativeLines.push(`${winnerName} dominated, winning by ${margin} pts despite the model giving ${predictedName} a ${probPct}% chance.`);
      }
    }

    // Sentence 2 — home/away context
    if (winnerWasHome) {
      narrativeLines.push(`${winnerName} took full advantage of home court/field, turning home-team status into a genuine performance edge.`);
    } else {
      narrativeLines.push(`${winnerName} pulled this off on the road — away wins are harder to predict and often reflect a team peaking at the right time.`);
    }
  } else {
    if (modelWasCorrect) {
      narrativeLines.push(`The model gave ${predictedName} a ${probPct}% win probability — the correct call.`);
    } else {
      narrativeLines.push(`The model gave ${predictedName} a ${probPct}% win probability, but ${winnerName} won.`);
    }
  }

  // ── Pre-game efficiency table ────────────────────────────────────────────
  let effHtml = '';
  const hasEff = e.home_off_rating != null || e.away_off_rating != null;
  const wNet = winnerWasHome ? e.home_net_rating : e.away_net_rating;
  const lNet = winnerWasHome ? e.away_net_rating : e.home_net_rating;
  const wOff = winnerWasHome ? e.home_off_rating : e.away_off_rating;
  const lOff = winnerWasHome ? e.away_off_rating : e.home_off_rating;

  if (hasEff) {
    const fmt = (v, sign) => v != null ? `${sign && v >= 0 ? '+' : ''}${v.toFixed(1)}` : '—';
    effHtml = `
    <div class="pm-section-label">Pre-game Efficiency</div>
    <div class="pm-eff-table">
      <div class="pm-eff-row pm-eff-header">
        <span></span><span>Off Rtg</span><span>Def Rtg</span><span>Net Rtg</span>
      </div>
      <div class="pm-eff-row ${winnerWasHome ? 'pm-winner-row' : ''}">
        <span>${e.home_name || e.home_team}</span>
        <span>${fmt(e.home_off_rating)}</span>
        <span>${fmt(e.home_def_rating)}</span>
        <span>${fmt(e.home_net_rating, true)}</span>
      </div>
      <div class="pm-eff-row ${!winnerWasHome ? 'pm-winner-row' : ''}">
        <span>${e.away_name || e.away_team}</span>
        <span>${fmt(e.away_off_rating)}</span>
        <span>${fmt(e.away_def_rating)}</span>
        <span>${fmt(e.away_net_rating, true)}</span>
      </div>
    </div>`;

    // Sentence 3 — efficiency insight
    if (wNet != null && lNet != null) {
      const gap = lNet - wNet;
      if (gap > 3) {
        narrativeLines.push(`${winnerName} overcame a ${gap.toFixed(1)}-pt pre-game net rating disadvantage — the stats favoured ${loserName} on paper.`);
      } else if (gap < -3 && !modelWasCorrect) {
        // QC fix: only note model failure when model was actually wrong (gap<-3 means
        // winner had the stronger net rating, which alone does not imply a model error).
        narrativeLines.push(`${winnerName} had the stronger net rating (+${Math.abs(gap).toFixed(1)} advantage) yet the model still went wrong — efficiency didn't tell the whole story.`);
      } else if (lOff != null && wOff != null && lOff - wOff > 3) {
        narrativeLines.push(`${loserName}'s offense (${lOff.toFixed(1)}) looked superior on paper but couldn't convert that into points on the night.`);
      }
    }
  }

  // ── Model vote breakdown ──────────────────────────────────────────────
  const modelDefs = [
    { label: 'ELO',   pick: e.elo_pick,   prob: e.elo_prob   },
    { label: 'Bayes', pick: e.bayes_pick, prob: e.bayes_prob },
    { label: 'LR',    pick: e.lr_pick,    prob: e.lr_prob    },
    { label: 'Pyth',  pick: e.pyth_pick,  prob: e.pyth_prob  },
    { label: 'Eff',   pick: e.eff_pick,   prob: e.eff_prob   },
  ].filter(m => m.pick);

  let votesHtml = '';
  let insightText = '';
  if (modelDefs.length) {
    const correctModels = modelDefs.filter(m => m.pick === winnerTeam);
    const wrongModels   = modelDefs.filter(m => m.pick !== winnerTeam);
    const correctCount  = correctModels.length;

    const votePills = modelDefs.map(m => {
      const correct = m.pick === winnerTeam;
      let probLabel = '';
      if (m.prob != null) {
        const pickIsHome = m.pick === e.home_team;
        const pct = Math.round((pickIsHome ? m.prob : 1 - m.prob) * 100);
        probLabel = ` (${pct}%)`;
      }
      return `<span class="pm-vote ${correct ? 'pm-vote-correct' : 'pm-vote-wrong'}">${m.label} → ${m.pick}${probLabel}</span>`;
    }).join('');

    // Named consensus text
    let consensusText = '';
    if (correctCount === 0) {
      consensusText = `All ${modelDefs.length} sub-models were unanimous on ${loserTeam} — a genuine statistical upset.`;
    } else if (correctCount === modelDefs.length) {
      if (modelWasCorrect) {
        consensusText = `All ${modelDefs.length} sub-models unanimously called ${winnerTeam} — the ensemble agreed.`;
      } else {
        consensusText = `All ${modelDefs.length} visible sub-models called ${winnerTeam}, but the ensemble still favoured ${loserTeam} — likely driven by the XGBoost component.`;
      }
    } else if (correctCount >= Math.ceil(modelDefs.length / 2)) {
      const rightNames = correctModels.map(m => m.label).join(', ');
      const wrongNames = wrongModels.map(m => m.label).join(', ');
      if (modelWasCorrect) {
        consensusText = `${rightNames} correctly called ${winnerTeam}; ${wrongNames} backed ${loserTeam} but the ensemble sided with the majority.`;
      } else {
        consensusText = `${rightNames} correctly called ${winnerTeam}, but ${wrongNames} backed ${loserTeam} and pulled the ensemble vote with them.`;
      }
    } else {
      const rightNames = correctModels.map(m => m.label).join(', ');
      const wrongNames = wrongModels.map(m => m.label).join(', ');
      consensusText = `Models were split — ${rightNames} called ${winnerTeam}; ${wrongNames} backed ${loserTeam}. The ensemble sided with the majority.`;
    }

    // Closing insight
    if (correctCount === 0 && probPct >= 75) {
      insightText = `High-confidence unanimous misses are the model's blind spot for genuine upsets — worth watching if this team keeps defying expectations.`;
    } else if (correctCount > 0 && correctCount < modelDefs.length && correctCount < Math.ceil(modelDefs.length / 2)) {
      insightText = `When sub-models disagree this strongly, treat the ensemble confidence with extra scepticism.`;
    } else if (margin != null && margin <= 5 && !modelWasCorrect) {
      insightText = `Coin-flip games like this are inherently hard to predict — a miss in a 5-pt game is expected noise, not a model failure.`;
    } else if (correctCount === 0 && probPct < 65) {
      insightText = `The model wasn't hugely confident here — this kind of miss is within normal variance.`;
    }

    votesHtml = `
    <div class="pm-section-label">Model Votes</div>
    <div class="pm-model-votes">${votePills}</div>
    <div class="pm-vote-summary">${consensusText}</div>
    ${insightText ? `<div class="pm-reason">${insightText}</div>` : ''}`;
  }

  const narrativeHtml = narrativeLines.map(t => `<p class="pm-narrative-line">${t}</p>`).join('');

  return `<div class="pm-content">
    ${scoreHeaderHtml}
    <div class="pm-narrative">${narrativeHtml}</div>
    ${effHtml}
    ${votesHtml}
  </div>`;
}

// Kalshi no-op stubs (entry points removed from index.html + renderAll in main.js)
function renderKalshiIfOpen() { /* SANDBOXED — see sandbox/kalshi/ */ }
function setKalshiLeague() { /* SANDBOXED */ }
function renderKalshiTab() { /* SANDBOXED */ }
