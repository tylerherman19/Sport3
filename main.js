/* ═══════════════════════════════════════════════════════════════
   Sports Prediction Dashboard — main.js
   Supports NFL and NBA with live switching
   ═══════════════════════════════════════════════════════════════ */

'use strict';

/* ── State ────────────────────────────────────────────────────── */
const state = {
  league: 'nfl',
  // NFL data
  predictions: null,
  eloRatings: null,
  leaderboard: null,
  modelMetrics: null,
  // NBA data
  nba: {
    predictions: null,
    leaderboard: null,
    modelMetrics: null,
  },
  sortCol: 'elo',
  sortDir: 'desc',
  gameFilter: 'all', // 'all' | 'today' | 'future'
  weights: { logistic: 0.30, xgboost: 0.25, elo: 0.20, pyth: 0.15, eff: 0.10 },
  params: { k: 20, hfa: 65, mov: 1.0, form: 0.30, sos: 0.5, h2h: 0.5, to: 1.0, rt: 1.0, lambda: 0.10 },
  gameOverrides: {}, // { [game_id]: { homeBoost, awayBoost, hfaMult, momentumBoost, restFactor } }
};

/* ── Chart instances ──────────────────────────────────────────── */
let calibrationChart = null;
let accuracyChart = null;

/* ── Helpers ──────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const pct = v => (v == null ? '—' : (v * 100).toFixed(1) + '%');
const eloFmt = v => (v == null ? '—' : Math.round(v).toString());
const fixed2 = v => (v == null ? '—' : v.toFixed(2));

function fmtDate(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' });
  } catch { return iso; }
}

function fmtUpdated(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return 'Updated ' + d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  } catch { return iso; }
}

function isToday(iso) {
  if (!iso) return false;
  const d = new Date(iso);
  const now = new Date();
  return d.getFullYear() === now.getFullYear() &&
         d.getMonth()    === now.getMonth()    &&
         d.getDate()     === now.getDate();
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/* ── Team Timezone Offsets (standard/non-DST UTC offset) ─────── */
const TEAM_TZ = {
  // NBA — Eastern
  BOS:-5, NYK:-5, BKN:-5, PHI:-5, TOR:-5, MIA:-5, ORL:-5, ATL:-5, CHA:-5, WAS:-5,
  CLE:-5, DET:-5, IND:-5,
  // NBA — Central
  CHI:-6, MIL:-6, MEM:-6, NOP:-6, SAS:-6, HOU:-6, DAL:-6, OKC:-6, MIN:-6,
  // NBA — Mountain
  DEN:-7, UTA:-7, PHX:-7,
  // NBA — Pacific
  LAL:-8, LAC:-8, SAC:-8, POR:-8, GSW:-8, SEA:-8,
  // NFL — Eastern
  NE:-5, NYG:-5, NYJ:-5, PHI:-5, DAL:-6, WAS:-5, MIA:-5, BUF:-5, BAL:-5, PIT:-5,
  CLE:-5, CIN:-5, ATL:-5, CAR:-5, TB:-5, NO:-6, JAX:-5,
  // NFL — Central
  GB:-6, CHI:-6, MIN:-6, DET:-5, HOU:-6, TEN:-6, IND:-5,
  // NFL — Mountain
  DEN:-7, KC:-6, LV:-8, LAR:-8, LAC:-8, ARI:-7,
  // NFL — Pacific
  SF:-8, SEA:-8,
};

function ensembleFromProbs(probs, weights) {
  const w = weights;
  const total = w.logistic + w.xgboost + w.elo + w.pyth + w.eff;
  if (total === 0) return 0.5;
  const xgb = probs.xgb_prob != null ? probs.xgb_prob : probs.logistic_prob;
  const val = (
    w.logistic * (probs.logistic_prob || 0.5) +
    w.xgboost  * (xgb || 0.5) +
    w.elo      * (probs.elo_prob || 0.5) +
    w.pyth     * (probs.pyth_prob || 0.5) +
    w.eff      * (probs.eff_prob || 0.5)
  ) / total;
  return clamp(val, 0.01, 0.99);
}

/* ── League switching ─────────────────────────────────────────── */
document.querySelectorAll('.league-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const league = btn.dataset.league;
    if (league === state.league) return;
    state.league = league;
    document.body.dataset.league = league;

    // Update active state
    document.querySelectorAll('.league-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    // Update header
    if (league === 'nba') {
      $('header-icon').textContent = '🏀';
      $('header-title').textContent = 'NBA Prediction Dashboard';
      $('tab-btn-games').textContent = "Tonight's Games";
      $('tab-btn-predictor').textContent = 'Matchup Predictor';
      $('hfa-label').textContent = 'Home Court Advantage (ELO pts)';
      $('sl-hfa').value = 100;
      $('val-hfa').textContent = 100;
      state.params.hfa = 100;
      $('leaderboard-badge').textContent = 'All 30 Teams';
      $('col-net-eff').textContent = 'Net Rtg';
      $('col-off-eff').textContent = 'Off Rtg';
      $('col-def-eff').textContent = 'Def Rtg';
      $('col-sb-prob').textContent = 'Champ %';
      // Show NBA rest options, hide NFL
      $('nfl-rest-options').style.display = 'none';
      $('nba-rest-options').style.display = 'block';
      $('predictor-empty-icon').textContent = '🏀';
    } else {
      $('header-icon').textContent = '🏈';
      $('header-title').textContent = 'NFL Prediction Dashboard';
      $('tab-btn-games').textContent = "This Week's Games";
      $('tab-btn-predictor').textContent = 'Matchup Predictor';
      $('hfa-label').textContent = 'Home Field Advantage (ELO pts)';
      $('sl-hfa').value = 65;
      $('val-hfa').textContent = 65;
      state.params.hfa = 65;
      $('leaderboard-badge').textContent = 'All 32 Teams';
      $('col-net-eff').textContent = 'Net Eff';
      $('col-off-eff').textContent = 'Off Eff';
      $('col-def-eff').textContent = 'Def Eff';
      $('col-sb-prob').textContent = 'SB %';
      $('nfl-rest-options').style.display = 'block';
      $('nba-rest-options').style.display = 'none';
      $('predictor-empty-icon').textContent = '🏈';
    }

    // Reset sort
    state.sortCol = 'elo';
    state.sortDir = 'desc';

    // Reload data if not yet loaded
    if (league === 'nba' && !state.nba.predictions) {
      loadNbaData();
    } else {
      renderAll();
    }
  });
});

/* ── Nav tabs ─────────────────────────────────────────────────── */
document.querySelectorAll('.nav-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    const id = 'tab-' + btn.dataset.tab;
    const sec = $(id);
    if (sec) sec.classList.add('active');

    // Lazy render charts on first visit to performance tab
    if (btn.dataset.tab === 'performance') {
      const metrics = state.league === 'nba' ? state.nba.modelMetrics : state.modelMetrics;
      if (metrics) renderCharts();
    }
    // Render prediction log on accuracy tab
    if (btn.dataset.tab === 'accuracy') {
      renderAccuracyTab();
    }
  });
});

/* ── Fetch data ───────────────────────────────────────────────── */
async function loadAll() {
  await Promise.all([loadNflData(), loadNbaData()]);
}

async function loadNflData() {
  const base = './data/';
  const t = Date.now();
  const [pred, elo, lb, metrics] = await Promise.all([
    fetch(base + 'nfl_predictions.json?t=' + t).then(r => r.json()).catch(() =>
      fetch(base + 'predictions.json?t=' + t).then(r => r.json()).catch(() => null)
    ),
    fetch(base + 'elo_ratings.json?t=' + t).then(r => r.json()).catch(() => null),
    fetch(base + 'nfl_leaderboard.json?t=' + t).then(r => r.json()).catch(() =>
      fetch(base + 'leaderboard.json?t=' + t).then(r => r.json()).catch(() => null)
    ),
    fetch(base + 'model_metrics.json?t=' + t).then(r => r.json()).catch(() => null),
  ]);

  state.predictions = pred;
  state.eloRatings = elo;
  state.leaderboard = lb;
  state.modelMetrics = metrics;

  if (state.league === 'nfl') {
    updateHeader(pred, 'nfl');
    renderAll();
  }
}

async function loadNbaData() {
  const base = './data/';
  const t = Date.now();
  const [pred, lb, metrics] = await Promise.all([
    fetch(base + 'nba_predictions.json?t=' + t).then(r => r.json()).catch(() => null),
    fetch(base + 'nba_leaderboard.json?t=' + t).then(r => r.json()).catch(() => null),
    fetch(base + 'nba_model_metrics.json?t=' + t).then(r => r.json()).catch(() => null),
  ]);

  state.nba.predictions = pred;
  state.nba.leaderboard = lb;
  state.nba.modelMetrics = metrics;

  if (state.league === 'nba') {
    updateHeader(pred, 'nba');
    renderAll();
  }
}

function updateHeader(pred, league) {
  const updated = pred?.updated || '';
  $('last-updated').textContent = fmtUpdated(updated);
  const season = pred?.season || '';
  const week = pred?.week;
  if (league === 'nfl') {
    $('season-week').textContent = week ? `${season} Season — Week ${week}` : (season ? `${season} Season` : '');
  } else {
    $('season-week').textContent = season ? `${season} NBA Season` : '';
  }
}

function renderAll() {
  renderGames();
  renderLeaderboard();
  renderMetrics();
  populatePredictor();
}

function forceRefresh() {
  loadNflData();
  loadNbaData();
}

// Start loading both leagues immediately
loadNflData();
loadNbaData();

/* ══════════════════════════════════════════════════════════════
   SECTION 1 — THIS WEEK'S GAMES
══════════════════════════════════════════════════════════════ */
function renderGames() {
  const grid = $('games-grid');
  const isNba = state.league === 'nba';
  const pred = isNba ? state.nba.predictions : state.predictions;

  if (!pred) {
    grid.innerHTML = '<div class="empty-state"><div class="empty-state-icon">⚠️</div><h3>Failed to load</h3><p>Could not load predictions data.</p></div>';
    return;
  }

  const offseasonMsg = isNba
    ? 'The NBA season is not currently active. Ratings below reflect final standings from the <strong>' + (pred.season || '') + ' season</strong>. Predictions will populate automatically when the new season begins.'
    : 'The NFL season is not currently active. Ratings below reflect final standings from the <strong>' + (pred.season || '') + ' season</strong>. Predictions will populate automatically when the new season begins.';

  const bannerIcon = isNba ? '🏀' : '🏈';
  const emptyIcon = isNba ? '🏀' : '🏈';
  const emptyTitle = isNba ? 'No Games Tonight' : 'No Games This Week';
  const emptyMsg = isNba
    ? 'Check back later. The model updates daily at 8 AM UTC.'
    : 'Check back when the season is active. The model updates daily at 8 AM UTC.';

  // Determine if we're effectively in offseason (explicit flag or all games are old finals)
  const allOldFinals = pred.games.length > 0 && pred.games.every(g => {
    if (g.status !== 'STATUS_FINAL') return false;
    const gameDate = g.game_time ? new Date(g.game_time) : null;
    return gameDate && (Date.now() - gameDate.getTime()) > 21 * 24 * 60 * 60 * 1000;
  });
  const effectiveOffseason = pred.is_offseason || allOldFinals;

  $('offseason-banner').innerHTML = '';
  if (effectiveOffseason || pred.games.length === 0) {
    $('offseason-banner').innerHTML = `
      <div class="offseason-banner">
        <div class="offseason-banner-icon">${bannerIcon}</div>
        <div>
          <strong>Offseason</strong>
          <p>${offseasonMsg}</p>
        </div>
      </div>`;
  }

  // When in offseason with only stale completed games, don't show the cards
  const displayGames = effectiveOffseason ? [] : pred.games;
  $('games-count-badge').textContent = displayGames.length + ' game' + (displayGames.length !== 1 ? 's' : '');

  // Apply game filter
  let filteredGames = displayGames;
  if (state.gameFilter === 'today') {
    filteredGames = pred.games.filter(g => !g.is_future && isToday(g.game_time));
  } else if (state.gameFilter === 'future') {
    const now = new Date();
    filteredGames = pred.games.filter(g =>
      g.is_future ||
      (!isToday(g.game_time) && new Date(g.game_time) > now &&
       g.status !== 'STATUS_FINAL' && g.status !== 'STATUS_IN_PROGRESS')
    );
  }

  // Update filter pill active state
  document.querySelectorAll('.filter-pill').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.filter === state.gameFilter);
  });

  if (displayGames.length === 0) {
    grid.innerHTML = `<div class="empty-state">
      <div class="empty-state-icon">${emptyIcon}</div>
      <h3>${emptyTitle}</h3>
      <p>${emptyMsg}</p>
    </div>`;
    return;
  }

  if (filteredGames.length === 0) {
    const filterLabels = { today: 'today / live games', future: 'upcoming scheduled games' };
    grid.innerHTML = `<div class="empty-state">
      <div class="empty-state-icon">${emptyIcon}</div>
      <h3>No ${filterLabels[state.gameFilter] || 'games'}</h3>
      <p>Try switching the filter above to see all games.</p>
    </div>`;
    return;
  }

  grid.innerHTML = filteredGames.map(game => buildGameCard(game, isNba)).join('');
  logPredictions(pred.games, isNba ? 'nba' : 'nfl');
}

/* ── Score display (FINAL / live) ─────────────────────────────── */
function buildScoreHtml(game) {
  const status = game.status || '';
  const isFinal = status === 'STATUS_FINAL';
  const isLive  = status === 'STATUS_IN_PROGRESS' || status === 'STATUS_HALFTIME';
  const homeScore = game.home_score;
  const awayScore = game.away_score;

  if ((isFinal || isLive) && homeScore != null && awayScore != null) {
    const winner = isFinal
      ? (homeScore > awayScore ? 'home' : awayScore > homeScore ? 'away' : 'tie')
      : '';
    return `
  <div class="score-display ${isLive ? 'score-live' : 'score-final'}">
    <span class="score-away ${winner === 'away' ? 'score-winner' : ''}">${game.away_team} <strong>${awayScore}</strong></span>
    <span class="score-sep">${isLive ? '<span class="live-dot"></span>LIVE' : 'FINAL'}</span>
    <span class="score-home ${winner === 'home' ? 'score-winner' : ''}"><strong>${homeScore}</strong> ${game.home_team}</span>
  </div>`;
  }
  return '';
}

/* ── Key factors strip (always-visible rest/timezone/B2B) ──────── */
function buildKeyFactorsStrip(game, adj, isNba) {
  const items = [];
  const restHome = adj.rest_home ?? adj.rest_home_days;
  const restAway = adj.rest_away ?? adj.rest_away_days;
  const tzHome = TEAM_TZ[game.home_team] || -5;
  const tzAway = TEAM_TZ[game.away_team] || -5;
  const tzShift = tzAway - tzHome; // positive = away travelling east (gaining time)
  const absTz = Math.abs(tzShift);

  // Rest days — show "B2B" text for back-to-back teams instead of stale rest day count
  if (restHome != null || restAway != null) {
    const rh = restHome ?? 7;
    const ra = restAway ?? 7;
    const homeIsB2b = isNba && adj.b2b_home;
    const awayIsB2b = isNba && adj.b2b_away;
    const rhDisplay = homeIsB2b ? 'B2B' : `${rh}d`;
    const raDisplay = awayIsB2b ? 'B2B' : `${ra}d`;
    const homeRestClass = homeIsB2b ? 'kf-bad' : (rh <= 2 ? 'kf-bad' : rh >= 7 ? 'kf-good' : '');
    const awayRestClass = awayIsB2b ? 'kf-bad' : (ra <= 2 ? 'kf-bad' : ra >= 7 ? 'kf-good' : '');
    items.push(`<div class="kf-item"><span class="kf-label">Rest</span>
      <span class="kf-val ${homeRestClass}">${game.home_team} ${rhDisplay}</span>
      <span class="kf-sep">vs</span>
      <span class="kf-val ${awayRestClass}">${game.away_team} ${raDisplay}</span></div>`);
  }

  // Timezone shift
  if (absTz >= 2) {
    const dir = tzShift > 0 ? `${game.away_team} travels east +${absTz}hr` : `${game.away_team} travels west ${absTz}hr`;
    items.push(`<div class="kf-item kf-tz"><span class="kf-label">TZ</span><span class="kf-val kf-warn">${dir}</span></div>`);
  }

  // Back-to-back (NBA) — using correct field names from JSON
  if (isNba && adj.b2b_home) items.push(`<div class="kf-item"><span class="kf-val kf-bad">${game.home_team} B2B</span></div>`);
  if (isNba && adj.b2b_away) items.push(`<div class="kf-item"><span class="kf-val kf-bad">${game.away_team} B2B</span></div>`);

  // Travel distance (NFL)
  if (!isNba && adj.travel_dist_miles > 1000) {
    items.push(`<div class="kf-item"><span class="kf-label">Travel</span><span class="kf-val kf-warn">${Math.round(adj.travel_dist_miles)} mi</span></div>`);
  }

  if (!items.length) return '';
  return `<div class="key-factors-strip">${items.join('')}</div>`;
}

function buildGameCard(game, isNba) {
  const p = game.predictions || {};
  const ensProb = ensembleFromProbs(p, state.weights);
  const awayProb = 1 - ensProb;
  const mc = game.monte_carlo || {};
  const adj = game.adjustments || {};
  const elo = game.elo || {};
  const market = game.market || {};
  const edge = market.edge != null ? market.edge : null;

  const edgeBadge = edge != null
    ? `<span class="edge-badge ${edge > 0.02 ? 'positive' : edge < -0.02 ? 'negative' : 'neutral'}">
        ${edge > 0 ? '▲' : '▼'} Edge: ${(Math.abs(edge) * 100).toFixed(1)}%
        ${market.kelly_pct != null ? ` · Kelly ${(market.kelly_pct*100).toFixed(1)}%` : ''}
      </span>`
    : `<span class="edge-badge neutral">No odds available</span>`;

  const lbData = (isNba ? state.nba.leaderboard : state.leaderboard)?.teams || [];
  const homeData = lbData.find(t => t.team === game.home_team) || {};
  const awayData = lbData.find(t => t.team === game.away_team) || {};

  const homeWins = homeData.wins ?? 0;
  const homeLosses = homeData.losses ?? 0;
  const homeTies = homeData.ties ?? 0;
  const awayWins = awayData.wins ?? 0;
  const awayLosses = awayData.losses ?? 0;
  const awayTies = awayData.ties ?? 0;
  const homeRecordStr = `${homeWins}-${homeLosses}${homeTies > 0 ? `-${homeTies}` : ''}`;
  const awayRecordStr = `${awayWins}-${awayLosses}${awayTies > 0 ? `-${awayTies}` : ''}`;

  // Streak badges
  const homeStreakType = homeData.streak_type || '';
  const homeStreakCount = homeData.streak_count || 0;
  const awayStreakType = awayData.streak_type || '';
  const awayStreakCount = awayData.streak_count || 0;
  const homeStreakBadge = homeStreakType && homeStreakType !== 'N' && homeStreakCount > 0
    ? `<span class="streak-badge streak-${homeStreakType.toLowerCase()}">${homeStreakType}${homeStreakCount}</span>` : '';
  const awayStreakBadge = awayStreakType && awayStreakType !== 'N' && awayStreakCount > 0
    ? `<span class="streak-badge streak-${awayStreakType.toLowerCase()}">${awayStreakType}${awayStreakCount}</span>` : '';

  // H2H strip
  const h2h = game.h2h;
  const h2hHtml = h2h && h2h.total_meetings > 0 ? (() => {
    const leader = h2h.home_wins > h2h.away_wins ? game.home_team
                 : h2h.away_wins > h2h.home_wins ? game.away_team : null;
    const leaderLabel = leader ? `<span class="h2h-leader">${leader} leads</span>` : `<span class="h2h-leader">Even</span>`;
    const last5 = h2h.last_n ? h2h.last_n.slice(-5) : [];
    const last5Dots = last5.map(m =>
      `<span class="h2h-dot ${m.winner === game.home_team ? 'h2h-dot-home' : 'h2h-dot-away'}" title="${m.date}: ${m.winner} +${m.margin}"></span>`
    ).join('');
    return `<div class="h2h-strip">
      <span class="h2h-label">H2H</span>
      ${leaderLabel}
      <span class="h2h-record">${game.home_team} ${h2h.home_wins}–${h2h.away_wins} ${game.away_team}</span>
      ${last5Dots.length ? `<span class="h2h-dots-label">Last ${last5.length}:</span>${last5Dots}` : ''}
    </div>`;
  })() : '';

  // Score display
  const scoreHtml = buildScoreHtml(game);

  // Key factors strip (always visible: rest, timezone, B2B, travel)
  const keyFactorsHtml = buildKeyFactorsStrip(game, adj, isNba);

  // Game context: rest, streak, travel, timezone
  const homeTrend = homeData.trend || 'neutral';
  const awayTrend = awayData.trend || 'neutral';
  const tzShift = Math.abs((TEAM_TZ[game.home_team] || -5) - (TEAM_TZ[game.away_team] || -5));
  const contextHtml = buildContextHtml(game, adj, isNba, homeTrend, awayTrend, tzShift);

  // Injury panel
  const injuryHtml = buildInjuryHtml(game, isNba);

  // Prediction drivers
  const drivers = game.prediction_drivers || [];
  const driversHtml = drivers.length ? `
    <div class="drivers-section">
      <div class="drivers-title">Prediction Drivers</div>
      ${drivers.map(d => {
        const cls = d.includes('OUT') || d.includes('DOUBTFUL') ? 'negative'
                  : d.includes('advantage') ? 'positive'
                  : '';
        return `<div class="driver-item ${cls}">${d}</div>`;
      }).join('')}
    </div>` : '';

  // Explanation
  const explanation = game.explanation || '';
  const explanationHtml = explanation
    ? `<div class="explanation-box">${explanation}</div>` : '';

  // Efficiency stats (original display)
  const effHtml = isNba && game.efficiency ? `
    <div class="game-stats">
      ${game.efficiency.home_off_rating != null ? `<div class="stat-pill"><span class="stat-pill-label">Home ORtg</span><span class="stat-pill-value">${game.efficiency.home_off_rating.toFixed(1)}</span></div>` : ''}
      ${game.efficiency.away_off_rating != null ? `<div class="stat-pill"><span class="stat-pill-label">Away ORtg</span><span class="stat-pill-value">${game.efficiency.away_off_rating.toFixed(1)}</span></div>` : ''}
      ${game.efficiency.home_net_rating != null ? `<div class="stat-pill"><span class="stat-pill-label">Home Net</span><span class="stat-pill-value">${game.efficiency.home_net_rating >= 0 ? '+' : ''}${game.efficiency.home_net_rating.toFixed(1)}</span></div>` : ''}
      ${game.efficiency.away_net_rating != null ? `<div class="stat-pill"><span class="stat-pill-label">Away Net</span><span class="stat-pill-value">${game.efficiency.away_net_rating >= 0 ? '+' : ''}${game.efficiency.away_net_rating.toFixed(1)}</span></div>` : ''}
    </div>` : (adj.rest_diff != null || adj.travel_dist_miles ? `
    <div class="game-stats">
      ${adj.rest_diff != null ? `<div class="stat-pill"><span class="stat-pill-label">Rest</span><span class="stat-pill-value">${adj.rest_home}d / ${adj.rest_away}d</span></div>` : ''}
      ${adj.travel_dist_miles ? `<div class="stat-pill"><span class="stat-pill-label">Travel</span><span class="stat-pill-value">${Math.round(adj.travel_dist_miles)} mi</span></div>` : ''}
      ${mc.exp_margin != null ? `<div class="stat-pill"><span class="stat-pill-label">Exp Margin</span><span class="stat-pill-value">${mc.exp_margin > 0 ? '+' : ''}${mc.exp_margin?.toFixed(1)} pts</span></div>` : ''}
      ${market.home_prob != null ? `<div class="stat-pill"><span class="stat-pill-label">Market</span><span class="stat-pill-value">${pct(market.home_prob)}</span></div>` : ''}
      ${market.home_american != null ? `<div class="stat-pill"><span class="stat-pill-label">ML</span><span class="stat-pill-value">${market.home_american > 0 ? '+' : ''}${market.home_american?.toFixed(0)}</span></div>` : ''}
    </div>` : '');

  // Adjustment panel
  const adjPanelHtml = buildAdjPanelHtml(game.game_id, elo, game);

  return `
<div class="game-card${game.is_future ? ' game-card-future' : ''}" data-game-id="${game.game_id}">
  <div class="game-header">
    <div class="game-meta">
      ${game.status === 'STATUS_IN_PROGRESS' || game.status === 'STATUS_HALFTIME' ? '<span class="text-live">● LIVE</span> · ' : ''}
      ${game.status === 'STATUS_FINAL' ? '<span class="text-red">FINAL</span> · ' : ''}
      ${game.is_future ? '<span class="future-badge">Upcoming</span> · ' : ''}
      ${fmtDate(game.game_time)}
      ${game.neutral ? ' · Neutral Site' : ''}
    </div>
    <div class="game-header-right">
      <span class="adj-badge" style="display:none">Adjusted</span>
      ${edgeBadge}
      <button class="adj-toggle-btn" data-adj-toggle="${game.game_id}" title="Adjust factors">⚙ Adjust</button>
    </div>
  </div>

  <div class="matchup-row">
    <div class="team-side away">
      <img class="team-logo" src="${game.away_logo}" alt="${game.away_team}" onerror="this.style.display='none'" loading="lazy" />
      <div class="team-name-abbrev" title="${game.away_name}">${game.away_team}</div>
      <div class="team-record">${awayRecordStr} ${awayStreakBadge}</div>
      <div class="team-elo">ELO ${eloFmt(elo.away)}</div>
    </div>
    <div class="vs-label">@</div>
    <div class="team-side home">
      <img class="team-logo" src="${game.home_logo}" alt="${game.home_team}" onerror="this.style.display='none'" loading="lazy" />
      <div class="team-name-abbrev" title="${game.home_name}">${game.home_team}</div>
      <div class="team-record">${homeRecordStr} ${homeStreakBadge}</div>
      <div class="team-elo">ELO ${eloFmt(elo.home)}</div>
    </div>
  </div>

  ${scoreHtml}
  ${keyFactorsHtml}

  <div class="prob-bar-section">
    <div class="prob-labels">
      <span class="prob-away-label ${awayProb > ensProb ? 'text-blue' : ''}">${game.away_team} ${pct(awayProb)}</span>
      <span class="prob-home-label ${ensProb >= awayProb ? 'text-blue' : ''}">${game.home_team} ${pct(ensProb)}</span>
    </div>
    <div class="prob-bar">
      <div class="prob-fill ${edge != null ? (edge > 0.02 ? 'fill-green' : edge < -0.02 ? 'fill-red' : '') : ''}"
           style="width:${(ensProb * 100).toFixed(1)}%;margin-left:${((1-ensProb)*100).toFixed(1)}%;background:linear-gradient(90deg,var(--blue),var(--blue-light));border-radius:99px;position:absolute;right:0;"></div>
    </div>
  </div>

  ${effHtml}
  ${h2hHtml}
  ${contextHtml}

  ${mc.win_prob != null ? `
  <div class="mc-row">
    <div class="mc-cell">
      <div class="mc-cell-val">${pct(mc.win_prob)}</div>
      <div class="mc-cell-lbl">Win %</div>
    </div>
    <div class="mc-cell">
      <div class="mc-cell-val">${pct(mc.prob_7plus)}</div>
      <div class="mc-cell-lbl">Win 7+</div>
    </div>
    <div class="mc-cell">
      <div class="mc-cell-val">${pct(mc.prob_14plus)}</div>
      <div class="mc-cell-lbl">Win 14+</div>
    </div>
    <div class="mc-cell">
      <div class="mc-cell-val">${pct(mc.prob_21plus)}</div>
      <div class="mc-cell-lbl">Win 21+</div>
    </div>
  </div>` : ''}

  <div class="model-probs">
    ${p.logistic_prob != null ? `<div class="model-prob-pill"><span class="model-prob-label">LR</span><span class="model-prob-val">${pct(p.logistic_prob)}</span></div>` : ''}
    ${p.xgb_prob != null ? `<div class="model-prob-pill"><span class="model-prob-label">XGB</span><span class="model-prob-val">${pct(p.xgb_prob)}</span></div>` : ''}
    ${p.elo_prob != null ? `<div class="model-prob-pill"><span class="model-prob-label">ELO</span><span class="model-prob-val">${pct(p.elo_prob)}</span></div>` : ''}
    ${p.pyth_prob != null ? `<div class="model-prob-pill"><span class="model-prob-label">Pyth</span><span class="model-prob-val">${pct(p.pyth_prob)}</span></div>` : ''}
    ${p.eff_prob != null ? `<div class="model-prob-pill"><span class="model-prob-label">Eff</span><span class="model-prob-val">${pct(p.eff_prob)}</span></div>` : ''}
    ${p.bayesian_prob != null ? `<div class="model-prob-pill"><span class="model-prob-label">Bayes</span><span class="model-prob-val">${pct(p.bayesian_prob)}</span></div>` : ''}
  </div>

  ${injuryHtml}
  ${driversHtml}
  ${explanationHtml}
  ${adjPanelHtml}
</div>`;
}

/* ── Game context: rest, streak, travel, timezone ─────────────── */
function buildContextHtml(game, adj, isNba, homeTrend, awayTrend, tzShift) {
  const pills = [];

  // Rest days
  if (adj.rest_home != null && adj.rest_away != null) {
    const restClass = (adj.rest_home <= 1 || adj.rest_away <= 1) ? 'warn' : '';
    pills.push(`<div class="stat-pill ${restClass}"><span class="stat-pill-label">Rest</span><span class="stat-pill-value">${game.home_team} ${adj.rest_home}d / ${game.away_team} ${adj.rest_away}d</span></div>`);
  }

  // B2B (NBA)
  if (isNba) {
    if (adj.b2b_home) pills.push(`<div class="stat-pill warn"><span class="stat-pill-label">B2B</span><span class="stat-pill-value">${game.home_team}</span></div>`);
    if (adj.b2b_away) pills.push(`<div class="stat-pill warn"><span class="stat-pill-label">B2B</span><span class="stat-pill-value">${game.away_team}</span></div>`);
  }

  // Travel + timezone
  if (adj.travel_dist_miles > 0) {
    const longTrip = adj.travel_dist_miles > 2000 || tzShift >= 2;
    const travelClass = longTrip ? 'warn' : '';
    const tzStr = tzShift > 0 ? ` · ${tzShift}hr TZ` : '';
    pills.push(`<div class="stat-pill ${travelClass}"><span class="stat-pill-label">Travel</span><span class="stat-pill-value">${game.away_team}: ${Math.round(adj.travel_dist_miles).toLocaleString()} mi${tzStr}</span></div>`);
  }

  // Hot/cold streak
  if (homeTrend === 'up') pills.push(`<div class="stat-pill streak-hot"><span class="stat-pill-label">${game.home_team}</span><span class="stat-pill-value">🔥 Hot</span></div>`);
  else if (homeTrend === 'down') pills.push(`<div class="stat-pill streak-cold"><span class="stat-pill-label">${game.home_team}</span><span class="stat-pill-value">❄️ Cold</span></div>`);

  if (awayTrend === 'up') pills.push(`<div class="stat-pill streak-hot"><span class="stat-pill-label">${game.away_team}</span><span class="stat-pill-value">🔥 Hot</span></div>`);
  else if (awayTrend === 'down') pills.push(`<div class="stat-pill streak-cold"><span class="stat-pill-label">${game.away_team}</span><span class="stat-pill-value">❄️ Cold</span></div>`);

  if (!pills.length) return '';
  return `<div class="game-stats context-stats">${pills.join('')}</div>`;
}

/* ── Injury panel ─────────────────────────────────────────────── */
function buildInjuryHtml(game, isNba) {
  const injuries = game.injuries || {};
  const impact = game.injury_impact || {};
  const homeInjuries = injuries.home || [];
  const awayInjuries = injuries.away || [];
  const totalInjured = homeInjuries.length + awayInjuries.length;

  if (totalInjured === 0 && !impact.home_elo_penalty && !impact.away_elo_penalty) {
    return `
  <div class="injury-panel">
    <div class="inj-none">✅ No reported injuries for either team</div>
  </div>`;
  }

  const statusClass = s => {
    const sl = (s || '').toLowerCase();
    if (sl === 'out' || sl === 'injured reserve' || sl === 'ir') return 'out';
    if (sl === 'doubtful') return 'doubtful';
    if (sl === 'questionable') return 'questionable';
    return 'probable';
  };

  const playerChip = p => {
    const cls = statusClass(p.status);
    const name = p.name || p.player || 'Unknown';
    const pos = p.position || p.pos || '';
    const status = p.status || '';
    return `<span class="injury-chip ${cls}" title="${name} — ${status}">${pos ? `<span class="inj-pos">${pos}</span>` : ''}${name}<span class="inj-status">${status}</span></span>`;
  };

  const homePenalty = impact.home_elo_penalty || 0;
  const awayPenalty = impact.away_elo_penalty || 0;

  const homeSection = homeInjuries.length ? `
    <div class="inj-team-section">
      <div class="inj-team-label">${game.home_team} ${homePenalty > 0 ? `<span class="inj-elo-penalty">-${homePenalty.toFixed(0)} ELO</span>` : ''}</div>
      <div class="inj-chips">${homeInjuries.map(playerChip).join('')}</div>
    </div>` : '';

  const awaySection = awayInjuries.length ? `
    <div class="inj-team-section">
      <div class="inj-team-label">${game.away_team} ${awayPenalty > 0 ? `<span class="inj-elo-penalty">-${awayPenalty.toFixed(0)} ELO</span>` : ''}</div>
      <div class="inj-chips">${awayInjuries.map(playerChip).join('')}</div>
    </div>` : '';

  const keyOut = (impact.home_key_players_out || []).concat(impact.away_key_players_out || []);
  const keyOutHtml = keyOut.length ? `<div class="inj-key-out">Key out: ${keyOut.join(', ')}</div>` : '';

  return `
  <div class="injury-panel">
    <button class="inj-toggle-btn" data-injury-toggle="${game.game_id}">
      🏥 Injuries (${totalInjured}) ${(homePenalty + awayPenalty) > 0 ? '· Impact active' : ''}
    </button>
    <div class="inj-content" id="inj-${game.game_id}">
      ${keyOutHtml}
      ${homeSection}
      ${awaySection}
    </div>
  </div>`;
}

/* ── Per-game adjustment panel ────────────────────────────────── */
function buildAdjPanelHtml(gameId, elo, game) {
  const ov = state.gameOverrides[gameId] || {};
  const homeTeam = game?.home_team || 'Home';
  const awayTeam = game?.away_team || 'Away';
  return `
  <div class="adjust-panel" id="adj-${gameId}" style="display:none">
    <div class="adjust-panel-header">
      <span>Manual Factor Adjustments</span>
      <button class="adj-reset-btn" data-adj-reset="${gameId}">Reset</button>
    </div>
    <div class="adj-section-label">ELO &amp; Momentum</div>
    <div class="adjust-sliders">
      <div class="adjust-row">
        <div class="adjust-row-label">
          <span>${(elo.home || 1500).toFixed(0)} ELO</span>
          <label>${homeTeam} ELO Boost <span class="adj-val" id="adj-homeBoost-val-${gameId}">${ov.homeBoost || 0}</span></label>
          <span></span>
        </div>
        <input type="range" class="adj-slider" data-game-id="${gameId}" data-adj="homeBoost" min="-50" max="50" step="1" value="${ov.homeBoost || 0}" />
      </div>
      <div class="adjust-row">
        <div class="adjust-row-label">
          <span>${(elo.away || 1500).toFixed(0)} ELO</span>
          <label>${awayTeam} ELO Boost <span class="adj-val" id="adj-awayBoost-val-${gameId}">${ov.awayBoost || 0}</span></label>
          <span></span>
        </div>
        <input type="range" class="adj-slider" data-game-id="${gameId}" data-adj="awayBoost" min="-50" max="50" step="1" value="${ov.awayBoost || 0}" />
      </div>
      <div class="adjust-row">
        <div class="adjust-row-label">
          <span>0×</span>
          <label>Home Court/Field Multiplier <span class="adj-val" id="adj-hfaMult-val-${gameId}">${(ov.hfaMult != null ? ov.hfaMult : 1.0).toFixed(1)}</span>×</label>
          <span>2×</span>
        </div>
        <input type="range" class="adj-slider" data-game-id="${gameId}" data-adj="hfaMult" min="0" max="2" step="0.1" value="${ov.hfaMult != null ? ov.hfaMult : 1.0}" />
      </div>
      <div class="adjust-row">
        <div class="adjust-row-label">
          <span>-15</span>
          <label>Momentum (Home) <span class="adj-val" id="adj-momentumBoost-val-${gameId}">${ov.momentumBoost || 0}</span></label>
          <span>+15</span>
        </div>
        <input type="range" class="adj-slider" data-game-id="${gameId}" data-adj="momentumBoost" min="-15" max="15" step="1" value="${ov.momentumBoost || 0}" />
      </div>
      <div class="adjust-row">
        <div class="adjust-row-label">
          <span>-20</span>
          <label>Rest Advantage (Home) <span class="adj-val" id="adj-restFactor-val-${gameId}">${ov.restFactor || 0}</span></label>
          <span>+20</span>
        </div>
        <input type="range" class="adj-slider" data-game-id="${gameId}" data-adj="restFactor" min="-20" max="20" step="1" value="${ov.restFactor || 0}" />
      </div>
    </div>
    <div class="adj-section-label">Injury Overrides</div>
    <div class="adjust-sliders">
      <div class="adjust-row">
        <div class="adjust-row-label">
          <span>-30</span>
          <label>${homeTeam} Injury Override <span class="adj-val" id="adj-injHome-val-${gameId}">${ov.injHome || 0}</span></label>
          <span>+30</span>
        </div>
        <input type="range" class="adj-slider" data-game-id="${gameId}" data-adj="injHome" min="-30" max="30" step="1" value="${ov.injHome || 0}" />
      </div>
      <div class="adjust-row">
        <div class="adjust-row-label">
          <span>-30</span>
          <label>${awayTeam} Injury Override <span class="adj-val" id="adj-injAway-val-${gameId}">${ov.injAway || 0}</span></label>
          <span>+30</span>
        </div>
        <input type="range" class="adj-slider" data-game-id="${gameId}" data-adj="injAway" min="-30" max="30" step="1" value="${ov.injAway || 0}" />
      </div>
    </div>
    <div class="adj-section-label">Conditions</div>
    <div class="adjust-sliders">
      <div class="adjust-row">
        <div class="adjust-row-label">
          <span>-15</span>
          <label>Weather / Conditions (Home) <span class="adj-val" id="adj-weather-val-${gameId}">${ov.weather || 0}</span></label>
          <span>+15</span>
        </div>
        <input type="range" class="adj-slider" data-game-id="${gameId}" data-adj="weather" min="-15" max="15" step="1" value="${ov.weather || 0}" />
      </div>
    </div>
    <div class="adj-section-label">H2H &amp; Streak Overrides</div>
    <div class="adjust-sliders">
      <div class="adjust-row">
        <div class="adjust-row-label">
          <span>-15</span>
          <label>H2H Dominance (Home +) <span class="adj-val" id="adj-h2hFactor-val-${gameId}">${ov.h2hFactor || 0}</span></label>
          <span>+15</span>
        </div>
        <input type="range" class="adj-slider" data-game-id="${gameId}" data-adj="h2hFactor" min="-15" max="15" step="1" value="${ov.h2hFactor || 0}" />
      </div>
      <div class="adjust-row">
        <div class="adjust-row-label">
          <span>-10</span>
          <label>Streak Momentum (Home +) <span class="adj-val" id="adj-streakMomentum-val-${gameId}">${ov.streakMomentum || 0}</span></label>
          <span>+10</span>
        </div>
        <input type="range" class="adj-slider" data-game-id="${gameId}" data-adj="streakMomentum" min="-10" max="10" step="1" value="${ov.streakMomentum || 0}" />
      </div>
    </div>
    <div class="adj-notes-wrap">
      <label class="adj-notes-label">Notes / Reasoning</label>
      <textarea class="adj-notes-input" data-game-id="${gameId}" placeholder="e.g. Starting QB listed questionable, home crowd advantage in cold weather…" rows="2">${ov.notes || ''}</textarea>
    </div>
    <div class="adj-result" id="adj-result-${gameId}"></div>
  </div>`;
}

/* ── Recalculate a game card's probability with overrides ─────── */
function recalcGameCard(gameId) {
  const isNba = state.league === 'nba';
  const pred = isNba ? state.nba.predictions : state.predictions;
  const game = pred?.games?.find(g => g.game_id === gameId);
  if (!game) return;

  const ov = state.gameOverrides[gameId] || {};
  const homeBoost = ov.homeBoost || 0;
  const awayBoost = ov.awayBoost || 0;
  const hfaMult = ov.hfaMult != null ? ov.hfaMult : 1.0;
  const momentumBoost = ov.momentumBoost || 0;
  const restFactor = ov.restFactor || 0;
  const injHome = ov.injHome || 0;
  const injAway = ov.injAway || 0;
  const weather = ov.weather || 0;
  const h2hFactor = ov.h2hFactor || 0;
  const streakMomentum = ov.streakMomentum || 0;

  const elo = game.elo || {};
  const hfa = state.params.hfa;
  const adjHomeElo = (elo.home || 1500) + homeBoost + momentumBoost + restFactor + injHome + weather + h2hFactor + streakMomentum + hfa * hfaMult;
  const adjAwayElo = (elo.away || 1500) + awayBoost + injAway;
  const adjEloProb = clamp(1 / (1 + Math.pow(10, (adjAwayElo - adjHomeElo) / 400)), 0.01, 0.99);

  const p = game.predictions || {};
  const modifiedProbs = { ...p, elo_prob: adjEloProb };
  const newEns = ensembleFromProbs(modifiedProbs, state.weights);
  const newAway = 1 - newEns;

  const card = document.querySelector(`.game-card[data-game-id="${gameId}"]`);
  if (!card) return;

  const homeLabel = card.querySelector('.prob-home-label');
  const awayLabel = card.querySelector('.prob-away-label');
  const probFill = card.querySelector('.prob-fill');
  const adjResult = document.getElementById(`adj-result-${gameId}`);
  const adjBadge = card.querySelector('.adj-badge');

  if (homeLabel) {
    homeLabel.textContent = `${game.home_team} ${pct(newEns)}`;
    homeLabel.className = `prob-home-label ${newEns >= newAway ? 'text-blue' : ''}`;
  }
  if (awayLabel) {
    awayLabel.textContent = `${game.away_team} ${pct(newAway)}`;
    awayLabel.className = `prob-away-label ${newAway > newEns ? 'text-blue' : ''}`;
  }
  if (probFill) {
    probFill.style.width = `${(newEns * 100).toFixed(1)}%`;
    probFill.style.marginLeft = `${((1 - newEns) * 100).toFixed(1)}%`;
  }

  const isActive = homeBoost !== 0 || awayBoost !== 0 || hfaMult !== 1.0 || momentumBoost !== 0 || restFactor !== 0 || injHome !== 0 || injAway !== 0 || weather !== 0 || h2hFactor !== 0 || streakMomentum !== 0;
  if (adjResult) {
    adjResult.innerHTML = isActive
      ? `<div class="adj-result-row">Adjusted: <strong>${game.home_team} ${pct(newEns)}</strong> vs <strong>${game.away_team} ${pct(newAway)}</strong></div>`
      : '';
  }
  if (adjBadge) adjBadge.style.display = isActive ? 'inline-block' : 'none';
}

/* ── Event delegation for game card interactions ──────────────── */
document.addEventListener('click', e => {
  // Game filter pills
  const filterPill = e.target.closest('.filter-pill');
  if (filterPill) {
    state.gameFilter = filterPill.dataset.filter || 'all';
    renderGames();
    return;
  }

  // Injury toggle
  const injBtn = e.target.closest('[data-injury-toggle]');
  if (injBtn) {
    const panel = document.getElementById('inj-' + injBtn.dataset.injuryToggle);
    if (panel) panel.classList.toggle('hidden');
    return;
  }

  // Adjust toggle
  const adjBtn = e.target.closest('[data-adj-toggle]');
  if (adjBtn) {
    const panel = document.getElementById('adj-' + adjBtn.dataset.adjToggle);
    if (panel) {
      const isOpen = panel.style.display !== 'none';
      panel.style.display = isOpen ? 'none' : 'block';
      adjBtn.classList.toggle('active', !isOpen);
    }
    return;
  }

  // Adjust reset
  const resetBtn = e.target.closest('[data-adj-reset]');
  if (resetBtn) {
    const gameId = resetBtn.dataset.adjReset;
    delete state.gameOverrides[gameId];
    // Reset sliders and notes
    const panel = document.getElementById('adj-' + gameId);
    if (panel) {
      panel.querySelectorAll('.adj-slider').forEach(sl => {
        const adjType = sl.dataset.adj;
        sl.value = adjType === 'hfaMult' ? 1.0 : 0;
        const valEl = document.getElementById(`adj-${adjType}-val-${gameId}`);
        if (valEl) valEl.textContent = adjType === 'hfaMult' ? '1.0' : '0';
      });
      const notesEl = panel.querySelector('.adj-notes-input');
      if (notesEl) notesEl.value = '';
    }
    recalcGameCard(gameId);
    return;
  }
});

document.addEventListener('input', e => {
  const slider = e.target.closest('.adj-slider');
  if (!slider) return;
  const gameId = slider.dataset.gameId;
  const adjType = slider.dataset.adj;
  if (!gameId || !adjType) return;

  if (!state.gameOverrides[gameId]) {
    state.gameOverrides[gameId] = { homeBoost: 0, awayBoost: 0, hfaMult: 1.0, momentumBoost: 0, restFactor: 0, injHome: 0, injAway: 0, weather: 0, notes: '' };
  }
  state.gameOverrides[gameId][adjType] = parseFloat(slider.value);

  const valEl = document.getElementById(`adj-${adjType}-val-${gameId}`);
  if (valEl) {
    valEl.textContent = adjType === 'hfaMult'
      ? parseFloat(slider.value).toFixed(1)
      : slider.value;
  }
  recalcGameCard(gameId);
});

// Notes textarea handler
document.addEventListener('input', e => {
  const notes = e.target.closest('.adj-notes-input');
  if (!notes) return;
  const gameId = notes.dataset.gameId;
  if (!gameId) return;
  if (!state.gameOverrides[gameId]) state.gameOverrides[gameId] = {};
  state.gameOverrides[gameId].notes = notes.value;
});

/* ══════════════════════════════════════════════════════════════
   SECTION 2 — MODEL CONTROLS (sliders)
══════════════════════════════════════════════════════════════ */
function bindSlider(slId, valId, stateKey, stateObj, onUpdate, decimals = 0) {
  const sl = $(slId);
  const vd = $(valId);
  if (!sl || !vd) return;
  const fmt = v => decimals > 0 ? parseFloat(v).toFixed(decimals) : parseInt(v);
  sl.addEventListener('input', () => {
    const v = parseFloat(sl.value);
    vd.textContent = fmt(v);
    stateObj[stateKey] = v;
    if (onUpdate) onUpdate();
  });
}

bindSlider('sl-k', 'val-k', 'k', state.params, null, 0);
bindSlider('sl-hfa', 'val-hfa', 'hfa', state.params, null, 0);
bindSlider('sl-mov', 'val-mov', 'mov', state.params, null, 1);
bindSlider('sl-form', 'val-form', 'form', state.params, null, 2);
bindSlider('sl-sos', 'val-sos', 'sos', state.params, null, 2);
bindSlider('sl-h2h', 'val-h2h', 'h2h', state.params, null, 2);
bindSlider('sl-to', 'val-to', 'to', state.params, null, 1);
bindSlider('sl-rt', 'val-rt', 'rt', state.params, null, 1);
bindSlider('sl-lambda', 'val-lambda', 'lambda', state.params, null, 2);

function onWeightChange() { updateWeightTotal(); }

bindSlider('sl-w-log',  'val-w-log',  'logistic', state.weights, onWeightChange, 2);
bindSlider('sl-w-xgb',  'val-w-xgb',  'xgboost',  state.weights, onWeightChange, 2);
bindSlider('sl-w-elo',  'val-w-elo',  'elo',       state.weights, onWeightChange, 2);
bindSlider('sl-w-pyth', 'val-w-pyth', 'pyth',      state.weights, onWeightChange, 2);
bindSlider('sl-w-eff',  'val-w-eff',  'eff',       state.weights, onWeightChange, 2);

function updateWeightTotal() {
  const total = Object.values(state.weights).reduce((a, b) => a + b, 0);
  const el = $('weight-total-val');
  el.textContent = total.toFixed(2);
  el.className = 'weight-total-val ' + (Math.abs(total - 1) < 0.01 ? 'ok' : 'warn');
}

/* ══════════════════════════════════════════════════════════════
   SECTION 3 — MATCHUP PREDICTOR
══════════════════════════════════════════════════════════════ */
const NFL_TEAMS = [
  "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
  "HOU","IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG",
  "NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"
];

const TEAM_NAMES = {
  ARI:"Arizona Cardinals", ATL:"Atlanta Falcons", BAL:"Baltimore Ravens",
  BUF:"Buffalo Bills", CAR:"Carolina Panthers", CHI:"Chicago Bears",
  CIN:"Cincinnati Bengals", CLE:"Cleveland Browns", DAL:"Dallas Cowboys",
  DEN:"Denver Broncos", DET:"Detroit Lions", GB:"Green Bay Packers",
  HOU:"Houston Texans", IND:"Indianapolis Colts", JAX:"Jacksonville Jaguars",
  KC:"Kansas City Chiefs", LAC:"Los Angeles Chargers", LAR:"Los Angeles Rams",
  LV:"Las Vegas Raiders", MIA:"Miami Dolphins", MIN:"Minnesota Vikings",
  NE:"New England Patriots", NO:"New Orleans Saints", NYG:"New York Giants",
  NYJ:"New York Jets", PHI:"Philadelphia Eagles", PIT:"Pittsburgh Steelers",
  SEA:"Seattle Seahawks", SF:"San Francisco 49ers", TB:"Tampa Bay Buccaneers",
  TEN:"Tennessee Titans", WAS:"Washington Commanders"
};

const NBA_TEAMS_LIST = [
  "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET","GSW",
  "HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK",
  "OKC","ORL","PHI","PHX","POR","SAC","SAS","TOR","UTA","WAS"
];

const NBA_TEAM_NAMES = {
  ATL:"Atlanta Hawks", BOS:"Boston Celtics", BKN:"Brooklyn Nets",
  CHA:"Charlotte Hornets", CHI:"Chicago Bulls", CLE:"Cleveland Cavaliers",
  DAL:"Dallas Mavericks", DEN:"Denver Nuggets", DET:"Detroit Pistons",
  GSW:"Golden State Warriors", HOU:"Houston Rockets", IND:"Indiana Pacers",
  LAC:"LA Clippers", LAL:"Los Angeles Lakers", MEM:"Memphis Grizzlies",
  MIA:"Miami Heat", MIL:"Milwaukee Bucks", MIN:"Minnesota Timberwolves",
  NOP:"New Orleans Pelicans", NYK:"New York Knicks", OKC:"Oklahoma City Thunder",
  ORL:"Orlando Magic", PHI:"Philadelphia 76ers", PHX:"Phoenix Suns",
  POR:"Portland Trail Blazers", SAC:"Sacramento Kings", SAS:"San Antonio Spurs",
  TOR:"Toronto Raptors", UTA:"Utah Jazz", WAS:"Washington Wizards"
};

// Coordinates for travel distance
const NFL_COORDS = {
  ARI:[33.5276,-112.2626], ATL:[33.7553,-84.4006], BAL:[39.2780,-76.6227],
  BUF:[42.7738,-78.7870], CAR:[35.2258,-80.8528], CHI:[41.8623,-87.6167],
  CIN:[39.0954,-84.5160], CLE:[41.5061,-81.6995], DAL:[32.7473,-97.0945],
  DEN:[39.7439,-105.0201], DET:[42.3400,-83.0456], GB:[44.5013,-88.0622],
  HOU:[29.6847,-95.4107], IND:[39.7601,-86.1639], JAX:[30.3239,-81.6373],
  KC:[39.0490,-94.4839], LAC:[33.8644,-118.2611], LAR:[33.9534,-118.3392],
  LV:[36.0909,-115.1833], MIA:[25.9580,-80.2389], MIN:[44.9737,-93.2575],
  NE:[42.0909,-71.2643], NO:[29.9511,-90.0812], NYG:[40.8135,-74.0745],
  NYJ:[40.8135,-74.0745], PHI:[39.9008,-75.1675], PIT:[40.4468,-80.0158],
  SEA:[47.5952,-122.3316], SF:[37.4032,-121.9698], TB:[27.9759,-82.5033],
  TEN:[36.1665,-86.7713], WAS:[38.9078,-76.8645]
};

const NBA_COORDS = {
  ATL:[33.7490,-84.3880], BOS:[42.3601,-71.0589], BKN:[40.6826,-73.9754],
  CHA:[35.2271,-80.8431], CHI:[41.8781,-87.6298], CLE:[41.4993,-81.6944],
  DAL:[32.7767,-96.7970], DEN:[39.7392,-104.9903], DET:[42.3314,-83.0458],
  GSW:[37.7680,-122.3877], HOU:[29.7604,-95.3698], IND:[39.7684,-86.1581],
  LAC:[34.0430,-118.2673], LAL:[34.0430,-118.2673], MEM:[35.1495,-90.0490],
  MIA:[25.7617,-80.1918], MIL:[43.0389,-76.0253], MIN:[44.9778,-93.2650],
  NOP:[29.9511,-90.0715], NYK:[40.7505,-73.9934], OKC:[35.4634,-97.5151],
  ORL:[28.5383,-81.3792], PHI:[39.9012,-75.1720], PHX:[33.4484,-112.0740],
  POR:[45.5231,-122.6765], SAC:[38.5816,-121.4944], SAS:[29.4241,-98.4936],
  TOR:[43.6532,-79.3832], UTA:[40.7608,-111.8910], WAS:[38.9072,-77.0369]
};

function haversine(a, b) {
  const R = 3959;
  const dLat = (b[0]-a[0]) * Math.PI/180;
  const dLon = (b[1]-a[1]) * Math.PI/180;
  const x = Math.sin(dLat/2)**2 + Math.cos(a[0]*Math.PI/180)*Math.cos(b[0]*Math.PI/180)*Math.sin(dLon/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1-x));
}

function populatePredictor() {
  const homeEl = $('pred-home');
  const awayEl = $('pred-away');
  if (!homeEl || !awayEl) return;

  const isNba = state.league === 'nba';

  if (isNba) {
    const lb = state.nba.leaderboard?.teams || [];
    const teams = lb.length > 0
      ? lb.map(t => ({ abbrev: t.team, name: t.team_name }))
      : NBA_TEAMS_LIST.map(t => ({ abbrev: t, name: NBA_TEAM_NAMES[t] || t }));
    const opts = teams.map(t => `<option value="${t.abbrev}">${t.abbrev} — ${t.name}</option>`).join('');
    homeEl.innerHTML = '<option value="">Select home team...</option>' + opts;
    awayEl.innerHTML = '<option value="">Select away team...</option>' + opts;
    homeEl.value = 'LAL';
    awayEl.value = 'GSW';
  } else {
    const lb = state.leaderboard?.teams || [];
    const teams = lb.length > 0
      ? lb.map(t => ({ abbrev: t.team, name: t.team_name }))
      : NFL_TEAMS.map(t => ({ abbrev: t, name: TEAM_NAMES[t] || t }));
    const opts = teams.map(t => `<option value="${t.abbrev}">${t.abbrev} — ${t.name}</option>`).join('');
    homeEl.innerHTML = '<option value="">Select home team...</option>' + opts;
    awayEl.innerHTML = '<option value="">Select away team...</option>' + opts;
    homeEl.value = 'KC';
    awayEl.value = 'BUF';
  }
}

$('btn-predict')?.addEventListener('click', runPredictor);

function runPredictor() {
  const home = $('pred-home').value;
  const away = $('pred-away').value;
  const neutral = parseInt($('pred-neutral').value || '0') === 1;

  if (!home || !away) { alert('Please select both teams.'); return; }
  if (home === away) { alert('Please select two different teams.'); return; }

  if (state.league === 'nba') {
    runNbaPredictor(home, away, neutral);
  } else {
    runNflPredictor(home, away, neutral);
  }
}

function runNflPredictor(home, away, neutral) {
  const restHome = parseInt($('pred-rest-home').value || '7');
  const restAway = parseInt($('pred-rest-away').value || '7');

  const lb = state.leaderboard?.teams || [];
  const homeData = lb.find(t => t.team === home) || {};
  const awayData = lb.find(t => t.team === away) || {};

  const homeElo = homeData.elo || 1500;
  const awayElo = awayData.elo || 1500;
  const hfa = neutral ? 0 : state.params.hfa;

  const eloDiff = (homeElo + hfa) - awayElo;
  const eloProb = 1.0 / (1.0 + Math.pow(10, -eloDiff / 400));

  const homePyth = homeData.pyth || 0.5;
  const awayPyth = awayData.pyth || 0.5;
  const pythDiff = (1500 + (homePyth - 0.5) * 400 + hfa) - (1500 + (awayPyth - 0.5) * 400);
  const pythProb = 1.0 / (1.0 + Math.pow(10, -pythDiff / 400));

  const homeEff = homeData.net_eff || 0;
  const awayEff = awayData.net_eff || 0;
  const effDiff = (1500 + homeEff * 200 + hfa) - (1500 + awayEff * 200);
  const effProb = 1.0 / (1.0 + Math.pow(10, -effDiff / 400));

  const restDiff = restHome - restAway;
  const restAdj = restDiff * 1.5;
  const eloWithRest = ((homeElo + hfa + restAdj) - awayElo);
  const logisticProb = 1.0 / (1.0 + Math.pow(10, -eloWithRest / 400));

  const ensProb = ensembleFromProbs({
    logistic_prob: logisticProb, xgb_prob: null,
    elo_prob: eloProb, pyth_prob: pythProb, eff_prob: effProb,
  }, state.weights);

  const cHome = NFL_COORDS[home];
  const cAway = NFL_COORDS[away];
  const travelMi = (cHome && cAway) ? haversine(cAway, cHome) : 0;

  const winner = ensProb >= 0.5 ? home : away;
  const winnerProb = ensProb >= 0.5 ? ensProb : 1 - ensProb;
  const loserName = TEAM_NAMES[ensProb >= 0.5 ? away : home] || (ensProb >= 0.5 ? away : home);
  const winnerName = TEAM_NAMES[winner] || winner;
  const eloDiffAbs = Math.abs(homeElo - awayElo);
  const confidence = winnerProb > 0.7 ? 'strong' : winnerProb > 0.6 ? 'moderate' : 'slight';
  const pythFavorite = homePyth > awayPyth ? home : away;
  const effFavorite = homeEff > awayEff ? home : away;

  const explanation = `The model gives <strong>${winnerName}</strong> a <strong>${pct(winnerProb)}</strong> win probability — a <strong>${confidence} favorite</strong> over ${loserName}.
The ELO gap is <strong>${Math.round(eloDiffAbs)} points</strong>${!neutral ? `, with home field adding approximately <strong>${state.params.hfa} ELO points</strong>` : ' at a neutral site'}.
Pythagorean expectation favors <strong>${TEAM_NAMES[pythFavorite] || pythFavorite}</strong> (${pct(pythProb)} home win prob), and efficiency metrics favor <strong>${TEAM_NAMES[effFavorite] || effFavorite}</strong>.
${restDiff !== 0 ? `The rest advantage (<strong>${restDiff > 0 ? home : away}</strong> has ${Math.abs(restDiff)} more days) is factored in.` : 'Both teams have equal rest.'}
${travelMi > 500 ? `${away} travels approximately <strong>${Math.round(travelMi)} miles</strong>, which contributes a fatigue penalty.` : ''}`;

  // Prediction drivers
  const drivers = [];
  if (eloDiffAbs >= 50) drivers.push(`ELO advantage: ${winner} +${Math.round(eloDiffAbs)} rating points`);
  if (Math.abs(homeEff - awayEff) >= 0.1) drivers.push(`Efficiency gap: ${home} net ${homeEff >= 0 ? '+' : ''}${homeEff.toFixed(3)} vs ${away} net ${awayEff >= 0 ? '+' : ''}${awayEff.toFixed(3)}`);
  if (Math.abs(restDiff) >= 3) drivers.push(`Rest advantage: ${restDiff > 0 ? home : away} has ${Math.abs(restDiff)} extra days rest`);
  if (travelMi >= 1500) drivers.push(`Travel penalty: ${away} travels ${Math.round(travelMi)} miles`);
  if (!neutral) drivers.push(`Home field: ${home} +${state.params.hfa} ELO advantage`);

  renderPredictorResults({
    home, away, ensProb, eloProb, pythProb, effProb, logisticProb,
    homeElo, awayElo, homePyth, awayPyth, homeEff, awayEff,
    homeData, awayData, travelMi, restHome, restAway, neutral, explanation, hfa: state.params.hfa,
    teamNames: TEAM_NAMES, logoBase: 'nfl', drivers,
    isNba: false,
  });
}

function runNbaPredictor(home, away, neutral) {
  const homeRestVal = $('pred-nba-rest-home').value;
  const awayRestVal = $('pred-nba-rest-away').value;
  const b2bHome = homeRestVal === 'b2b';
  const b2bAway = awayRestVal === 'b2b';

  const lb = state.nba.leaderboard?.teams || [];
  const homeData = lb.find(t => t.team === home) || {};
  const awayData = lb.find(t => t.team === away) || {};

  const homeElo = homeData.elo || 1500;
  const awayElo = awayData.elo || 1500;
  const hfa = neutral ? 0 : 100;

  const b2bAdjHome = b2bHome ? -5 : 0;
  const b2bAdjAway = b2bAway ? -5 : 0;

  const eloDiff = (homeElo + hfa + b2bAdjHome) - (awayElo + b2bAdjAway);
  const eloProb = 1.0 / (1.0 + Math.pow(10, -eloDiff / 400));

  const homePyth = homeData.pyth || 0.5;
  const awayPyth = awayData.pyth || 0.5;
  const pythDiff = (1500 + (homePyth - 0.5) * 400 + hfa) - (1500 + (awayPyth - 0.5) * 400);
  const pythProb = 1.0 / (1.0 + Math.pow(10, -pythDiff / 400));

  const homeNetRtg = homeData.net_rating || homeData.net_eff || 0;
  const awayNetRtg = awayData.net_rating || awayData.net_eff || 0;
  const effDiff = (1500 + homeNetRtg * 10 + hfa) - (1500 + awayNetRtg * 10);
  const effProb = 1.0 / (1.0 + Math.pow(10, -effDiff / 400));

  const logisticProb = eloProb;

  const ensProb = ensembleFromProbs({
    logistic_prob: logisticProb, xgb_prob: null,
    elo_prob: eloProb, pyth_prob: pythProb, eff_prob: effProb,
  }, state.weights);

  const cHome = NBA_COORDS[home];
  const cAway = NBA_COORDS[away];
  const travelMi = (cHome && cAway) ? haversine(cAway, cHome) : 0;

  const winner = ensProb >= 0.5 ? home : away;
  const winnerProb = ensProb >= 0.5 ? ensProb : 1 - ensProb;
  const loserName = NBA_TEAM_NAMES[ensProb >= 0.5 ? away : home] || (ensProb >= 0.5 ? away : home);
  const winnerName = NBA_TEAM_NAMES[winner] || winner;
  const eloDiffAbs = Math.abs(homeElo - awayElo);
  const confidence = winnerProb > 0.7 ? 'strong' : winnerProb > 0.6 ? 'moderate' : 'slight';

  const explanation = `The model gives <strong>${winnerName}</strong> a <strong>${pct(winnerProb)}</strong> win probability — a <strong>${confidence} favorite</strong> over ${loserName}.
The ELO gap is <strong>${Math.round(eloDiffAbs)} points</strong>${!neutral ? `, with home court adding approximately <strong>100 ELO points</strong>` : ' at a neutral site'}.
Net rating: <strong>${home} ${homeNetRtg >= 0 ? '+' : ''}${homeNetRtg.toFixed(1)}</strong> vs <strong>${away} ${awayNetRtg >= 0 ? '+' : ''}${awayNetRtg.toFixed(1)}</strong>.
${b2bHome ? `<strong>${home}</strong> is on a back-to-back (−5 rating penalty).` : ''}
${b2bAway ? `<strong>${away}</strong> is on a back-to-back (−5 rating penalty).` : ''}
${travelMi > 500 ? `The away team travels approximately <strong>${Math.round(travelMi)} miles</strong>.` : ''}`;

  const drivers = [];
  if (eloDiffAbs >= 75) drivers.push(`ELO advantage: ${winner} +${Math.round(eloDiffAbs)} rating points`);
  if (Math.abs(homeNetRtg - awayNetRtg) >= 3) drivers.push(`Net rating gap: ${home} ${homeNetRtg >= 0 ? '+' : ''}${homeNetRtg.toFixed(1)} vs ${away} ${awayNetRtg >= 0 ? '+' : ''}${awayNetRtg.toFixed(1)}`);
  if (b2bHome) drivers.push(`Back-to-back: ${home} playing on zero days rest (−5 rating)`);
  if (b2bAway) drivers.push(`Back-to-back: ${away} playing on zero days rest (−5 rating)`);
  if (travelMi >= 1500) drivers.push(`Travel: away team travels ${Math.round(travelMi)} miles`);
  if (!neutral) drivers.push(`Home court: ${home} +100 ELO advantage`);

  renderPredictorResults({
    home, away, ensProb, eloProb, pythProb, effProb, logisticProb,
    homeElo, awayElo, homePyth, awayPyth, homeEff: homeNetRtg, awayEff: awayNetRtg,
    homeData, awayData, travelMi, restHome: homeRestVal, restAway: awayRestVal,
    neutral, explanation, hfa,
    teamNames: NBA_TEAM_NAMES, logoBase: 'nba', drivers,
    isNba: true, b2bHome, b2bAway,
  });
}

function renderPredictorResults(d) {
  const el = $('predictor-results');
  const logoBase = d.logoBase || 'nfl';
  const teamNames = d.teamNames || TEAM_NAMES;
  const effLabel = d.isNba ? 'Net Rating' : 'Net Efficiency';
  const hfaLabel = d.isNba ? 'Home Court Bonus' : 'Home Field Bonus';

  const driversHtml = d.drivers && d.drivers.length ? `
    <div class="drivers-section">
      <div class="drivers-title">Prediction Drivers</div>
      ${d.drivers.map(dr => `<div class="driver-item">${dr}</div>`).join('')}
    </div>` : '';

  el.innerHTML = `
<div>
  <div class="predictor-matchup">
    <div>
      <img class="predictor-team-logo" src="https://a.espncdn.com/i/teamlogos/${logoBase}/500/${d.away.toLowerCase()}.png" alt="${d.away}" onerror="this.style.display='none'" />
      <div class="predictor-team-name">${d.away}</div>
      <div class="muted" style="font-size:0.8rem;">${teamNames[d.away] || ''}</div>
      <div class="predictor-prob text-blue">${pct(1 - d.ensProb)}</div>
      <div class="predictor-prob-lbl">Win probability</div>
    </div>
    <div style="text-align:center;">
      <div class="vs-label" style="font-size:1.5rem;">@</div>
      <div style="font-size:0.7rem;color:var(--text-muted);margin-top:0.5rem;">${d.neutral ? 'Neutral' : 'Home'}</div>
    </div>
    <div>
      <img class="predictor-team-logo" src="https://a.espncdn.com/i/teamlogos/${logoBase}/500/${d.home.toLowerCase()}.png" alt="${d.home}" onerror="this.style.display='none'" />
      <div class="predictor-team-name">${d.home}</div>
      <div class="muted" style="font-size:0.8rem;">${teamNames[d.home] || ''}</div>
      <div class="predictor-prob text-blue">${pct(d.ensProb)}</div>
      <div class="predictor-prob-lbl">Win probability</div>
    </div>
  </div>

  <div class="prob-bar-section" style="margin-bottom:1rem;">
    <div class="prob-labels">
      <span>${d.away} ${pct(1-d.ensProb)}</span>
      <span>${d.home} ${pct(d.ensProb)}</span>
    </div>
    <div class="prob-bar" style="height:14px;">
      <div style="position:absolute;right:0;height:100%;width:${(d.ensProb*100).toFixed(1)}%;background:linear-gradient(90deg,var(--blue),var(--blue-light));border-radius:99px;"></div>
    </div>
  </div>

  <div class="predictor-details">
    <div class="detail-block">
      <div class="detail-block-label">ELO</div>
      <div class="detail-block-value">${d.away}: ${eloFmt(d.awayElo)} / ${d.home}: ${eloFmt(d.homeElo)}</div>
    </div>
    <div class="detail-block">
      <div class="detail-block-label">ELO Win Prob</div>
      <div class="detail-block-value">${pct(d.eloProb)}</div>
    </div>
    <div class="detail-block">
      <div class="detail-block-label">Pythagorean</div>
      <div class="detail-block-value">${d.away}: ${(d.awayPyth*100).toFixed(1)}% / ${d.home}: ${(d.homePyth*100).toFixed(1)}%</div>
    </div>
    <div class="detail-block">
      <div class="detail-block-label">Pyth Win Prob</div>
      <div class="detail-block-value">${pct(d.pythProb)}</div>
    </div>
    <div class="detail-block">
      <div class="detail-block-label">${effLabel}</div>
      <div class="detail-block-value">${d.away}: ${d.awayEff.toFixed(d.isNba ? 1 : 3)} / ${d.home}: ${d.homeEff.toFixed(d.isNba ? 1 : 3)}</div>
    </div>
    <div class="detail-block">
      <div class="detail-block-label">Eff Win Prob</div>
      <div class="detail-block-value">${pct(d.effProb)}</div>
    </div>
    <div class="detail-block">
      <div class="detail-block-label">Rest</div>
      <div class="detail-block-value">${d.away}: ${d.restAway} / ${d.home}: ${d.restHome}</div>
    </div>
    <div class="detail-block">
      <div class="detail-block-label">Travel Distance</div>
      <div class="detail-block-value">${d.travelMi > 0 ? Math.round(d.travelMi) + ' mi' : 'N/A'}</div>
    </div>
    <div class="detail-block">
      <div class="detail-block-label">${hfaLabel}</div>
      <div class="detail-block-value">${d.neutral ? 'N/A (neutral)' : d.hfa + ' ELO pts'}</div>
    </div>
    <div class="detail-block">
      <div class="detail-block-label">Ensemble Prob</div>
      <div class="detail-block-value text-blue">${pct(d.ensProb)} (${d.home})</div>
    </div>
    <div class="detail-block">
      <div class="detail-block-label">Playoff Prob</div>
      <div class="detail-block-value">${d.away}: ${pct(d.awayData.playoff_prob)} / ${d.home}: ${pct(d.homeData.playoff_prob)}</div>
    </div>
    <div class="detail-block">
      <div class="detail-block-label">${d.isNba ? 'Champ Prob' : 'SB Prob'}</div>
      <div class="detail-block-value">${d.away}: ${pct(d.awayData.sb_prob || d.awayData.champ_prob)} / ${d.home}: ${pct(d.homeData.sb_prob || d.homeData.champ_prob)}</div>
    </div>
  </div>

  ${driversHtml}
  <div class="explanation-box">${d.explanation}</div>
</div>`;
}

/* ══════════════════════════════════════════════════════════════
   SECTION 4 — ELO LEADERBOARD
══════════════════════════════════════════════════════════════ */
function renderLeaderboard() {
  sortAndRenderLeaderboard();
}

function sortAndRenderLeaderboard() {
  const isNba = state.league === 'nba';
  const lb = isNba ? state.nba.leaderboard : state.leaderboard;
  if (!lb || !lb.teams) return;

  let teams = [...lb.teams];
  const col = state.sortCol;
  const dir = state.sortDir;

  teams.sort((a, b) => {
    let va = a[col], vb = b[col];
    if (typeof va === 'string') va = va.toLowerCase();
    if (typeof vb === 'string') vb = vb.toLowerCase();
    if (va < vb) return dir === 'asc' ? -1 : 1;
    if (va > vb) return dir === 'asc' ? 1 : -1;
    return 0;
  });

  const body = $('leaderboard-body');
  if (!body) return;

  body.innerHTML = teams.map((t, i) => {
    const rank = i + 1;
    const rankClass = rank === 1 ? 'rank-1' : rank === 2 ? 'rank-2' : rank === 3 ? 'rank-3' : 'rank-n';
    const trendIcon = t.trend === 'up' ? '▲' : t.trend === 'down' ? '▼' : '—';
    const trendClass = t.trend === 'up' ? 'trend-up' : t.trend === 'down' ? 'trend-down' : 'trend-neutral';
    const playoffPct = (t.playoff_prob || 0) * 100;

    const netEffVal = isNba
      ? (t.net_rating != null ? (t.net_rating >= 0 ? '+' : '') + t.net_rating.toFixed(1) : '—')
      : (t.net_eff != null ? ((t.net_eff >= 0 ? '+' : '') + t.net_eff.toFixed(3)) : '—');
    const netEffClass = isNba
      ? (t.net_rating >= 0 ? 'text-green' : 'text-red')
      : (t.net_eff >= 0 ? 'text-green' : 'text-red');

    const offEffVal = isNba
      ? (t.offensive_rating != null ? t.offensive_rating.toFixed(1) : '—')
      : (t.off_eff != null ? t.off_eff.toFixed(3) : '—');
    const defEffVal = isNba
      ? (t.defensive_rating != null ? t.defensive_rating.toFixed(1) : '—')
      : (t.def_eff != null ? t.def_eff.toFixed(3) : '—');
    const sbProbVal = t.champ_prob != null ? pct(t.champ_prob) : pct(t.sb_prob);

    return `<tr>
      <td><span class="rank-badge ${rankClass}">${rank}</span></td>
      <td>
        <div class="team-cell">
          <img class="team-logo-sm" src="${t.logo}" alt="${t.team}" onerror="this.style.display='none'" loading="lazy" />
          <div>
            <div class="team-abbrev">${t.team}</div>
            <div class="team-full">${t.team_name || ''}</div>
          </div>
        </div>
      </td>
      <td>
        <div class="elo-cell">
          <span class="elo-value">${eloFmt(t.elo)}</span>
        </div>
      </td>
      <td class="elo-band">${eloFmt(t.lower_band)} – ${eloFmt(t.upper_band)}</td>
      <td class="mono">${t.wins}-${t.losses}${t.ties > 0 ? '-' + t.ties : ''}</td>
      <td class="mono">${(t.pyth * 100).toFixed(1)}%</td>
      <td class="mono ${netEffClass}">${netEffVal}</td>
      <td class="mono">${offEffVal}</td>
      <td class="mono">${defEffVal}</td>
      <td>
        <div style="display:flex;align-items:center;gap:0.5rem;">
          <span class="mono">${pct(t.playoff_prob)}</span>
          <div class="playoff-bar"><div class="playoff-fill" style="width:${playoffPct.toFixed(1)}%"></div></div>
        </div>
      </td>
      <td class="mono">${sbProbVal}</td>
      <td><span class="trend-icon ${trendClass}">${trendIcon}</span></td>
    </tr>`;
  }).join('');
}

// Sortable column headers
document.querySelectorAll('#leaderboard-table th[data-col]').forEach(th => {
  th.addEventListener('click', () => {
    const col = th.dataset.col;
    if (col === 'rank') return;
    if (state.sortCol === col) {
      state.sortDir = state.sortDir === 'desc' ? 'asc' : 'desc';
    } else {
      state.sortCol = col;
      state.sortDir = 'desc';
    }
    document.querySelectorAll('#leaderboard-table th').forEach(h => {
      h.classList.remove('sorted');
      const si = h.querySelector('.sort-icon');
      if (si) si.textContent = '▼';
    });
    th.classList.add('sorted');
    const si = th.querySelector('.sort-icon');
    if (si) si.textContent = state.sortDir === 'asc' ? '▲' : '▼';
    sortAndRenderLeaderboard();
  });
});

/* ══════════════════════════════════════════════════════════════
   SECTION 5 — MODEL PERFORMANCE
══════════════════════════════════════════════════════════════ */
function renderMetrics() {
  const isNba = state.league === 'nba';
  const m = isNba ? state.nba.modelMetrics : state.modelMetrics;
  if (!m) return;

  const label = isNba ? 'NBA model' : `Trained on ${(m.n_training_games || 0).toLocaleString()} games`;
  $('metrics-trained-on').textContent = m.n_training_games
    ? `Trained on ${m.n_training_games.toLocaleString()} games`
    : (isNba ? 'NBA model' : 'Model metrics');

  $('metrics-row').innerHTML = `
    <div class="metric-card">
      <div class="metric-label">Log Loss</div>
      <div class="metric-value">${m.log_loss != null ? m.log_loss.toFixed(4) : '—'}</div>
      <div class="metric-desc">Lower is better. Perfect = 0</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Brier Score</div>
      <div class="metric-value">${m.brier_score != null ? m.brier_score.toFixed(4) : '—'}</div>
      <div class="metric-desc">Lower is better. Perfect = 0</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">AUC-ROC</div>
      <div class="metric-value">${m.auc != null ? m.auc.toFixed(4) : '—'}</div>
      <div class="metric-desc">Higher is better. Perfect = 1</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">XGBoost</div>
      <div class="metric-value" style="font-size:1.25rem;">${m.xgboost_available ? '✅ Active' : '⚠️ Off'}</div>
      <div class="metric-desc">Gradient boosting model status</div>
    </div>`;
}

function renderCharts() {
  const isNba = state.league === 'nba';
  const m = isNba ? state.nba.modelMetrics : state.modelMetrics;
  if (!m) return;

  renderCalibrationChart(m.calibration_buckets || []);
  renderAccuracyChart(m.historical_accuracy || []);
}

function renderCalibrationChart(buckets) {
  const ctx = $('calibration-chart')?.getContext('2d');
  if (!ctx) return;

  if (calibrationChart) { calibrationChart.destroy(); }

  if (!buckets.length) {
    calibrationChart = null;
    return;
  }

  const labels = buckets.map(b => b.bucket);
  const predicted = buckets.map(b => b.predicted * 100);
  const actual    = buckets.map(b => b.actual * 100);

  calibrationChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Actual',
          data: actual,
          borderColor: '#3fb950',
          backgroundColor: 'rgba(63,185,80,0.15)',
          borderWidth: 2, fill: false, tension: 0.3, pointRadius: 4,
        },
        {
          label: 'Predicted',
          data: predicted,
          borderColor: '#58a6ff',
          backgroundColor: 'rgba(88,166,255,0.1)',
          borderWidth: 2, fill: false, tension: 0.3, pointRadius: 4,
        },
        {
          label: 'Perfect',
          data: predicted,
          borderColor: '#30363d',
          borderDash: [4, 4],
          borderWidth: 1, fill: false, pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#8b949e', font: { size: 11 } } },
        tooltip: { mode: 'index', intersect: false },
      },
      scales: {
        x: { ticks: { color: '#8b949e', font: { size: 10 } }, grid: { color: '#30363d' } },
        y: {
          ticks: { color: '#8b949e', font: { size: 10 }, callback: v => v.toFixed(0) + '%' },
          grid: { color: '#30363d' },
          min: 0, max: 100,
        },
      },
    },
  });
}

function renderAccuracyChart(history) {
  const ctx = $('accuracy-chart')?.getContext('2d');
  if (!ctx) return;

  if (accuracyChart) { accuracyChart.destroy(); }

  if (!history.length) {
    accuracyChart = null;
    return;
  }

  const labels = history.map(h => h.year.toString());
  const values  = history.map(h => (h.accuracy * 100));

  accuracyChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Accuracy',
        data: values,
        backgroundColor: values.map(v =>
          v >= 68 ? 'rgba(63,185,80,0.6)' : v >= 65 ? 'rgba(88,166,255,0.6)' : 'rgba(218,54,51,0.6)'
        ),
        borderColor: values.map(v =>
          v >= 68 ? '#3fb950' : v >= 65 ? '#58a6ff' : '#f85149'
        ),
        borderWidth: 1, borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ' ' + ctx.parsed.y.toFixed(2) + '%' } },
      },
      scales: {
        x: { ticks: { color: '#8b949e', font: { size: 11 } }, grid: { display: false } },
        y: {
          ticks: { color: '#8b949e', font: { size: 10 }, callback: v => v.toFixed(0) + '%' },
          grid: { color: '#30363d' },
          min: 50, max: 75,
        },
      },
    },
  });
}

// Fix prob bar layout
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.prob-bar').forEach(bar => {
    bar.style.position = 'relative';
    const fill = bar.querySelector('.prob-fill');
    if (fill) {
      fill.style.position = 'absolute';
      fill.style.right = '0';
      fill.style.height = '100%';
    }
  });
});

/* ══════════════════════════════════════════════════════════════
   SECTION 6 — PREDICTION LOG & ACCURACY TRACKING
══════════════════════════════════════════════════════════════ */

function logPredictions(games, league) {
  if (!games || !games.length) return;
  const key = `sport3_log_${league}`;
  let existing = [];
  try { existing = JSON.parse(localStorage.getItem(key) || '[]'); } catch { existing = []; }
  const existingIds = new Set(existing.map(e => e.game_id));

  let added = false;
  games.forEach(game => {
    if (existingIds.has(game.game_id)) return;
    const p = game.predictions || {};
    const ensProb = ensembleFromProbs(p, state.weights);
    const predictedWinner = ensProb >= 0.5 ? game.home_team : game.away_team;
    const predictedProb = ensProb >= 0.5 ? ensProb : (1 - ensProb);
    // Per-model picks for accuracy breakdown
    const eloProb = p.elo_prob || 0.5;
    const bayesProb = p.bayesian_prob || 0.5;
    const lrProb = p.logistic_prob || 0.5;
    const pythProb = p.pyth_prob || 0.5;
    const effProb = p.eff_prob || 0.5;
    existing.push({
      game_id: game.game_id,
      league,
      saved_at: new Date().toISOString(),
      home_team: game.home_team,
      home_name: game.home_name || game.home_team,
      away_team: game.away_team,
      away_name: game.away_name || game.away_team,
      game_time: game.game_time,
      predicted_winner: predictedWinner,
      predicted_prob: predictedProb,
      status: game.status,
      actual_winner: null,
      // Individual model picks (home if prob >= 0.5)
      elo_pick: eloProb >= 0.5 ? game.home_team : game.away_team,
      bayes_pick: bayesProb >= 0.5 ? game.home_team : game.away_team,
      lr_pick: lrProb >= 0.5 ? game.home_team : game.away_team,
      pyth_pick: pythProb >= 0.5 ? game.home_team : game.away_team,
      eff_pick: effProb >= 0.5 ? game.home_team : game.away_team,
    });
    added = true;
  });

  if (added) {
    try { localStorage.setItem(key, JSON.stringify(existing)); } catch {}
  }
}

async function resolveActualWinners(league) {
  const key = `sport3_log_${league}`;
  let entries = [];
  try { entries = JSON.parse(localStorage.getItem(key) || '[]'); } catch { return; }

  const unresolved = entries.filter(e => e.status === 'STATUS_FINAL' && !e.actual_winner);
  if (!unresolved.length) return;

  // Group by date
  const dateGroups = {};
  unresolved.forEach(e => {
    const d = e.game_time ? e.game_time.slice(0, 10).replace(/-/g, '') : null;
    if (!d) return;
    if (!dateGroups[d]) dateGroups[d] = [];
    dateGroups[d].push(e);
  });

  const sport = league === 'nba' ? 'basketball/nba' : 'football/nfl';

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
        const winner = competitors.find(c => c.winner);
        if (!winner) return;
        const winnerAbbrev = winner.team?.abbreviation?.toUpperCase();

        // Match to our log entry by game_id or team names
        const match = dayEntries.find(e =>
          e.game_id === ev.id ||
          (e.home_team === (competitors.find(c => c.homeAway === 'home')?.team?.abbreviation?.toUpperCase()) &&
           e.away_team === (competitors.find(c => c.homeAway === 'away')?.team?.abbreviation?.toUpperCase()))
        );
        if (match && winnerAbbrev) {
          const entry = entries.find(e => e.game_id === match.game_id);
          if (entry) entry.actual_winner = winnerAbbrev;
        }
      });
    } catch { /* network failure, skip */ }
  }

  try { localStorage.setItem(key, JSON.stringify(entries)); } catch {}
}

async function renderAccuracyTab() {
  const container = $('tab-accuracy');
  if (!container) return;

  // Resolve winners in background
  await Promise.all([resolveActualWinners('nfl'), resolveActualWinners('nba')]);

  let nflLog = [], nbaLog = [];
  try { nflLog = JSON.parse(localStorage.getItem('sport3_log_nfl') || '[]'); } catch {}
  try { nbaLog = JSON.parse(localStorage.getItem('sport3_log_nba') || '[]'); } catch {}

  const renderLeagueSection = (entries, label) => {
    if (!entries.length) return `<div class="log-empty">No ${label} predictions logged yet. Predictions are saved automatically when you view games.</div>`;
    const hasResolved = entries.some(e => e.actual_winner);
    if (!hasResolved) return `<div class="log-empty">No resolved ${label} predictions yet. Results will appear here once games finish.</div>`;

    const resolved = entries.filter(e => e.actual_winner);
    const correct = resolved.filter(e => e.actual_winner === e.predicted_winner);
    const accuracy = resolved.length > 0 ? (correct.length / resolved.length * 100).toFixed(1) : null;

    // Season record banner
    const recordBannerHtml = resolved.length > 0 ? (() => {
      const wins = correct.length;
      const losses = resolved.length - correct.length;
      const accNum = parseFloat(accuracy);
      const accClass = accNum >= 60 ? 'record-good' : accNum < 50 ? 'record-bad' : 'record-neutral';
      // Recent streak (last 10 resolved)
      const last10 = [...resolved].sort((a, b) => new Date(b.game_time) - new Date(a.game_time)).slice(0, 10);
      const streakDots = last10.map(e => {
        const isW = e.actual_winner === e.predicted_winner;
        return `<span class="streak-dot ${isW ? 'streak-w' : 'streak-l'}" title="${e.away_team} @ ${e.home_team}: ${isW ? 'W' : 'L'}">${isW ? 'W' : 'L'}</span>`;
      }).join('');
      return `
      <div class="season-record-banner ${accClass}">
        <div class="season-record-main">Season Record: <strong>${wins}–${losses}</strong> <span class="season-acc">(${accuracy}%)</span></div>
        <div class="streak-row"><span class="streak-label">Last ${last10.length}:</span> ${streakDots}</div>
      </div>`;
    })() : '';

    // Confidence tier breakdown
    const tiers = [
      { label: '<55%', min: 0, max: 0.55 },
      { label: '55–65%', min: 0.55, max: 0.65 },
      { label: '65–75%', min: 0.65, max: 0.75 },
      { label: '75%+', min: 0.75, max: 1 },
    ];
    const tierStats = tiers.map(t => {
      const tier = resolved.filter(e => e.predicted_prob >= t.min && e.predicted_prob < t.max);
      const tierCorrect = tier.filter(e => e.actual_winner === e.predicted_winner);
      return { ...t, n: tier.length, correct: tierCorrect.length };
    });

    // Per-model accuracy breakdown
    const modelKeys = [
      { key: 'predicted_winner', label: 'Ensemble' },
      { key: 'elo_pick', label: 'ELO' },
      { key: 'bayes_pick', label: 'Bayes' },
      { key: 'lr_pick', label: 'LR' },
      { key: 'pyth_pick', label: 'Pyth' },
      { key: 'eff_pick', label: 'Eff' },
    ];
    const modelStatsHtml = resolved.length > 0 ? `
      <div class="model-accuracy-section">
        <div class="model-accuracy-title">Per-Model Accuracy</div>
        <div class="model-accuracy-grid">
          ${modelKeys.map(m => {
            const modelResolved = resolved.filter(e => e[m.key] != null);
            if (modelResolved.length === 0) return '';
            const modelCorrect = modelResolved.filter(e => e.actual_winner === e[m.key]);
            const modelAcc = (modelCorrect.length / modelResolved.length * 100).toFixed(0);
            const accClass = modelAcc >= 60 ? 'good' : modelAcc < 50 ? 'bad' : '';
            return `<div class="model-acc-cell ${accClass}">
              <div class="model-acc-val">${modelAcc}%</div>
              <div class="model-acc-lbl">${m.label}</div>
              <div class="model-acc-sub">${modelCorrect.length}/${modelResolved.length}</div>
            </div>`;
          }).join('')}
        </div>
      </div>` : '';

    const summaryHtml = `
      ${recordBannerHtml}
      <div class="accuracy-summary">
        <div class="acc-stat">
          <div class="acc-stat-val">${entries.length}</div>
          <div class="acc-stat-lbl">Logged</div>
        </div>
        <div class="acc-stat">
          <div class="acc-stat-val">${resolved.length}</div>
          <div class="acc-stat-lbl">Resolved</div>
        </div>
        <div class="acc-stat">
          <div class="acc-stat-val">${correct.length}</div>
          <div class="acc-stat-lbl">Correct</div>
        </div>
        <div class="acc-stat ${accuracy >= 60 ? 'good' : accuracy !== null && accuracy < 50 ? 'bad' : ''}">
          <div class="acc-stat-val">${accuracy !== null ? accuracy + '%' : '—'}</div>
          <div class="acc-stat-lbl">Accuracy</div>
        </div>
      </div>
      ${modelStatsHtml}
      <div class="tier-breakdown">
        ${tierStats.map(t => `
          <div class="tier-item">
            <div class="tier-label">${t.label}</div>
            <div class="tier-val">${t.n > 0 ? (t.correct / t.n * 100).toFixed(0) + '% ('+t.correct+'/'+t.n+')' : '—'}</div>
          </div>`).join('')}
      </div>`;

    const sortedEntries = [...entries]
      .filter(e => e.actual_winner)
      .sort((a, b) => new Date(b.game_time) - new Date(a.game_time));
    if (sortedEntries.length === 0) {
      return `<div class="log-empty">No resolved predictions yet. Results will appear here once games finish.</div>`;
    }
    const rowsHtml = sortedEntries.map(e => {
      let resultHtml = '';
      if (e.actual_winner) {
        const isCorrect = e.actual_winner === e.predicted_winner;
        resultHtml = isCorrect
          ? `<span class="log-correct">✓ ${e.actual_winner}</span>`
          : `<span class="log-incorrect">✗ ${e.actual_winner}</span>`;
      }
      const gameDate = e.game_time ? new Date(e.game_time).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : '—';
      return `<tr>
        <td class="log-date">${gameDate}</td>
        <td class="log-matchup">${e.away_team} @ ${e.home_team}</td>
        <td class="log-pick ${e.predicted_winner === e.home_team ? 'home-pick' : 'away-pick'}">${e.predicted_winner}</td>
        <td class="log-conf">${(e.predicted_prob * 100).toFixed(1)}%</td>
        <td>${resultHtml}</td>
      </tr>`;
    }).join('');

    return `
      ${summaryHtml}
      <div class="log-table-wrap">
        <table class="log-table">
          <thead><tr>
            <th>Date</th>
            <th>Matchup</th>
            <th>Model Pick</th>
            <th>Confidence</th>
            <th>Result</th>
          </tr></thead>
          <tbody>${rowsHtml}</tbody>
        </table>
      </div>`;
  };

  container.innerHTML = `
    <div class="section-header">
      <div class="section-title">
        <h2>Prediction Log</h2>
        <span class="section-badge">Auto-tracked</span>
      </div>
    </div>
    <p class="muted" style="font-size:0.8125rem;margin-bottom:1.5rem;">
      Model picks are logged automatically each time you view games. Actual results are fetched from ESPN for completed games.
    </p>

    <div class="log-league-section">
      <h3 class="log-league-title">🏈 NFL</h3>
      ${renderLeagueSection(nflLog, 'NFL')}
    </div>

    <div class="log-league-section" style="margin-top:2rem;">
      <h3 class="log-league-title">🏀 NBA</h3>
      ${renderLeagueSection(nbaLog, 'NBA')}
    </div>

    <div style="margin-top:2rem;display:flex;gap:1rem;">
      <button class="btn-clear-log" onclick="clearLog('nfl')">Clear NFL Log</button>
      <button class="btn-clear-log" onclick="clearLog('nba')">Clear NBA Log</button>
    </div>`;
}

function clearLog(league) {
  localStorage.removeItem(`sport3_log_${league}`);
  renderAccuracyTab();
}
