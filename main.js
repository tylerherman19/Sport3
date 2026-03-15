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
  weights: { logistic: 0.30, xgboost: 0.25, elo: 0.20, pyth: 0.15, eff: 0.10 },
  params: { k: 20, hfa: 65, mov: 1.0, form: 0.30, sos: 0.5, h2h: 0.5, to: 1.0, rt: 1.0, lambda: 0.10 },
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

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

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
  });
});

/* ── Fetch data ───────────────────────────────────────────────── */
async function loadAll() {
  await Promise.all([loadNflData(), loadNbaData()]);
}

async function loadNflData() {
  const base = './data/';
  const [pred, elo, lb, metrics] = await Promise.all([
    fetch(base + 'nfl_predictions.json').then(r => r.json()).catch(() =>
      fetch(base + 'predictions.json').then(r => r.json()).catch(() => null)
    ),
    fetch(base + 'elo_ratings.json').then(r => r.json()).catch(() => null),
    fetch(base + 'nfl_leaderboard.json').then(r => r.json()).catch(() =>
      fetch(base + 'leaderboard.json').then(r => r.json()).catch(() => null)
    ),
    fetch(base + 'model_metrics.json').then(r => r.json()).catch(() => null),
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
  const [pred, lb, metrics] = await Promise.all([
    fetch(base + 'nba_predictions.json').then(r => r.json()).catch(() => null),
    fetch(base + 'nba_leaderboard.json').then(r => r.json()).catch(() => null),
    fetch(base + 'nba_model_metrics.json').then(r => r.json()).catch(() => null),
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

  $('offseason-banner').innerHTML = '';
  if (pred.is_offseason || pred.games.length === 0) {
    $('offseason-banner').innerHTML = `
      <div class="offseason-banner">
        <div class="offseason-banner-icon">${bannerIcon}</div>
        <div>
          <strong>Offseason</strong>
          <p>${offseasonMsg}</p>
        </div>
      </div>`;
  }

  $('games-count-badge').textContent = pred.games.length + ' game' + (pred.games.length !== 1 ? 's' : '');

  if (pred.games.length === 0) {
    grid.innerHTML = `<div class="empty-state">
      <div class="empty-state-icon">${emptyIcon}</div>
      <h3>${emptyTitle}</h3>
      <p>${emptyMsg}</p>
    </div>`;
    return;
  }

  grid.innerHTML = pred.games.map(game => buildGameCard(game, isNba)).join('');
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

  const homeWins = homeData.wins ?? '';
  const homeLosses = homeData.losses ?? '';
  const awayWins = awayData.wins ?? '';
  const awayLosses = awayData.losses ?? '';
  const homeRecordStr = (homeWins !== '' ? `${homeWins}-${homeLosses}` : '');
  const awayRecordStr = (awayWins !== '' ? `${awayWins}-${awayLosses}` : '');

  // Prediction drivers
  const drivers = game.prediction_drivers || [];
  const driversHtml = drivers.length ? `
    <div class="drivers-section">
      <div class="drivers-title">Prediction Drivers</div>
      ${drivers.map(d => {
        const cls = d.includes('OUT') || d.includes('DOUBTFUL') ? 'negative'
                  : d.includes('advantage') || d.includes('advantage') ? 'positive'
                  : '';
        return `<div class="driver-item ${cls}">${d}</div>`;
      }).join('')}
    </div>` : '';

  // Explanation
  const explanation = game.explanation || '';
  const explanationHtml = explanation
    ? `<div class="explanation-box">${explanation}</div>` : '';

  // NBA-specific efficiency display
  const effHtml = isNba && game.efficiency ? `
    <div class="game-stats">
      ${game.efficiency.home_off_rating != null ? `<div class="stat-pill"><span class="stat-pill-label">Home ORtg</span><span class="stat-pill-value">${game.efficiency.home_off_rating.toFixed(1)}</span></div>` : ''}
      ${game.efficiency.away_off_rating != null ? `<div class="stat-pill"><span class="stat-pill-label">Away ORtg</span><span class="stat-pill-value">${game.efficiency.away_off_rating.toFixed(1)}</span></div>` : ''}
      ${game.efficiency.home_net_rating != null ? `<div class="stat-pill"><span class="stat-pill-label">Home Net</span><span class="stat-pill-value">${game.efficiency.home_net_rating >= 0 ? '+' : ''}${game.efficiency.home_net_rating.toFixed(1)}</span></div>` : ''}
      ${game.efficiency.away_net_rating != null ? `<div class="stat-pill"><span class="stat-pill-label">Away Net</span><span class="stat-pill-value">${game.efficiency.away_net_rating >= 0 ? '+' : ''}${game.efficiency.away_net_rating.toFixed(1)}</span></div>` : ''}
      ${adj.b2b_home ? `<div class="stat-pill warn"><span class="stat-pill-label">B2B</span><span class="stat-pill-value">${game.home_team}</span></div>` : ''}
      ${adj.b2b_away ? `<div class="stat-pill warn"><span class="stat-pill-label">B2B</span><span class="stat-pill-value">${game.away_team}</span></div>` : ''}
    </div>` : `
    <div class="game-stats">
      ${adj.rest_diff != null ? `<div class="stat-pill"><span class="stat-pill-label">Rest</span><span class="stat-pill-value">${adj.rest_home}d / ${adj.rest_away}d</span></div>` : ''}
      ${adj.travel_dist_miles ? `<div class="stat-pill"><span class="stat-pill-label">Travel</span><span class="stat-pill-value">${Math.round(adj.travel_dist_miles)} mi</span></div>` : ''}
      ${mc.exp_margin != null ? `<div class="stat-pill"><span class="stat-pill-label">Exp Margin</span><span class="stat-pill-value">${mc.exp_margin > 0 ? '+' : ''}${mc.exp_margin?.toFixed(1)} pts</span></div>` : ''}
      ${market.home_prob != null ? `<div class="stat-pill"><span class="stat-pill-label">Market</span><span class="stat-pill-value">${pct(market.home_prob)}</span></div>` : ''}
      ${market.home_american != null ? `<div class="stat-pill"><span class="stat-pill-label">ML</span><span class="stat-pill-value">${market.home_american > 0 ? '+' : ''}${market.home_american?.toFixed(0)}</span></div>` : ''}
    </div>`;

  return `
<div class="game-card">
  <div class="game-header">
    <div class="game-meta">
      ${game.status === 'STATUS_FINAL' ? '<span class="text-red">FINAL</span> · ' : ''}
      ${fmtDate(game.game_time)}
      ${game.neutral ? ' · Neutral Site' : ''}
    </div>
    ${edgeBadge}
  </div>

  <div class="matchup-row">
    <div class="team-side away">
      <img class="team-logo" src="${game.away_logo}" alt="${game.away_team}" onerror="this.style.display='none'" loading="lazy" />
      <div class="team-name-abbrev">${game.away_team}</div>
      ${awayRecordStr ? `<div class="team-record">${awayRecordStr}</div>` : ''}
      <div class="team-elo">ELO ${eloFmt(elo.away)}</div>
    </div>
    <div class="vs-label">@</div>
    <div class="team-side home">
      <img class="team-logo" src="${game.home_logo}" alt="${game.home_team}" onerror="this.style.display='none'" loading="lazy" />
      <div class="team-name-abbrev">${game.home_team}</div>
      ${homeRecordStr ? `<div class="team-record">${homeRecordStr}</div>` : ''}
      <div class="team-elo">ELO ${eloFmt(elo.home)}</div>
    </div>
  </div>

  <div class="prob-bar-section">
    <div class="prob-labels">
      <span class="${awayProb > ensProb ? 'text-blue' : ''}">${game.away_team} ${pct(awayProb)}</span>
      <span class="${ensProb >= awayProb ? 'text-blue' : ''}">${game.home_team} ${pct(ensProb)}</span>
    </div>
    <div class="prob-bar">
      <div class="prob-fill ${edge != null ? (edge > 0.02 ? 'fill-green' : edge < -0.02 ? 'fill-red' : '') : ''}"
           style="width:${(ensProb * 100).toFixed(1)}%;margin-left:${((1-ensProb)*100).toFixed(1)}%;background:linear-gradient(90deg,var(--blue),var(--blue-light));border-radius:99px;position:absolute;right:0;"></div>
    </div>
  </div>

  ${effHtml}

  ${mc.win_prob != null ? `
  <div class="mc-row">
    <div class="mc-cell">
      <div class="mc-cell-val">${pct(mc.win_prob)}</div>
      <div class="mc-cell-lbl">Win %</div>
    </div>
    <div class="mc-cell">
      <div class="mc-cell-val">${pct(mc.prob_7plus)}</div>
      <div class="mc-cell-lbl">${isNba ? 'Win 7+' : 'Win 7+'}</div>
    </div>
    <div class="mc-cell">
      <div class="mc-cell-val">${pct(mc.prob_14plus)}</div>
      <div class="mc-cell-lbl">${isNba ? 'Win 14+' : 'Win 14+'}</div>
    </div>
    <div class="mc-cell">
      <div class="mc-cell-val">${pct(mc.prob_21plus)}</div>
      <div class="mc-cell-lbl">${isNba ? 'Win 21+' : 'Win 21+'}</div>
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

  ${driversHtml}
  ${explanationHtml}
</div>`;
}

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
${travelMi > 500 ? `The away team travels approximately <strong>${Math.round(travelMi)} miles</strong>, which contributes a fatigue penalty.` : ''}`;

  // Prediction drivers
  const drivers = [];
  if (eloDiffAbs >= 50) drivers.push(`ELO advantage: ${winner} +${Math.round(eloDiffAbs)} rating points`);
  if (Math.abs(homeEff - awayEff) >= 0.1) drivers.push(`Efficiency gap: ${home} net ${homeEff >= 0 ? '+' : ''}${homeEff.toFixed(3)} vs ${away} net ${awayEff >= 0 ? '+' : ''}${awayEff.toFixed(3)}`);
  if (Math.abs(restDiff) >= 3) drivers.push(`Rest advantage: ${restDiff > 0 ? home : away} has ${Math.abs(restDiff)} extra days rest`);
  if (travelMi >= 1500) drivers.push(`Travel penalty: away team travels ${Math.round(travelMi)} miles`);
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
