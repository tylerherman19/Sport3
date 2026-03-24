/* ═══════════════════════════════════════════════════════════════
   Sports Prediction Dashboard — main.js
   Supports NFL and NBA with live switching
   ═══════════════════════════════════════════════════════════════ */

'use strict';

/* ── State ────────────────────────────────────────────────────── */
const state = {
  league: 'nba',
  // NFL data
  predictions: null,
  eloRatings: null,
  leaderboard: null,
  modelMetrics: null,
  // NFL injury map: { TEAM: [{player, status, position}, ...] }
  injuriesMap: {},
  // NBA data
  nba: {
    predictions: null,
    leaderboard: null,
    modelMetrics: null,
    // NBA injury map: { TEAM: [{player, status, position}, ...] }
    injuriesMap: {},
  },
  sortCol: 'elo',
  sortDir: 'desc',
  gameFilter: 'live', // 'upcoming' | 'live' | 'completed'
  weights: { logistic: 0.30, xgboost: 0.25, elo: 0.20, pyth: 0.15, eff: 0.10 },
  // Note: weights are overwritten from localStorage below if saved values exist
  params: { k: 20, hfa: 65, mov: 1.0, form: 0.30, sos: 0.5, h2h: 0.5, to: 1.0, rt: 1.0, lambda: 0.10 },
  gameOverrides: {}, // { [game_id]: { homeBoost, awayBoost, hfaMult, momentumBoost, restFactor } }
  kalshiData: null,    // cached Kalshi market list (shared; re-matched per league)
  kalshiUpdated: null, // ISO timestamp from kalshidata.json
  kalshi: {
    rows: [],
    sortCol: 'mismatch',
    sortDir: 'desc',
    league: 'nba', // which league to display in the tab
  },
};

/* ── Load persisted weights ───────────────────────────────────── */
try {
  const _sw = localStorage.getItem('sport3_weights');
  if (_sw) Object.assign(state.weights, JSON.parse(_sw));
} catch {}

/* ── Chart instances ──────────────────────────────────────────── */
let calibrationChart = null;
let accuracyChart = null;

/* ── Live score state ─────────────────────────────────────────── */
let liveScoreCache = {};       // game_id → {home_score, away_score, period, display_clock, status, status_detail}
let livePollingInterval = null;
const LIVE_POLL_MS = 30000;    // poll every 30 seconds

/* ── Math helpers for in-game win probability ────────────────── */
function erfApprox(x) {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
}
function normalCDF(z) {
  return 0.5 * (1 + erfApprox(z / Math.SQRT2));
}
function normalInv(p) {
  if (p <= 0) return -6;
  if (p >= 1) return 6;
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
              1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
              6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
              -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const pLow = 0.02425;
  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  }
  const q = p - 0.5, r = q * q;
  return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
}

function calcLiveWinProb(pregameProb, homePts, awayPts, period, clockStr) {
  if (!period || period === 0) return pregameProb;
  const NBA_STD = 11.5;    // typical NBA final score std dev in points
  const TOTAL_SECS = 2880; // 48 min regulation
  const parts = (clockStr || '0:00').split(':');
  const clockSecs = (parseInt(parts[0]) || 0) * 60 + (parseInt(parts[1]) || 0);
  // For OT periods, each OT is 5 min (300 sec)
  const isOT = period > 4;
  const secRemaining = isOT
    ? clockSecs  // just this OT period
    : Math.max(0, 4 - period) * 720 + clockSecs;
  const frac = isOT ? clockSecs / TOTAL_SECS : secRemaining / TOTAL_SECS;
  if (frac <= 0.001) return homePts > awayPts ? 0.99 : homePts < awayPts ? 0.01 : 0.5;
  const pregameSpread = normalInv(Math.max(0.01, Math.min(0.99, pregameProb))) * NBA_STD;
  const scoreDiff = homePts - awayPts;
  const blendedSpread = scoreDiff + pregameSpread * frac;
  const liveStd = NBA_STD * Math.sqrt(frac);
  return Math.max(0.01, Math.min(0.99, normalCDF(blendedSpread / liveStd)));
}

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

function normalizeGame(game) {
  if (!game) return {};
  game.predictions  = game.predictions  || {};
  game.adjustments  = game.adjustments  || {
    rest_home: 0, rest_away: 0, rest_diff: 0,
    b2b_home: false, b2b_away: false,
    home_elo_bonus: 0, travel_dist_miles: 0,
  };
  game.elo          = game.elo          || {};
  game.efficiency   = game.efficiency   || {};
  game.market       = game.market       || {};
  game.monte_carlo  = game.monte_carlo  || {};
  game.injuries     = game.injuries     || { home: [], away: [] };
  game.h2h          = game.h2h          || {};
  return game;
}

function normalizePredictionPayload(p) {
  if (!p) return {};
  const fields = ['logistic_prob','xgb_prob','elo_prob','pyth_prob','eff_prob','bayesian_prob'];
  for (const f of fields) {
    if (p[f] === undefined) p[f] = null;
  }
  return p;
}

function ensembleFromProbs(probs, weights) {
  const w = { ...weights };
  // If XGBoost unavailable, redistribute its weight proportionally to remaining models
  // (previously fell back to logistic_prob, giving logistic double weight)
  if (probs.xgb_prob == null) {
    const xgbW = w.xgboost;
    const rem = w.logistic + w.elo + w.pyth + w.eff;
    if (rem > 0) {
      w.logistic += xgbW * (w.logistic / rem);
      w.elo      += xgbW * (w.elo / rem);
      w.pyth     += xgbW * (w.pyth / rem);
      w.eff      += xgbW * (w.eff / rem);
    }
    w.xgboost = 0;
  }
  const total = w.logistic + w.xgboost + w.elo + w.pyth + w.eff;
  if (total === 0) return 0.5;
  const val = (
    w.logistic * (probs.logistic_prob ?? 0.5) +
    w.xgboost  * (probs.xgb_prob     ?? 0.5) +
    w.elo      * (probs.elo_prob      ?? 0.5) +
    w.pyth     * (probs.pyth_prob     ?? 0.5) +
    w.eff      * (probs.eff_prob      ?? 0.5)
  ) / total;
  return clamp(val, 0.01, 0.99);
}

function isoDateKey(iso) {
  if (!iso) return '';
  try {
    return new Date(iso).toISOString().slice(0, 10);
  } catch {
    return String(iso).slice(0, 10);
  }
}

function findMatchingNbaGame(existingGames, liveGame) {
  if (!existingGames?.length || !liveGame) return null;
  return existingGames.find(g =>
    g.home_team === liveGame.home_team &&
    g.away_team === liveGame.away_team &&
    isoDateKey(g.game_time) === isoDateKey(liveGame.game_time)
  ) || null;
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
      if (state.nba.predictions) updateHeader(state.nba.predictions, 'nba');
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
      if (state.predictions) updateHeader(state.predictions, 'nfl');
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
    // Render Kalshi odds comparison on kalshi tab
    if (btn.dataset.tab === 'kalshi') {
      renderKalshiTab();
    }
  });
});

/* ── Fetch data ───────────────────────────────────────────────── */
async function loadAll() {
  await Promise.all([loadNflData(), loadNbaData()]);
}

/* Build a team-keyed injury map from the flat injuries JSON file.
   Returns { TEAM: [{player, status, position, injury_description}, ...] } */
function buildInjuriesMap(injData) {
  const map = {};
  if (!injData?.injuries) return map;
  for (const p of injData.injuries) {
    const team = p.team;
    if (!team) continue;
    if (!map[team]) map[team] = [];
    map[team].push({ player: p.player, status: p.status, position: p.position, injury_description: p.injury_description || p.status, value_tier: p.value_tier || null });
  }
  return map;
}

async function loadNflData() {
  const base = './data/';
  const t = Date.now();
  const [pred, elo, lb, metrics, injData] = await Promise.all([
    fetch(base + 'nfl_predictions.json?t=' + t).then(r => r.json()).catch(() =>
      fetch(base + 'predictions.json?t=' + t).then(r => r.json()).catch(() => null)
    ),
    fetch(base + 'elo_ratings.json?t=' + t).then(r => r.json()).catch(() => null),
    fetch(base + 'nfl_leaderboard.json?t=' + t).then(r => r.json()).catch(() =>
      fetch(base + 'leaderboard.json?t=' + t).then(r => r.json()).catch(() => null)
    ),
    fetch(base + 'model_metrics.json?t=' + t).then(r => r.json()).catch(() => null),
    fetch(base + 'nfl_injuries.json?t=' + t).then(r => r.json()).catch(() => null),
  ]);

  state.predictions = pred;
  state.eloRatings = elo;
  state.leaderboard = lb;
  state.modelMetrics = metrics;
  state.injuriesMap = buildInjuriesMap(injData);

  // Log all games at load time so predictions are captured even if user never views Games tab
  if (pred?.games) logPredictions(pred.games, 'nfl');

  if (state.league === 'nfl') {
    updateHeader(pred, 'nfl');
    renderAll();
  }
}

async function loadNbaData() {
  const base = './data/';
  const t = Date.now();
  const [pred, lb, metrics, injData] = await Promise.all([
    fetch(base + 'nba_predictions.json?t=' + t).then(r => r.json()).catch(() => null),
    fetch(base + 'nba_leaderboard.json?t=' + t).then(r => r.json()).catch(() => null),
    fetch(base + 'nba_model_metrics.json?t=' + t).then(r => r.json()).catch(() => null),
    fetch(base + 'nba_injuries.json?t=' + t).then(r => r.json()).catch(() => null),
  ]);

  state.nba.predictions = pred;
  state.nba.leaderboard = lb;
  state.nba.modelMetrics = metrics;
  state.nba.injuriesMap = buildInjuriesMap(injData);

  // Log all games at load time so predictions are captured even if user never views Games tab
  if (pred?.games) logPredictions(pred.games, 'nba');

  if (state.league === 'nba') {
    updateHeader(pred, 'nba');
    renderAll();
  }

  // Augment with live ESPN data (async, re-renders when ready)
  fetchLiveNBAStandings();
  fetchLiveNBAData();
}

/* ── ESPN abbreviation normalization (mirrors Python backend) ─── */
const ESPN_NBA_ABBREV_MAP = {
  'GS': 'GSW', 'NY': 'NYK', 'NO': 'NOP', 'SA': 'SAS',
  'WSH': 'WAS', 'PHO': 'PHX', 'UTAH': 'UTA',
};
function normNBAabbrev(a) {
  const u = (a || '').toUpperCase().trim();
  return ESPN_NBA_ABBREV_MAP[u] || u;
}

/* ── Parse a single ESPN NBA event into a game object ─────────── */
function parseLiveNBAEvent(event) {
  try {
    const comp = (event.competitions || [])[0];
    if (!comp) return null;
    const competitors = comp.competitors || [];
    const home = competitors.find(c => c.homeAway === 'home');
    const away = competitors.find(c => c.homeAway === 'away');
    if (!home || !away) return null;

    const homeAbbrev = normNBAabbrev(home.team.abbreviation);
    const awayAbbrev = normNBAabbrev(away.team.abbreviation);
    const status = event.status?.type?.name || '';
    const gameTime = event.date || '';
    const isFuture = status === 'STATUS_SCHEDULED' ||
      (gameTime && new Date(gameTime).getTime() > Date.now() + 1800000); // >30 min away

    const homeScore = parseInt(home.score || 0) || 0;
    const awayScore = parseInt(away.score || 0) || 0;

    return {
      game_id: event.id,
      game_time: gameTime,
      status,
      is_future: isFuture,
      home_team: homeAbbrev,
      away_team: awayAbbrev,
      home_name: home.team.displayName || homeAbbrev,
      away_name: away.team.displayName || awayAbbrev,
      home_logo: `https://a.espncdn.com/i/teamlogos/nba/500/${home.team.abbreviation.toLowerCase()}.png`,
      away_logo: `https://a.espncdn.com/i/teamlogos/nba/500/${away.team.abbreviation.toLowerCase()}.png`,
      neutral: comp.neutralSite || false,
      home_score: homeScore,
      away_score: awayScore,
      // Default prediction fields for new games not in static JSON
      predictions: { ensemble_prob: 0.5, elo_prob: 0.5, pyth_prob: 0.5, eff_prob: 0.5, bayesian_prob: 0.5, logistic_prob: 0.5, xgb_prob: null },
      market: { home_prob: null, edge: null, kelly_pct: null, home_american: null, away_american: null },
      monte_carlo: null,
      adjustments: { rest_home: 2, rest_away: 2, rest_diff: 0, travel_dist_miles: 0, b2b_home: false, b2b_away: false, home_elo_bonus: 100 },
      elo: { home: 1500, away: 1500, diff: 0 },
      efficiency: null,
      injuries: { home: [], away: [] },
      injury_impact: { home_elo_penalty: 0, away_elo_penalty: 0, home_key_players_out: [], away_key_players_out: [] },
      prediction_drivers: [],
      explanation: '',
      h2h: null,
    };
  } catch { return null; }
}

/* ── Fetch live NBA scoreboard: today + next 7 days ───────────── */
async function fetchLiveNBAData() {
  const ESPN = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba';
  const liveMap = new Map();

  for (let delta = 0; delta <= 7; delta++) {
    const d = new Date();
    d.setDate(d.getDate() + delta);
    const dateStr = d.toISOString().slice(0, 10).replace(/-/g, '');
    try {
      const data = await fetch(`${ESPN}/scoreboard?dates=${dateStr}`).then(r => r.json());
      for (const event of data.events || []) {
        const parsed = parseLiveNBAEvent(event);
        if (parsed) liveMap.set(parsed.game_id, parsed);
      }
    } catch { /* network failure — skip day */ }
  }

  if (!liveMap.size) return;

  // Ensure predictions container exists
  if (!state.nba.predictions) {
    state.nba.predictions = { games: [], is_offseason: false, season: new Date().getFullYear(), league: 'nba', updated: new Date().toISOString() };
  }

  const existingMap = new Map(state.nba.predictions.games.map(g => [g.game_id, g]));
  const existingGames = state.nba.predictions.games;

  for (const [id, live] of liveMap) {
    const matched = existingMap.get(id) || findMatchingNbaGame(existingGames, live);
    if (matched) {
      // Update mutable fields on the existing enriched game object
      const g = matched;
      g.status = live.status;
      g.home_score = live.home_score;
      g.away_score = live.away_score;
      g.is_future = live.is_future;
    } else {
      // Truly new game not present in static JSON — add live placeholder row
      state.nba.predictions.games.push(live);
    }
  }

  if (state.league === 'nba') renderGames();
}

/* ── Fetch live NBA standings and patch leaderboard ───────────── */
async function fetchLiveNBAStandings() {
  try {
    const data = await fetch('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/standings').then(r => r.json());
    const liveTeams = {};

    for (const group of data.children || []) {
      for (const div of (group.children || [group])) {
        for (const entry of div.standings?.entries || []) {
          const abbrev = normNBAabbrev(entry.team?.abbreviation || '');
          if (!abbrev) continue;
          const stats = Object.fromEntries((entry.stats || []).map(s => [s.name, parseFloat(s.value) || 0]));
          const offRtg = stats.offensiveRating || stats.pointsFor || 110;
          const defRtg = stats.defensiveRating || stats.pointsAgainst || 110;
          liveTeams[abbrev] = {
            wins: Math.round(stats.wins || 0),
            losses: Math.round(stats.losses || 0),
            offensive_rating: offRtg,
            defensive_rating: defRtg,
            net_rating: offRtg - defRtg,
          };
        }
      }
    }

    if (!Object.keys(liveTeams).length) return;

    // Ensure leaderboard exists (may be null if static JSON failed)
    if (!state.nba.leaderboard) {
      state.nba.leaderboard = { teams: [] };
    }

    // Patch existing leaderboard entries with live records + ratings
    const patchedTeams = new Set();
    state.nba.leaderboard.teams.forEach(t => {
      const live = liveTeams[t.team];
      if (!live) return;
      t.wins = live.wins;
      t.losses = live.losses;
      t.offensive_rating = live.offensive_rating;
      t.off_eff = live.offensive_rating;
      t.defensive_rating = live.defensive_rating;
      t.def_eff = live.defensive_rating;
      t.net_rating = live.net_rating;
      t.net_eff = live.net_rating;
      patchedTeams.add(t.team);
    });

    // Add any teams missing from leaderboard (e.g. all-default JSON)
    for (const [abbrev, live] of Object.entries(liveTeams)) {
      if (patchedTeams.has(abbrev)) continue;
      state.nba.leaderboard.teams.push({
        team: abbrev, team_name: abbrev, abbrev: abbrev.toLowerCase(),
        logo: `https://a.espncdn.com/i/teamlogos/nba/500/${abbrev.toLowerCase()}.png`,
        elo: 1500, sigma: 75, mu: 1500, lower_band: 1425, upper_band: 1575,
        pyth: 0.5, trend: 'neutral', streak_type: 'N', streak_count: 0,
        playoff_prob: 0, sb_prob: 0, champ_prob: 0, ties: 0,
        injury_elo_penalty: 0, injury_impact_score: 0, injury_players_count: 0,
        ...live,
      });
    }

    if (state.league === 'nba') {
      sortAndRenderLeaderboard();
    }
  } catch { /* network failure — keep static JSON data */ }
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
  renderKalshiIfOpen();
}

function forceRefresh() {
  loadNflData();
  loadNbaData().then(() => fetchLiveScores());
}

/* ── Live score overlay ─────────���──────────────────────────────── */
function mergeLiveGame(game) {
  const live = liveScoreCache[game.game_id];
  if (!live) return game;
  return { ...game, ...live };
}

async function fetchLiveScores() {
  const now = new Date();
  const dates = [now, new Date(now - 86400000)].map(d =>
    d.toISOString().slice(0, 10).replace(/-/g, '')
  );
  let updated = false;
  for (const dateStr of dates) {
    try {
      const data = await fetch(
        `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=${dateStr}`
      ).then(r => r.json());
      for (const event of (data.events || [])) {
        const comp = event.competitions?.[0];
        if (!comp) continue;
        const home = comp.competitors?.find(c => c.homeAway === 'home');
        const away = comp.competitors?.find(c => c.homeAway === 'away');
        const statusObj = event.status || {};
        liveScoreCache[event.id] = {
          home_score: parseInt(home?.score || 0),
          away_score: parseInt(away?.score || 0),
          status: statusObj.type?.name || '',
          period: statusObj.period || 0,
          display_clock: statusObj.displayClock || '',
          status_detail: statusObj.type?.detail || '',
        };
        updated = true;
      }
    } catch (_) { /* network errors are non-fatal */ }
  }
  if (updated && state.league === 'nba') renderAll();
  manageLivePolling();
  updateRefreshButtonState();
  // Ensure any newly-FINAL games are logged before resolving winners
  if (updated) {
    const nbaGames = state.nba.predictions?.games;
    if (nbaGames) logPredictions(nbaGames, 'nba');
    resolveActualWinners('nba');
  }
}

function manageLivePolling() {
  const games = state.nba.predictions?.games || [];
  const hasLive = games.some(g => {
    const s = (liveScoreCache[g.game_id]?.status) || g.status;
    return s === 'STATUS_IN_PROGRESS' || s === 'STATUS_HALFTIME';
  });
  if (hasLive && !livePollingInterval) {
    livePollingInterval = setInterval(fetchLiveScores, LIVE_POLL_MS);
  } else if (!hasLive && livePollingInterval) {
    clearInterval(livePollingInterval);
    livePollingInterval = null;
  }
  updateRefreshButtonState();
}

function updateRefreshButtonState() {
  const btn = $('btn-refresh');
  if (!btn) return;
  if (livePollingInterval) {
    btn.innerHTML = '<span class="live-pulse"></span> Live';
    btn.title = 'Auto-updating live scores every 30s — click to force refresh';
  } else {
    btn.innerHTML = '↻ Refresh';
    btn.title = 'Force data reload';
  }
}

// Start loading both leagues immediately
loadNflData();
loadNbaData().then(() => fetchLiveScores());

/* ══════════════════════════════════════════════════════════════
   SECTION 1 — THIS WEEK'S GAMES
══════════════════════════════════════════════════════════════ */
function renderGames() {
  const grid = $('games-grid');
  const isNba = state.league === 'nba';
  const pred = isNba ? state.nba.predictions : state.predictions;

  if (!pred) {
    grid.innerHTML = '<div class="empty-state"><div class="empty-state-icon">��️</div><h3>Failed to load</h3><p>Could not load predictions data.</p></div>';
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
  $('games-count-badge').textContent = pred.games.length + ' game' + (pred.games.length !== 1 ? 's' : '');

  // Apply game filter — use status + game_time in addition to is_future flag
  const nowMs = Date.now();
  const todayDateStr = new Date().toISOString().slice(0, 10);
  let filteredGames = pred.games;
  if (state.gameFilter === 'live') {
    filteredGames = pred.games.filter(g =>
      g.status === 'STATUS_IN_PROGRESS' || g.status === 'STATUS_HALFTIME'
    );
  } else if (state.gameFilter === 'upcoming') {
    filteredGames = pred.games.filter(g =>
      g.is_future ||
      g.status === 'STATUS_SCHEDULED' ||
      (g.game_time && new Date(g.game_time).getTime() > nowMs + 1800000)
    );
  } else if (state.gameFilter === 'completed') {
    filteredGames = pred.games.filter(g => g.status === 'STATUS_FINAL');
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
    const filterLabels = { upcoming: 'upcoming scheduled games', live: 'live games right now', completed: 'completed games' };
    grid.innerHTML = `<div class="empty-state">
      <div class="empty-state-icon">${emptyIcon}</div>
      <h3>No ${filterLabels[state.gameFilter] || 'games'}</h3>
      <p>Try switching the filter above to see all games.</p>
    </div>`;
    return;
  }

  grid.innerHTML = filteredGames.map(game => buildGameCard(game, isNba)).join('');
  logPredictions(pred.games, isNba ? 'nba' : 'nfl');
  // Auto-resolve any completed games in the background without waiting
  resolveActualWinners(isNba ? 'nba' : 'nfl');
}

/* ── Score display (FINAL / live) ─────────────────────────────── */
function buildScoreHtml(game) {
  const g = mergeLiveGame(game);
  const status = g.status || '';
  const isFinal = status === 'STATUS_FINAL';
  const isLive  = status === 'STATUS_IN_PROGRESS' || status === 'STATUS_HALFTIME';
  const homeScore = g.home_score;
  const awayScore = g.away_score;

  if ((isFinal || isLive) && homeScore != null && awayScore != null) {
    const winner = isFinal
      ? (homeScore > awayScore ? 'home' : awayScore > homeScore ? 'away' : 'tie')
      : '';
    let liveSepContent;
    if (isLive) {
      let clockLabel = '';
      if (g.period) {
        const periodLabel = g.period > 4
          ? `OT${g.period > 5 ? g.period - 4 : ''}`
          : `Q${g.period}`;
        clockLabel = g.display_clock ? `${periodLabel} ${g.display_clock}` : periodLabel;
      } else if (status === 'STATUS_HALFTIME') {
        clockLabel = 'HT';
      }
      liveSepContent = `<span class="live-dot"></span>LIVE`;
    } else {
      liveSepContent = 'FINAL';
    }
    return `
  <div class="score-display ${isLive ? 'score-live' : 'score-final'}">
    <span class="score-away ${winner === 'away' ? 'score-winner' : ''}">${g.away_team} <strong>${awayScore}</strong></span>
    <span class="score-sep">${liveSepContent}</span>
    <span class="score-home ${winner === 'home' ? 'score-winner' : ''}"><strong>${homeScore}</strong> ${g.home_team}</span>
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

  // Travel distance — show for both NBA and NFL when significant
  if (adj.travel_dist_miles > 300) {
    const travelClass = adj.travel_dist_miles > 2000 ? 'kf-bad' : 'kf-warn';
    items.push(`<div class="kf-item"><span class="kf-label">Travel</span><span class="kf-val ${travelClass}">${game.away_team} ${Math.round(adj.travel_dist_miles)} mi</span></div>`);
  }

  if (!items.length) return '';
  return `<div class="key-factors-strip">${items.join('')}</div>`;
}

function computeLeagueRanks(lbData) {
  if (!lbData || !lbData.length) return {};
  const sorted_off = [...lbData].sort((a, b) => (b.off_eff || 0) - (a.off_eff || 0));
  const sorted_def = [...lbData].sort((a, b) => (a.def_eff || 0) - (b.def_eff || 0)); // lower def allowed = better
  const sorted_net = [...lbData].sort((a, b) => (b.net_eff || 0) - (a.net_eff || 0));
  const ranks = {};
  sorted_off.forEach((t, i) => { ranks[t.team] = ranks[t.team] || {}; ranks[t.team].offRank = i + 1; });
  sorted_def.forEach((t, i) => { ranks[t.team] = ranks[t.team] || {}; ranks[t.team].defRank = i + 1; });
  sorted_net.forEach((t, i) => { ranks[t.team] = ranks[t.team] || {}; ranks[t.team].netRank = i + 1; });
  return ranks;
}

function buildRichExplanationHtml(game, homeData, awayData, lbData, isNba) {
  const explanation = game.explanation || '';
  const eff = game.efficiency;
  if (!explanation && (!isNba || !eff)) return '';

  const narrativeHtml = explanation ? `<p class="explanation-narrative">${explanation}</p>` : '';

  let effPanelHtml = '';
  if (isNba && eff) {
    const ranks = computeLeagueRanks(lbData);
    const hr = ranks[game.home_team] || {};
    const ar = ranks[game.away_team] || {};
    const sign = v => v >= 0 ? '+' : '';
    const rank = n => n ? `<span class="eff-rank">#${n}</span>` : '';

    const homeOffAdv = eff.home_off_rating != null && eff.away_def_rating != null
      ? eff.home_off_rating - eff.away_def_rating : null;
    const awayOffAdv = eff.away_off_rating != null && eff.home_def_rating != null
      ? eff.away_off_rating - eff.home_def_rating : null;

    const matchupRows = [];
    if (homeOffAdv != null) {
      const cls = homeOffAdv > 3 ? 'positive' : homeOffAdv < -3 ? 'negative' : '';
      const leader = homeOffAdv > 0 ? game.home_team : game.away_team;
      matchupRows.push(`<div class="explanation-matchup-row ${cls}">
        ${game.home_team} offense (${eff.home_off_rating.toFixed(1)}) vs ${game.away_team} defense (${eff.away_def_rating.toFixed(1)})
        → <strong>${leader}</strong> ${sign(homeOffAdv)}${homeOffAdv.toFixed(1)} edge
      </div>`);
    }
    if (awayOffAdv != null) {
      const cls = awayOffAdv > 3 ? 'positive' : awayOffAdv < -3 ? 'negative' : '';
      const leader = awayOffAdv > 0 ? game.away_team : game.home_team;
      matchupRows.push(`<div class="explanation-matchup-row ${cls}">
        ${game.away_team} offense (${eff.away_off_rating.toFixed(1)}) vs ${game.home_team} defense (${eff.home_def_rating.toFixed(1)})
        → <strong>${leader}</strong> ${sign(awayOffAdv)}${awayOffAdv.toFixed(1)} edge
      </div>`);
    }

    effPanelHtml = `
    <div class="explanation-eff-panel">
      <div class="explanation-eff-header">Efficiency Breakdown</div>
      <div class="explanation-eff-team-row">
        <span class="eff-team-name">${game.home_team}</span>
        ${eff.home_off_rating != null ? `<span class="eff-stat">Off <strong>${eff.home_off_rating.toFixed(1)}</strong>${rank(hr.offRank)}</span>` : ''}
        ${eff.home_def_rating != null ? `<span class="eff-stat">Def <strong>${eff.home_def_rating.toFixed(1)}</strong>${rank(hr.defRank)}</span>` : ''}
        ${eff.home_net_rating != null ? `<span class="eff-stat">Net <strong>${sign(eff.home_net_rating)}${eff.home_net_rating.toFixed(1)}</strong>${rank(hr.netRank)}</span>` : ''}
      </div>
      <div class="explanation-eff-team-row">
        <span class="eff-team-name">${game.away_team}</span>
        ${eff.away_off_rating != null ? `<span class="eff-stat">Off <strong>${eff.away_off_rating.toFixed(1)}</strong>${rank(ar.offRank)}</span>` : ''}
        ${eff.away_def_rating != null ? `<span class="eff-stat">Def <strong>${eff.away_def_rating.toFixed(1)}</strong>${rank(ar.defRank)}</span>` : ''}
        ${eff.away_net_rating != null ? `<span class="eff-stat">Net <strong>${sign(eff.away_net_rating)}${eff.away_net_rating.toFixed(1)}</strong>${rank(ar.netRank)}</span>` : ''}
      </div>
      ${matchupRows.join('')}
    </div>`;
  }

  if (!narrativeHtml && !effPanelHtml) return '';
  return `<div class="explanation-box">${narrativeHtml}${effPanelHtml}</div>`;
}

function isEarlySeason(game, isNba) {
  if (isNba) {
    const gameDate = new Date(game.game_time);
    const month = gameDate.getMonth(); // 0-indexed
    const day = gameDate.getDate();
    // NBA season starts early October; first ~45 days are high-uncertainty
    return (month === 9) || (month === 10 && day <= 15);
  } else {
    return game.week && game.week <= 3;
  }
}

function buildGameCard(game, isNba) {
  let p = game.predictions || {};
  const g = mergeLiveGame(game);
  const isLiveGame = isNba && (g.status === 'STATUS_IN_PROGRESS' || g.status === 'STATUS_HALFTIME');
  let ensProb = ensembleFromProbs(p, state.weights);
  if (isLiveGame && g.period) {
    ensProb = calcLiveWinProb(ensProb, g.home_score, g.away_score, g.period, g.display_clock);
  }
  // For completed games, restore the pre-game prediction from the log
  // (backend regenerates predictions daily with current stats, overwriting pre-game values)
  if (g.status === 'STATUS_FINAL') {
    const logKey = `sport3_log_${isNba ? 'nba' : 'nfl'}`;
    let logEntries = [];
    try { logEntries = JSON.parse(localStorage.getItem(logKey) || '[]'); } catch {}
    const logEntry = logEntries.find(e => e.game_id === game.game_id);
    if (logEntry) {
      p = {
        ...p,
        elo_prob:      logEntry.elo_prob,
        bayesian_prob: logEntry.bayes_prob,
        logistic_prob: logEntry.lr_prob,
        pyth_prob:     logEntry.pyth_prob,
        eff_prob:      logEntry.eff_prob,
      };
      // Use stored ensemble probability (captured before game, not recalculated with today's ratings)
      ensProb = logEntry.predicted_winner === game.home_team
        ? logEntry.predicted_prob
        : (1 - logEntry.predicted_prob);
      // Restore pre-game injuries (ESPN clears injury report after game ends)
      if (logEntry.home_injuries !== undefined || logEntry.away_injuries !== undefined) {
        game = {
          ...game,
          injuries: {
            home: logEntry.home_injuries || [],
            away: logEntry.away_injuries || [],
          },
          injury_impact: logEntry.injury_impact || {},
        };
      }
    }
  }
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

  // Issue 8: early-season warning badge — ratings stabilize after ~4 weeks
  const earlySeasonHtml = isEarlySeason(game, isNba)
    ? `<span class="edge-badge neutral" title="Early-season predictions carry higher uncertainty — ratings stabilize after ~4 weeks">⚠ Early Season</span>`
    : '';

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

  // Always-visible injury tier impact summary
  const injuryTierSummaryHtml = buildInjuryTierSummaryHtml(game);

  // Injury panel (collapsible)
  const injuryHtml = buildInjuryHtml(game, isNba);

  // Prediction drivers
  const drivers = game.prediction_drivers || [];
  const driversHtml = drivers.length ? `
    <div class="drivers-section">
      <div class="drivers-title">Prediction Drivers</div>
      ${drivers.map(d => {
        let cls = '';
        if (d.includes('[Superstar]')) cls = 'negative superstar-injury';
        else if (d.includes('[All-Star]') || d.includes('[All-star]')) cls = 'negative allstar-injury';
        else if (d.includes('star stack')) cls = 'stack-bonus';
        else if (d.includes('OUT') || d.includes('DOUBTFUL')) cls = 'negative';
        else if (d.includes('advantage')) cls = 'positive';
        return `<div class="driver-item ${cls}">${d}</div>`;
      }).join('')}
    </div>` : '';

  // Rich explanation with efficiency breakdown + league ranks
  const explanationHtml = buildRichExplanationHtml(game, homeData, awayData, lbData, isNba);

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
      ${adj.travel_dist_miles ? `<div class="stat-pill"><span class="stat-pill-label">Travel</span><span class="stat-pill-value">${game.away_team} ${Math.round(adj.travel_dist_miles)} mi</span></div>` : ''}
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
      ${g.status === 'STATUS_IN_PROGRESS' || g.status === 'STATUS_HALFTIME' ? '<span class="text-live">● LIVE</span> · ' : ''}
      ${g.status === 'STATUS_FINAL' ? '<span class="text-red">FINAL</span> · ' : ''}
      ${game.is_future && g.status !== 'STATUS_IN_PROGRESS' && g.status !== 'STATUS_FINAL' ? '<span class="future-badge">Upcoming</span> · ' : ''}
      ${fmtDate(game.game_time)}
      ${game.neutral ? ' · Neutral Site' : ''}
    </div>
    <div class="game-header-right">
      <span class="adj-badge" style="display:none">Adjusted</span>
      ${edgeBadge}
      ${earlySeasonHtml}
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

  ${(() => {
    const ensWinner = ensProb >= 0.5 ? game.home_team : game.away_team;
    function modelPill(prob, label, fullName) {
      if (prob == null) return '';
      const pick = prob >= 0.5 ? game.home_team : game.away_team;
      const conf = prob >= 0.5 ? prob : 1 - prob;
      const cls = pick === ensWinner ? 'agrees' : 'disagrees';
      return `<div class="model-prob-pill ${cls}" title="${fullName}"><span class="model-prob-label">${label}</span><span class="model-prob-team">${pick}</span><span class="model-prob-val">${pct(conf)}</span></div>`;
    }
    const pills = [
      modelPill(p.logistic_prob, 'Logistic',     'Logistic Regression'),
      modelPill(p.xgb_prob,      'XGBoost',      'XGBoost'),
      modelPill(p.elo_prob,      'ELO',          'ELO Rating'),
      modelPill(p.pyth_prob,     'Pythagorean',  'Pythagorean Expectation'),
      modelPill(p.eff_prob,      'Efficiency',   'Efficiency Model'),
      modelPill(p.bayesian_prob, 'Bayesian',     'Bayesian Model'),
    ].filter(Boolean).join('');
    return pills ? `<div class="model-probs-header">Sub-models</div><div class="model-probs">${pills}</div>` : '';
  })()}

  ${injuryTierSummaryHtml}
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

  // Travel + timezone — always label the away team as the traveler
  if (adj.travel_dist_miles > 0) {
    const absTzShift = Math.abs(tzShift);
    const longTrip = adj.travel_dist_miles > 2000 || absTzShift >= 2;
    const travelClass = longTrip ? 'warn' : '';
    const tzStr = absTzShift >= 1 ? ` · ${absTzShift}hr TZ` : '';
    pills.push(`<div class="stat-pill ${travelClass}"><span class="stat-pill-label">Travel</span><span class="stat-pill-value">${game.away_team} ${Math.round(adj.travel_dist_miles).toLocaleString()} mi${tzStr}</span></div>`);
  }

  // Hot/cold streak
  if (homeTrend === 'up') pills.push(`<div class="stat-pill streak-hot"><span class="stat-pill-label">${game.home_team}</span><span class="stat-pill-value">🔥 Hot</span></div>`);
  else if (homeTrend === 'down') pills.push(`<div class="stat-pill streak-cold"><span class="stat-pill-label">${game.home_team}</span><span class="stat-pill-value">❄️ Cold</span></div>`);

  if (awayTrend === 'up') pills.push(`<div class="stat-pill streak-hot"><span class="stat-pill-label">${game.away_team}</span><span class="stat-pill-value">🔥 Hot</span></div>`);
  else if (awayTrend === 'down') pills.push(`<div class="stat-pill streak-cold"><span class="stat-pill-label">${game.away_team}</span><span class="stat-pill-value">❄️ Cold</span></div>`);

  if (!pills.length) return '';
  return `<div class="game-stats context-stats">${pills.join('')}</div>`;
}

/* ── Always-visible injury tier impact summary ────────────────── */
function buildInjuryTierSummaryHtml(game) {
  const impact = game.injury_impact || {};
  const TIER_BADGE_CLASS = { superstar: 'tier-superstar', 'all-star': 'tier-allstar', starter: 'tier-starter', backup: 'tier-backup', rotation: 'tier-rotation' };

  const renderTeamRows = (team, keyPlayers, eloPenalty, starCount, stackMult) => {
    if (!keyPlayers || !keyPlayers.length) return '';
    const rows = keyPlayers.map(p => {
      const tier = p.value_tier || null;
      const badgeClass = tier ? (TIER_BADGE_CLASS[tier] || '') : '';
      const badge = badgeClass ? `<span class="inj-tier-badge ${badgeClass}">${tier}</span>` : '';
      return `<div class="inj-tier-row">
        <span class="inj-player-name">${badge} ${p.player}</span>
        <span class="inj-tier-status">${p.position} · ${p.status}</span>
        <span class="inj-tier-elo">−${p.elo_impact} ELO</span>
      </div>`;
    }).join('');

    const stackNote = (starCount >= 2 && stackMult > 1.0)
      ? `<div class="inj-stack-note">★ Star stack ×${stackMult.toFixed(2)} applied (${starCount} elite players out)</div>`
      : '';

    const total = `<div class="inj-tier-total">
      <span>${team} total impact</span>
      <span class="inj-tier-total-elo">−${Math.round(eloPenalty)} ELO</span>
    </div>`;

    return `<div class="inj-tier-team-label">${team}</div>${rows}${stackNote}${total}`;
  };

  const homeRows = renderTeamRows(
    game.home_team,
    impact.home_key_players_out || [],
    impact.home_elo_penalty || 0,
    impact.home_star_count || 0,
    impact.home_star_stack_multiplier || 1.0
  );
  const awayRows = renderTeamRows(
    game.away_team,
    impact.away_key_players_out || [],
    impact.away_elo_penalty || 0,
    impact.away_star_count || 0,
    impact.away_star_stack_multiplier || 1.0
  );

  if (!homeRows && !awayRows) return '';

  return `<div class="inj-tier-box">
    <div class="inj-tier-box-title">Injury Impact</div>
    ${homeRows}${awayRows}
  </div>`;
}

/* ── Injury panel ─────────────────────────────────────────────── */
function buildInjuryHtml(game, isNba) {
  // For non-completed games, prefer live injury data from the injuries JSON file
  // (loaded into state.injuriesMap / state.nba.injuriesMap). This ensures the display
  // always reflects current injuries without needing to rebuild the predictions JSON.
  // For completed games, game.injuries is restored from the prediction log snapshot.
  const isCompleted = (game.status || '').includes('FINAL');
  const liveMap = isNba ? state.nba.injuriesMap : state.injuriesMap;
  let homeInjuries, awayInjuries;
  if (!isCompleted && liveMap && Object.keys(liveMap).length > 0) {
    homeInjuries = (liveMap[game.home_team] || []).slice(0, 5);
    awayInjuries = (liveMap[game.away_team] || []).slice(0, 5);
  } else {
    const injuries = game.injuries || {};
    homeInjuries = injuries.home || [];
    awayInjuries = injuries.away || [];
  }
  const impact = game.injury_impact || {};
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

  const tierBadge = tier => {
    if (!tier) return '';
    const badgeClass = { superstar: 'tier-superstar', 'all-star': 'tier-allstar', starter: 'tier-starter', backup: 'tier-backup', rotation: 'tier-rotation' }[tier] || '';
    if (!badgeClass) return '';
    return `<span class="inj-tier-badge ${badgeClass}">${tier}</span>`;
  };

  const playerChip = p => {
    const cls = statusClass(p.status);
    const name = p.name || p.player || 'Unknown';
    const pos = p.position || p.pos || '';
    const status = p.status || '';
    const tier = p.value_tier || '';
    return `<span class="injury-chip ${cls}" title="${name} — ${status}${tier ? ' · ' + tier : ''}">${pos ? `<span class="inj-pos">${pos}</span>` : ''}${tierBadge(tier)}${name}<span class="inj-status">${status}</span></span>`;
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
  const keyOutHtml = keyOut.length ? `<div class="inj-key-out">Key out: ${keyOut.map(p => {
    const tier = p.value_tier || '';
    return `<strong>${p.player}</strong>${tierBadge(tier)} (${p.position}, ${p.status}, −${p.elo_impact} ELO)`;
  }).join(' · ')}</div>` : '';

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
          <span class="adj-hint">← More injured</span>
          <label>${homeTeam} Injury Impact <span class="adj-val" id="adj-injHome-val-${gameId}">${ov.injHome || 0}</span></label>
          <span class="adj-hint">Healthier →</span>
        </div>
        <input type="range" class="adj-slider" data-game-id="${gameId}" data-adj="injHome" min="-30" max="30" step="1" value="${ov.injHome || 0}" />
      </div>
      <div class="adjust-row">
        <div class="adjust-row-label">
          <span class="adj-hint">← More injured</span>
          <label>${awayTeam} Injury Impact <span class="adj-val" id="adj-injAway-val-${gameId}">${ov.injAway || 0}</span></label>
          <span class="adj-hint">Healthier →</span>
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
    state.gameFilter = filterPill.dataset.filter || 'live';
    renderGames();
    return;
  }

  // Post-mortem toggle (prediction log)
  const pmBtn = e.target.closest('[data-postmortem]');
  if (pmBtn) {
    const row = document.getElementById('pm-' + pmBtn.dataset.postmortem);
    if (row) {
      const isOpen = row.style.display !== 'none';
      row.style.display = isOpen ? 'none' : 'table-row';
      pmBtn.classList.toggle('open', !isOpen);
    }
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

function saveWeights() {
  try { localStorage.setItem('sport3_weights', JSON.stringify(state.weights)); } catch {}
}

function onWeightChange() { updateWeightTotal(); saveWeights(); }

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

function syncWeightSliders() {
  const map = { logistic: 'sl-w-log', xgboost: 'sl-w-xgb', elo: 'sl-w-elo', pyth: 'sl-w-pyth', eff: 'sl-w-eff' };
  const valMap = { logistic: 'val-w-log', xgboost: 'val-w-xgb', elo: 'val-w-elo', pyth: 'val-w-pyth', eff: 'val-w-eff' };
  for (const [key, slId] of Object.entries(map)) {
    const sl = $(slId); const vd = $(valMap[key]);
    if (sl) sl.value = state.weights[key].toFixed(2);
    if (vd) vd.textContent = state.weights[key].toFixed(2);
  }
  updateWeightTotal();
}

// Sync sliders to any restored weight values from localStorage
syncWeightSliders();

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
  MIA:[25.7617,-80.1918], MIL:[43.0436,-87.9166], MIN:[44.9778,-93.2650],
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

  const homeElo = homeData.elo ?? homeData.mu ?? 1500;
  const awayElo = awayData.elo ?? awayData.mu ?? 1500;
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

  const homeElo = homeData.elo ?? homeData.mu ?? 1500;
  const awayElo = awayData.elo ?? awayData.mu ?? 1500;
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
    const eloProb = p.elo_prob ?? 0.5;
    const bayesProb = p.bayesian_prob ?? 0.5;
    const lrProb = p.logistic_prob ?? 0.5;
    const pythProb = p.pyth_prob ?? 0.5;
    const effProb = p.eff_prob ?? 0.5;
    const eff = game.efficiency || {};
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
      home_score: null,
      away_score: null,
      // Pre-game efficiency data for post-mortem analysis
      home_off_rating: eff.home_off_rating ?? null,
      home_def_rating: eff.home_def_rating ?? null,
      home_net_rating: eff.home_net_rating ?? null,
      away_off_rating: eff.away_off_rating ?? null,
      away_def_rating: eff.away_def_rating ?? null,
      away_net_rating: eff.away_net_rating ?? null,
      // Individual model picks (home if prob >= 0.5)
      elo_pick: eloProb >= 0.5 ? game.home_team : game.away_team,
      bayes_pick: bayesProb >= 0.5 ? game.home_team : game.away_team,
      lr_pick: lrProb >= 0.5 ? game.home_team : game.away_team,
      pyth_pick: pythProb >= 0.5 ? game.home_team : game.away_team,
      eff_pick: effProb >= 0.5 ? game.home_team : game.away_team,
      // Individual model probabilities (home team win %)
      elo_prob: eloProb,
      bayes_prob: bayesProb,
      lr_prob: lrProb,
      pyth_prob: pythProb,
      eff_prob: effProb,
      // Pre-game injury snapshot (ESPN clears this after game ends)
      home_injuries: game.injuries?.home || [],
      away_injuries: game.injuries?.away || [],
      injury_impact: game.injury_impact || {},
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

  const unresolved = entries.filter(e => !e.actual_winner);
  if (!unresolved.length) return;

  // Group by date
  const dateGroups = {};
  unresolved.forEach(e => {
    const d = e.game_time ? (() => {
      const dt = new Date(e.game_time);
      // Use UTC methods — game_time is a UTC ISO string; local getDate()
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

// ��─ Weight feedback loop ──────────────────────────────────────────
// After each batch of resolved games, nudge sub-model weights toward
// whichever models have been most accurate recently (last 20 games).
// Adjustments are intentionally tiny (±0.02 max per call).
function adjustWeightsFromLog(league) {
  const key = `sport3_log_${league}`;
  let entries = [];
  try { entries = JSON.parse(localStorage.getItem(key) || '[]'); } catch { return; }

  const resolved = entries.filter(e => e.actual_winner != null);
  if (resolved.length < 50) return; // need enough data before nudging (Issue 6 fix: 10 → 50)

  // Use the most recent 50 resolved games (Issue 6 fix: 20 → 50 for statistical stability)
  const recent = resolved.slice(-50);

  // Map log pick fields to ensemble weight keys.
  // bayes_pick removed — the Bayesian model is not part of the ensemble
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
  // Issue 6 fix: reduced delta cap ±0.02→±0.01 and sensitivity 0.1→0.05 to avoid overcorrecting
  accuracies.forEach(({ weightKey, acc }) => {
    const delta = Math.max(-0.01, Math.min(0.01, (acc - avgAcc) * 0.05));
    state.weights[weightKey] = Math.max(0.01, state.weights[weightKey] + delta);
  });

  // Re-normalise so weights sum to 1.0
  const total = Object.values(state.weights).reduce((s, v) => s + v, 0);
  for (const k of Object.keys(state.weights)) state.weights[k] /= total;

  saveWeights();
  syncWeightSliders();
}

async function renderAccuracyTab() {
  const container = $('tab-accuracy');
  if (!container) return;

  // Log any games not yet captured (handles case where user skips Games tab)
  if (state.predictions?.games) logPredictions(state.predictions.games, 'nfl');
  if (state.nba.predictions?.games) logPredictions(state.nba.predictions.games, 'nba');

  // Resolve winners in background
  await Promise.all([resolveActualWinners('nfl'), resolveActualWinners('nba')]);

  const logLeague = window._logLeaguePref || state.league;
  let nflLog = [], nbaLog = [];
  try { nflLog = JSON.parse(localStorage.getItem('sport3_log_nfl') || '[]'); } catch {}
  try { nbaLog = JSON.parse(localStorage.getItem('sport3_log_nba') || '[]'); } catch {}
  const activeLog = logLeague === 'nba' ? nbaLog : nflLog;

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
      return `<div class="log-empty">No predictions logged yet. Results will appear here once games finish.</div>`;
    }
    const rowsHtml = sortedEntries.map(e => {
      let resultHtml = '';
      let caretHtml = '';
      let postMortemRow = '';
      if (e.actual_winner) {
        const isCorrect = e.actual_winner === e.predicted_winner;
        const pmText = generatePostMortemExplanation(e);
        caretHtml = `<button class="log-postmortem-btn" data-postmortem="${e.game_id}" title="Game analysis">▼</button>`;
        postMortemRow = `<tr class="log-postmortem-row" id="pm-${e.game_id}" style="display:none">
            <td colspan="5"><div class="log-postmortem-text">${pmText}</div></td>
          </tr>`;
        if (isCorrect) {
          resultHtml = `<span class="log-correct">✓ ${e.actual_winner}</span>`;
        } else {
          resultHtml = `<span class="log-incorrect">✗ ${e.actual_winner}</span>`;
        }
      } else {
        resultHtml = `<span class="log-pending">Pending</span>`;
      }
      const gameDate = e.game_time
        ? (() => {
            const [y, m, d] = e.game_time.slice(0, 10).split('-').map(Number);
            return new Date(y, m - 1, d).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
          })()
        : '—';
      return `<tr>
        <td class="log-date">${gameDate}</td>
        <td class="log-matchup">${e.away_team} @ ${e.home_team}</td>
        <td class="log-pick ${e.predicted_winner === e.home_team ? 'home-pick' : 'away-pick'}">${e.predicted_winner}</td>
        <td class="log-conf">${(e.predicted_prob * 100).toFixed(1)}%</td>
        <td style="white-space:nowrap">${resultHtml}${caretHtml}</td>
      </tr>${postMortemRow}`;
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
    <p class="muted" style="font-size:0.8125rem;margin-bottom:1.25rem;">
      Model picks are logged automatically each time you view games. Results appear once games finish.
    </p>

    <div class="log-league-toggle" style="display:flex;gap:0.5rem;margin-bottom:1.5rem;">
      <button class="log-league-btn${logLeague === 'nfl' ? ' active' : ''}" onclick="setLogLeague('nfl')">🏈 NFL</button>
      <button class="log-league-btn${logLeague === 'nba' ? ' active' : ''}" onclick="setLogLeague('nba')">🏀 NBA</button>
    </div>

    <div class="log-league-section">
      ${renderLeagueSection(activeLog, logLeague.toUpperCase())}
    </div>

    <div style="margin-top:2rem;display:flex;gap:0.75rem;flex-wrap:wrap;">
      <button class="btn-clear-log" onclick="clearLog('${logLeague}')">Clear ${logLeague.toUpperCase()} Log</button>
      <button class="btn-clear-log" style="background:var(--accent-blue,#3b82f6)" onclick="relogAllGames('${logLeague}')">Re-log All Games</button>
    </div>`;
}

function generatePostMortemExplanation(e) {
  const winnerName = e.actual_winner === e.home_team ? e.home_name : e.away_name;
  const loserName  = e.actual_winner === e.home_team ? e.away_name : e.home_name;
  const winnerTeam = e.actual_winner;
  const loserTeam  = e.actual_winner === e.home_team ? e.away_team : e.home_team;
  const probPct = Math.round(e.predicted_prob * 100);
  const winnerWasHome = e.actual_winner === e.home_team;
  const modelWasCorrect = e.predicted_winner === e.actual_winner;
  const predictedName   = e.predicted_winner === e.home_team ? e.home_name : e.away_name;

  // ── Score header ──────────────────────────────────────────────
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

  // ── Pre-game efficiency table ─────────────────────────────────
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
        // Only note model failure when model was actually wrong (gap<-3 means
        // winner had the stronger net rating, which alone does not imply a model error).
        narrativeLines.push(`${winnerName} had the stronger net rating (+${Math.abs(gap).toFixed(1)} advantage) yet the model still went wrong — efficiency didn't tell the whole story.`);
      } else if (lOff != null && wOff != null && lOff - wOff > 3) {
        narrativeLines.push(`${loserName}'s offense (${lOff.toFixed(1)}) looked superior on paper but couldn't convert that into points on the night.`);
      }
    }
  }

  // ── Model vote breakdown ──────────────────────────────────────
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
      // Show probability if stored, oriented toward the pick
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
      // All visible sub-models wrong
      consensusText = `All ${modelDefs.length} sub-models were unanimous on ${loserTeam} — a genuine statistical upset.`;
    } else if (correctCount === modelDefs.length) {
      // All visible sub-models correct
      if (modelWasCorrect) {
        consensusText = `All ${modelDefs.length} sub-models unanimously called ${winnerTeam} — the ensemble agreed.`;
      } else {
        // Hidden XGBoost component overrode unanimous sub-model vote
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

function clearLog(league) {
  localStorage.removeItem(`sport3_log_${league}`);
  renderAccuracyTab();
}

async function relogAllGames(league) {
  // If predictions are not yet loaded, force a re-fetch before proceeding
  let games = league === 'nba'
    ? (state.nba.predictions?.games || [])
    : (state.predictions?.games || []);
  if (!games.length) {
    if (league === 'nba') {
      await loadNbaData();
      games = state.nba.predictions?.games || [];
    } else {
      await loadNflData();
      games = state.predictions?.games || [];
    }
  }
  if (!games.length) { alert('No games loaded — data fetch returned no games.'); return; }

  const key = `sport3_log_${league}`;
  let existing = [];
  try { existing = JSON.parse(localStorage.getItem(key) || '[]'); } catch { existing = []; }

  const existingMap = new Map(existing.map(e => [e.game_id, e]));
  let changed = false;

  games.forEach(game => {
    const p = game.predictions || {};
    const ensProb = ensembleFromProbs(p, state.weights);
    const predictedWinner = ensProb >= 0.5 ? game.home_team : game.away_team;
    const predictedProb = ensProb >= 0.5 ? ensProb : (1 - ensProb);
    const eloProb = p.elo_prob ?? 0.5;
    const bayesProb = p.bayesian_prob ?? 0.5;
    const lrProb = p.logistic_prob ?? 0.5;
    const pythProb = p.pyth_prob ?? 0.5;
    const effProb = p.eff_prob ?? 0.5;
    const eff = game.efficiency || {};
    const homeInj = game.injuries?.home || [];
    const awayInj = game.injuries?.away || [];
    const injImpact = game.injury_impact || {};

    if (!existingMap.has(game.game_id)) {
      // New entry — add it
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
        home_score: null,
        away_score: null,
        home_off_rating: eff.home_off_rating ?? null,
        home_def_rating: eff.home_def_rating ?? null,
        home_net_rating: eff.home_net_rating ?? null,
        away_off_rating: eff.away_off_rating ?? null,
        away_def_rating: eff.away_def_rating ?? null,
        away_net_rating: eff.away_net_rating ?? null,
        elo_pick: eloProb >= 0.5 ? game.home_team : game.away_team,
        bayes_pick: bayesProb >= 0.5 ? game.home_team : game.away_team,
        lr_pick: lrProb >= 0.5 ? game.home_team : game.away_team,
        pyth_pick: pythProb >= 0.5 ? game.home_team : game.away_team,
        eff_pick: effProb >= 0.5 ? game.home_team : game.away_team,
        elo_prob: eloProb,
        bayes_prob: bayesProb,
        lr_prob: lrProb,
        pyth_prob: pythProb,
        eff_prob: effProb,
        home_injuries: homeInj,
        away_injuries: awayInj,
        injury_impact: injImpact,
      });
      existingMap.set(game.game_id, existing[existing.length - 1]);
      changed = true;
    } else {
      // Existing entry — refresh predictions snapshot AND injury data with latest values
      const entry = existingMap.get(game.game_id);
      entry.predicted_winner = predictedWinner;
      entry.predicted_prob = predictedProb;
      entry.elo_pick = eloProb >= 0.5 ? game.home_team : game.away_team;
      entry.bayes_pick = bayesProb >= 0.5 ? game.home_team : game.away_team;
      entry.lr_pick = lrProb >= 0.5 ? game.home_team : game.away_team;
      entry.pyth_pick = pythProb >= 0.5 ? game.home_team : game.away_team;
      entry.eff_pick = effProb >= 0.5 ? game.home_team : game.away_team;
      entry.elo_prob = eloProb;
      entry.bayes_prob = bayesProb;
      entry.lr_prob = lrProb;
      entry.pyth_prob = pythProb;
      entry.eff_prob = effProb;
      entry.home_injuries = homeInj;
      entry.away_injuries = awayInj;
      entry.injury_impact = injImpact;
      changed = true;
    }
  });

  if (changed) {
    try { localStorage.setItem(key, JSON.stringify(existing)); } catch {}
  }
  // Resolve actual winners for any newly added (or previously unresolved) entries
  await resolveActualWinners(league);
  renderAccuracyTab();
}

function setLogLeague(league) {
  window._logLeaguePref = league;
  renderAccuracyTab();
}

/* ── Kalshi Odds Comparison ───────────────────────────────────── */

function kalshiRelativeTime(isoStr) {
  try {
    const diff = Date.now() - new Date(isoStr).getTime();
    if (diff < 0 || new Date(isoStr).getFullYear() < 2000) return null;
    const mins = Math.floor(diff / 60000);
    if (mins < 2) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
  } catch { return null; }
}

const KALSHI_EXCLUDE_KWS = [
  'super bowl', 'championship', 'mvp', 'season wins', 'total wins',
  'make playoffs', 'win division', 'draft', 'spread', 'cover',
  'over/under', 'first quarter', 'first half', 'halftime',
];

/* Build token→abbrev lookup from TEAM_NAMES or NBA_TEAM_NAMES */
function buildKalshiTokenMap(teamNames) {
  const map = {};
  for (const [abbrev, fullName] of Object.entries(teamNames)) {
    const parts = fullName.toLowerCase().split(/\s+/);
    map[parts[parts.length - 1]] = abbrev;              // nickname: "chiefs"
    if (parts.length >= 2) map[parts.slice(0, -1).join(' ')] = abbrev; // city: "kansas city"
    map[abbrev.toLowerCase()] = abbrev;                 // abbrev: "kc"
  }
  return map;
}

/* Extract up to 2 team abbrevs from a market object.
   For KXNBAGAME / KXNFLGAME markets, parses team codes directly from event_ticker
   (e.g. "KXNBAGAME-26MAR18LALHOU" → LAL, HOU) which is unambiguous.
   Falls back to title token-matching for other market types.
   Returns { teamA, teamB, yesIsTeamA } or null if fewer than 2 found. */
function kalshiExtractTeams(market, tokenMap) {
  // Prefer ticker-based extraction for game-winner markets
  const eventTicker = market.event_ticker || '';
  const tickerMatch = eventTicker.match(/^KX(?:NBA|NFL)GAME-\d{2}[A-Z]{3}\d{2}([A-Z]{3})([A-Z]{3})$/i);
  if (tickerMatch) {
    const teamA = tickerMatch[1].toUpperCase();
    const teamB = tickerMatch[2].toUpperCase();
    // The market ticker suffix (-LAL / -HOU) identifies which team "yes" resolves on
    const yesSuffix = (market.ticker || '').split('-').pop().toUpperCase();
    const yesIsTeamA = yesSuffix === teamA;
    return { teamA, teamB, yesIsTeamA };
  }
  // Fall back to title token matching
  const title = market.title || '';
  const lower = title.toLowerCase().replace(/[^a-z0-9 ]/g, ' ');
  // Sort tokens longest-first to prefer "kansas city" over "city"
  const tokens = Object.keys(tokenMap).sort((a, b) => b.length - a.length);
  const found = [];
  let scratch = lower;
  for (const tok of tokens) {
    if (found.length >= 2) break;
    if (scratch.includes(tok)) {
      const abbrev = tokenMap[tok];
      if (!found.some(f => f.abbrev === abbrev)) {
        found.push({ abbrev, idx: lower.indexOf(tok) });
        scratch = scratch.replaceAll(tok, ' ');
      }
    }
  }
  if (found.length < 2) return null;
  // Team mentioned first in title is typically the "yes" side
  found.sort((a, b) => a.idx - b.idx);
  return { teamA: found[0].abbrev, teamB: found[1].abbrev, yesIsTeamA: true };
}

/* Load pre-fetched Kalshi markets from data/kalshidata.json (generated server-side by updatekalshi.py) */
async function loadKalshiData() {
  const t = Date.now();
  const res = await fetch(`./data/kalshidata.json?t=${t}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/* Match Kalshi markets to prediction games and compute mismatch rows */
function buildKalshiRows(markets, games, tokenMap) {
  const gameMap = new Map();
  for (const g of games) {
    gameMap.set(`${g.home_team}|${g.away_team}`, g);
    gameMap.set(`${g.away_team}|${g.home_team}`, g);
  }

  const matched = new Set();
  const rows = [];

  for (const m of markets) {
    const titleLower = (m.title || '').toLowerCase();
    if (KALSHI_EXCLUDE_KWS.some(kw => titleLower.includes(kw))) continue;
    const teams = kalshiExtractTeams(m, tokenMap);
    if (!teams) continue;
    const { teamA, teamB, yesIsTeamA } = teams;
    const game = gameMap.get(`${teamA}|${teamB}`) || gameMap.get(`${teamB}|${teamA}`);
    if (!game || matched.has(game.game_id)) continue;
    matched.add(game.game_id);

    // Kalshi mid-price (yes_bid / yes_ask in cents 0–99)
    const bid = m.yes_bid != null ? m.yes_bid : (m.yes_bid_dollars || 0) * 100;
    const ask = m.yes_ask != null ? m.yes_ask : (m.yes_ask_dollars || 0) * 100;
    const yesMid = Math.max(1, Math.min(99, (bid + ask) / 2)) / 100;
    const yesTeam = yesIsTeamA ? teamA : teamB;
    const kalshiHomeProb = yesTeam === game.home_team ? yesMid : 1 - yesMid;

    const modelProb = ensembleFromProbs(game.predictions || {}, state.weights);
    const mismatch = modelProb - kalshiHomeProb; // signed: + = model more bullish on home

    rows.push({ game, market: m, modelProb, kalshiHomeProb, mismatch, volume: parseFloat(m.volume_fp || m.volume || 0) });
  }

  // Append unmatched upcoming games
  for (const g of games) {
    if (!matched.has(g.game_id)) {
      rows.push({ game: g, market: null, modelProb: ensembleFromProbs(g.predictions || {}, state.weights), kalshiHomeProb: null, mismatch: null, volume: 0 });
    }
  }

  return rows;
}

/* Sort rows per state.kalshi sort settings */
function sortKalshiRows(rows) {
  const { sortCol, sortDir } = state.kalshi;
  const dir = sortDir === 'asc' ? 1 : -1;
  rows.sort((a, b) => {
    // Nulls (unmatched) always sink to bottom
    if (a.mismatch == null && b.mismatch == null) return 0;
    if (sortCol !== 'matchup') {
      if (a.market == null) return 1;
      if (b.market == null) return -1;
    }
    if (sortCol === 'mismatch') return dir * (Math.abs(b.mismatch) - Math.abs(a.mismatch));
    if (sortCol === 'model')   return dir * (a.modelProb - b.modelProb);
    if (sortCol === 'kalshi')  return dir * (a.kalshiHomeProb - b.kalshiHomeProb);
    if (sortCol === 'volume')  return dir * (a.volume - b.volume);
    if (sortCol === 'matchup') return dir * (`${a.game.away_team}@${a.game.home_team}`).localeCompare(`${b.game.away_team}@${b.game.home_team}`);
    return 0;
  });
}

/* Render the Kalshi table into container (called from renderKalshiTab and on re-sort) */
function renderKalshiTable(rows, isNba) {
  const container = $('kalshi-content');
  const badge = $('kalshi-badge');
  if (!container) return;

  if (!rows.length) {
    container.innerHTML = `<div class="empty-state"><div class="empty-state-icon">${isNba ? '🏀' : '🏈'}</div><h3>No Upcoming Games</h3><p>No upcoming or live games to compare with Kalshi markets.</p></div>`;
    return;
  }

  const matchedCount = rows.filter(r => r.market !== null).length;
  if (badge) badge.textContent = `${matchedCount} of ${rows.length} matched`;

  const { sortCol, sortDir } = state.kalshi;
  const arrow = col => sortCol === col ? (sortDir === 'desc' ? ' ▼' : ' ▲') : '';
  const cls   = col => sortCol === col ? 'sorted' : '';

  const rowsHtml = rows.map(r => {
    const g = r.game;
    const live = g.status === 'STATUS_IN_PROGRESS' || g.status === 'STATUS_HALFTIME';
    const matchup = `${live ? '● ' : ''}${g.away_team} @ ${g.home_team}`;

    if (!r.market) {
      return `<tr class="kalshi-row-unmatched">
        <td><span class="mono" style="font-weight:600;">${matchup}</span></td>
        <td class="num-cell mono">${(r.modelProb * 100).toFixed(1)}%</td>
        <td class="num-cell muted">—</td>
        <td class="num-cell muted">—</td>
        <td><span class="muted">No market</span></td>
        <td class="num-cell muted">—</td>
      </tr>`;
    }

    const absMis = Math.abs(r.mismatch);
    const mismatchClass = absMis >= 0.10 ? 'mismatch-high' : absMis >= 0.05 ? 'mismatch-mid' : 'mismatch-low';
    let edgeHtml;
    if (absMis < 0.02) {
      edgeHtml = '<span class="muted">≈ Aligned</span>';
    } else if (r.mismatch > 0) {
      edgeHtml = `<span class="edge-model">▲ Model favors ${g.home_team}</span>`;
    } else {
      edgeHtml = `<span class="edge-market">▲ Market favors ${g.away_team}</span>`;
    }

    const marketTicker = r.market.event_ticker || r.market.ticker || '';
    const volStr = r.volume > 0 ? r.volume.toLocaleString() : '—';

    return `<tr>
      <td>
        <span class="mono" style="font-weight:600;">${matchup}</span>
        <a class="kalshi-link" href="https://kalshi.com/markets/${marketTicker}" target="_blank" rel="noopener" style="display:block;">${r.market.ticker || marketTicker}</a>
      </td>
      <td class="num-cell mono">${(r.modelProb * 100).toFixed(1)}%</td>
      <td class="num-cell mono">${(r.kalshiHomeProb * 100).toFixed(1)}%</td>
      <td class="num-cell mono ${mismatchClass}">${(absMis * 100).toFixed(1)}%</td>
      <td>${edgeHtml}</td>
      <td class="num-cell mono muted">${volStr}</td>
    </tr>`;
  }).join('');

  const updatedStr = state.kalshiUpdated ? kalshiRelativeTime(state.kalshiUpdated) : null;
  const updatedNote = updatedStr ? `<span style="color:var(--text-muted);font-size:0.75rem;">Updated ${updatedStr}</span>` : '';

  container.innerHTML = `
    <div class="kalshi-note">
      Mismatch = |Model% − Kalshi implied%| for home team winning. Kalshi % uses mid-price of bid+ask.
      Ranked by biggest gap first. <strong>Not financial advice.</strong>
    </div>
    <div class="table-wrapper">
      <table class="kalshi-table">
        <thead><tr>
          <th data-kcol="matchup" class="${cls('matchup')}" style="cursor:pointer;">Matchup${arrow('matchup')}</th>
          <th data-kcol="model" class="${cls('model')} num-cell" style="cursor:pointer;">Model (Home%)${arrow('model')}</th>
          <th data-kcol="kalshi" class="${cls('kalshi')} num-cell" style="cursor:pointer;">Kalshi (Home%)${arrow('kalshi')}</th>
          <th data-kcol="mismatch" class="${cls('mismatch')} num-cell" style="cursor:pointer;">Mismatch${arrow('mismatch')}</th>
          <th>Edge Direction</th>
          <th data-kcol="volume" class="${cls('volume')} num-cell" style="cursor:pointer;">Volume${arrow('volume')}</th>
        </tr></thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    </div>
    <div style="display:flex;align-items:center;justify-content:flex-end;gap:0.75rem;margin-top:0.75rem;">
      ${updatedNote}
      <button class="btn" onclick="state.kalshiData=null;renderKalshiTab(true)" style="font-size:0.75rem;padding:4px 10px;">↻ Refresh Markets</button>
      <a class="btn" href="https://github.com/tylerherman19/Sport3/actions/workflows/update-kalshi.yml" target="_blank" rel="noopener" style="font-size:0.75rem;padding:4px 10px;text-decoration:none;">⚡ Force Update</a>
    </div>`;

  // Column sort bindings
  container.querySelectorAll('th[data-kcol]').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.kcol;
      if (state.kalshi.sortCol === col) {
        state.kalshi.sortDir = state.kalshi.sortDir === 'desc' ? 'asc' : 'desc';
      } else {
        state.kalshi.sortCol = col;
        state.kalshi.sortDir = col === 'matchup' ? 'asc' : 'desc';
      }
      sortKalshiRows(state.kalshi.rows);
      renderKalshiTable(state.kalshi.rows, state.league === 'nba');
    });
  });
}

/* Switch the Kalshi tab's own league filter (independent of global league) */
function setKalshiLeague(league) {
  state.kalshi.league = league;
  document.querySelectorAll('#kalshi-league-pills .filter-pill').forEach(p => {
    p.classList.toggle('active', p.dataset.kleague === league);
  });
  renderKalshiTab(false);
}

async function renderKalshiTab(forceRefetch) {
  const container = $('kalshi-content');
  if (!container) return;

  const isNba = state.kalshi.league === 'nba';
  const pred = isNba ? state.nba.predictions : state.predictions;
  const teamNames = isNba ? NBA_TEAM_NAMES : TEAM_NAMES;
  const tokenMap = buildKalshiTokenMap(teamNames);
  const games = (pred?.games || []).filter(g => g.status !== 'STATUS_FINAL');

  // Re-render from cache on league toggle or repeated visits
  if (!forceRefetch && state.kalshiData) {
    state.kalshi.rows = buildKalshiRows(state.kalshiData, games, tokenMap);
    sortKalshiRows(state.kalshi.rows);
    renderKalshiTable(state.kalshi.rows, isNba);
    return;
  }

  container.innerHTML = '<div class="loading"><div class="spinner"></div><span>Loading Kalshi markets…</span></div>';

  try {
    const data = await loadKalshiData();
    state.kalshiData = data.markets || [];
    state.kalshiUpdated = data.updated || null;
    state.kalshi.rows = buildKalshiRows(state.kalshiData, games, tokenMap);
    sortKalshiRows(state.kalshi.rows);
    renderKalshiTable(state.kalshi.rows, isNba);
  } catch (err) {
    container.innerHTML = `<div class="empty-state">
      <div class="empty-state-icon">⚠️</div>
      <h3>Could not load Kalshi data</h3>
      <p>kalshidata.json is unavailable. Use the Force Update button to trigger a fresh fetch via GitHub Actions.</p>
      <div style="display:flex;gap:0.75rem;justify-content:center;margin-top:0.75rem;flex-wrap:wrap;">
        <button class="btn btn-primary" onclick="state.kalshiData=null;renderKalshiTab(true)">Retry</button>
        <a class="btn" href="https://github.com/tylerherman19/Sport3/actions/workflows/update-kalshi.yml" target="_blank" rel="noopener" style="text-decoration:none;">⚡ Force Update on GitHub</a>
      </div>
    </div>`;
    const badge = $('kalshi-badge');
    if (badge) badge.textContent = 'Error';
  }
}

/* Re-render Kalshi tab if it is currently active (called from renderAll) */
function renderKalshiIfOpen() {
  const sec = $('tab-kalshi');
  if (sec && sec.classList.contains('active')) renderKalshiTab(false);
}
