/* Kalshi Odds Comparison — sandboxed from main.js
 * To use: copy back into main.js and re-enable entry points.
 * Also requires: Kalshi nav tab + section in index.html (see kalshi_tab.html),
 *   and .github/workflows/update-kalshi.yml (see update-kalshi.yml).
 * Requires state fields: kalshiData, kalshiUpdated, kalshi{rows,sortCol,sortDir,league}
 * Requires TEAM_NAMES, NBA_TEAM_NAMES, ensembleFromProbs, state.weights from main.js.
 */

/* ── Kalshi Odds Comparison ─────────────────────────────────────────────── */

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
    const yesSuffix = (market.ticker || '').split('-').pop().toUpperCase();
    const yesIsTeamA = yesSuffix === teamA;
    return { teamA, teamB, yesIsTeamA };
  }
  // Fall back to title token matching
  const title = market.title || '';
  const lower = title.toLowerCase().replace(/[^a-z0-9 ]/g, ' ');
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

    const bid = m.yes_bid != null ? m.yes_bid : (m.yes_bid_dollars || 0) * 100;
    const ask = m.yes_ask != null ? m.yes_ask : (m.yes_ask_dollars || 0) * 100;
    const yesMid = Math.max(1, Math.min(99, (bid + ask) / 2)) / 100;
    const yesTeam = yesIsTeamA ? teamA : teamB;
    const kalshiHomeProb = yesTeam === game.home_team ? yesMid : 1 - yesMid;

    const modelProb = ensembleFromProbs(game.predictions || {}, state.weights);
    const mismatch = modelProb - kalshiHomeProb;

    rows.push({ game, market: m, modelProb, kalshiHomeProb, mismatch, volume: parseFloat(m.volume_fp || m.volume || 0) });
  }

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