/* Sport3 — broadsheet frontend. Reads the same ./data/*.json the pipeline writes. */

const TZ = "America/Chicago";
const state = {
  predictions: null,
  leaderboard: null,
  metrics: null,
  injuries: [],
  week: null,
  sortCol: "elo",
  sortDir: -1,
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
const pct1 = (p) => (p == null ? "—" : (p * 100).toFixed(1) + "%");
const pct0 = (p) => (p == null ? "—" : Math.round(p * 100) + "%");
const signed = (n, d = 1) => (n == null ? "—" : (n > 0 ? "+" : "") + n.toFixed(d));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function getJSON(path) {
  try {
    const r = await fetch(path + "?t=" + Date.now());
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

/* ── Masthead + lede ─────────────────────────────────────── */

function renderMasthead(pred) {
  $("#mast-week").textContent = `WEEK ${state.week} · ${pred.season} NFL SEASON`;
  const upd = pred.updated ? new Date(pred.updated) : null;
  $("#mast-updated").textContent = upd
    ? "UPDATED " + upd.toLocaleString("en-US", { timeZone: TZ, month: "short", day: "numeric" }).toUpperCase() +
      ", " + upd.toLocaleString("en-US", { timeZone: TZ, hour: "numeric", minute: "2-digit" }).toLowerCase() + " CT"
    : "";
}

function renderLede(pred) {
  const games = (pred.games || []).filter((g) => g.week === state.week);
  if (!games.length) { $("#lede").textContent = ""; return; }
  let fav = null, tight = null;
  for (const g of games) {
    const p = g.predictions?.ensemble_prob;
    if (p == null) continue;
    const top = Math.max(p, 1 - p);
    if (!fav || top > fav.top) fav = { g, top };
    if (!tight || top < tight.top) tight = { g, top };
  }
  const bits = [];
  if (fav) {
    const fTeam = fav.g.predictions.ensemble_prob >= 0.5 ? fav.g.home_team : fav.g.away_team;
    bits.push(`${fTeam} is the week's strongest favorite at ${pct1(fav.top)}`);
  }
  if (tight && tight !== fav) {
    const tTeams = `${tight.g.away_team} at ${tight.g.home_team}`;
    bits.push(`${tTeams} is the coin flip at ${pct1(tight.top)}`);
  }
  const m = state.metrics;
  if (m?.accuracy != null && m?.vegas_benchmark?.accuracy != null) {
    const gap = (m.vegas_benchmark.accuracy - m.accuracy) * 100;
    bits.push(`the model trails Vegas by ${gap.toFixed(1)} points over ${fmtInt(m.n_scored_games)} games`);
  }
  $("#lede").innerHTML =
    `${games.length} games on the board this week: ${esc(bits[0])}, ${esc(bits[1] || "no coin flips on file")}, and ${esc(bits[2] || "the ledger is closed")}.`;
}

const fmtInt = (n) => (n == null ? "—" : Number(n).toLocaleString("en-US"));

/* ── The Board ───────────────────────────────────────────── */

function renderWeekLine(pred) {
  const weeks = [...new Set((pred.games || []).map((g) => g.week))].sort((a, b) => a - b);
  const wrap = $("#week-line");
  wrap.innerHTML = "";
  for (const w of weeks) {
    const b = el("button", null, `WK ${w}`);
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(w === state.week));
    b.title = w === pred.week ? "Current week" : "";
    b.addEventListener("click", () => { state.week = w; renderWeekLine(pred); renderGames(pred); renderMasthead(pred); renderLede(pred); });
    wrap.appendChild(b);
  }
}

function dayKey(g) {
  const d = new Date(g.game_time);
  return d.toLocaleDateString("en-US", { timeZone: TZ, weekday: "long", month: "long", day: "numeric" });
}
function timeStr(g) {
  return new Date(g.game_time).toLocaleTimeString("en-US", { timeZone: TZ, hour: "numeric", minute: "2-digit" }).replace(" ", "").toLowerCase();
}

function favorite(g) {
  const p = g.predictions?.ensemble_prob;
  if (p == null) return null;
  return p >= 0.5
    ? { team: g.home_team, name: g.home_name, p }
    : { team: g.away_team, name: g.away_name, p: 1 - p };
}

function renderGames(pred) {
  const wrap = $("#games");
  wrap.innerHTML = "";
  const games = (pred.games || [])
    .filter((g) => g.week === state.week)
    .sort((a, b) => new Date(a.game_time) - new Date(b.game_time));

  $("#board-dek").textContent = `${games.length} games, kickoff to kickoff. All times Central. Tap a game for the full scouting line.`;

  let lastDay = null;
  for (const g of games) {
    const dk = dayKey(g);
    if (dk !== lastDay) {
      wrap.appendChild(el("h3", "day-head", dk));
      lastDay = dk;
    }
    wrap.appendChild(gameBlock(g));
  }
}

function gameBlock(g) {
  const box = el("div", "game");
  const row = el("button", "game-row");
  row.type = "button";
  row.setAttribute("aria-expanded", "false");

  const when = el("div", "game-when");
  when.innerHTML = `${esc(timeStr(g))}<br>${g.neutral ? "neutral" : ""}`;

  const teams = el("div", "game-teams");
  teams.innerHTML = `${esc(g.away_name)} at <span class="home">${esc(g.home_name)}</span>`;

  const f = favorite(g);
  const sub = el("div", "game-sub");
  const qbs = [];
  if (g.player_form?.away_qb) qbs.push(esc(g.player_form.away_qb));
  if (g.player_form?.home_qb) qbs.push(esc(g.player_form.home_qb));
  sub.textContent = qbs.length ? qbs.join(" vs ") : "";

  const pick = el("div", "game-pick");
  if (f) {
    const mc = g.monte_carlo?.exp_margin;
    pick.innerHTML =
      `<span class="pick-team">${esc(f.team)}</span><span class="pick-pct">${pct1(f.p)}</span>` +
      `<span class="pick-margin">${mc != null ? "by " + Math.abs(mc).toFixed(1) : ""}</span>`;
  } else {
    pick.innerHTML = `<span class="pick-margin">no line</span>`;
  }

  row.append(when, teams, sub, pick);
  row.addEventListener("click", () => {
    const open = box.classList.toggle("open");
    row.setAttribute("aria-expanded", String(open));
  });

  box.append(row, gameDetail(g));
  return box;
}

function gameDetail(g) {
  const d = el("div", "game-detail");
  const lines = [];

  const elo = g.elo || {};
  if (elo.home != null)
    lines.push(`<b>Elo</b> ${esc(g.home_team)} ${Math.round(elo.home)} · ${esc(g.away_team)} ${Math.round(elo.away)}`);

  const adj = g.adjustments || {};
  const rest = `Rest ${adj.rest_home ?? "—"} vs ${adj.rest_away ?? "—"} days`;
  const travel = adj.travel_dist_miles ? ` · ${esc(g.away_team)} travels ${fmtInt(Math.round(adj.travel_dist_miles))} mi${adj.travel_adj ? " (" + signed(adj.travel_adj, 0) + " Elo)" : ""}` : "";
  lines.push(`<b>Camp</b> ${rest}${travel}`);

  const pf = g.player_form || {};
  if (pf.home_qb || pf.away_qb) {
    lines.push(
      `<b>Quarterbacks</b> ${esc(pf.home_qb || "—")} (${signed(pf.home_qb_epa_l8, 3)} EPA L8) vs ${esc(pf.away_qb || "—")} (${signed(pf.away_qb_epa_l8, 3)})`
    );
  }

  const mc = g.monte_carlo || {};
  if (mc.exp_margin != null) {
    const f = favorite(g);
    lines.push(
      `<b>Simulation</b> 10,000 runs: ${esc(f ? f.team : "")} by ${Math.abs(mc.exp_margin).toFixed(1)}, ${pct0(mc.prob_7plus)} to win by 7+, ${pct0(mc.prob_14plus)} by 14+`
    );
  }

  if (g.explanation) lines.push(esc(g.explanation).replace(/—/g, "-"));

  const models = [];
  const p = g.predictions || {};
  const add = (k, label, shipped) => { if (p[k] != null) models.push(`<span class="m${shipped ? " shipped" : ""}">${label} ${pct0(p[k])}</span>`); };
  add("ensemble_prob", "MODEL", true);
  add("elo_prob", "ELO"); add("logistic_prob", "LOG"); add("xgb_prob", "XGB");
  add("pyth_prob", "PYTH"); add("eff_prob", "EFF"); add("bayesian_prob", "BAY");
  if (mc.win_prob != null) models.push(`<span class="m">MC ${pct0(mc.win_prob)}</span>`);

  d.innerHTML = lines.map((l) => `<p class="detail-line">${l}</p>`).join("");

  const dr = g.prediction_drivers || [];
  if (dr.length) {
    const ul = el("ul", "drivers");
    ul.innerHTML = dr.slice(0, 8).map((x) => `<li>${esc(x).replace(/—/g, "-")}</li>`).join("");
    d.appendChild(ul);
  }

  const ii = g.injury_impact || {};
  const injBits = [];
  for (const side of ["home", "away"]) {
    const team = side === "home" ? g.home_team : g.away_team;
    const kp = ii[`${side}_key_players_out`] || [];
    const pen = ii[`${side}_elo_penalty`];
    if (kp.length) {
      injBits.push(`<b>${esc(team)}</b> ${kp.map((x) => `${esc(x.player)} (${esc(x.position)}, ${esc(String(x.status).toLowerCase())})`).join(", ")}${pen ? `, ${signed(-pen, 1)} Elo` : ""}`);
    }
  }
  if (injBits.length) d.insertAdjacentHTML("beforeend", `<p class="detail-line"><b>Missing</b> ${injBits.join(" · ")}</p>`);

  if (models.length) d.insertAdjacentHTML("beforeend", `<p class="model-line">${models.join(" · ")}</p>`);
  return d;
}

/* ── The Ledger ──────────────────────────────────────────── */

function renderLedger(m) {
  if (!m) return;
  const sr = $("#stat-row");
  sr.innerHTML = "";
  const stats = [
    { n: pct1(m.accuracy), l: `Model accuracy · ${fmtInt(m.n_scored_games)} games` },
    { n: pct1(m.vegas_benchmark?.accuracy), l: `Vegas accuracy · ${fmtInt(m.vegas_benchmark?.n_games)} games` },
    { n: m.log_loss?.toFixed(3), l: "Log loss (Vegas " + (m.vegas_benchmark?.log_loss?.toFixed(3) ?? "—") + ")" },
    { n: m.auc?.toFixed(3), l: "AUC (Vegas " + (m.vegas_benchmark?.auc?.toFixed(3) ?? "—") + ")" },
  ];
  for (const s of stats) {
    const d = el("div", "stat");
    d.innerHTML = `<span class="n">${esc(s.n ?? "—")}</span><span class="l">${esc(s.l)}</span>`;
    sr.appendChild(d);
  }

  const yt = $("#year-table");
  const hist = (m.historical_accuracy || []).slice().sort((a, b) => a.year - b.year);
  const min = 0.55, max = 0.72;
  yt.innerHTML =
    `<tr><th>Season</th><th class="num">Accuracy</th><th class="bar-cell"></th></tr>` +
    hist.map((h) => {
      const w = Math.max(2, Math.round(((h.accuracy - min) / (max - min)) * 100));
      return `<tr><td>${h.year}</td><td class="num">${pct1(h.accuracy)}</td><td class="bar-cell"><span class="bar" style="width:${w}%"></span></td></tr>`;
    }).join("") +
    `<tr><td>Vegas, same span</td><td class="num">${pct1(m.vegas_benchmark?.accuracy)}</td><td class="bar-cell"><span class="bar accent-bar" style="width:${Math.round(((m.vegas_benchmark?.accuracy - min) / (max - min)) * 100)}%"></span></td></tr>`;

  const ct = $("#cal-table");
  const buckets = (m.calibration_buckets || []).filter((b) => b.count > 0);
  ct.innerHTML =
    `<tr><th>It said</th><th class="num">Predicted</th><th class="num">Happened</th><th class="num">Games</th></tr>` +
    buckets.map((b) =>
      `<tr><td>${esc(b.bucket)}</td><td class="num">${pct1(b.predicted)}</td><td class="num">${pct1(b.actual)}</td><td class="num">${fmtInt(b.count)}</td></tr>`
    ).join("");

  $("#ledger-foot").textContent =
    `${m.evaluation || "Walk-forward backtest"}. Trained on ${fmtInt(m.n_training_games)} games, scored on ${fmtInt(m.n_scored_games)}. Brier ${m.brier_score?.toFixed(4) ?? "—"}.`;
}

/* ── Power Rankings ──────────────────────────────────────── */

const RANK_COLS = [
  { key: "team", label: "Team", get: (t) => t.team, text: (t) => t.team },
  { key: "elo", label: "Elo", num: true, get: (t) => t.elo, text: (t) => Math.round(t.elo) },
  { key: "band", label: "True band", num: true, get: (t) => t.mu, text: (t) => `${Math.round(t.lower_band)}–${Math.round(t.upper_band)}` },
  { key: "wins", label: "W–L", num: true, get: (t) => (t.wins ?? 0) - (t.losses ?? 0), text: (t) => `${t.wins ?? 0}–${t.losses ?? 0}` },
  { key: "playoff_prob", label: "Playoffs", num: true, get: (t) => t.playoff_prob, text: (t) => pct0(t.playoff_prob) },
  { key: "sb_prob", label: "Title", num: true, get: (t) => t.sb_prob, text: (t) => pct0(t.sb_prob) },
  { key: "injury_elo_penalty", label: "Inj. cost", num: true, get: (t) => t.injury_elo_penalty, text: (t) => (t.injury_elo_penalty ? "−" + t.injury_elo_penalty.toFixed(1) : "0") },
];

function renderRankings(lb) {
  const tbl = $("#rank-table");
  if (!lb?.teams) return;
  const teams = lb.teams.slice().sort((a, b) => {
    const c = RANK_COLS.find((x) => x.key === state.sortCol) || RANK_COLS[1];
    const av = c.get(a) ?? -1e9, bv = c.get(b) ?? -1e9;
    return (av > bv ? 1 : av < bv ? -1 : 0) * state.sortDir;
  });

  tbl.innerHTML = "";
  const head = el("tr");
  head.innerHTML = `<th class="num">Rk</th>` + RANK_COLS.map((c) =>
    `<th class="${c.num ? "num" : ""}" data-col="${c.key}" ${c.key === state.sortCol ? `aria-sort="${state.sortDir < 0 ? "descending" : "ascending"}"` : ""}>${c.label}</th>`
  ).join("");
  tbl.appendChild(head);

  teams.forEach((t, i) => {
    const tr = el("tr");
    tr.innerHTML = `<td class="num rk">${i + 1}</td>` + RANK_COLS.map((c) =>
      `<td class="${c.num ? "num" : ""} ${c.key === "team" ? "team" : ""}" title="${esc(t.team_name)}">${esc(c.text(t))}</td>`
    ).join("");
    tbl.appendChild(tr);
  });

  tbl.querySelectorAll("th[data-col]").forEach((th) =>
    th.addEventListener("click", () => {
      const k = th.dataset.col;
      if (state.sortCol === k) state.sortDir *= -1;
      else { state.sortCol = k; state.sortDir = k === "team" ? 1 : -1; }
      renderRankings(state.leaderboard);
    })
  );
}

/* ── Injury Report ───────────────────────────────────────── */

const STATUS_RANK = { "Out": 0, "Injured Reserve": 1, "Suspension": 2, "Questionable": 3 };
const STATUS_SHORT = { "Out": "OUT", "Injured Reserve": "IR", "Suspension": "SUSP", "Questionable": "Q" };

function renderInjuries(inj) {
  const grid = $("#inj-grid");
  grid.innerHTML = "";
  const byTeam = {};
  for (const p of inj) {
    if (!p.team || !(p.status in STATUS_RANK)) continue;
    (byTeam[p.team] = byTeam[p.team] || []).push(p);
  }
  const teams = Object.keys(byTeam).sort();
  let total = 0;
  for (const t of teams) {
    const players = byTeam[t].sort((a, b) =>
      (STATUS_RANK[a.status] - STATUS_RANK[b.status]) ||
      ((b.value_tier === "starter") - (a.value_tier === "starter")) ||
      a.player.localeCompare(b.player)
    );
    total += players.length;
    const block = el("div", "inj-team");
    const counts = {};
    for (const p of players) counts[p.status] = (counts[p.status] || 0) + 1;
    const countStr = Object.entries(counts).map(([s, n]) => `${n} ${STATUS_SHORT[s]}`).join(" · ");
    const lines = players.slice(0, 12).map((p) => {
      const cls = p.status === "Out" ? "inj-status out" : "inj-status";
      const name = p.value_tier === "starter" ? `<b>${esc(p.player)}</b>` : esc(p.player);
      return `<span class="${cls}">${STATUS_SHORT[p.status]}</span> ${name} (${esc(p.position)})`;
    });
    if (players.length > 12) lines.push(`+ ${players.length - 12} more questionable`);
    block.innerHTML = `<h4>${esc(t)} <span class="count">${countStr}</span></h4><p class="inj-line">${lines.join(" · ")}</p>`;
    grid.appendChild(block);
  }
  $("#inj-dek").textContent = `${total} players not Active across ${teams.length} teams. Starters in bold. Q means Sunday decides.`;
}

/* ── The Toy ─────────────────────────────────────────────── */

const NFL_COORDS = {
  ARI:[33.5276,-112.2626], ATL:[33.7553,-84.4006], BAL:[39.2780,-76.6227],
  BUF:[42.7738,-78.7870], CAR:[35.2258,-80.8528], CHI:[41.8623,-87.6167],
  CIN:[39.0954,-84.5160], CLE:[41.5061,-81.6995], DAL:[32.7473,-97.0945],
  DEN:[39.7439,-105.0201], DET:[42.3400,-83.0456], GB:[44.5013,-88.0622],
  HOU:[29.6847,-95.4107], IND:[39.7601,-86.1639], JAX:[30.3240,-81.6373],
  KC:[39.0489,-94.4839], LAC:[33.9534,-118.3392], LAR:[33.9534,-118.3392],
  LV:[36.0909,-115.1833], MIA:[25.9580,-80.2389], MIN:[44.9738,-93.2581],
  NE:[42.0909,-71.2643], NO:[29.9509,-90.0814], NYG:[40.8128,-74.0742],
  NYJ:[40.8128,-74.0742], PHI:[39.9008,-75.1675], PIT:[40.4468,-80.0158],
  SEA:[47.5952,-122.3316], SF:[37.4030,-121.9700], TB:[27.9759,-82.5033],
  TEN:[36.1665,-86.7713], WAS:[38.9076,-76.8645],
};
const HFA = 65;
const WEIGHTS = { logistic: 0.30, xgboost: 0.25, elo: 0.20, pyth: 0.15, eff: 0.10 };

function haversine(a, b) {
  const R = 3958.8, rad = Math.PI / 180;
  const dLat = (b[0] - a[0]) * rad, dLon = (b[1] - a[1]) * rad;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a[0] * rad) * Math.cos(b[0] * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}
const eloProb = (diff) => 1 / (1 + Math.pow(10, -diff / 400));

function setupToy(lb) {
  const teams = (lb?.teams || []).slice().sort((a, b) => a.team.localeCompare(b.team));
  for (const sel of [$("#toy-home"), $("#toy-away")]) {
    sel.innerHTML = teams.map((t) => `<option value="${esc(t.team)}">${esc(t.team_name)}</option>`).join("");
  }
  $("#toy-home").value = "MIN";
  $("#toy-away").value = "CHI";

  $("#toy-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const home = $("#toy-home").value, away = $("#toy-away").value;
    if (home === away) return;
    const neutral = $("#toy-neutral").checked;
    const restH = parseInt($("#toy-rest-home").value || "7", 10);
    const restA = parseInt($("#toy-rest-away").value || "7", 10);
    const hd = teams.find((t) => t.team === home), ad = teams.find((t) => t.team === away);

    const hfa = neutral ? 0 : HFA;
    const homeElo = hd?.elo ?? 1500, awayElo = ad?.elo ?? 1500;
    const eP = eloProb(homeElo + hfa - awayElo);
    const pythDiff = ((hd?.pyth ?? 0.5) - (ad?.pyth ?? 0.5)) * 400 + hfa;
    const pP = eloProb(pythDiff);
    const effDiff = ((hd?.net_eff ?? 0) - (ad?.net_eff ?? 0)) * 200 + hfa;
    const fP = eloProb(effDiff);
    const restAdj = (restH - restA) * 1.5;
    const lP = eloProb(homeElo + hfa + restAdj - awayElo);

    const xgbW = WEIGHTS.xgboost, rem = WEIGHTS.logistic + WEIGHTS.elo + WEIGHTS.pyth + WEIGHTS.eff;
    const w = {
      logistic: WEIGHTS.logistic + xgbW * (WEIGHTS.logistic / rem),
      elo: WEIGHTS.elo + xgbW * (WEIGHTS.elo / rem),
      pyth: WEIGHTS.pyth + xgbW * (WEIGHTS.pyth / rem),
      eff: WEIGHTS.eff + xgbW * (WEIGHTS.eff / rem),
    };
    const ens = lP * w.logistic + eP * w.elo + pP * w.pyth + fP * w.eff;

    const winner = ens >= 0.5 ? home : away;
    const winnerName = ens >= 0.5 ? hd.team_name : ad.team_name;
    const wp = Math.max(ens, 1 - ens);
    const conf = wp > 0.7 ? "a strong favorite" : wp > 0.6 ? "a clear favorite" : "barely favored";
    const miles = NFL_COORDS[home] && NFL_COORDS[away] ? Math.round(haversine(NFL_COORDS[away], NFL_COORDS[home])) : 0;

    const res = $("#toy-result");
    res.hidden = false;
    res.innerHTML =
      `<p class="verdict"><span class="accent">${esc(winner)}</span> ${pct1(wp)} · ${esc(winnerName)}, ${conf}.</p>` +
      `<p class="why">Elo ${Math.round(homeElo)} vs ${Math.round(awayElo)}${neutral ? ", neutral site, no home field" : `, home field worth ${HFA} Elo`}. Rest ${restH} vs ${restA} days. ${miles > 500 ? `${esc(away)} travels ${fmtInt(miles)} miles.` : "Short trip."}</p>` +
      `<p class="sub-models">ELO ${pct0(eP)} · LOG ${pct0(lP)} · PYTH ${pct0(pP)} · EFF ${pct0(fP)} · blend of four, XGB weight split among them. Same arithmetic the nightly pipeline runs.</p>`;
    res.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
}

/* ── Boot ────────────────────────────────────────────────── */

(async function boot() {
  const [pred, lb, metrics, inj] = await Promise.all([
    getJSON("./data/nfl_predictions.json").then((d) => d || getJSON("./data/predictions.json")),
    getJSON("./data/nfl_leaderboard.json").then((d) => d || getJSON("./data/leaderboard.json")),
    getJSON("./data/model_metrics.json"),
    getJSON("./data/nfl_injuries.json"),
  ]);
  if (!pred) {
    $("#lede").textContent = "The data robot has not published tonight's file. Try again shortly.";
    return;
  }
  state.predictions = pred;
  state.leaderboard = lb;
  state.metrics = metrics;
  state.injuries = inj?.injuries || [];
  state.week = pred.week ?? Math.min(...(pred.games || []).map((g) => g.week));

  renderMasthead(pred);
  renderLede(pred);
  renderWeekLine(pred);
  renderGames(pred);
  renderLedger(metrics);
  renderRankings(lb);
  renderInjuries(state.injuries);
  setupToy(lb);
})();
