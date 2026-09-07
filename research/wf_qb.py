"""538-style QB adjustment walk-forward. Fair: only pre-game info used."""
import math
import numpy as np
import pandas as pd
import wf_elo as W

def load_qb_stats():
    frames = []
    cols = ["player_id","player_display_name","season","week","season_type","game_id","team","opponent_team",
            "completions","attempts","passing_yards","passing_tds","passing_interceptions","sacks_suffered",
            "carries","rushing_yards","rushing_tds"]
    for y in range(2015, 2026):
        try:
            f = pd.read_csv(f"/tmp/nflqb/stats_player_week_{y}.csv", usecols=lambda c: c in cols, low_memory=False)
            frames.append(f)
        except Exception as e:
            print(f"skip {y}: {e}")
    qb = pd.concat(frames)
    qb = qb[qb["attempts"].fillna(0) + qb["carries"].fillna(0) > 0]
    # game VALUE per player-game (538 formula)
    qb["value"] = (-2.2*qb["attempts"].fillna(0) + 3.7*qb["completions"].fillna(0)
                   + qb["passing_yards"].fillna(0)/5 + 11.3*qb["passing_tds"].fillna(0)
                   - 14.1*qb["passing_interceptions"].fillna(0) - 8*qb["sacks_suffered"].fillna(0)
                   - 1.1*qb["carries"].fillna(0) + 0.6*qb["rushing_yards"].fillna(0)
                   + 15.9*qb["rushing_tds"].fillna(0))
    return qb

def load_draft():
    d = pd.read_csv("/tmp/nflqb/draft_picks.csv", usecols=["season","round","pick","gsis_id"])
    return {r.gsis_id: (int(r.season), int(r.round), int(r.pick)) for r in d.itertuples() if isinstance(r.gsis_id, str)}

DRAFT_INIT_ELO = {1: 113, 2: 40, 3: 25}  # round -> elo pts; later/undrafted 0

def prep(qb):
    """Per (game_id, team): starter (most attempts) id + total team QB value; per (game_id,team): opp team."""
    g = qb.sort_values("attempts", ascending=False).groupby(["game_id","team"])
    starter = g.first()[["player_id","value"]].reset_index()
    teamval = qb.groupby(["game_id","team"])["value"].sum().reset_index().rename(columns={"value":"team_value"})
    opp = qb.groupby(["game_id","team"])["opponent_team"].first().reset_index()
    return starter.merge(teamval, on=["game_id","team"]).merge(opp, on=["game_id","team"])

def run_with_qb(df, starter_df, draft, hfa=48.0, era_hfa=False, k_base=20.0, regress_pct=0.33,
                qb_mult=3.3, since=2017, qb_on=True, travel_per_1000=0.0, _hfa_hist=None):
    skey = { (r.game_id, r.team): r for r in starter_df.itertuples() }
    # rebuild game ids to join: our games df lacks game_id; re-derive from raw
    raw = pd.read_csv(W.GAMES, usecols=["game_id","gameday","home_team","away_team","home_score","away_score","season"])
    raw = raw.dropna(subset=["home_score","away_score"])
    gid = { (str(r.gameday), W.norm(r.home_team), W.norm(r.away_team)): r.game_id for r in raw.itertuples() }

    elo = {}
    qb_rating = {}   # player_id -> rolling VALUE
    qb_starts = {}
    team_val = {}    # team -> rolling team VALUE (0.95/0.05)
    def_allowed = {} # team -> rolling VALUE allowed
    league_allowed = []
    cur_season = None
    recs = []
    for g in df.itertuples():
        if cur_season != g.season:
            # season rollover: regress elo; revert qb ratings (10-100 starts -> 25% toward league avg)
            avg_qb = np.mean(list(qb_rating.values())) if qb_rating else 0.0
            for p in list(qb_rating):
                st = qb_starts.get(p, 0)
                if 10 <= st <= 100:
                    qb_rating[p] = qb_rating[p]*0.75 + avg_qb*0.25
            for t in list(elo):
                elo[t] = elo[t]*(1-regress_pct) + 1500.0*regress_pct
            cur_season = g.season
        t1, t2 = g.team1, g.team2
        e1 = elo.get(t1, 1500.0); e2 = elo.get(t2, 1500.0)
        game_id = gid.get((str(g.date.date()), t1, t2))
        adj1 = adj2 = 0.0
        s1 = s2 = None
        trav = travel_per_1000 * (getattr(g, "travel", 0.0) or 0.0) / 1000.0
        if qb_on and game_id is not None:
            r1 = skey.get((game_id, t1)); r2 = skey.get((game_id, t2))
            if r1 is not None and r2 is not None:
                s1, s2 = r1, r2
                tv1 = team_val.get(t1, 0.0); tv2 = team_val.get(t2, 0.0)
                q1 = qb_rating.get(r1.player_id, None); q2 = qb_rating.get(r2.player_id, None)
                if q1 is None:
                    q1 = _draft_init(draft.get(r1.player_id), g.season, avg_qb)
                if q2 is None:
                    q2 = _draft_init(draft.get(r2.player_id), g.season, avg_qb)
                adj1 = qb_mult * (q1 - tv1); adj2 = qb_mult * (q2 - tv2)
        if era_hfa and _hfa_hist is not None:
            vals = [_hfa_hist[s_] for s_ in range(g.season-10, g.season) if s_ in _hfa_hist]
            hfa_now = 400.0*math.log10(np.mean(vals)/(1-np.mean(vals))) if len(vals) >= 3 else hfa
        else:
            hfa_now = hfa
        hfa_adj = 0.0 if g.neutral else hfa_now
        diff = (e1 + adj1 + hfa_adj) - (e2 + adj2 - trav)
        prob = 1.0/(1.0+10**(-diff/400.0))
        if g.season >= since:
            actual = 1.0 if g.score1 > g.score2 else (0.5 if g.score1 == g.score2 else 0.0)
            recs.append({"season": g.season, "prob": prob, "vegas": g.vegas, "actual": actual})
        # --- updates (post-game) ---
        adj_e1 = e1 + adj1 + hfa_adj
        adj_e2 = e2 + adj2 - trav
        exp1 = 1.0/(1.0+10**(-(adj_e1-adj_e2)/400.0))
        a1 = 1.0 if g.score1 > g.score2 else (0.5 if g.score1 == g.score2 else 0.0)
        pdiff = abs(g.score1-g.score2)
        if pdiff > 0:
            winner_diff = (adj_e1-adj_e2) if a1 == 1.0 else (adj_e2-adj_e1)
            mov = math.log(pdiff+1) * (2.2/(winner_diff*0.001+2.2))
        else:
            mov = 1.0
        elo[t1] = e1 + k_base*mov*(a1-exp1)
        elo[t2] = e2 + k_base*mov*((1.0-a1)-(1.0-exp1))
        if qb_on and s1 is not None:
            lg_avg = np.mean(league_allowed[-500:]) if league_allowed else 0.0
            for t, srow, opp in ((t1,s1,t2),(t2,s2,t1)):
                opp_allow = def_allowed.get(opp, lg_avg)
                val_adj = srow.team_value - (opp_allow - lg_avg)
                # team rolling (0.05) uses team total; qb rolling (0.1) uses starter's own game value
                team_val[t] = 0.95*team_val.get(t, 0.0) + 0.05*val_adj
                def_allowed[t] = 0.9*def_allowed.get(t, 0.0) + 0.1*srow.team_value
                league_allowed.append(srow.team_value)
                pid = srow.player_id
                qv = srow.value - (opp_allow - lg_avg)
                qb_rating[pid] = 0.9*qb_rating.get(pid, _draft_init(draft.get(pid), g.season, avg_qb)) + 0.1*qv
                qb_starts[pid] = qb_starts.get(pid, 0) + 1
    return pd.DataFrame(recs)

def _draft_init(dinfo, season, avg_qb):
    if not dinfo: return 0.0
    dseason, rnd, pick = dinfo
    if season > dseason + 3: return 0.0  # old undrafted-ish unknown
    elo = DRAFT_INIT_ELO.get(rnd, 0)
    return elo/3.3

if __name__ == "__main__":
    df = W.load()
    # add game_id join via raw
    raw = pd.read_csv(W.GAMES, usecols=["game_id","gameday","home_team","away_team","home_score","away_score"])
    raw = raw.dropna(subset=["home_score","away_score"])
    gid_map = { (str(r.gameday), W.norm(r.home_team), W.norm(r.away_team)): r.game_id for r in raw.itertuples() }
    df["game_id"] = [gid_map.get((str(d.date()), h, a)) for d,h,a in zip(df["date"], df["team1"], df["team2"])]
    print("games with id:", df["game_id"].notna().sum(), "/", len(df))
    qb = load_qb_stats()
    starter = prep(qb)
    draft = load_draft()
    print("starter rows:", len(starter))
    # pass game_id through
    df2 = df.copy()
    recs = run_with_qb(df2, starter, draft)
    s, n = W.score(recs, "qb")
    m, v = s['prob'], s['vegas']
    print(f"QB-ADJ  acc={m[0]:.4f} ll={m[1]:.4f} br={m[2]:.4f} | vegas acc={v[0]:.4f} N={n}")
