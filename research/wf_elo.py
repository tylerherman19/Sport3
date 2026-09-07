"""
Walk-forward ELO lab for Sport3 NFL.
Fully fair: every game predicted BEFORE its result updates any rating.
Metrics: accuracy / log loss / brier, per season 2016-2025 and pooled, vs Vegas devig closing moneyline.
"""
import sys, math, json
import numpy as np
import pandas as pd

GAMES = "/tmp/Sport3/data/nflverse_cache/games.csv"
SINCE = 2016

ABBR = {"WSH":"WAS","JAC":"JAX","LVR":"LV","LA":"LAR","OAK":"LV","SD":"LAC","STL":"LAR","ARZ":"ARI"}
def norm(t): return ABBR.get(t, t)

def american_to_prob(o):
    if o is None or (isinstance(o,float) and math.isnan(o)): return None
    o = float(o)
    return 100.0/(o+100.0) if o > 0 else -o/(-o+100.0)

def devig(a, b):
    if a is None or b is None: return None
    return a/(a+b)

def load():
    raw = pd.read_csv(GAMES)
    raw = raw.dropna(subset=["home_score","away_score"]).copy()
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["gameday"]),
        "season": raw["season"].astype(int),
        "playoff": (raw["game_type"] != "REG").astype(int),
        "team1": raw["home_team"].map(norm),
        "team2": raw["away_team"].map(norm),
        "score1": raw["home_score"].astype(float),
        "score2": raw["away_score"].astype(float),
        "neutral": (raw["location"] == "Neutral").astype(int),
        "vegas": [devig(american_to_prob(h), american_to_prob(a)) for h,a in zip(raw["home_moneyline"], raw["away_moneyline"])],
        "week": raw["week"].astype(int),
    }).sort_values(["date"]).reset_index(drop=True)
    return df

def mov_multiplier(point_diff, elo_diff):
    return math.log(abs(point_diff)+1) * (2.2/(elo_diff*0.001+2.2))

def run(df, hfa=65.0, k_base=20.0, k_decay=True, regress_pct=0.33, regress_target=1500.0,
        playoff_mult=1.0, bye_rest=0.0, initial=1500.0):
    elo = {}
    last_date = {}
    cur_season = None
    team_games = {}
    recs = []
    for g in df.itertuples():
        if cur_season != g.season:
            cur_season = g.season
            for t in list(elo):
                elo[t] = elo[t]*(1-regress_pct) + regress_target*regress_pct
            team_games = {}
        t1, t2 = g.team1, g.team2
        e1 = elo.get(t1, initial); e2 = elo.get(t2, initial)
        # rest
        r1 = (g.date - last_date[t1]).days if t1 in last_date else 7
        r2 = (g.date - last_date[t2]).days if t2 in last_date else 7
        hfa_adj = 0.0 if g.neutral else hfa
        # bye rest bonus (coming off >=12 days rest)
        b1 = bye_rest if r1 >= 12 else 0.0
        b2 = bye_rest if r2 >= 12 else 0.0
        diff = (e1 + hfa_adj + b1) - (e2 + b2)
        if g.playoff: diff *= playoff_mult
        prob = 1.0/(1.0+10**(-diff/400.0))
        if g.season >= SINCE:
            actual = 1.0 if g.score1 > g.score2 else (0.5 if g.score1 == g.score2 else 0.0)
            recs.append({"season": g.season, "prob": prob, "vegas": g.vegas, "actual": actual})
        # update
        adj_e1 = e1 + hfa_adj + b1
        exp1 = 1.0/(1.0+10**(-(adj_e1-(e2+b2))/400.0))
        a1 = 1.0 if g.score1 > g.score2 else (0.5 if g.score1 == g.score2 else 0.0)
        pdiff = abs(g.score1-g.score2)
        mov = mov_multiplier(pdiff, abs(adj_e1-(e2+b2))) if pdiff > 0 else 1.0
        if k_decay:
            prog = min(1.0, team_games.get(t1,0)/17)
            k = k_base*(1.0-0.5*prog)
        else:
            k = k_base
        elo[t1] = e1 + k*mov*(a1-exp1)
        elo[t2] = e2 + k*mov*((1.0-a1)-(1.0-exp1))
        team_games[t1] = team_games.get(t1,0)+1
        team_games[t2] = team_games.get(t2,0)+1
        last_date[t1] = g.date; last_date[t2] = g.date
    return pd.DataFrame(recs)

def score(recs, label):
    d = recs.dropna(subset=["vegas"])
    d = d[d["actual"] != 0.5]  # drop ties
    out = {}
    for col in ["prob","vegas"]:
        p = np.clip(d[col].values, 1e-6, 1-1e-6); y = d["actual"].values
        acc = float(np.mean((p>=0.5)==y))
        ll = float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
        br = float(np.mean((p-y)**2))
        out[col] = (acc, ll, br)
    return out, len(d)

if __name__ == "__main__":
    df = load()
    print(f"{len(df)} games loaded, scoring {SINCE}-2025")
    base, n = score(run(df), "base")
    print(f"N={n}  MODEL acc={base['prob'][0]:.3f} ll={base['prob'][1]:.4f} br={base['prob'][2]:.4f} | VEGAS acc={base['vegas'][0]:.3f} ll={base['vegas'][1]:.4f} br={base['vegas'][2]:.4f}")
