"""
nba_data.py — NBA feature engineering pipeline.
All nba_api / stats.nba.com calls replaced with:
  - cdn.nba.com/stats/leaguegamelog (LeagueGameLog)
  - cdn.nba.com/stats/leaguedashteamstats (LeagueDashTeamStats)
  - cdn.nba.com/static/json/staticData/standings.json (standings)
  - ESPN scoreboard day-by-day fallback (site.api.espn.com)
  - ESPN teams/{id}/athletes for roster data

Outputs:
  data/nba_features.json
  data/nba_results.json
  data/nba_fetch_log.txt
"""

import os
import sys
import json
import math
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR  = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
FETCH_LOG = DATA_DIR / "nba_fetch_log.txt"

ESPN_NBA_BASE  = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CDN_NBA_STATS  = "https://cdn.nba.com/stats"
CDN_NBA_STATIC = "https://cdn.nba.com/static/json"

_CDN_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": "Mozilla/5.0 (compatible; Sport3DataBot/2.0)",
}

ARENA_INFO = {
    "ATL":{"name":"State Farm Arena","city":"Atlanta","lat":33.7573,"lon":-84.3963,"capacity":18118,"utc_offset":-5},
    "BOS":{"name":"TD Garden","city":"Boston","lat":42.3662,"lon":-71.0621,"capacity":19156,"utc_offset":-5},
    "BKN":{"name":"Barclays Center","city":"Brooklyn","lat":40.6826,"lon":-73.9754,"capacity":17732,"utc_offset":-5},
    "CHA":{"name":"Spectrum Center","city":"Charlotte","lat":35.2251,"lon":-80.8392,"capacity":19077,"utc_offset":-5},
    "CHI":{"name":"United Center","city":"Chicago","lat":41.8807,"lon":-87.6742,"capacity":20917,"utc_offset":-6},
    "CLE":{"name":"Rocket Mortgage FieldHouse","city":"Cleveland","lat":41.4964,"lon":-81.6882,"capacity":19432,"utc_offset":-5},
    "DAL":{"name":"American Airlines Center","city":"Dallas","lat":32.7905,"lon":-96.8103,"capacity":19200,"utc_offset":-6},
    "DEN":{"name":"Ball Arena","city":"Denver","lat":39.7487,"lon":-105.0077,"capacity":19520,"utc_offset":-7},
    "DET":{"name":"Little Caesars Arena","city":"Detroit","lat":42.3410,"lon":-83.0554,"capacity":20332,"utc_offset":-5},
    "GSW":{"name":"Chase Center","city":"San Francisco","lat":37.7680,"lon":-122.3877,"capacity":18064,"utc_offset":-8},
    "HOU":{"name":"Toyota Center","city":"Houston","lat":29.7508,"lon":-95.3621,"capacity":18055,"utc_offset":-6},
    "IND":{"name":"Gainbridge Fieldhouse","city":"Indianapolis","lat":39.7640,"lon":-86.1555,"capacity":17923,"utc_offset":-5},
    "LAC":{"name":"Intuit Dome","city":"Inglewood","lat":33.9578,"lon":-118.3417,"capacity":18000,"utc_offset":-8},
    "LAL":{"name":"Crypto.com Arena","city":"Los Angeles","lat":34.0430,"lon":-118.2673,"capacity":19068,"utc_offset":-8},
    "MEM":{"name":"FedExForum","city":"Memphis","lat":35.1383,"lon":-90.0505,"capacity":17794,"utc_offset":-6},
    "MIA":{"name":"Kaseya Center","city":"Miami","lat":25.7814,"lon":-80.1870,"capacity":19600,"utc_offset":-5},
    "MIL":{"name":"Fiserv Forum","city":"Milwaukee","lat":43.0450,"lon":-87.9170,"capacity":17341,"utc_offset":-6},
    "MIN":{"name":"Target Center","city":"Minneapolis","lat":44.9795,"lon":-93.2762,"capacity":18978,"utc_offset":-6},
    "NOP":{"name":"Smoothie King Center","city":"New Orleans","lat":29.9490,"lon":-90.0821,"capacity":17791,"utc_offset":-6},
    "NYK":{"name":"Madison Square Garden","city":"New York","lat":40.7505,"lon":-73.9934,"capacity":19812,"utc_offset":-5},
    "OKC":{"name":"Paycom Center","city":"Oklahoma City","lat":35.4634,"lon":-97.5151,"capacity":18203,"utc_offset":-6},
    "ORL":{"name":"Kia Center","city":"Orlando","lat":28.5392,"lon":-81.3839,"capacity":18846,"utc_offset":-5},
    "PHI":{"name":"Wells Fargo Center","city":"Philadelphia","lat":39.9012,"lon":-75.1720,"capacity":20478,"utc_offset":-5},
    "PHX":{"name":"Footprint Center","city":"Phoenix","lat":33.4457,"lon":-112.0712,"capacity":17072,"utc_offset":-7},
    "POR":{"name":"Moda Center","city":"Portland","lat":45.5316,"lon":-122.6668,"capacity":19393,"utc_offset":-8},
    "SAC":{"name":"Golden 1 Center","city":"Sacramento","lat":38.5490,"lon":-121.5002,"capacity":17608,"utc_offset":-8},
    "SAS":{"name":"Frost Bank Center","city":"San Antonio","lat":29.4270,"lon":-98.4375,"capacity":18418,"utc_offset":-6},
    "TOR":{"name":"Scotiabank Arena","city":"Toronto","lat":43.6435,"lon":-79.3791,"capacity":19800,"utc_offset":-5},
    "UTA":{"name":"Delta Center","city":"Salt Lake City","lat":40.7683,"lon":-111.9011,"capacity":18306,"utc_offset":-7},
    "WAS":{"name":"Capital One Arena","city":"Washington","lat":38.8981,"lon":-77.0209,"capacity":20356,"utc_offset":-5},
}

NBA_DIVISIONS = {
    "Atlantic":["BOS","BKN","NYK","PHI","TOR"],"Central":["CHI","CLE","DET","IND","MIL"],
    "Southeast":["ATL","CHA","MIA","ORL","WAS"],"Northwest":["DEN","MIN","OKC","POR","UTA"],
    "Pacific":["GSW","LAC","LAL","PHX","SAC"],"Southwest":["DAL","HOU","MEM","NOP","SAS"],
}
NBA_CONFERENCES = {
    "East":["BOS","BKN","NYK","PHI","TOR","CHI","CLE","DET","IND","MIL","ATL","CHA","MIA","ORL","WAS"],
    "West":["DEN","MIN","OKC","POR","UTA","GSW","LAC","LAL","PHX","SAC","DAL","HOU","MEM","NOP","SAS"],
}
TEAM_DIVISION   = {t: div  for div,  ts in NBA_DIVISIONS.items()   for t in ts}
TEAM_CONFERENCE = {t: conf for conf, ts in NBA_CONFERENCES.items() for t in ts}
BUBBLE_START = date(2020, 3, 11); BUBBLE_END = date(2020, 10, 12)
ESPN_TO_NBA = {"GS":"GSW","NY":"NYK","NO":"NOP","SA":"SAS","WSH":"WAS","UTAH":"UTA"}
NBA_TEAMS = list(ARENA_INFO.keys())


def log_fetch_error(msg):
    with open(FETCH_LOG, "a") as f: f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    log.warning(msg)


def safe_get(url, params=None, headers=None, timeout=30):
    try:
        r = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
        r.raise_for_status(); return r.json()
    except Exception as e:
        log_fetch_error(f"GET {url}: {e}"); return None


def abbrev_norm(a): return ESPN_TO_NBA.get(a.upper().strip(), a.upper().strip())
def haversine_miles(lat1,lon1,lat2,lon2):
    R=3959.0; phi1=math.radians(lat1); phi2=math.radians(lat2)
    dphi=math.radians(lat2-lat1); dlambda=math.radians(lon2-lon1)
    x=math.sin(dphi/2)**2+math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R*2*math.atan2(math.sqrt(x),math.sqrt(1-x))
def nba_season_str(yr): return f"{yr-1}-{str(yr)[2:]}"
def is_bubble(gd):
    if isinstance(gd,str):
        try: gd=date.fromisoformat(gd[:10])
        except ValueError: return 0
    return 1 if BUBBLE_START<=gd<=BUBBLE_END else 0
def is_division_rival(a,b): return 1 if TEAM_DIVISION.get(a)==TEAM_DIVISION.get(b) else 0
def is_interconference(a,b): return 1 if TEAM_CONFERENCE.get(a)!=TEAM_CONFERENCE.get(b) else 0
def parse_min(v):
    if v is None: return 240.0
    if isinstance(v,(int,float)): return float(v)
    s=str(v)
    if ":" in s: p=s.split(":"); return float(p[0])+float(p[1])/60.0
    try: return float(s)
    except: return 240.0


# ---- cdn.nba.com endpoints (replace stats.nba.com) ----

def fetch_cdn_league_game_log(season_year, season_type="Regular Season"):
    """cdn.nba.com/stats/leaguegamelog replaces nba_api LeagueGameLog."""
    url    = f"{CDN_NBA_STATS}/leaguegamelog"
    params = {"Counter":"1000","Direction":"DESC","LeagueID":"00",
              "PlayerOrTeam":"T","Season":nba_season_str(season_year),
              "SeasonType":season_type,"Sorter":"DATE"}
    data = safe_get(url, params=params, headers=_CDN_HEADERS, timeout=30)
    if not data:
        log.warning(f"cdn game log unavailable for {season_year} {season_type} — ESPN fallback")
        return _espn_game_log_fallback(season_year)
    try:
        rs = data["resultSets"][0]
        hl = rs["headers"]; rows = rs["rowSet"]
        df = pd.DataFrame(rows, columns=hl)
        if df.empty: return pd.DataFrame()
        if "TEAM_ABBREVIATION" in df.columns:
            df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].apply(abbrev_norm)
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df
    except Exception as e:
        log_fetch_error(f"cdn game log parse {season_year}: {e}")
        return _espn_game_log_fallback(season_year)


def _espn_game_log_fallback(season_year):
    """ESPN scoreboard day-by-day fallback for game logs."""
    season_start = date(season_year-1,10,1); season_end = min(date.today(),date(season_year,6,30))
    if season_start > date.today(): return pd.DataFrame()
    is_cur = (season_year >= datetime.now().year); stride = 1 if is_cur else 7
    date_list = []
    cur = season_start
    while cur <= season_end:
        date_list.append(cur.strftime("%Y%m%d")); cur += timedelta(days=stride)
    def _fd(ds): return ds, safe_get(f"{ESPN_NBA_BASE}/scoreboard?dates={ds}&limit=20")
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_fd,ds):ds for ds in date_list}
        for fut in as_completed(futs):
            ds, data = fut.result()
            if not data: continue
            for event in data.get("events",[]):
                try:
                    if event.get("status",{}).get("type",{}).get("name","")!="STATUS_FINAL": continue
                    comp=event["competitions"][0]; comps=comp["competitors"]
                    home=next((c for c in comps if c["homeAway"]=="home"),None)
                    away=next((c for c in comps if c["homeAway"]=="away"),None)
                    if not home or not away: continue
                    hs=int(home.get("score",0) or 0); as_=int(away.get("score",0) or 0)
                    if hs==0 and as_==0: continue
                    gid=event.get("id",""); gdt=event.get("date","")[:10]
                    ha=abbrev_norm(home["team"]["abbreviation"])
                    aa=abbrev_norm(away["team"]["abbreviation"])
                    rows.append({"GAME_ID":gid,"GAME_DATE":gdt,"TEAM_ABBREVIATION":ha,
                                 "MATCHUP":f"{ha} vs. {aa}","WL":"W" if hs>as_ else "L",
                                 "PTS":hs,"FGA":None,"FGM":None,"FTA":None,"OREB":None,
                                 "DREB":None,"TOV":None,"MIN":240,"SEASON_YEAR":season_year})
                    rows.append({"GAME_ID":gid,"GAME_DATE":gdt,"TEAM_ABBREVIATION":aa,
                                 "MATCHUP":f"{aa} @ {ha}","WL":"W" if as_>hs else "L",
                                 "PTS":as_,"FGA":None,"FGM":None,"FTA":None,"OREB":None,
                                 "DREB":None,"TOV":None,"MIN":240,"SEASON_YEAR":season_year})
                except (KeyError,IndexError,TypeError): pass
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df.sort_values("GAME_DATE",inplace=True); return df


def fetch_cdn_team_stats(season_year, last_n_games=0, measure_type="Advanced"):
    """cdn.nba.com/stats/leaguedashteamstats replaces nba_api LeagueDashTeamStats."""
    url    = f"{CDN_NBA_STATS}/leaguedashteamstats"
    params = {"MeasureType":measure_type,"PerMode":"Per100Possessions",
              "Season":nba_season_str(season_year),"SeasonType":"Regular Season",
              "LeagueID":"00","LastNGames":str(last_n_games)}
    data = safe_get(url, params=params, headers=_CDN_HEADERS, timeout=30)
    if not data: return pd.DataFrame()
    try:
        rs=data["resultSets"][0]; df=pd.DataFrame(rs["rowSet"],columns=rs["headers"])
        if "TEAM_ABBREVIATION" in df.columns:
            df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].apply(abbrev_norm)
        return df
    except Exception as e:
        log_fetch_error(f"cdn team stats parse {season_year}: {e}"); return pd.DataFrame()


def fetch_cdn_standings(season_year):
    """cdn.nba.com static standings replaces nba_api LeagueStandingsV3."""
    data = safe_get(f"{CDN_NBA_STATIC}/staticData/standings.json", headers=_CDN_HEADERS)
    if not data: return pd.DataFrame()
    rows = []
    try:
        raw = data.get("standings", [])
        if raw and isinstance(raw[0], dict):
            for row in raw:
                abbrev = abbrev_norm(row.get("teamAbbreviation",""))
                rows.append({"TeamAbbreviation":abbrev,
                             "WINS":int(row.get("wins",0)),"LOSSES":int(row.get("losses",0)),
                             "W":int(row.get("wins",0)),"L":int(row.get("losses",0)),
                             "WinPct":float(row.get("winPct",row.get("winPercentage",0)))})
    except Exception as e:
        log_fetch_error(f"cdn standings parse: {e}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_roster_espn(team_abbrev):
    """ESPN teams/{id}/athletes replaces nba_api CommonTeamRoster."""
    td = safe_get(f"{ESPN_NBA_BASE}/teams")
    if not td: return pd.DataFrame()
    tlist = td.get("sports",[{}])[0].get("leagues",[{}])[0].get("teams",[])
    team_id = None
    for te in tlist:
        if abbrev_norm(te.get("team",{}).get("abbreviation","")) == team_abbrev.upper():
            team_id = te.get("team",{}).get("id",""); break
    if not team_id: return pd.DataFrame()
    data = safe_get(f"{ESPN_NBA_BASE}/teams/{team_id}/athletes")
    if not data: return pd.DataFrame()
    rows = [{"PLAYER_ID":a.get("id",""),"PLAYER":a.get("displayName","")} for a in data.get("athletes",[])]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---- Game pairing and advanced stats ----

def pair_games(games_df):
    if games_df.empty or "MATCHUP" not in games_df.columns: return pd.DataFrame()
    df = games_df.copy()
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs\\.")
    hd = df[df["IS_HOME"]].copy().add_prefix("home_")
    ad = df[~df["IS_HOME"]].copy().add_prefix("away_")
    return pd.merge(hd.rename(columns={"home_GAME_ID":"GAME_ID"}),
                    ad.rename(columns={"away_GAME_ID":"GAME_ID"}), on="GAME_ID", how="inner")


def compute_possessions(fga,fta,oreb,tov):
    return float(fga or 0)+0.44*float(fta or 0)-float(oreb or 0)+float(tov or 0)


def compute_advanced(tr,or_):
    def s(x):
        try: return float(x) if x is not None else 0.0
        except: return 0.0
    pts=s(tr.get("PTS",0)); opp=s(or_.get("PTS",0))
    fga=max(s(tr.get("FGA",1)),1.0); fta=s(tr.get("FTA",0))
    oreb=s(tr.get("OREB",0)); tov=s(tr.get("TOV",0))
    ofga=max(s(or_.get("FGA",1)),1.0); ofta=s(or_.get("FTA",0))
    ooreb=s(or_.get("OREB",0)); otov=s(or_.get("TOV",0)); odreb=s(or_.get("DREB",0))
    mn=max(parse_min(tr.get("MIN",240)),1.0)
    tp=compute_possessions(fga,fta,oreb,tov); op=compute_possessions(ofga,ofta,ooreb,otov)
    ap=max((tp+op)/2,1.0)
    offr=(pts/ap)*100; defr=(opp/ap)*100; pace=(ap/mn)*48
    ts=pts/(2*(fga+0.44*fta)) if (fga+0.44*fta)>0 else 0.5
    td2=fga+0.44*fta+tov; tvr=tov/td2 if td2>0 else 0.12
    od=oreb+odreb; op2=oreb/od if od>0 else 0.25; ftr=fta/fga if fga>0 else 0.2
    return {"off_rating":round(offr,2),"def_rating":round(defr,2),"net_rating":round(offr-defr,2),
            "pace":round(pace,2),"ts_pct":round(ts,4),"tov_rate":round(tvr,4),
            "oreb_pct":round(op2,4),"ft_rate":round(ftr,4),"point_diff":int(pts-opp),
            "win":1 if pts>opp else 0}


# ---- Schedule, travel, rolling metrics ----

def compute_schedule_features(team_game_log, team, game_date):
    empty={"days_rest":7,"is_back_to_back":0,"games_last_7":0,"games_last_14":0,
           "schedule_density":0.0,"home_stand_length":0,"road_trip_length":0}
    if team_game_log.empty: return empty
    prior=team_game_log[(team_game_log["TEAM_ABBREVIATION"]==team)&
                        (team_game_log["GAME_DATE"].dt.date<game_date)].sort_values("GAME_DATE")
    if prior.empty: return empty
    last_date=prior.iloc[-1]["GAME_DATE"].date(); days_rest=(game_date-last_date).days
    tgt=pd.Timestamp(game_date)
    g7=int((prior["GAME_DATE"]>=tgt-pd.Timedelta(days=7)).sum())
    g14=int((prior["GAME_DATE"]>=tgt-pd.Timedelta(days=14)).sum())
    g10=int((prior["GAME_DATE"]>=tgt-pd.Timedelta(days=10)).sum())
    hs=rt=0; lih="vs." in str(prior.iloc[-1].get("MATCHUP",""))
    for _,row in prior.iloc[::-1].iterrows():
        ih="vs." in str(row.get("MATCHUP",""))
        if ih==lih: hs+=1 if lih else 0; rt+=0 if lih else 1
        else: break
    return {"days_rest":max(days_rest,0),"is_back_to_back":1 if days_rest==1 else 0,
            "games_last_7":g7,"games_last_14":g14,"schedule_density":round(g10/10.0,3),
            "home_stand_length":hs,"road_trip_length":rt}


def _last_arena(prior, team):
    if prior.empty: return team
    m=str(prior.iloc[-1].get("MATCHUP",""))
    if "@" in m: p=m.split("@"); return abbrev_norm(p[1].strip()) if len(p)>1 else team
    return team


def compute_travel_features(tgl, team, game_date, home_team):
    prior=tgl[(tgl["TEAM_ABBREVIATION"]==team)&(tgl["GAME_DATE"].dt.date<game_date)].sort_values("GAME_DATE")
    dest=ARENA_INFO.get(home_team,{}); prev=ARENA_INFO.get(_last_arena(prior,team),{})
    tm=tz=0
    if dest and prev:
        tm=haversine_miles(prev["lat"],prev["lon"],dest["lat"],dest["lon"])
        tz=dest["utc_offset"]-prev["utc_offset"]
    tgt=pd.Timestamp(game_date); recent=prior[prior["GAME_DATE"]>=tgt-pd.Timedelta(days=7)]
    cum=0.0; pl=_last_arena(prior,team)
    for _,row in recent.iterrows():
        m=str(row.get("MATCHUP","")); cl=abbrev_norm(m.split("@")[1].strip()) if "@" in m else team
        p=ARENA_INFO.get(pl,{}); c=ARENA_INFO.get(cl,{})
        if p and c: cum+=haversine_miles(p["lat"],p["lon"],c["lat"],c["lon"])
        pl=cl
    return {"travel_miles":round(tm,1),"timezone_change":int(tz),
            "travel_direction":1 if tz>0 else (-1 if tz<0 else 0),
            "cumulative_travel_miles_7d":round(cum,1)}


def compute_rolling_metrics(paired_df, team, game_date, n):
    empty={f"{k}_l{n}":None for k in ["off_rating","def_rating","net_rating","pace",
           "ts_pct","tov_rate","oreb_pct","ft_rate","point_diff","win_pct"]}
    if paired_df.empty: return empty
    dc="home_GAME_DATE" if "home_GAME_DATE" in paired_df.columns else None
    if not dc: return empty
    gts=pd.Timestamp(game_date)
    hm=paired_df.get("home_TEAM_ABBREVIATION",pd.Series(dtype=str))==team
    am=paired_df.get("away_TEAM_ABBREVIATION",pd.Series(dtype=str))==team
    sc=["PTS","FGA","FTA","OREB","DREB","TOV","MIN"]; ar=[]
    for _,row in paired_df[hm&(paired_df[dc]<gts)].iterrows():
        adv=compute_advanced({c:row.get(f"home_{c}",0) for c in sc},{c:row.get(f"away_{c}",0) for c in sc})
        adv["date"]=row[dc]; ar.append(adv)
    for _,row in paired_df[am&(paired_df[dc]<gts)].iterrows():
        adv=compute_advanced({c:row.get(f"away_{c}",0) for c in sc},{c:row.get(f"home_{c}",0) for c in sc})
        adv["date"]=row[dc]; ar.append(adv)
    if not ar: return empty
    w=pd.DataFrame(ar).sort_values("date").tail(n)
    def avg(col): v=w[col].dropna(); return round(float(v.mean()),4) if len(v)>0 else None
    return {f"off_rating_l{n}":avg("off_rating"),f"def_rating_l{n}":avg("def_rating"),
            f"net_rating_l{n}":avg("net_rating"),f"pace_l{n}":avg("pace"),
            f"ts_pct_l{n}":avg("ts_pct"),f"tov_rate_l{n}":avg("tov_rate"),
            f"oreb_pct_l{n}":avg("oreb_pct"),f"ft_rate_l{n}":avg("ft_rate"),
            f"point_diff_l{n}":avg("point_diff"),f"win_pct_l{n}":avg("win")}


def rolling_delta(hm, am, n):
    keys=["off_rating","def_rating","net_rating","pace","ts_pct","tov_rate","oreb_pct","ft_rate","point_diff","win_pct"]
    return {f"delta_{k}_l{n}":(round(hm.get(f"{k}_l{n}")-am.get(f"{k}_l{n}"),4)
             if hm.get(f"{k}_l{n}") is not None and am.get(f"{k}_l{n}") is not None else None)
            for k in keys}


def compute_h2h(home, away, all_games, season_year):
    ms=[g for g in all_games if (g["team1"]==home and g["team2"]==away) or
                                  (g["team1"]==away and g["team2"]==home)]
    if not ms: return {"h2h_win_pct_season":0.5,"h2h_win_pct_3yr":0.5,"h2h_avg_margin_5":0.0}
    def hw(g): return 1 if (g["team1"]==home and g["score1"]>g["score2"]) or (g["team2"]==home and g["score2"]>g["score1"]) else 0
    def mg(g): return (g["score1"]-g["score2"]) if g["team1"]==home else (g["score2"]-g["score1"])
    sg=[g for g in ms if g.get("season")==season_year]; r3=[g for g in ms if g.get("season",0)>=season_year-2]
    l5=sorted(ms,key=lambda g:g.get("date",""))[-5:]
    return {"h2h_win_pct_season":round(sum(hw(g) for g in sg)/len(sg),4) if sg else 0.5,
            "h2h_win_pct_3yr":round(sum(hw(g) for g in r3)/len(r3),4) if r3 else 0.5,
            "h2h_avg_margin_5":round(sum(mg(g) for g in l5)/len(l5),2) if l5 else 0.0}


def fetch_roster_availability(team_abbrev):
    try:
        rd=fetch_roster_espn(team_abbrev)
        if rd.empty: return {"top3_available":1.0,"roster_depth":1.0}
        return {"top3_available":1.0,"roster_depth":round(min(len(rd),15)/13.0,3)}
    except Exception as e:
        log_fetch_error(f"Roster availability {team_abbrev}: {e}")
        return {"top3_available":1.0,"roster_depth":1.0}


def compute_game_importance(team, standings_df, season_year):
    if standings_df.empty: return 0.0
    try:
        col="TeamAbbreviation" if "TeamAbbreviation" in standings_df.columns else None
        if not col: return 0.0
        row=standings_df[standings_df[col]==team]
        if row.empty: return 0.0
        row=row.iloc[0]
        wins=float(row.get("WINS",row.get("W",0)) or 0)
        losses=float(row.get("LOSSES",row.get("L",0)) or 0)
        gr=max(82-(wins+losses),1)
        conf=TEAM_CONFERENCE.get(team,"East"); ct=NBA_CONFERENCES.get(conf,[])
        cr=standings_df[standings_df[col].isin(ct)].copy()
        if "WinPct" in cr.columns: cr=cr.sort_values("WinPct",ascending=False)
        elif "W" in cr.columns: cr=cr.sort_values("W",ascending=False)
        gb=0.0
        if len(cr)>=8:
            e=cr.iloc[7]; ew=float(e.get("WINS",e.get("W",0)) or 0); el=float(e.get("LOSSES",e.get("L",0)) or 0)
            gb=max(0,(ew-wins+losses-el)/2.0)
        return round((1.0/gr)*(1.0/max(1.0,gb)),6)
    except Exception as e:
        log_fetch_error(f"game_importance {team}: {e}"); return 0.0


# ---- Main feature builder ----

def build_features(season_years):
    log.info(f"Building NBA features for {season_years}...")
    now=datetime.now(); cy=now.year+(1 if now.month>=10 else 0)
    main_season=season_years[-1] if season_years else cy

    log.info("Fetching game logs (cdn.nba.com)...")
    all_dfs=[]
    for yr in season_years:
        df=fetch_cdn_league_game_log(yr,"Regular Season")
        if not df.empty: df["SEASON_YEAR"]=yr; all_dfs.append(df)
        dfp=fetch_cdn_league_game_log(yr,"Playoffs")
        if not dfp.empty: dfp["SEASON_YEAR"]=yr; dfp["IS_PLAYOFF"]=True; all_dfs.append(dfp)
    if not all_dfs: log.error("No game log data"); return [],[]
    combined_log=pd.concat(all_dfs,ignore_index=True)
    combined_log["GAME_DATE"]=pd.to_datetime(combined_log["GAME_DATE"])
    combined_log.sort_values("GAME_DATE",inplace=True)
    paired_df=pair_games(combined_log)

    log.info("Fetching standings (cdn.nba.com)...")
    standings_df=fetch_cdn_standings(main_season)
    log.info("Fetching team stats (cdn.nba.com)...")
    dash_s=fetch_cdn_team_stats(main_season,0); dash_l5=fetch_cdn_team_stats(main_season,5)
    dash_l10=fetch_cdn_team_stats(main_season,10); dash_l20=fetch_cdn_team_stats(main_season,20)
    def dr(df,team):
        if df.empty or "TEAM_ABBREVIATION" not in df.columns: return {}
        r=df[df["TEAM_ABBREVIATION"]==team]; return r.iloc[0].to_dict() if not r.empty else {}
    all_gr=[]
    if not paired_df.empty:
        for _,row in paired_df.iterrows():
            all_gr.append({"date":str(row.get("home_GAME_DATE",""))[:10],
                           "season":int(row.get("home_SEASON_YEAR",main_season)),
                           "team1":row.get("home_TEAM_ABBREVIATION",""),
                           "team2":row.get("away_TEAM_ABBREVIATION",""),
                           "score1":float(row.get("home_PTS",0) or 0),
                           "score2":float(row.get("away_PTS",0) or 0)})
    cur_log=combined_log[combined_log["SEASON_YEAR"]==main_season].copy()
    if cur_log.empty: log.warning("No games for current season"); return [],[]
    cur_paired=pair_games(cur_log)
    log.info(f"Featurizing {len(cur_paired)} games...")
    features_list=[]; results_list=[]; seen=set(); rc={}
    def _da(ddf,team,sfx):
        r=dr(ddf,team)
        if not r: return {}
        return {f"off_rtg_dash{sfx}":r.get("OFF_RATING"),f"def_rtg_dash{sfx}":r.get("DEF_RATING"),
                f"net_rtg_dash{sfx}":r.get("NET_RATING"),f"pace_dash{sfx}":r.get("PACE"),
                f"ts_pct_dash{sfx}":r.get("TS_PCT")}
    for _,row in cur_paired.iterrows():
        gid=str(row.get("GAME_ID",""))
        if gid in seen: continue
        seen.add(gid)
        home=row.get("home_TEAM_ABBREVIATION",""); away=row.get("away_TEAM_ABBREVIATION","")
        if not home or not away or home not in ARENA_INFO or away not in ARENA_INFO: continue
        gts=row.get("home_GAME_DATE",pd.NaT)
        if pd.isnull(gts): continue
        gdo=gts.date(); gds=gts.strftime("%Y-%m-%d")
        syn=int(row.get("home_SEASON_YEAR",main_season)); isp=bool(row.get("home_IS_PLAYOFF",False))
        hs=float(row.get("home_PTS",0) or 0); as_=float(row.get("away_PTS",0) or 0)
        ic=hs>0 or as_>0
        arena=ARENA_INFO.get(home,{}); bub=is_bubble(gdo)
        sh=compute_schedule_features(combined_log,home,gdo); sa=compute_schedule_features(combined_log,away,gdo)
        tr=compute_travel_features(combined_log,away,gdo,home) if not bub else {"travel_miles":0,"timezone_change":0,"travel_direction":0,"cumulative_travel_miles_7d":0}
        hr={}; ar2={}; dl={}
        for n in [5,10,20]:
            hm=compute_rolling_metrics(paired_df,home,gdo,n); am=compute_rolling_metrics(paired_df,away,gdo,n)
            hr.update(hm); ar2.update(am); dl.update(rolling_delta(hm,am,n))
        div=is_division_rival(home,away); ic2=is_interconference(home,away)
        cap=arena.get("capacity",18000); att=round(cap*0.85); atp=round(att/max(cap,1),4)
        gih=compute_game_importance(home,standings_df,syn); gia=compute_game_importance(away,standings_df,syn)
        h2h=compute_h2h(home,away,all_gr,syn)
        if home not in rc: rc[home]=fetch_roster_availability(home)
        if away not in rc: rc[away]=fetch_roster_availability(away)
        rh=rc[home]; ra=rc[away]
        feat={"game_id":gid,"game_date":gds,"season":syn,
              "season_type":"Playoffs" if isp else "Regular Season",
              "home_team":home,"away_team":away,
              "home_score":int(hs) if ic else None,"away_score":int(as_) if ic else None,
              "is_completed":ic,"arena_name":arena.get("name",""),"arena_city":arena.get("city",""),
              "arena_capacity":arena.get("capacity",0),"day_of_week":gts.day_name(),"month":gts.month,
              "is_bubble":bub,"home_days_rest":sh["days_rest"],"home_is_back_to_back":sh["is_back_to_back"],
              "home_games_last_7":sh["games_last_7"],"home_games_last_14":sh["games_last_14"],
              "home_schedule_density":sh["schedule_density"],"home_stand_length":sh["home_stand_length"],
              "away_days_rest":sa["days_rest"],"away_is_back_to_back":sa["is_back_to_back"],
              "away_games_last_7":sa["games_last_7"],"away_games_last_14":sa["games_last_14"],
              "away_schedule_density":sa["schedule_density"],"away_road_trip_length":sa["road_trip_length"],
              "rest_differential":sh["days_rest"]-sa["days_rest"],
              "travel_miles":tr["travel_miles"],"timezone_change":tr["timezone_change"],
              "travel_direction":tr["travel_direction"],"cumulative_travel_miles_7d":tr["cumulative_travel_miles_7d"],
              "is_national_tv":0,"is_division_rival":div,"is_interconference":ic2,
              "attendance_pct":atp,"game_importance_home":gih,"game_importance_away":gia,
              **h2h,
              "home_top3_available":rh["top3_available"],"home_roster_depth":rh["roster_depth"],
              "away_top3_available":ra["top3_available"],"away_roster_depth":ra["roster_depth"]}
        feat.update({f"home_{k}":v for k,v in hr.items()})
        feat.update({f"away_{k}":v for k,v in ar2.items()})
        feat.update(dl)
        for sfx, ddf in [("_season",dash_s),("_l5",dash_l5),("_l10",dash_l10),("_l20",dash_l20)]:
            feat.update({f"home_{k}":v for k,v in _da(ddf,home,sfx).items()})
            feat.update({f"away_{k}":v for k,v in _da(ddf,away,sfx).items()})
        features_list.append(feat)
        if ic:
            results_list.append({"game_id":gid,"game_date":gds,"season":syn,
                                  "home_team":home,"away_team":away,
                                  "home_score":int(hs),"away_score":int(as_),
                                  "home_win":1 if hs>as_ else 0})
    log.info(f"Built {len(features_list)} feature records ({len(results_list)} completed)")
    return features_list, results_list


def main():
    now=datetime.now(); cy=now.year+(1 if now.month>=10 else 0)
    season_years=[cy-2,cy-1,cy]
    log.info(f"nba_data.py — cdn.nba.com endpoints — seasons {season_years}")
    try: features,results=build_features(season_years)
    except Exception as e: log_fetch_error(f"build_features failed: {e}"); raise
    ts=now.isoformat()
    (DATA_DIR/"nba_features.json").write_text(json.dumps({"updated":ts,"seasons":season_years,"count":len(features),"features":features},indent=2,default=str))
    (DATA_DIR/"nba_results.json").write_text(json.dumps({"updated":ts,"count":len(results),"results":results},indent=2,default=str))
    log.info(f"Wrote {len(features)} records to nba_features.json")
    log.info(f"Wrote {len(results)} results to nba_results.json")


if __name__ == "__main__":
    main()
