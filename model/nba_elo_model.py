"""
NBA ELO Model — mirrors structure of model/elo_model.py for NFL.
Implements Dynamic ELO for all 30 NBA teams with:
  - FiveThirtyEight historical initialization (1946–present)
  - K = 20 (NBA standard per spec)
  - HFA = 100 ELO points (home court advantage, probability calc only)
  - MOV multiplier: ln(|margin|+1) * (2.2 / (0.001 * elo_diff + 2.2))
  - Season regression: 25% toward 1500 at start of each season
  - REST_ELO_BONUS = 15 points added/subtracted for rest differential >= 2

Outputs:
  data/nba_elo_ratings.json  — current ELO ratings for all 30 teams
  data/fte_nba_elo.csv       — cached FiveThirtyEight data
"""

import json
import logging
import os
from datetime import datetime, timezone
from math import log, exp
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ─── Parameters ───────────────────────────────────────────────────────────────
K_BASE          = 20.0     # K-factor base (NBA standard)
HFA             = 100.0    # Home court advantage (added to home ELO for prob calc only)
INITIAL_ELO     = 1500.0   # Starting ELO for every team
REGRESS_PCT     = 0.25     # Season regression toward mean (25%)
REST_ELO_BONUS  = 15.0     # ELO bonus for rested team when |rest_diff| >= 2

FTE_NBA_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"
)
FTE_CACHE_PATH = DATA_DIR / "fte_nba_elo.csv"

# Canonical 30-team abbreviation set (maps FTE names → our abbreviations)
FTE_TO_ABBREV = {
    "ATL": "ATL", "BOS": "BOS", "BRK": "BKN", "BKN": "BKN",
    "CHA": "CHA", "CHH": "CHA", "CHI": "CHI", "CLE": "CLE",
    "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GSW": "GSW",
    "HOU": "HOU", "IND": "IND", "LAC": "LAC", "LAL": "LAL",
    "MEM": "MEM", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN",
    "NJN": "BKN", "NOH": "NOP", "NOK": "NOP", "NOP": "NOP",
    "NYK": "NYK", "OKC": "OKC", "ORL": "ORL", "PHI": "PHI",
    "PHO": "PHX", "PHX": "PHX", "POR": "POR", "SAC": "SAC",
    "SAS": "SAS", "SEA": "OKC", "TOR": "TOR", "UTA": "UTA",
    "VAN": "MEM", "WAS": "WAS", "WSB": "WAS",
    # Older franchises that merged or relocated
    "AND": None, "BAL": None, "BUF": None, "CAP": "WAS",
    "CHZ": "WAS", "CIN": None, "CLE": "CLE", "FTW": "DET",
    "INO": "IND", "KCK": "SAC", "KCO": "SAC", "MLH": "ATL",
    "MNL": "LAL", "MNB": None, "NOJ": "UTA", "SDC": "LAC",
    "SDR": "HOU", "SFW": "GSW", "STL": "ATL", "SYR": "PHI",
    "TRI": None, "WSC": "WAS",
}

NBA_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]


# ─── Core ELO functions (mirror elo_model.py structure) ──────────────────────

def expected_score(elo_a: float, elo_b: float) -> float:
    """Win probability for team A given ELO ratings."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def mov_multiplier(point_diff: float, elo_diff: float) -> float:
    """
    Margin of Victory multiplier applied to K before ELO update.
    Uses NFL-equivalent constant 2.2 per spec.
    """
    return log(abs(point_diff) + 1) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))


def apply_season_regression(elo_dict: dict) -> dict:
    """Regress all team ELOs 25% toward the mean (1500) at season start."""
    return {
        team: elo * (1.0 - REGRESS_PCT) + INITIAL_ELO * REGRESS_PCT
        for team, elo in elo_dict.items()
    }


def compute_nba_elo(
    games: list,
    k_base: float = K_BASE,
    hfa: float = HFA,
    initial_elo: float = INITIAL_ELO,
    regress_pct: float = REGRESS_PCT,
) -> tuple:
    """
    Process a list of NBA games chronologically and compute ELO ratings.
    Each game dict must have: team1 (home), team2 (away), score1, score2,
    date (ISO string), season (int), neutral (0/1).
    Returns (elo_dict, game_history) mirroring elo_model.py.
    """
    # Sort chronologically
    sorted_games = sorted(games, key=lambda g: g.get("date", ""))

    elo_dict     = {}
    game_history = {}   # {team: [{"result": 0/1, "elo_diff": x, "date": iso}]}
    last_season  = None

    for g in sorted_games:
        team1   = g.get("team1", "")
        team2   = g.get("team2", "")
        score1  = g.get("score1", 0)
        score2  = g.get("score2", 0)
        neutral = g.get("neutral", 0)
        season  = g.get("season", 0)
        gdate   = g.get("date", "")

        if not team1 or not team2:
            continue
        try:
            score1 = float(score1)
            score2 = float(score2)
        except (TypeError, ValueError):
            continue
        if score1 == score2:
            continue  # skip ties (extremely rare in NBA)

        # Season regression at season boundary
        if last_season is not None and season != last_season:
            regressed = apply_season_regression(elo_dict)
            elo_dict.update(regressed)
            # Clear game history at season start (mirror elo_model.py)
            for t in list(game_history.keys()):
                game_history[t] = []

        last_season = season

        # Initialise new teams
        for t in (team1, team2):
            if t not in elo_dict:
                elo_dict[t]     = initial_elo
                game_history[t] = []

        e1 = elo_dict[team1]
        e2 = elo_dict[team2]

        # Home court adjustment (prob calc only)
        hfa_adj = 0.0 if neutral else hfa
        adj_e1  = e1 + hfa_adj

        exp1 = expected_score(adj_e1, e2)
        exp2 = 1.0 - exp1

        actual1 = 1.0 if score1 > score2 else 0.0
        actual2 = 1.0 - actual1

        point_diff   = abs(score1 - score2)
        elo_diff_abs = abs(adj_e1 - e2)
        mov           = mov_multiplier(point_diff, elo_diff_abs) if point_diff > 0 else 1.0

        k = k_base
        elo_dict[team1] = e1 + k * mov * (actual1 - exp1)
        elo_dict[team2] = e2 + k * mov * (actual2 - exp2)

        game_history[team1].append({
            "result":   actual1,
            "elo_diff": float(adj_e1 - e2),
            "date":     gdate,
        })
        game_history[team2].append({
            "result":   actual2,
            "elo_diff": float(e2 - adj_e1),
            "date":     gdate,
        })

    return elo_dict, game_history


def recent_form(game_history: dict, team: str, n: int = 5) -> float:
    """Win rate over last n games. Mirrors elo_model.py recent_form."""
    hist = game_history.get(team, [])
    if not hist:
        return 0.5
    recent = hist[-n:]
    return sum(g["result"] for g in recent) / len(recent)


def get_trend(game_history: dict, team: str, window: int = 5) -> str:
    """Return 'up', 'down', or 'neutral'. Mirrors elo_model.py get_trend."""
    hist = game_history.get(team, [])
    if len(hist) < window * 2:
        return "neutral"
    recent = sum(g["result"] for g in hist[-window:]) / window
    older  = sum(g["result"] for g in hist[-window * 2:-window]) / window
    if recent > older + 0.15:
        return "up"
    elif recent < older - 0.15:
        return "down"
    return "neutral"


def predict_game(
    home_team: str,
    away_team: str,
    elo_dict: dict,
    game_history: dict,
    neutral: bool = False,
    rest_diff: int = 0,       # home_days_rest − away_days_rest
    hfa: float = HFA,
    form_blend: float = 0.2,
) -> dict:
    """
    Predict win probability for home_team vs away_team.
    REST_ELO_BONUS applied when |rest_diff| >= 2 (prob calc only).
    Returns dict mirroring elo_model.py predict_game.
    """
    base_elo_home = elo_dict.get(home_team, INITIAL_ELO)
    base_elo_away = elo_dict.get(away_team, INITIAL_ELO)

    # Recent form adjustment (same formula as elo_model.py)
    form_home = recent_form(game_history, home_team)
    form_away = recent_form(game_history, away_team)
    form_adj_home = (form_home - 0.5) * form_blend * 100.0
    form_adj_away = (form_away - 0.5) * form_blend * 100.0

    adj_home = base_elo_home + form_adj_home
    adj_away = base_elo_away + form_adj_away

    # Home court advantage (prob calc only)
    if not neutral:
        adj_home += hfa

    # Rest adjustment (prob calc only, parameterized by REST_ELO_BONUS)
    if rest_diff >= 2:
        adj_home += REST_ELO_BONUS
    elif rest_diff <= -2:
        adj_away += REST_ELO_BONUS

    prob_home = expected_score(adj_home, adj_away)

    return {
        "prob":         float(prob_home),
        "elo_home":     float(base_elo_home),
        "elo_away":     float(base_elo_away),
        "adj_elo_home": float(adj_home),
        "adj_elo_away": float(adj_away),
        "form_home":    float(form_home),
        "form_away":    float(form_away),
        "elo_diff":     float(adj_home - adj_away),
    }


# ─── FiveThirtyEight data ─────────────────────────────────────────────────────

def load_fte_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load FiveThirtyEight historical NBA ELO CSV.
    Downloads once and caches locally at data/fte_nba_elo.csv.
    """
    if FTE_CACHE_PATH.exists() and not force_download:
        log.info(f"Loading FTE NBA ELO from cache: {FTE_CACHE_PATH}")
        try:
            return pd.read_csv(FTE_CACHE_PATH, low_memory=False)
        except Exception as e:
            log.warning(f"Cache read failed ({e}), re-downloading...")

    log.info(f"Downloading FTE NBA ELO from {FTE_NBA_URL}")
    try:
        resp = requests.get(FTE_NBA_URL, timeout=60)
        resp.raise_for_status()
        FTE_CACHE_PATH.write_bytes(resp.content)
        log.info(f"FTE NBA ELO cached at {FTE_CACHE_PATH}")
        return pd.read_csv(FTE_CACHE_PATH, low_memory=False)
    except Exception as e:
        log.error(f"Failed to download FTE NBA ELO: {e}")
        return pd.DataFrame()


def fte_to_game_list(df: pd.DataFrame) -> list:
    """
    Convert FiveThirtyEight nbaallelo.csv to the format expected by compute_nba_elo().
    FTE columns: date, season, team_id, fran_id, pts, opp_pts, game_location, game_result.
    Each row is one team's view — we reconstruct full games.
    """
    if df.empty:
        return []

    df = df.copy()
    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    required = {"date", "season", "team_id", "pts", "opp_pts"}
    if not required.issubset(df.columns):
        log.warning(f"FTE CSV missing columns. Found: {list(df.columns)}")
        return []

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "pts", "opp_pts"])
    df = df.sort_values("date")

    # Filter to home games only to avoid duplicates (game_location == 'H')
    if "game_location" in df.columns:
        home_df = df[df["game_location"] == "H"].copy()
    elif "is_playoffs" in df.columns or "playoff" in df.columns:
        # Fallback: take every-other row by game_id grouping
        home_df = df.drop_duplicates(subset=["date", "season", "pts", "opp_pts"])
    else:
        # If no location column, deduplicate using game_id or pts pairing
        home_df = df.copy()

    games = []
    for _, row in home_df.iterrows():
        raw_team = str(row.get("team_id", "") or row.get("fran_id", "")).upper()
        team1    = FTE_TO_ABBREV.get(raw_team)
        if team1 is None:
            continue

        # Opponent is harder — FTE only has opponent's pts in this row
        # We reconstruct from the pair in the dataframe
        # Use game_id if available; otherwise match by date+pts
        raw_opp = str(row.get("opp_id", "") or "").upper()
        team2   = FTE_TO_ABBREV.get(raw_opp)
        if team2 is None:
            continue  # skip games with unknown/historical teams

        games.append({
            "date":    row["date"].strftime("%Y-%m-%d"),
            "season":  int(row.get("season", 1946)),
            "team1":   team1,
            "team2":   team2,
            "score1":  float(row["pts"]),
            "score2":  float(row["opp_pts"]),
            "neutral": int(row.get("neutral", 0) if "neutral" in row.index else 0),
        })

    log.info(f"Parsed {len(games)} games from FTE NBA ELO data")
    return games


# ─── Save ratings ─────────────────────────────────────────────────────────────

def save_ratings(
    elo_dict: dict,
    game_history: dict,
    season_year: int,
    path: Path = None,
) -> None:
    """
    Write nba_elo_ratings.json mirroring the format of elo_ratings.json.
    """
    if path is None:
        path = DATA_DIR / "nba_elo_ratings.json"

    ratings = []
    for team in sorted(elo_dict.keys()):
        hist = game_history.get(team, [])
        wins   = sum(1 for g in hist if g["result"] == 1.0)
        losses = sum(1 for g in hist if g["result"] == 0.0)
        elo    = elo_dict[team]
        ratings.append({
            "team":         team,
            "elo":          round(elo, 1),
            "games_played": wins + losses,
            "wins":         wins,
            "losses":       losses,
            "trend":        get_trend(game_history, team),
        })

    # Sort by ELO descending
    ratings.sort(key=lambda r: r["elo"], reverse=True)

    output = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "season":  season_year,
        "ratings": ratings,
    }

    path.write_text(json.dumps(output, indent=2))
    log.info(f"Wrote NBA ELO ratings for {len(ratings)} teams to {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    from datetime import datetime as dt
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    now          = dt.now()
    season_year  = now.year + (1 if now.month >= 10 else 0)

    log.info("=== NBA ELO Model (nba_elo_model.py) starting ===")
    log.info(f"K_BASE={K_BASE}, HFA={HFA}, REGRESS_PCT={REGRESS_PCT}, REST_ELO_BONUS={REST_ELO_BONUS}")

    # ── 1. Load FiveThirtyEight historical data ───────────────────────────────
    log.info("Loading FiveThirtyEight NBA ELO data...")
    fte_df = load_fte_data()

    if not fte_df.empty:
        log.info(f"FTE data: {len(fte_df)} rows, seasons {fte_df['season'].min()}–{fte_df['season'].max()}" if "season" in fte_df.columns else f"FTE data: {len(fte_df)} rows")
        fte_games = fte_to_game_list(fte_df)
    else:
        log.warning("FTE data unavailable — starting from scratch with all teams at 1500.")
        fte_games = []

    # ── 2. Compute ELO from FTE history ──────────────────────────────────────
    if fte_games:
        log.info(f"Computing ELO from {len(fte_games)} historical FTE games...")
        elo_dict, game_history = compute_nba_elo(fte_games)
    else:
        elo_dict     = {t: INITIAL_ELO for t in NBA_TEAMS}
        game_history = {t: [] for t in NBA_TEAMS}

    # Ensure all 30 current teams are present
    for team in NBA_TEAMS:
        if team not in elo_dict:
            elo_dict[team]     = INITIAL_ELO
            game_history[team] = []

    # ── 3. Load recent season data for final ELO updates ─────────────────────
    # Try to import ESPN-fetched game list from update_nba_data to extend
    try:
        recent_json = DATA_DIR / "nba_results.json"
        if recent_json.exists():
            results = json.loads(recent_json.read_text()).get("results", [])
            if results:
                log.info(f"Extending ELO with {len(results)} recent games from nba_results.json...")
                # Only use games after the FTE cutoff date
                fte_cutoff = "2024-10-01"  # FTE data typically lags; use recent games
                recent_only = [g for g in results if g.get("game_date", "") >= fte_cutoff]
                if recent_only:
                    recent_games = [
                        {
                            "date":    g["game_date"],
                            "season":  g["season"],
                            "team1":   g["home_team"],
                            "team2":   g["away_team"],
                            "score1":  g["home_score"],
                            "score2":  g["away_score"],
                            "neutral": 0,
                        }
                        for g in recent_only
                        if g.get("home_score") and g.get("away_score")
                    ]
                    if recent_games:
                        # Apply one more season regression if crossing a season boundary
                        elo_dict, game_history = compute_nba_elo(
                            recent_games,
                            k_base=K_BASE,
                            hfa=HFA,
                            initial_elo=INITIAL_ELO,
                            regress_pct=REGRESS_PCT,
                        )
                        log.info("ELO extended with recent ESPN results.")
    except Exception as e:
        log.warning(f"Could not extend ELO with recent results: {e}")

    # ── 4. Save ratings ───────────────────────────────────────────────────────
    log.info("Saving NBA ELO ratings...")
    save_ratings(elo_dict, game_history, season_year)

    # Quick sanity check
    if elo_dict:
        elos  = list(elo_dict.values())
        log.info(f"ELO range: {min(elos):.0f} – {max(elos):.0f}, mean: {sum(elos)/len(elos):.0f}")

    log.info("=== NBA ELO Model complete ===")


if __name__ == "__main__":
    main()
