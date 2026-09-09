"""
SumerSports team-table feed (open gate, no auth).

Fetches the public, server-rendered team stat tables from sumersports.com and
caches them as JSON under data/sumersports_cache/. Verified live 2026-09-09:

  https://sumersports.com/teams/offensive/?season=2024   -> 32-row <table>
  https://sumersports.com/teams/defensive/?season=2024   -> same shape
  seasons 2022-2025 served; 2021 and earlier render an empty page.

The tables carry per-team EPA/Play, Total EPA, Success %, EPA/Pass, EPA/Rush,
yardage/TD splits, ADoT, scramble/int rates, and (defense) 3-/4-man rush rates.
Pages carry a "Last Updated MM-DD-YYYY" stamp and refresh in-season (observed
09-02-2026 on the 2025 table), so the CURRENT season table is a live weekly
feed while completed seasons are static.

This is a raw-stats feed (play-level EPA aggregates), not odds and not someone
else's win probabilities - it is eligible as a model input under the repo's
no-market-data rule (memory: nfl-model-no-odds-data), same class as nflverse.

Stdlib only (urllib + html.parser) so the GitHub Actions runner needs nothing new.

Usage:
    python3 scripts/sumersports_feed.py                  # all seasons 2022..current
    python3 scripts/sumersports_feed.py --season 2025    # one season
    python3 scripts/sumersports_feed.py --current-only   # just the live season
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "sumersports_cache"

BASE = "https://sumersports.com/teams/{side}/?season={season}"
FIRST_SEASON = 2022  # verified: 2021 and earlier render with no data rows
USER_AGENT = "Mozilla/5.0 (compatible; TheModel research bot; +https://tylerherman19.github.io/Sport3/)"

# SumerSports prints full team names ("1.Kansas City Chiefs"); the repo keys
# everything on these abbreviations. Fail loudly on an unmapped name - a silent
# drop would feed the model a partial league.
TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    # Historical name variants inside the 2022+ window
    "Washington Football Team": "WAS",
}

# Column header -> (json field, kind). kinds: float, int, pct (stored as fraction).
COLUMN_MAP = {
    "EPA/Play": ("epa_play", "float"),
    "Total EPA": ("total_epa", "float"),
    "Success %": ("success_pct", "pct"),
    "EPA/Pass": ("epa_pass", "float"),
    "EPA/Rush": ("epa_rush", "float"),
    "Pass Yards": ("pass_yards", "int"),
    "Pass TD": ("pass_td", "int"),
    "Rush Yards": ("rush_yards", "int"),
    "Rush TD": ("rush_td", "int"),
    "ADoT": ("adot", "float"),
    "Scramble %": ("scramble_pct", "pct"),
    "Int %": ("int_pct", "pct"),
    "Man Run %": ("man_run_pct", "pct"),
    "Power Run %": ("power_run_pct", "pct"),
    "3-man Rush %": ("three_man_rush_pct", "pct"),
    "4-man Rush %": ("four_man_rush_pct", "pct"),
}


class _TableParser(HTMLParser):
    """Extracts rows from the single server-rendered <table> on each page."""

    def __init__(self):
        super().__init__()
        self.in_table = 0
        self.in_cell = False
        self.in_row = False
        self.cur_cell = ""
        self.cur_row = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table += 1
        elif self.in_table:
            if tag == "tr":
                self.in_row = True
                self.cur_row = []
            elif tag in ("td", "th"):
                self.in_cell = True
                self.cur_cell = ""

    def handle_endtag(self, tag):
        if tag == "table":
            self.in_table -= 1
        elif self.in_table:
            if tag in ("td", "th") and self.in_cell:
                self.cur_row.append(self.cur_cell.strip())
                self.in_cell = False
            elif tag == "tr" and self.in_row:
                if self.cur_row:
                    self.rows.append(self.cur_row)
                self.in_row = False

    def handle_data(self, data):
        if self.in_cell:
            self.cur_cell += data


def _fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read().decode("utf-8", "replace")


def _convert(raw, kind):
    raw = raw.strip()
    if raw == "":
        return None
    try:
        if kind == "pct":
            return round(float(raw.rstrip("%")) / 100.0, 6)
        if kind == "int":
            return int(raw.replace(",", ""))
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def parse_team_table(html, expected_season=None):
    """Parse one SumerSports team-table page.

    Returns (teams, last_updated) where teams is {ABBR: {field: value}} keyed by
    the COLUMN_MAP fields present on the page. Raises ValueError when the page
    has no data rows (an unserved season) or names a team we cannot map - both
    are feed-integrity failures, not data.
    """
    parser = _TableParser()
    parser.feed(html)
    if len(parser.rows) < 2:
        raise ValueError("no data rows on page (season not served?)")

    header = parser.rows[0]
    col_idx = {}
    for i, h in enumerate(header):
        h_clean = h.strip()
        if h_clean in COLUMN_MAP:
            col_idx[i] = COLUMN_MAP[h_clean]

    m = re.search(r"Last Updated[^0-9]*([0-9]{2}-[0-9]{2}-[0-9]{4})", html)
    last_updated = m.group(1) if m else None

    teams = {}
    for row in parser.rows[1:]:
        if len(row) < 3:
            continue
        name = re.sub(r"^\d+\.", "", row[0]).strip()
        season_str = row[1].strip() if len(row) > 1 else ""
        if expected_season is not None and season_str and int(season_str) != int(expected_season):
            raise ValueError(f"page season mismatch: row says {season_str}, asked {expected_season}")
        abbr = TEAM_NAME_TO_ABBR.get(name)
        if abbr is None:
            raise ValueError(f"unmapped team name: {name!r}")
        fields = {}
        for i, (field, kind) in col_idx.items():
            if i < len(row):
                fields[field] = _convert(row[i], kind)
        teams[abbr] = fields
    return teams, last_updated


def fetch_season(season, pause=1.0):
    """Fetch offensive + defensive tables for one season.

    Returns the cache dict, or None when the season is not served (e.g. asking
    for a season that has not started). Historical seasons are static, so the
    caller decides refresh policy.
    """
    out = {"season": int(season), "teams": {}}
    last_updated = None
    for side in ("offensive", "defensive"):
        url = BASE.format(side=side, season=int(season))
        html = _fetch(url)
        teams, stamp = parse_team_table(html, expected_season=season)
        if not teams:
            return None
        key = "off" if side == "offensive" else "def"
        for abbr, fields in teams.items():
            out["teams"].setdefault(abbr, {})[key] = fields
        last_updated = stamp or last_updated
        time.sleep(pause)  # be polite; this is an unofficial gate
    if len(out["teams"]) != 32:
        raise ValueError(f"season {season}: expected 32 teams, got {len(out['teams'])}")
    out["source"] = [BASE.format(side=s, season=int(season)) for s in ("offensive", "defensive")]
    out["fetched_at"] = datetime.now(timezone.utc).isoformat()
    out["last_updated"] = last_updated
    return out


def cache_path(season):
    return CACHE_DIR / f"team_tables_{int(season)}.json"


def write_cache(data):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(data["season"])
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--current-only", action="store_true",
                    help="refresh only the current NFL season (started Aug+)")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    current_season = now.year if now.month >= 3 else now.year - 1

    if args.season:
        seasons = [args.season]
    elif args.current_only:
        seasons = [current_season]
    else:
        seasons = list(range(FIRST_SEASON, current_season + 1))

    ok, skipped, failed = 0, [], []
    for season in seasons:
        path = cache_path(season)
        # Completed seasons are static; only the live season needs refreshing.
        if season < current_season and path.exists():
            skipped.append(season)
            continue
        try:
            data = fetch_season(season)
        except ValueError as e:
            print(f"  {season}: skipped ({e})")
            continue
        except Exception as e:
            failed.append((season, e))
            print(f"  {season}: FETCH FAILED ({e})", file=sys.stderr)
            continue
        if data is None:
            print(f"  {season}: not served yet")
            continue
        p = write_cache(data)
        ok += 1
        print(f"  {season}: {len(data['teams'])} teams -> {p.name} "
              f"(site stamp {data.get('last_updated')})")

    print(f"done: {ok} refreshed, {len(skipped)} cached, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
