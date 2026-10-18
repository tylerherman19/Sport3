#!/usr/bin/env python3
"""Build data/nfl_record.json - the public, graded prediction record.

The record answers one question honestly: what did the site publish before
each game, and did it hit?

Pick of record: the last prediction published before kickoff.
  - Backfill (one-time, --backfill): recovered from git history. For each game,
    the pick comes from the last commit touching the NFL predictions file
    before game_time. Provenance (commit hash + timestamp) is stored per pick.
  - Ongoing (daily CI): the pick starts as whatever the current feed holds and
    is refreshed every run until kickoff, when it freezes. The daily commit is
    the public, timestamped evidence.

Grading: ESPN scoreboard API, final games only. Ties are recorded but excluded
from accuracy. A game with no pre-kickoff published pick is recorded with
result NO_PICK and excluded from accuracy (never invent a pick).

Entries are immutable once graded; existing picks are never rewritten after
kickoff.
"""
import argparse
import json
import subprocess
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RECORD_PATH = REPO / "data" / "nfl_record.json"
PRED_FILES = ["data/nfl_predictions.json", "data/predictions.json"]
ESPN = ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/"
        "scoreboard?dates=")
SEASON_START = "2026-09-01"  # regular-season feeds only
ESPNFIX = {"JAC": "JAX", "ARZ": "ARI", "CLV": "CLE", "HST": "HOU",
           "LVR": "LV", "WSH": "WAS"}


def norm_abbr(a):
    u = (a or "").upper()
    return ESPNFIX.get(u, u)


def parse_iso(s):
    """Parse the feed's loose ISO timestamps (e.g. 2026-10-18T13:30Z)."""
    if not s:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def iso_z(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_history_entries():
    """Yield (commit_hash, commit_dt, predictions_json_text) oldest first."""
    out = subprocess.run(
        ["git", "log", "--reverse", "--format=%H%x00%cI", "--"] + PRED_FILES,
        cwd=REPO, capture_output=True, text=True, check=True).stdout
    commits = []
    for line in out.splitlines():
        if not line.strip():
            continue
        h, _, ts = line.partition("\x00")
        commits.append((h.strip(), parse_iso(ts.strip())))
    for h, dt in commits:
        for f in PRED_FILES:
            r = subprocess.run(["git", "show", f"{h}:{f}"], cwd=REPO,
                               capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                yield h, dt, r.stdout
                break


def feed_games(doc):
    """Extract {game_id: game-dict} for regular-season 2026 games."""
    games = {}
    if not isinstance(doc, dict):
        return games
    if doc.get("season") not in (2026, "2026"):
        return games
    for g in doc.get("games") or []:
        gt = parse_iso(g.get("game_time"))
        if not gt or gt < parse_iso(SEASON_START):
            continue
        if not g.get("game_id"):
            continue
        games[str(g["game_id"])] = g
    return games


def backfill(record):
    """Recover pre-kickoff picks for past games from git history."""
    # game_id -> list of (commit_dt, hash, home_prob)
    sightings = defaultdict(list)
    for h, dt, text in git_history_entries():
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            continue
        for gid, g in feed_games(doc).items():
            p = (g.get("predictions") or {}).get("ensemble_prob")
            if isinstance(p, (int, float)):
                sightings[gid].append((dt, h, float(p), g))
    now = datetime.now(timezone.utc)
    added = updated = 0
    for gid, seen in sightings.items():
        seen.sort(key=lambda s: s[0])
        game_time = parse_iso(seen[-1][3].get("game_time"))
        pre = [s for s in seen if s[0] < game_time]
        entry = record["games"].get(gid)
        if entry is None:
            g = seen[-1][3]
            entry = {
                "game_id": gid,
                "week": g.get("week"),
                "game_time": iso_z(game_time),
                "home_team": g.get("home_team"),
                "away_team": g.get("away_team"),
                "home_prob": None,
                "predicted_winner": None,
                "predicted_prob": None,
                "pick_source": None,
                "pick_published_at": None,
                "actual_winner": None,
                "home_score": None,
                "away_score": None,
                "result": "PENDING",
            }
            record["games"][gid] = entry
            added += 1
        if entry["result"] not in ("PENDING", "NO_PICK"):
            continue  # frozen
        if game_time >= now:
            continue  # future game: incremental mode owns it
        if pre:
            dt, h, p, _ = pre[-1]
            entry["home_prob"] = p
            entry["pick_source"] = f"git:{h[:10]}"
            entry["pick_published_at"] = iso_z(dt)
            updated += 1
        else:
            entry["result"] = "NO_PICK"
    return added, updated


def incremental(record, feed_doc):
    """Snapshot/refresh picks from the current feed until kickoff."""
    now = datetime.now(timezone.utc)
    feed_updated = parse_iso(feed_doc.get("updated")) or now
    added = updated = 0
    for gid, g in feed_games(feed_doc).items():
        gt = parse_iso(g["game_time"])
        p = (g.get("predictions") or {}).get("ensemble_prob")
        if not isinstance(p, (int, float)):
            continue
        entry = record["games"].get(gid)
        if entry is None:
            entry = {
                "game_id": gid,
                "week": g.get("week"),
                "game_time": iso_z(gt),
                "home_team": g.get("home_team"),
                "away_team": g.get("away_team"),
                "home_prob": float(p),
                "predicted_winner": None,
                "predicted_prob": None,
                "pick_source": "feed",
                "pick_published_at": iso_z(feed_updated),
                "actual_winner": None,
                "home_score": None,
                "away_score": None,
                "result": "PENDING",
            }
            record["games"][gid] = entry
            added += 1
        elif entry["result"] in ("PENDING", "NO_PICK") and gt > now:
            # still pre-kickoff: track the latest published pick
            if entry["home_prob"] != float(p) or not entry.get("pick_source"):
                entry["home_prob"] = float(p)
                entry["pick_source"] = "feed"
                entry["pick_published_at"] = iso_z(feed_updated)
                if entry["result"] == "NO_PICK":
                    entry["result"] = "PENDING"
                updated += 1
        # else: kicked off - pick is frozen
    return added, updated


def finalize_picks(record):
    """Derive winner/confidence from home_prob once, at render boundary."""
    for e in record["games"].values():
        p = e.get("home_prob")
        if isinstance(p, (int, float)):
            e["predicted_winner"] = e["home_team"] if p >= 0.5 else e["away_team"]
            e["predicted_prob"] = max(p, 1 - p)


def fetch_scoreboard(date_str):
    url = ESPN + date_str
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  scoreboard fetch failed for {date_str}: {e}",
              file=sys.stderr)
        return None


def grade(record):
    """Fill final results from ESPN for past, ungraded games."""
    now = datetime.now(timezone.utc)
    todo = [e for e in record["games"].values()
            if e["result"] in ("PENDING", "NO_PICK")
            and parse_iso(e["game_time"]) < now]
    by_day = defaultdict(list)
    for e in todo:
        dt = parse_iso(e["game_time"])
        # UTC date and the day before (ESPN's dates param is US/Eastern-based;
        # prime-time games land after midnight UTC)
        by_day[dt.strftime("%Y%m%d")].append(e)
        by_day[(dt - timedelta(days=1)).strftime("%Y%m%d")].append(e)
    graded = 0
    for date_str in sorted(by_day):
        data = fetch_scoreboard(date_str)
        if not data:
            continue
        for ev in data.get("events") or []:
            comp = (ev.get("competitions") or [{}])[0]
            cs = comp.get("competitors") or []
            if len(cs) < 2:
                continue
            home_c = next((c for c in cs if c.get("homeAway") == "home"), None)
            away_c = next((c for c in cs if c.get("homeAway") == "away"), None)
            if not home_c or not away_c:
                continue
            entries = [e for e in by_day[date_str]
                       if e["game_id"] == str(ev.get("id"))]
            if not entries:
                h, a = norm_abbr(home_c["team"]["abbreviation"]), \
                    norm_abbr(away_c["team"]["abbreviation"])
                entries = [e for e in by_day[date_str]
                           if e["home_team"] == h and e["away_team"] == a]
            if not entries:
                continue
            status = (ev.get("status") or {}).get("type", {}).get("name", "")
            winner = next((c for c in cs if c.get("winner")), None)
            try:
                hs, as_ = int(home_c.get("score")), int(away_c.get("score"))
            except (TypeError, ValueError):
                continue
            if not winner and status == "STATUS_FINAL" and hs != as_:
                winner = home_c if hs > as_ else away_c
            if status != "STATUS_FINAL":
                continue
            for entry in entries:
                if entry["result"] not in ("PENDING", "NO_PICK"):
                    continue
                entry["home_score"] = hs
                entry["away_score"] = as_
                if not entry.get("predicted_winner"):
                    # no pick of record: scores fill, result stays NO_PICK
                    entry["actual_winner"] = (
                        norm_abbr(winner["team"]["abbreviation"])
                        if winner else "TIE")
                    entry["result"] = "NO_PICK"
                elif winner is None:
                    entry["actual_winner"] = "TIE"
                    entry["result"] = "TIE"
                else:
                    w = norm_abbr(winner["team"]["abbreviation"])
                    entry["actual_winner"] = w
                    entry["result"] = ("HIT" if w == entry["predicted_winner"]
                                       else "MISS")
                graded += 1
    return graded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="reconstruct past picks from git history (one-time)")
    args = ap.parse_args()

    if RECORD_PATH.exists():
        record = json.loads(RECORD_PATH.read_text())
    else:
        record = {"season": 2026, "games": {}}
    record.setdefault("games", {})
    if isinstance(record["games"], list):  # written back as a sorted list
        record["games"] = {str(e["game_id"]): e for e in record["games"]}

    if args.backfill:
        a, u = backfill(record)
        print(f"backfill: {a} games found, {u} picks recovered")

    feed_path = REPO / "data" / "nfl_predictions.json"
    if not feed_path.exists():
        feed_path = REPO / "data" / "predictions.json"
    if feed_path.exists():
        a, u = incremental(record, json.loads(feed_path.read_text()))
        print(f"incremental: {a} added, {u} picks refreshed")

    finalize_picks(record)
    g = grade(record)
    print(f"graded: {g} results")

    games = sorted(record["games"].values(), key=lambda e: e["game_time"])
    graded_games = [e for e in games if e["result"] in ("HIT", "MISS")]
    hits = [e for e in games if e["result"] == "HIT"]
    record["updated"] = iso_z(datetime.now(timezone.utc))
    record["summary"] = {
        "called": sum(1 for e in games if e["result"] != "NO_PICK"),
        "graded": len(graded_games),
        "hits": len(hits),
        "accuracy": (len(hits) / len(graded_games)) if graded_games else None,
    }
    out = dict(record)
    out["games"] = games
    RECORD_PATH.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {RECORD_PATH}: {record['summary']}")


if __name__ == "__main__":
    main()
