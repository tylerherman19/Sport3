#!/usr/bin/env python3
"""
updatekalshi.py — Fetch Kalshi sports markets server-side and write to data/kalshidata.json.

Runs in GitHub Actions to avoid CORS restrictions in the browser.
Mirrors the pattern of update_nba_data.py / update_data.py.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
KALSHI_BASE = "https://trading-api.kalshi.com/trade-api/v2"

EXCLUDE_KEYWORDS = [
    'super bowl', 'championship', 'mvp', 'season wins', 'total wins',
    'make playoffs', 'win division', 'draft', 'spread', 'cover',
    'over/under', 'first quarter', 'first half', 'halftime',
]

SPORTS_KEYWORDS = [
    # NFL team nicknames
    'chiefs', 'eagles', 'cowboys', 'patriots', 'packers', 'bears', '49ers',
    'ravens', 'broncos', 'seahawks', 'bills', 'dolphins', 'jets', 'giants',
    'commanders', 'steelers', 'bengals', 'browns', 'texans', 'colts',
    'jaguars', 'titans', 'raiders', 'chargers', 'rams', 'cardinals',
    'falcons', 'panthers', 'saints', 'buccaneers', 'vikings', 'lions',
    # NBA team nicknames
    'lakers', 'celtics', 'warriors', 'nets', 'knicks', 'heat', 'bulls',
    'spurs', 'nuggets', 'clippers', 'bucks', 'sixers', 'raptors', 'thunder',
    'jazz', 'grizzlies', 'pelicans', 'rockets', 'mavericks', 'suns', 'kings',
    'blazers', 'pistons', 'pacers', 'hawks', 'hornets', 'magic', 'wizards',
    'cavaliers', 'timberwolves',
]


def fetch_all_markets(api_key):
    """Fetch all open markets from Kalshi API with pagination and optional auth."""
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    markets, cursor, pages = [], None, 0
    while pages < 5:
        params = {'status': 'open', 'limit': 200}
        if cursor:
            params['cursor'] = cursor
        try:
            resp = requests.get(
                f'{KALSHI_BASE}/markets',
                params=params,
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error(f"Kalshi request failed (page {pages}): {e}")
            break

        data = resp.json()
        batch = data.get('markets', [])
        markets.extend(batch)
        log.info(f"Page {pages + 1}: {len(batch)} markets (total: {len(markets)})")

        cursor = data.get('cursor')
        if not cursor or not batch:
            break
        pages += 1
        time.sleep(0.5)  # respect rate limits

    return markets


def is_sports_market(market):
    """Return True if this market is likely an NFL or NBA game-winner market."""
    title = (market.get('title') or '').lower()
    if any(kw in title for kw in EXCLUDE_KEYWORDS):
        return False
    text = ' '.join([
        title,
        (market.get('subtitle') or '').lower(),
        (market.get('category') or '').lower(),
        (market.get('series_ticker') or '').lower(),
    ])
    return any(kw in text for kw in SPORTS_KEYWORDS) or 'nfl' in text or 'nba' in text


def main():
    api_key = os.environ.get('KALSHI_API_KEY')
    log.info("KALSHI_API_KEY: %s", "present" if api_key else "not set (unauthenticated)")

    all_markets = fetch_all_markets(api_key)
    log.info(f"Total markets fetched: {len(all_markets)}")

    sports_markets = [m for m in all_markets if is_sports_market(m)]
    log.info(f"Sports markets after filtering: {len(sports_markets)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        'updated': datetime.now(timezone.utc).isoformat(),
        'market_count': len(sports_markets),
        'markets': sports_markets,
    }
    out_path = DATA_DIR / 'kalshidata.json'
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info(f"Written {len(sports_markets)} markets to {out_path}")


if __name__ == '__main__':
    main()
