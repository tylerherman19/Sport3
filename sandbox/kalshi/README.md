# Kalshi Odds Comparison — Sandbox

This directory contains the Kalshi odds comparison feature, which has been
removed from the main production flow and preserved here for future reference
or re-integration.

## What was sandboxed

| File | Description |
|------|-------------|
| `kalshi_functions.js` | All Kalshi JS functions extracted from `main.js` |
| `kalshi_tab.html` | The Kalshi section HTML extracted from `index.html` |
| `updatekalshi.py` | Script that fetches Kalshi market data via API |
| `update-kalshi.yml` | GitHub Actions workflow (original, un-disabled copy) |

## What was changed in the live site

- `index.html`: Kalshi nav tab button and section commented out
- `main.js`:
  - `renderKalshiIfOpen()` call in `renderAll()` commented out
  - Kalshi nav tab handler commented out
  - Kalshi section marked with SANDBOXED banner
  - Kalshi state fields condensed with SANDBOXED note
- `.github/workflows/update-kalshi.yml`: job disabled with `if: false`

## To re-enable

1. Un-comment the nav tab button and section in `index.html`
2. Un-comment `renderKalshiIfOpen()` in `renderAll()` in `main.js`
3. Un-comment the Kalshi nav handler block in `main.js`
4. Remove `if: false` from `.github/workflows/update-kalshi.yml`
