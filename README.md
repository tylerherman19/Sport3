# NFL Prediction Dashboard

A fully automated, 12-model ensemble NFL prediction system that runs entirely in GitHub and is viewable via GitHub Pages.

---

## Live Dashboard

**[View Dashboard →](https://tylerherman19.github.io/Sport3/)**

---

## Setup — 3 Steps

### Step 1 — Enable GitHub Pages

1. Go to your repository on GitHub: `https://github.com/tylerherman19/Sport3`
2. Click **Settings** (top menu)
3. Click **Pages** (left sidebar)
4. Under **Source**, select **Deploy from a branch**
5. Set branch to **`main`**, folder to **`/ (root)`**
6. Click **Save**
7. Wait 2–3 minutes, then visit: `https://tylerherman19.github.io/Sport3/`

### Step 2 — Run the Model Manually (First Time)

1. Go to the **Actions** tab in your repository
2. Click **Update NFL Model** (left sidebar)
3. Click **Run workflow** → **Run workflow**
4. Wait ~5 minutes for it to complete
5. Refresh the dashboard — it will now show live data

### Step 3 (Optional) — Add Betting Odds

To enable live betting line comparisons:

1. Register for a free account at [The Odds API](https://the-odds-api.com) (500 free requests/month)
2. Copy your API key
3. Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Name: `ODDS_API_KEY`, Value: your API key
6. Click **Add secret**
7. Re-run the workflow (Step 2)

---

## How It Works

### Automation
The model runs automatically every day at 8 AM UTC via GitHub Actions. No action required.

### Data Sources
- **FiveThirtyEight** historical NFL ELO dataset (~11,000 games for model training)
- **ESPN public APIs** for live scores, standings, injuries
- **The Odds API** for betting lines (optional)

### 12 Model Systems

| System | Model | Weight |
|--------|-------|--------|
| 1 | Logistic Regression + Isotonic Calibration | 30% |
| 2 | ELO with MOV Multiplier | — |
| 3 | Bayesian Normal Rating | — |
| 4 | Pythagorean Win Expectation | 15% |
| 5 | Offensive / Defensive Efficiency | 10% |
| 6 | Turnover Regression | — |
| 7 | Rest & Travel Adjustments | — |
| 8 | Time Decay Weighting | — |
| 9 | XGBoost Gradient Boosting | 25% |
| 10 | Monte Carlo Simulation (10,000 runs) | — |
| 11 | Hierarchical Team Strength | — |
| 12 | Weighted Ensemble | Final output |

### Repository Structure

```
Sport3/
├── index.html              Dashboard frontend
├── style.css               Dark theme styles
├── main.js                 Dashboard logic
├── data/
│   ├── predictions.json    Game predictions (auto-updated)
│   ├── elo_ratings.json    Team ELO ratings (auto-updated)
│   ├── leaderboard.json    Team leaderboard (auto-updated)
│   └── model_metrics.json  Model accuracy metrics (auto-updated)
├── model/
│   ├── logistic_model.py   Logistic regression + calibration
│   ├── elo_model.py        ELO rating system
│   ├── bayesian_model.py   Bayesian team ratings
│   ├── efficiency_model.py Pythagorean + efficiency + adjustments
│   ├── monte_carlo.py      Monte Carlo simulation
│   └── ensemble_model.py   XGBoost + ensemble
├── scripts/
│   └── update_data.py      Daily orchestration script
└── .github/
    └── workflows/
        └── update-model.yml GitHub Actions workflow
```

---

## Dashboard Sections

1. **This Week's Games** — Predicted winner, win probability bar, model edge vs market, ELO, rest, travel, Monte Carlo margin
2. **Model Controls** — Sliders for K-factor, home field, ensemble weights, time decay λ
3. **Matchup Predictor** — Any two teams, full breakdown with plain-English explanation
4. **ELO Leaderboard** — All 32 teams sorted by ELO with uncertainty bands, playoff probabilities
5. **Model Performance** — Log loss, Brier score, AUC, calibration curve, historical accuracy
