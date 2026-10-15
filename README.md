# Teammate Delta: F1 one-lap pace from the teammate web

A static GitHub Pages site that ranks the 100 fastest Formula 1 drivers over one lap, 1950 to today, using the only fair comparison in the sport: the teammate.

- **Top 100** (`index.html`): peak or career one-lap pace, filterable by era.
- **Drivers** (`search.html`): search every driver since 1950.
- **Driver profiles** (`driver.html?id=…`): pace by season, the teammate web, gap to every teammate (race by race) and full team history.
- **Compare** (`compare.html`): pick any two drivers to see the strongest chain of teammates linking them and the model's verdict.
- **Method** (`methodology.html`): how the model works.

## How it works (short version)

1. Every qualifying session since 1950 (Indy 500 excluded) gives a teammate gap. Teammates share an entrant *and* a constructor. In the knockout era the gap comes from the last session both drivers set a time in.
2. Gaps are divided by that season's typical teammate gap, so every era sits on one scale.
3. The whole teammate web is solved at once: per-driver-season ratings with a shared age curve, a random-walk career drift, rookie terms, a light per-season anchor and Huber-robust weighting. Thin drivers are **not** shrunk inside the fit, so a win over a weak teammate is only worth what that teammate proved elsewhere in the web.
4. Uncertainty is applied only when ranking, via empirical-Bayes shrinkage of each driver's peak/career estimate by its standard error.

## Build

```bash
pip install -r requirements.txt
python3 scripts/build_data.py          # downloads the latest F1DB release, writes site/data/
python3 -m http.server -d site 8000    # preview at http://localhost:8000
```

`.github/workflows/pages.yml` rebuilds the data from the latest [F1DB](https://github.com/f1db/f1db) release and deploys `site/` to GitHub Pages on every push and every Tuesday.

Data: F1DB, CC BY 4.0. Not affiliated with Formula 1.
