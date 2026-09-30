# PLPredictor

Predicts Premier League match results (home/draw/away) from rolling team form and bookmaker odds, and tests whether either can beat the betting market.

## Findings

Evaluated on 1,950 held-out matches from the later seasons, trained on earlier seasons only. Metric: log loss (lower is better).

| Model | Log loss | Accuracy |
| --- | ---: | ---: |
| Class prior | 1.0695 | 43.9% |
| Form only (logistic) | 1.0355 | 48.6% |
| Odds only (logistic) | 0.9650 | 55.3% |
| Form + odds (logistic) | 0.9676 | 54.5% |
| XGBoost (form + odds) | 0.9789 | 53.1% |
| Bookmaker implied probs | 0.9627 | 54.9% |

- Recent form has real but modest signal: log loss is 0.0339 below the class prior (95% paired-bootstrap CI [-0.0476, -0.0219]).
- Form adds little once odds are included (+0.0025, CI [-0.0006, 0.0058]).
- No model beat the bookmaker. The odds-only model is statistically indistinguishable from it (+0.0023, CI [-0.0002, 0.0049]), which is what you would expect if the market already prices in recent form. Form + odds logistic regression is slightly but significantly worse (+0.0048, CI [0.0006, 0.0091]).
- XGBoost was significantly worse than logistic regression (+0.0114, CI [0.0046, 0.0182]) and than the bookmaker (+0.0162, CI [0.0096, 0.0232]) on this small feature set. With ~1,900 training matches and three noisy features, the extra flexibility mostly fits noise.

Full tables: [results/model_results.csv](results/model_results.csv), [results/bootstrap.csv](results/bootstrap.csv), [results/season_logloss.csv](results/season_logloss.csv). The test set includes 50 matches from the incomplete 2026-27 season.

## Method

- Data: football-data.co.uk season CSVs across 11 seasons.
- Features: mean points, goals for and goals against over each team's last 5 matches, using shift(1) so a match never sees its own result. Form resets each season. Bet365 odds are converted to normalised implied probabilities.
- Split: chronological by season. Training seasons come first; the last training season is held out as validation. It is used to tune XGBoost (grid search on validation log loss, including the number of trees) and to pick the "best learned model" for the per-season table; the final models are then refit on all training seasons. The test set is never used for selection.
- Tests: a match's result must not change its own or earlier matches' features; early-season matches have no form; form does not carry across seasons.

## Limitations

- Single chronological split, no walk-forward retraining.
- The first 5 matches of each team-season have no form; these are median-imputed for logistic regression, which affects roughly 13% of matches.
- Bootstrap CIs treat matches as independent; real uncertainty is likely a bit larger. Several comparisons were run without correction.
- XGBoost was tuned on a single validation season over a small grid (learning rate, depth, subsample, column sampling, min child weight), so the tuning itself is noisy.
- The validation-selected "best" model (XGBoost) is not the best model on the test set; the per-season table reports it as chosen, not as the test-set winner.

## Run it

```bash
pip install -r requirements.txt
python -m src.train   # ~1 min; regenerates results/
python -m pytest
```

For the interactive demo:

```bash
streamlit run app.py
```

## Project structure

- [src/features.py](src/features.py): feature generation and time-safe historical inputs.
- [src/train.py](src/train.py): model training, tuning, evaluation, bootstrap comparisons, and exports.
- [src/predict.py](src/predict.py): fits the final models and returns held-out predictions for the demo app.
- [app.py](app.py): Streamlit demo.
- [tests/](tests/): tests for date parsing, leakage, odds normalisation, the bootstrap, and metric helpers. Run in CI via GitHub Actions.
- [results/xgb_config.json](results/xgb_config.json): tuned XGBoost settings, reused by the demo.
- [results/model_results.csv](results/model_results.csv): benchmark table for the main models.
- [results/bootstrap.csv](results/bootstrap.csv): paired bootstrap comparison between model and market performance.
- [results/season_logloss.csv](results/season_logloss.csv): per-season log-loss comparison.

## Notes

This project is intentionally framed as a realistic forecasting exercise rather than a claim that a machine learning model can reliably beat the market. The most useful takeaway is not the single headline metric, but the way the project handles data leakage, season-aware evaluation, and honest benchmark comparison.

