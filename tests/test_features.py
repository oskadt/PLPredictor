from pathlib import Path

import pandas as pd

from src.features import generate_features
from src.train import evaluate_model


def _write_match_csv(path: Path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_generate_features_builds_team_form_and_result_columns(tmp_path):
    rows_2022 = [
        {
            "Div": "E0",
            "Date": "01/08/2022",
            "Time": "12:30",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Liverpool",
            "FTHG": 2,
            "FTAG": 1,
            "FTR": "H",
            "B365H": 1.8,
            "B365D": 3.5,
            "B365A": 4.5,
        },
        {
            "Div": "E0",
            "Date": "08/08/2022",
            "Time": "15:00",
            "HomeTeam": "Everton",
            "AwayTeam": "Arsenal",
            "FTHG": 1,
            "FTAG": 1,
            "FTR": "D",
            "B365H": 2.5,
            "B365D": 3.2,
            "B365A": 2.7,
        },
        {
            "Div": "E0",
            "Date": "15/08/2022",
            "Time": "17:30",
            "HomeTeam": "Liverpool",
            "AwayTeam": "Everton",
            "FTHG": 3,
            "FTAG": 0,
            "FTR": "H",
            "B365H": 1.4,
            "B365D": 5.0,
            "B365A": 7.5,
        },
    ]
    rows_2023 = [
        {
            "Div": "E0",
            "Date": "01/08/2023",
            "Time": "12:30",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "FTHG": 1,
            "FTAG": 0,
            "FTR": "H",
            "B365H": 1.7,
            "B365D": 3.7,
            "B365A": 5.5,
        }
    ]

    _write_match_csv(tmp_path / "2022.csv", rows_2022)
    _write_match_csv(tmp_path / "2023.csv", rows_2023)

    features = generate_features(tmp_path)

    assert {"home_team", "away_team", "result"}.issubset(features.columns)
    assert {"home_last_5_points", "away_last_5_points", "home_last_5_goals_for", "away_last_5_goals_for"}.issubset(features.columns)
    assert features["result"].isin([0, 1, 2]).all()
    assert features["home_last_5_points"].notna().all()
    assert features["away_last_5_points"].notna().all()


def test_generate_features_handles_mixed_date_formats(tmp_path):
    rows = [
        {
            "Div": "E0",
            "Date": "2022-08-01",
            "Time": "12:30",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Liverpool",
            "FTHG": 2,
            "FTAG": 1,
            "FTR": "H",
            "B365H": 1.8,
            "B365D": 3.5,
            "B365A": 4.5,
        },
        {
            "Div": "E0",
            "Date": "02/08/2022",
            "Time": "15:00",
            "HomeTeam": "Liverpool",
            "AwayTeam": "Arsenal",
            "FTHG": 1,
            "FTAG": 1,
            "FTR": "D",
            "B365H": 2.5,
            "B365D": 3.2,
            "B365A": 2.7,
        },
    ]

    _write_match_csv(tmp_path / "2022.csv", rows)

    features = generate_features(tmp_path)

    assert features["Date"].notna().all()
    assert len(features) == 2


def test_evaluate_model_includes_bookmaker_accuracy_and_f1():
    test_df = pd.DataFrame(
        {
            "away_implied_prob": [0.5, 0.2],
            "draw_implied_prob": [0.3, 0.5],
            "home_implied_prob": [0.2, 0.3],
        }
    )
    X_test = pd.DataFrame({"feature_1": [1, 2]})
    y_test = pd.Series([0, 1])

    class DummyModel:
        def predict_proba(self, X):
            return [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]

        def predict(self, X):
            return [0, 1]

    result = evaluate_model(DummyModel(), X_test, y_test, test_df)

    assert "bookmaker_accuracy" in result["metrics"]
    assert "bookmaker_macro_f1" in result["metrics"]
    assert 0 <= result["metrics"]["bookmaker_accuracy"] <= 1
    assert 0 <= result["metrics"]["bookmaker_macro_f1"] <= 1
