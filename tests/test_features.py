from pathlib import Path

import numpy as np
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
    assert features["home_last_5_points"].isna().all()
    assert features["away_last_5_points"].isna().all()


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
        {
            "Div": "E0",
            "Date": "05/08/00",
            "Time": "15:00",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "FTHG": 0,
            "FTAG": 2,
            "FTR": "A",
            "B365H": 1.7,
            "B365D": 3.6,
            "B365A": 4.2,
        },
    ]

    _write_match_csv(tmp_path / "2022.csv", rows)

    features = generate_features(tmp_path)

    parsed_dates = pd.to_datetime(features["Date"])
    assert parsed_dates.notna().all()
    assert len(features) == 3
    assert pd.Timestamp("2000-08-05") in parsed_dates.values
    assert pd.Timestamp("2022-08-02") in parsed_dates.values


def test_generate_features_does_not_use_current_match_result_for_its_own_features(tmp_path):
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
            "Date": "2022-08-08",
            "Time": "15:00",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "FTHG": 1,
            "FTAG": 0,
            "FTR": "H",
            "B365H": 1.9,
            "B365D": 3.4,
            "B365A": 4.0,
        },
        {
            "Div": "E0",
            "Date": "2022-08-15",
            "Time": "15:00",
            "HomeTeam": "Chelsea",
            "AwayTeam": "Liverpool",
            "FTHG": 1,
            "FTAG": 1,
            "FTR": "D",
            "B365H": 2.0,
            "B365D": 3.3,
            "B365A": 3.8,
        },
    ]

    base_dir = tmp_path / "base_dir"
    modified_dir = tmp_path / "modified_dir"
    base_dir.mkdir()
    modified_dir.mkdir()

    _write_match_csv(base_dir / "base.csv", rows)
    rows_modified = [dict(r) for r in rows]
    rows_modified[1]["FTR"] = "A"
    rows_modified[1]["FTHG"] = 0
    rows_modified[1]["FTAG"] = 2
    _write_match_csv(modified_dir / "modified.csv", rows_modified)

    baseline = generate_features(base_dir)
    modified = generate_features(modified_dir)

    baseline_row = baseline[baseline["HomeTeam"] == "Arsenal"].iloc[1]
    modified_row = modified[modified["HomeTeam"] == "Arsenal"].iloc[1]

    assert pd.isna(baseline_row["home_last_5_points"]) == pd.isna(modified_row["home_last_5_points"])
    assert pd.isna(baseline_row["away_last_5_points"]) == pd.isna(modified_row["away_last_5_points"])


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
        classes_ = np.array([0, 1, 2])

        def predict_proba(self, X):
            return [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]

        def predict(self, X):
            return [0, 1]

    result = evaluate_model(DummyModel(), X_test, y_test, test_df)

    assert "bookmaker_accuracy" in result["metrics"]
    assert "bookmaker_macro_f1" in result["metrics"]
    assert 0 <= result["metrics"]["bookmaker_accuracy"] <= 1
    assert 0 <= result["metrics"]["bookmaker_macro_f1"] <= 1
