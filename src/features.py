from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _parse_date(value: object) -> pd.Timestamp | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y"]

    for fmt in formats:
        try:
            return pd.to_datetime(text, format=fmt)
        except (TypeError, ValueError):
            continue

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    return parsed if pd.notna(parsed) else None


def load_match_data(data_dir: str | Path) -> pd.DataFrame:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(data_path.glob("*.csv")):
        season_df = pd.read_csv(csv_path)
        season_df = season_df.copy()
        season_df["season"] = csv_path.stem
        frames.append(season_df)

    if not frames:
        raise ValueError(f"No CSV files found in: {data_path}")

    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)
    df["Date"] = df["Date"].map(_parse_date)
    df = df.dropna(subset=["Date"]).copy()
    if len(df) != n_raw:
        print(f"Warning: dropped {n_raw - len(df)} rows with unparseable dates")

    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="mergesort").reset_index(drop=True)
    return df


def _build_team_history(df: pd.DataFrame, window: int) -> pd.DataFrame:
    # Shift(1) ensures that a match never uses its own result when building rolling form.
    home_rows = df[["season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].rename(
        columns={
            "HomeTeam": "team",
            "AwayTeam": "opponent",
            "FTHG": "goals_for",
            "FTAG": "goals_against",
        }
    )
    home_rows["is_home"] = 1

    away_rows = df[["season", "Date", "AwayTeam", "HomeTeam", "FTAG", "FTHG"]].rename(
        columns={
            "AwayTeam": "team",
            "HomeTeam": "opponent",
            "FTAG": "goals_for",
            "FTHG": "goals_against",
        }
    )
    away_rows["is_home"] = 0

    team_history = pd.concat([home_rows, away_rows], ignore_index=True, sort=False)
    team_history = team_history.assign(
        points=(team_history["goals_for"] > team_history["goals_against"]).astype(int) * 3
        + (
            team_history["goals_for"].eq(team_history["goals_against"]) & team_history["goals_for"].notna()
        ).astype(int)
    )
    team_history = team_history.sort_values(["team", "season", "Date"]).reset_index(drop=True)

    rolling_features = {}
    for column, agg in [("points", "mean"), ("goals_for", "mean"), ("goals_against", "mean")]:
        rolling_features[f"last_{window}_{column}"] = (
            team_history.groupby(["team", "season"], group_keys=False)[column]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).agg(agg))
        )

    team_history = team_history.assign(**rolling_features)
    return team_history


def _add_bookmaker_features(df: pd.DataFrame) -> pd.DataFrame:
    odds_columns = ["B365H", "B365D", "B365A"]
    implied_columns = {}
    for column in odds_columns:
        if column not in df.columns:
            continue

        implied_columns[f"{column.lower()}_implied"] = (
            1.0 / df[column].replace(0, np.nan).astype(float)
        )

    df = df.assign(**implied_columns)

    if {"b365h_implied", "b365d_implied", "b365a_implied"}.issubset(df.columns):
        implied_total = df[["b365h_implied", "b365d_implied", "b365a_implied"]].sum(axis=1).replace(0, np.nan)
        df = df.assign(
            home_implied_prob=df["b365h_implied"].astype(float) / implied_total.astype(float),
            draw_implied_prob=df["b365d_implied"].astype(float) / implied_total.astype(float),
            away_implied_prob=df["b365a_implied"].astype(float) / implied_total.astype(float),
        )

    return df


def generate_features(data_dir: str | Path, window: int = 5) -> pd.DataFrame:
    df = load_match_data(data_dir)
    team_history = _build_team_history(df, window=window)

    home_stats = team_history[
        ["team", "season", "Date", f"last_{window}_points", f"last_{window}_goals_for", f"last_{window}_goals_against"]
    ].rename(
        columns={
            "team": "HomeTeam",
            "season": "season",
            f"last_{window}_points": f"home_last_{window}_points",
            f"last_{window}_goals_for": f"home_last_{window}_goals_for",
            f"last_{window}_goals_against": f"home_last_{window}_goals_against",
        }
    )
    away_stats = team_history[
        ["team", "season", "Date", f"last_{window}_points", f"last_{window}_goals_for", f"last_{window}_goals_against"]
    ].rename(
        columns={
            "team": "AwayTeam",
            "season": "season",
            f"last_{window}_points": f"away_last_{window}_points",
            f"last_{window}_goals_for": f"away_last_{window}_goals_for",
            f"last_{window}_goals_against": f"away_last_{window}_goals_against",
        }
    )

    n_before = len(df)
    df = df.merge(home_stats, on=["HomeTeam", "season", "Date"], how="left")
    assert len(df) == n_before, "Merge on HomeTeam/season/Date duplicated rows"

    n_before = len(df)
    df = df.merge(away_stats, on=["AwayTeam", "season", "Date"], how="left")
    assert len(df) == n_before, "Merge on AwayTeam/season/Date duplicated rows"

    df = _add_bookmaker_features(df)

    df = df.assign(
        result=df["FTR"].map({"H": 2, "D": 1, "A": 0}),
        target=df["FTR"],
        home_team=df["HomeTeam"],
        away_team=df["AwayTeam"],
    )

    feature_columns = [
        "season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "home_team",
        "away_team",
        f"home_last_{window}_points",
        f"away_last_{window}_points",
        f"home_last_{window}_goals_for",
        f"away_last_{window}_goals_for",
        f"home_last_{window}_goals_against",
        f"away_last_{window}_goals_against",
        "home_implied_prob",
        "draw_implied_prob",
        "away_implied_prob",
        "result",
        "target",
    ]

    return df[feature_columns].copy()
