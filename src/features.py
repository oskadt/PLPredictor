from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _parse_date(value: object) -> pd.Timestamp | pd.NaT:
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%m-%d-%Y"]

    for fmt in formats:
        try:
            return pd.to_datetime(text, format=fmt)
        except (TypeError, ValueError):
            continue

    return pd.to_datetime(text, errors="coerce")


def load_match_data(data_dir: str | Path) -> pd.DataFrame:
    # Load all CSV season files from a data directory into one dataframe.
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
    df["Date"] = df["Date"].map(_parse_date)
    df = df.dropna(subset=["Date"]).copy()
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="mergesort").reset_index(drop=True)
    return df


def _build_team_history(df: pd.DataFrame, window: int) -> pd.DataFrame:
    # Create rolling team-level stats using only information available before each match.
    home_rows = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].rename(
        columns={
            "HomeTeam": "team",
            "AwayTeam": "opponent",
            "FTHG": "goals_for",
            "FTAG": "goals_against",
        }
    )
    home_rows["is_home"] = 1

    away_rows = df[["Date", "AwayTeam", "HomeTeam", "FTAG", "FTHG"]].rename(
        columns={
            "AwayTeam": "team",
            "HomeTeam": "opponent",
            "FTAG": "goals_for",
            "FTHG": "goals_against",
        }
    )
    away_rows["is_home"] = 0

    team_history = pd.concat([home_rows, away_rows], ignore_index=True, sort=False)
    # 3 points for win, 1 for draw, 0 for loss, NaN for unplayed
    team_history = team_history.assign(
        points=(team_history["goals_for"] > team_history["goals_against"]).astype(int) * 3
        + (
            team_history["goals_for"].eq(team_history["goals_against"]) & team_history["goals_for"].notna()
        ).astype(int)
    )
    team_history = team_history.sort_values(["team", "Date"]).reset_index(drop=True)

    rolling_features = {}
    for column, agg in [("points", "sum"), ("goals_for", "mean"), ("goals_against", "mean")]:
        rolling_features[f"last_{window}_{column}"] = (
            team_history.groupby("team")[column]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).agg(agg))
        )

    team_history = team_history.assign(**rolling_features)

    return team_history


def _add_bookmaker_features(df: pd.DataFrame) -> pd.DataFrame:
    # Convert bookmaker decimal odds into normalised implied probabilities.
    odds_columns = ["B365H", "B365D", "B365A"]
    implied_columns = {}
    for column in odds_columns:
        if column not in df.columns:
            continue

        implied_columns[f"{column.lower()}_implied"] = 1.0 / df[column].replace(0, pd.NA)

    df = df.assign(**implied_columns)

    if {"b365h_implied", "b365d_implied", "b365a_implied"}.issubset(df.columns):
        implied_total = (
            df[["b365h_implied", "b365d_implied", "b365a_implied"]].sum(axis=1).replace(0, pd.NA)
        )
        df = df.assign(
            home_implied_prob=df["b365h_implied"] / implied_total,
            draw_implied_prob=df["b365d_implied"] / implied_total,
            away_implied_prob=df["b365a_implied"] / implied_total,
        )

    return df


def generate_features(data_dir: str | Path, window: int = 5) -> pd.DataFrame:
    # Generate leakage-free rolling-form and bookmaker features for all matches.
    df = load_match_data(data_dir)
    team_history = _build_team_history(df, window=window)

    home_stats = team_history[
        ["team", "Date", f"last_{window}_points", f"last_{window}_goals_for", f"last_{window}_goals_against"]
    ].rename(
        columns={
            "team": "HomeTeam",
            f"last_{window}_points": f"home_last_{window}_points",
            f"last_{window}_goals_for": f"home_last_{window}_goals_for",
            f"last_{window}_goals_against": f"home_last_{window}_goals_against",
        }
    )
    away_stats = team_history[
        ["team", "Date", f"last_{window}_points", f"last_{window}_goals_for", f"last_{window}_goals_against"]
    ].rename(
        columns={
            "team": "AwayTeam",
            f"last_{window}_points": f"away_last_{window}_points",
            f"last_{window}_goals_for": f"away_last_{window}_goals_for",
            f"last_{window}_goals_against": f"away_last_{window}_goals_against",
        }
    )

    df = df.merge(home_stats, on=["HomeTeam", "Date"], how="left")
    df = df.merge(away_stats, on=["AwayTeam", "Date"], how="left")

    for column in [
        f"home_last_{window}_points",
        f"away_last_{window}_points",
        f"home_last_{window}_goals_for",
        f"away_last_{window}_goals_for",
        f"home_last_{window}_goals_against",
        f"away_last_{window}_goals_against",
    ]:
        df[column] = df[column].fillna(0)

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
