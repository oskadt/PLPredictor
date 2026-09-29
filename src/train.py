from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

try:
    from src.features import generate_features
except ModuleNotFoundError:  # pragma: no cover
    from features import generate_features


def prepare_training_data(data_dir: str | Path, split_date: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a time-based train/test split for the football prediction dataset."""
    data = generate_features(data_dir)
    data = data.sort_values("Date").reset_index(drop=True)

    if split_date is None:
        split_date = data["Date"].sort_values().iloc[len(data) // 2]

    split_date = pd.Timestamp(split_date)
    train_df = data[data["Date"] < split_date].copy()
    test_df = data[data["Date"] >= split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Not enough data for a valid split. Check the dataset range or pass a different split_date. "
            f"Min date: {data['Date'].min()}, max date: {data['Date'].max()}"
        )

    feature_columns = [
        col
        for col in train_df.columns
        if col not in {"season", "Date", "HomeTeam", "AwayTeam", "home_team", "away_team", "target", "result"}
    ]

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    y_train = train_df["result"].astype(int)
    y_test = test_df["result"].astype(int)

    return X_train, X_test, y_train, y_test, train_df, test_df


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Train a multiclass logistic regression model for home/draw/away prediction."""
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series, test_df: pd.DataFrame) -> dict:
    """Evaluate the model and compare it to bookmaker probabilities."""
    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    bookmaker_probabilities = test_df[["away_implied_prob", "draw_implied_prob", "home_implied_prob"]].to_numpy()

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "log_loss": log_loss(y_test, probabilities, labels=[0, 1, 2]),
        "bookmaker_log_loss": log_loss(y_test, bookmaker_probabilities, labels=[0, 1, 2]),
    }

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metrics,
    }


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    X_train, X_test, y_train, y_test, train_df, test_df = prepare_training_data(data_dir)

    model = train_logistic_regression(X_train, y_train)
    evaluation = evaluate_model(model, X_test, y_test, test_df)

    print("Training rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("Model accuracy:", round(evaluation["metrics"]["accuracy"], 4))
    print("Model macro F1:", round(evaluation["metrics"]["macro_f1"], 4))
    print("Model log loss:", round(evaluation["metrics"]["log_loss"], 4))
    print("Bookmaker log loss:", round(evaluation["metrics"]["bookmaker_log_loss"], 4))

    print("\nExample predictions:")
    for match_index, pred in enumerate(evaluation["predictions"][:5]):
        print(f"Match {match_index + 1}: predicted result {pred}")


if __name__ == "__main__":
    main()
