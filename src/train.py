from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from src.features import generate_features
except ModuleNotFoundError:  # pragma: no cover
    from features import generate_features


# This function prepares the data in a way that mirrors how model evaluation should work in real life.
# We split by time, not randomly, so the model is trained on earlier matches and tested on later ones.
def prepare_training_data(data_dir: str | Path, split_date: str | None = None):
    data = generate_features(data_dir)
    data = data.sort_values("Date").reset_index(drop=True)

    if split_date is None:
        split_date = data["Date"].sort_values().iloc[len(data) // 2]

    split_date = pd.Timestamp(split_date)
    train_df = data[data["Date"] < split_date].copy()
    test_df = data[data["Date"] >= split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Not enough data for a proper time-based split. "
            f"Date range: {data['Date'].min()} to {data['Date'].max()}"
        )

    # Keep only the actual model inputs. We remove labels, dates, and names.
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


# This is the baseline model: simple, interpretable, and useful as a benchmark.
def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
    )
    model.fit(X_train, y_train)
    return model


# This is the more powerful tree-based model we add after the baseline.
def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Install it with: pip install xgboost")

    model = XGBClassifier(
        # gives one probability per class for home/draw/away
        objective="multi:softprob",
        # evaluates probability quality using multiclass log loss
        eval_metric="mlogloss",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# This function compares the model to the bookmaker probabilities, which is the business-relevant benchmark.
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, test_df: pd.DataFrame) -> dict:
    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    bookmaker_probabilities = test_df[["away_implied_prob", "draw_implied_prob", "home_implied_prob"]].to_numpy()
    bookmaker_predictions = bookmaker_probabilities.argmax(axis=1).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "log_loss": log_loss(y_test, probabilities, labels=[0, 1, 2]),
        "bookmaker_accuracy": accuracy_score(y_test, bookmaker_predictions),
        "bookmaker_macro_f1": f1_score(y_test, bookmaker_predictions, average="macro"),
        "bookmaker_log_loss": log_loss(y_test, bookmaker_probabilities, labels=[0, 1, 2]),
    }

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metrics,
    }



# The main script runs both models and prints their results side-by-side.
def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    X_train, X_test, y_train, y_test, train_df, test_df = prepare_training_data(data_dir)

    # Baseline model
    logit_model = train_logistic_regression(X_train, y_train)
    logit_eval = evaluate_model(logit_model, X_test, y_test, test_df)

    # Stronger model
    xgb_model = train_xgboost(X_train, y_train)
    xgb_eval = evaluate_model(xgb_model, X_test, y_test, test_df)

    print("Training rows:", len(train_df))
    print("Test rows:", len(test_df))
    print()
    print("Logistic Regression")
    print("Accuracy:", round(logit_eval["metrics"]["accuracy"], 4))
    print("Macro F1:", round(logit_eval["metrics"]["macro_f1"], 4))
    print("Log loss:", round(logit_eval["metrics"]["log_loss"], 4))
    print()
    print("Bookmaker benchmark")
    print("Bookmaker accuracy:", round(logit_eval["metrics"]["bookmaker_accuracy"], 4))
    print("Bookmaker macro F1:", round(logit_eval["metrics"]["bookmaker_macro_f1"], 4))
    print("Bookmaker log loss:", round(logit_eval["metrics"]["bookmaker_log_loss"], 4))
    print()
    print("XGBoost")
    print("Accuracy:", round(xgb_eval["metrics"]["accuracy"], 4))
    print("Macro F1:", round(xgb_eval["metrics"]["macro_f1"], 4))
    print("Log loss:", round(xgb_eval["metrics"]["log_loss"], 4))
    




if __name__ == "__main__":
    main()
