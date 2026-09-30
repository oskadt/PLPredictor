from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

try:
    from src.features import generate_features
except ModuleNotFoundError:  # pragma: no cover
    from features import generate_features


def _to_markdown_table(df: pd.DataFrame) -> str:
    table = df.copy()
    if table.empty:
        return "| no rows |"

    headers = [str(col) for col in table.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rows)


def prepare_training_data(data_dir: str | Path):
    data = generate_features(data_dir)
    n_raw = len(data)
    data = data.dropna(subset=["home_implied_prob", "draw_implied_prob", "away_implied_prob"]).copy()
    dropped_odds = n_raw - len(data)
    if dropped_odds:
        print(f"Warning: dropped {dropped_odds} rows with missing bookmaker odds")

    data = data.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="mergesort").reset_index(drop=True)
    season_order = data.groupby("season")["Date"].min().sort_values().index.tolist()
    if len(season_order) < 2:
        raise ValueError("Need at least two seasons to create a proper time-based split.")
    split_season = season_order[len(season_order) // 2]
    train_df = data[data["season"] < split_season].copy()
    test_df = data[data["season"] >= split_season].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Not enough data for a proper time-based split. "
            f"Date range: {data['Date'].min()} to {data['Date'].max()}"
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

    return X_train, X_test, y_train, y_test, train_df, test_df, dropped_odds


def _prepare_logistic_inputs(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_medians = X_train.median(numeric_only=True)
    return X_train.fillna(train_medians), X_test.fillna(train_medians)


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(solver="lbfgs", max_iter=5000)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def compute_bookmaker_metrics(y_true: pd.Series, bookmaker_probabilities: np.ndarray) -> dict:
    bookmaker_predictions = bookmaker_probabilities.argmax(axis=1).astype(int)
    return {
        "predictions": bookmaker_predictions,
        "probabilities": bookmaker_probabilities,
        "accuracy": accuracy_score(y_true, bookmaker_predictions),
        "macro_f1": f1_score(y_true, bookmaker_predictions, average="macro"),
        "log_loss": log_loss(y_true, bookmaker_probabilities, labels=[0, 1, 2]),
    }


def paired_bootstrap(y, p_a, p_b, n=2000, seed=42):
    y = np.asarray(y)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)

    assert p_a.shape == p_b.shape, "Probability matrices must have the same shape."
    assert p_a.shape[0] == len(y), "Probability rows must match the number of labels."
    assert p_a.shape[1] == 3, "Probability matrices must have three columns for home, draw, away."
    assert np.allclose(p_a.sum(axis=1), 1.0, atol=1e-5), "p_a rows must sum to 1."
    assert np.allclose(p_b.sum(axis=1), 1.0, atol=1e-5), "p_b rows must sum to 1."
    assert np.all(p_a > 0), "p_a must contain strictly positive probabilities."
    assert np.all(p_b > 0), "p_b must contain strictly positive probabilities."
    assert set(np.unique(y)).issubset({0, 1, 2}), "y labels must be in {0, 1, 2}."

    idx = np.arange(len(y))
    loss_a = -np.log(p_a[idx, y])
    loss_b = -np.log(p_b[idx, y])
    diff = loss_a - loss_b

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n)
    for i in range(n):
        sample = rng.choice(idx, size=len(idx), replace=True)
        boot_means[i] = diff[sample].mean()

    return loss_a.mean(), loss_b.mean(), diff.mean(), np.percentile(boot_means, [2.5, 97.5])


def evaluate_class_prior_baseline(y_train: pd.Series, y_test: pd.Series) -> dict:
    prior_class = int(y_train.value_counts().idxmax())
    predictions = np.repeat(prior_class, len(y_test))
    class_probabilities = y_train.value_counts(normalize=True).reindex([0, 1, 2], fill_value=0.0).to_numpy()
    probability_matrix = np.tile(class_probabilities, (len(y_test), 1))

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "log_loss": log_loss(y_test, probability_matrix, labels=[0, 1, 2]),
    }

    return {
        "predictions": predictions,
        "probabilities": probability_matrix,
        "metrics": metrics,
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, test_df: pd.DataFrame, bookmaker_probabilities: np.ndarray | None = None) -> dict:
    assert hasattr(model, "classes_"), "Model must expose classes_ after fitting."
    assert list(model.classes_) == [0, 1, 2], "Model classes are not in the expected home/draw/away order."

    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    if bookmaker_probabilities is None:
        bookmaker_probabilities = test_df[["away_implied_prob", "draw_implied_prob", "home_implied_prob"]].to_numpy(dtype=float)
    bookmaker_metrics = compute_bookmaker_metrics(y_test, bookmaker_probabilities)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "log_loss": log_loss(y_test, probabilities, labels=[0, 1, 2]),
        "bookmaker_accuracy": bookmaker_metrics["accuracy"],
        "bookmaker_macro_f1": bookmaker_metrics["macro_f1"],
        "bookmaker_log_loss": bookmaker_metrics["log_loss"],
    }

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metrics,
    }


def evaluate_feature_subset(model_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, test_df: pd.DataFrame, feature_columns: list[str], bookmaker_probabilities: np.ndarray | None = None):
    X_train_subset = X_train[feature_columns].copy()
    X_test_subset = X_test[feature_columns].copy()

    if model_name == "logistic":
        X_train_subset, X_test_subset = _prepare_logistic_inputs(X_train_subset, X_test_subset)
        model = train_logistic_regression(X_train_subset, y_train)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return evaluate_model(model, X_test_subset, y_test, test_df, bookmaker_probabilities=bookmaker_probabilities)


def _summarise_metrics(metrics: dict) -> dict:
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "log_loss": metrics["log_loss"],
    }


def _train_xgboost_with_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_train_full: pd.DataFrame | None = None,
    y_train_full: pd.Series | None = None,
):
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Install it with: pip install xgboost")

    param_grid = {
        "learning_rate": [0.03, 0.05],
        "max_depth": [2, 3],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 2, 4],
    }

    best_score = np.inf
    best_params = None
    best_n_estimators = None

    for learning_rate in param_grid["learning_rate"]:
        for max_depth in param_grid["max_depth"]:
            for subsample in param_grid["subsample"]:
                for colsample_bytree in param_grid["colsample_bytree"]:
                    for min_child_weight in param_grid["min_child_weight"]:
                        candidate = XGBClassifier(
                            objective="multi:softprob",
                            eval_metric="mlogloss",
                            n_estimators=500,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            min_child_weight=min_child_weight,
                            random_state=42,
                        )
                        candidate.fit(
                            X_train,
                            y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False,
                        )
                        validation_loss = candidate.evals_result_["validation_0"]["mlogloss"]
                        current_best_n = int(np.argmin(validation_loss) + 1)

                        candidate_best = XGBClassifier(
                            objective="multi:softprob",
                            eval_metric="mlogloss",
                            n_estimators=current_best_n,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            min_child_weight=min_child_weight,
                            random_state=42,
                        )
                        candidate_best.fit(X_train, y_train)
                        val_predictions = candidate_best.predict(X_val)
                        val_probabilities = candidate_best.predict_proba(X_val)
                        val_log_loss = log_loss(y_val, val_probabilities, labels=[0, 1, 2])
                        val_macro_f1 = f1_score(y_val, val_predictions, average="macro")
                        current_score = val_log_loss - val_macro_f1

                        if current_score < best_score:
                            best_score = current_score
                            best_params = {
                                "learning_rate": learning_rate,
                                "max_depth": max_depth,
                                "subsample": subsample,
                                "colsample_bytree": colsample_bytree,
                                "min_child_weight": min_child_weight,
                            }
                            best_n_estimators = current_best_n

    refit_X = X_train_full if X_train_full is not None else X_train
    refit_y = y_train_full if y_train_full is not None else y_train

    refit_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=best_n_estimators,
        **best_params,
        random_state=42,
    )
    refit_model.fit(refit_X, refit_y)
    return refit_model


def _build_comparison_table(model_name: str, model_metrics: dict) -> dict:
    return {
        "model": model_name,
        "accuracy": model_metrics["accuracy"],
        "macro_f1": model_metrics["macro_f1"],
        "log_loss": model_metrics["log_loss"],
    }


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    X_train, X_test, y_train, y_test, train_df, test_df, dropped_odds = prepare_training_data(data_dir)
    bookmaker_probabilities = test_df[["away_implied_prob", "draw_implied_prob", "home_implied_prob"]].to_numpy(dtype=float)

    all_columns = list(X_train.columns)
    form_columns = [col for col in all_columns if "implied_prob" not in col]
    odds_columns = [col for col in all_columns if "implied_prob" in col]

    logistic_runs = {}
    for name, columns in {"form": form_columns, "odds": odds_columns, "both": all_columns}.items():
        logistic_runs[name] = evaluate_feature_subset(
            "logistic",
            X_train,
            X_test,
            y_train,
            y_test,
            test_df,
            columns,
            bookmaker_probabilities=bookmaker_probabilities,
        )

    class_prior = evaluate_class_prior_baseline(y_train, y_test)

    season_order = train_df.groupby("season")["Date"].min().sort_values().index.tolist()
    validation_season = season_order[-1]
    validation_mask = train_df["season"] == validation_season
    X_train_tune = X_train.loc[~validation_mask].copy()
    y_train_tune = y_train.loc[~validation_mask].copy()
    X_val = X_train.loc[validation_mask].copy()
    y_val = y_train.loc[validation_mask].copy()

    xgb_runs = {}
    for name, columns in {"form": form_columns, "odds": odds_columns, "both": all_columns}.items():
        xgb_model = _train_xgboost_with_validation(
            X_train_tune[columns].copy(),
            y_train_tune.copy(),
            X_val[columns].copy(),
            y_val.copy(),
            X_train_full=X_train[columns].copy(),
            y_train_full=y_train.copy(),
        )
        xgb_runs[name] = evaluate_model(
            xgb_model,
            X_test[columns].copy(),
            y_test,
            test_df,
            bookmaker_probabilities=bookmaker_probabilities,
        )

    summary_rows = [
        {"model": "class_prior_baseline", **_summarise_metrics(class_prior["metrics"])},
        {"model": "form_only_logreg", **_summarise_metrics(logistic_runs["form"]["metrics"])},
        {"model": "odds_only_logreg", **_summarise_metrics(logistic_runs["odds"]["metrics"])},
        {"model": "logistic_regression", **_summarise_metrics(logistic_runs["both"]["metrics"])},
        {"model": "form_only_xgboost", **_summarise_metrics(xgb_runs["form"]["metrics"])},
        {"model": "odds_only_xgboost", **_summarise_metrics(xgb_runs["odds"]["metrics"])},
        {"model": "xgboost", **_summarise_metrics(xgb_runs["both"]["metrics"])},
        {
            "model": "bookmaker",
            "accuracy": accuracy_score(y_test, bookmaker_probabilities.argmax(axis=1).astype(int)),
            "macro_f1": f1_score(y_test, bookmaker_probabilities.argmax(axis=1).astype(int), average="macro"),
            "log_loss": log_loss(y_test, bookmaker_probabilities, labels=[0, 1, 2]),
        },
    ]

    summary_df = pd.DataFrame(summary_rows)
    output_dir = Path(__file__).resolve().parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "model_results.csv"
    summary_df.to_csv(output_path, index=False)

    best_learned_label, best_learned_model = min(
        [
            ("form_only_logreg", logistic_runs["form"]),
            ("odds_only_logreg", logistic_runs["odds"]),
            ("logistic_regression", logistic_runs["both"]),
            ("form_only_xgboost", xgb_runs["form"]),
            ("odds_only_xgboost", xgb_runs["odds"]),
            ("xgboost", xgb_runs["both"]),
        ],
        key=lambda item: item[1]["metrics"]["log_loss"],
    )

    y = y_test.to_numpy()
    comparisons = {
        "form_logreg_vs_prior": (logistic_runs["form"]["probabilities"], class_prior["probabilities"]),
        "both_logreg_vs_odds_only_logreg": (logistic_runs["both"]["probabilities"], logistic_runs["odds"]["probabilities"]),
        "both_logreg_vs_bookmaker": (logistic_runs["both"]["probabilities"], bookmaker_probabilities),
        "odds_only_logreg_vs_bookmaker": (logistic_runs["odds"]["probabilities"], bookmaker_probabilities),
        "xgboost_vs_logistic_regression": (xgb_runs["both"]["probabilities"], logistic_runs["both"]["probabilities"]),
        "xgboost_vs_bookmaker": (xgb_runs["both"]["probabilities"], bookmaker_probabilities),
    }

    bootstrap_rows = []
    for name, (pa, pb) in comparisons.items():
        la, lb, diff, ci = paired_bootstrap(y, pa, pb)
        bootstrap_rows.append(
            {
                "comparison": name,
                "loss_a": la,
                "loss_b": lb,
                "diff": diff,
                "ci_low": ci[0],
                "ci_high": ci[1],
            }
        )

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_output = output_dir / "bootstrap.csv"
    bootstrap_df.to_csv(bootstrap_output, index=False)

    season_probabilities = best_learned_model["probabilities"]
    season_rows = []
    for season in sorted(test_df["season"].unique()):
        season_mask = test_df["season"] == season
        season_y = y_test.loc[season_mask].to_numpy()
        season_model_probs = season_probabilities[season_mask.to_numpy()]
        season_book_probs = bookmaker_probabilities[season_mask.to_numpy()]
        season_rows.append(
            {
                "season": season,
                "best_model_log_loss": log_loss(season_y, season_model_probs, labels=[0, 1, 2]),
                "bookmaker_log_loss": log_loss(season_y, season_book_probs, labels=[0, 1, 2]),
            }
        )
    season_logloss_df = pd.DataFrame(season_rows)
    season_logloss_path = output_dir / "season_logloss.csv"
    season_logloss_df.to_csv(season_logloss_path, index=False)

    complete_form_mask = X_test[form_columns].notna().all(axis=1)
    if complete_form_mask.any():
        complete_form_model = train_logistic_regression(
            _prepare_logistic_inputs(X_train[form_columns].copy(), X_test[form_columns].copy())[0],
            y_train,
        )
        complete_form_eval = evaluate_model(
            complete_form_model,
            X_test[form_columns].loc[complete_form_mask].copy(),
            y_test.loc[complete_form_mask].copy(),
            test_df.loc[complete_form_mask].copy(),
            bookmaker_probabilities=bookmaker_probabilities[complete_form_mask.to_numpy()],
        )
        complete_form_prior = evaluate_class_prior_baseline(y_train, y_test.loc[complete_form_mask].copy())
        robustness_result = complete_form_eval["metrics"]["log_loss"] < complete_form_prior["metrics"]["log_loss"]
    else:
        robustness_result = None

    print("Training rows:", len(train_df))
    print("Test rows:", len(test_df))
    print()
    print(_to_markdown_table(summary_df.round(4)))
    print()
    print(_to_markdown_table(bootstrap_df.round(4)))
    print()
    print(_to_markdown_table(season_logloss_df.round(4)))

    if robustness_result is not None:
        print(f"\nComplete-form robustness: form-logreg beats class prior on complete-form subset = {robustness_result}")
    else:
        print("\nNo rows with complete form features in the test set.")

    if "2020-21" in set(test_df["season"].astype(str)):
        print("\nNote: the test split includes the 2020-21 season, which is relevant because it was a no-crowd season.")

    print(f"\nSaved summary table to: {output_path}")
    print(f"Saved bootstrap table to: {bootstrap_output}")
    print(f"Saved per-season log-loss table to: {season_logloss_path}")

    if robustness_result is not None:
        print(
            "\nRobustness check: "
            f"form-only logistic regression on complete-form rows has log_loss {complete_form_eval['metrics']['log_loss']:.4f} "
            f"vs class-prior baseline {complete_form_prior['metrics']['log_loss']:.4f}."
        )


if __name__ == "__main__":
    main()
