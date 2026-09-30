from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import generate_features

ODDS_COLUMNS = ["away_implied_prob", "draw_implied_prob", "home_implied_prob"]
NON_FEATURE_COLUMNS = {"season", "Date", "HomeTeam", "AwayTeam", "home_team", "away_team", "target", "result"}
PARAM_GRID = {
    "learning_rate": [0.03, 0.05],
    "max_depth": [2, 3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 2, 4],
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "| no rows |"

    headers = [str(col) for col in df.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in df.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rows)


def load_model_data(data_dir: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Features for every match with odds, in chronological order, plus the season order."""
    data = generate_features(data_dir)
    n_raw = len(data)
    data = data.dropna(subset=ODDS_COLUMNS).copy()
    if n_raw != len(data):
        print(f"Warning: dropped {n_raw - len(data)} rows with missing bookmaker odds")

    data = data.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="mergesort").reset_index(drop=True)
    season_order = data.groupby("season")["Date"].min().sort_values().index.tolist()
    if len(season_order) < 2:
        raise ValueError("Need at least two seasons to create a proper time-based split.")
    return data, season_order


def prepare_training_data(data_dir: str | Path):
    data, season_order = load_model_data(data_dir)
    split_season = season_order[len(season_order) // 2]
    train_seasons = season_order[: len(season_order) // 2]
    train_df = data[data["season"].isin(train_seasons)].copy()
    test_df = data[~data["season"].isin(train_seasons)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Not enough data for a time-based split at {split_season}. "
            f"Date range: {data['Date'].min()} to {data['Date'].max()}"
        )

    feature_columns = [col for col in train_df.columns if col not in NON_FEATURE_COLUMNS]
    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    y_train = train_df["result"].astype(int)
    y_test = test_df["result"].astype(int)

    return X_train, X_test, y_train, y_test, train_df, test_df


def impute_with_train_medians(X_train: pd.DataFrame, *others: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    medians = X_train.median(numeric_only=True)
    return tuple(frame.fillna(medians) for frame in (X_train, *others))


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(solver="lbfgs", max_iter=5000)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def make_xgboost(n_estimators: int, **params) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=n_estimators,
        random_state=42,
        **params,
    )


def tune_xgboost(X_tune, y_tune, X_val, y_val) -> tuple[dict, int, float]:
    """Grid search on validation log loss. Returns (params, best n_estimators, validation log loss)."""
    best_loss, best_params, best_n = np.inf, None, None

    for values in itertools.product(*PARAM_GRID.values()):
        params = dict(zip(PARAM_GRID, values))
        candidate = make_xgboost(500, **params)
        candidate.fit(X_tune, y_tune, eval_set=[(X_val, y_val)], verbose=False)
        val_curve = candidate.evals_result_["validation_0"]["mlogloss"]
        loss = float(np.min(val_curve))
        if loss < best_loss:
            best_loss, best_params, best_n = loss, params, int(np.argmin(val_curve) + 1)

    return best_params, best_n, best_loss


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


def score_probabilities(y_true, probabilities: np.ndarray) -> dict:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)  # float32 XGBoost output
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, average="macro"),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1, 2]),
    }


def class_prior_probabilities(y_train: pd.Series, n_rows: int) -> np.ndarray:
    priors = y_train.value_counts(normalize=True).reindex([0, 1, 2], fill_value=0.0).to_numpy()
    return np.tile(priors, (n_rows, 1))


def main() -> None:
    X_train, X_test, y_train, y_test, train_df, test_df = prepare_training_data(PROJECT_ROOT / "data")
    bookmaker_probs = test_df[ODDS_COLUMNS].to_numpy(dtype=float)

    all_columns = list(X_train.columns)
    feature_sets = {
        "form": [c for c in all_columns if "implied_prob" not in c],
        "odds": [c for c in all_columns if "implied_prob" in c],
        "both": all_columns,
    }

    # The last training season is held out to tune XGBoost and to pick the best learned model.
    season_order = train_df.groupby("season")["Date"].min().sort_values().index.tolist()
    val_mask = (train_df["season"] == season_order[-1]).to_numpy()
    y_val, y_tune = y_train[val_mask], y_train[~val_mask]

    test_probs: dict[str, np.ndarray] = {}
    val_losses: dict[str, float] = {}
    xgb_configs: dict[str, dict] = {}

    for name, columns in feature_sets.items():
        Xtr, Xte = impute_with_train_medians(X_train[columns], X_test[columns])
        Xtune, Xval = impute_with_train_medians(X_train.loc[~val_mask, columns], X_train.loc[val_mask, columns])

        label = {"form": "form_only_logreg", "odds": "odds_only_logreg", "both": "logistic_regression"}[name]
        val_model = train_logistic_regression(Xtune, y_tune)
        val_losses[label] = log_loss(y_val, val_model.predict_proba(Xval), labels=[0, 1, 2])
        test_probs[label] = train_logistic_regression(Xtr, y_train).predict_proba(Xte)

        # XGBoost handles NaN natively, so it gets the raw (unimputed) features.
        params, n_estimators, val_loss = tune_xgboost(
            X_train.loc[~val_mask, columns], y_tune, X_train.loc[val_mask, columns], y_val
        )
        label = {"form": "form_only_xgboost", "odds": "odds_only_xgboost", "both": "xgboost"}[name]
        val_losses[label] = val_loss
        refit = make_xgboost(n_estimators, **params).fit(X_train[columns], y_train)
        test_probs[label] = refit.predict_proba(X_test[columns])
        xgb_configs[name] = {"params": params, "n_estimators": n_estimators, "columns": columns}

    class_prior = class_prior_probabilities(y_train, len(y_test))
    y = y_test.to_numpy()

    ordered = {
        "class_prior_baseline": class_prior,
        "form_only_logreg": test_probs["form_only_logreg"],
        "odds_only_logreg": test_probs["odds_only_logreg"],
        "logistic_regression": test_probs["logistic_regression"],
        "form_only_xgboost": test_probs["form_only_xgboost"],
        "odds_only_xgboost": test_probs["odds_only_xgboost"],
        "xgboost": test_probs["xgboost"],
        "bookmaker": bookmaker_probs,
    }
    summary_df = pd.DataFrame([{"model": name, **score_probabilities(y, p)} for name, p in ordered.items()])

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    summary_df.to_csv(output_dir / "model_results.csv", index=False)
    (output_dir / "xgb_config.json").write_text(json.dumps(xgb_configs, indent=2))

    comparisons = {
        "form_logreg_vs_prior": ("form_only_logreg", "class_prior_baseline"),
        "both_logreg_vs_odds_only_logreg": ("logistic_regression", "odds_only_logreg"),
        "both_logreg_vs_bookmaker": ("logistic_regression", "bookmaker"),
        "odds_only_logreg_vs_bookmaker": ("odds_only_logreg", "bookmaker"),
        "xgboost_vs_logistic_regression": ("xgboost", "logistic_regression"),
        "xgboost_vs_bookmaker": ("xgboost", "bookmaker"),
    }
    bootstrap_rows = []
    for name, (a, b) in comparisons.items():
        la, lb, diff, ci = paired_bootstrap(y, ordered[a], ordered[b])
        bootstrap_rows.append(
            {"comparison": name, "loss_a": la, "loss_b": lb, "diff": diff, "ci_low": ci[0], "ci_high": ci[1]}
        )
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(output_dir / "bootstrap.csv", index=False)

    # Best learned model is chosen on the validation season, never on the test set.
    best_label = min(val_losses, key=val_losses.get)
    season_rows = []
    for season in sorted(test_df["season"].unique()):
        mask = (test_df["season"] == season).to_numpy()
        season_rows.append(
            {
                "season": season,
                "matches": int(mask.sum()),
                "best_model_log_loss": log_loss(y[mask], ordered[best_label][mask], labels=[0, 1, 2]),
                "bookmaker_log_loss": log_loss(y[mask], bookmaker_probs[mask], labels=[0, 1, 2]),
            }
        )
    season_df = pd.DataFrame(season_rows)
    season_df.to_csv(output_dir / "season_logloss.csv", index=False)

    # Robustness: does form still beat the prior on matches where form is actually defined?
    complete = X_test[feature_sets["form"]].notna().all(axis=1).to_numpy()
    robustness = None
    if complete.any():
        form_loss = log_loss(y[complete], test_probs["form_only_logreg"][complete], labels=[0, 1, 2])
        prior_loss = log_loss(y[complete], class_prior[complete], labels=[0, 1, 2])
        robustness = (form_loss, prior_loss)

    print("Training rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("Best learned model on validation season:", best_label)
    print()
    print(_to_markdown_table(summary_df.round(4)))
    print()
    print(_to_markdown_table(bootstrap_df.round(4)))
    print()
    print(_to_markdown_table(season_df.round(4)))

    if robustness:
        print(
            f"\nComplete-form robustness: form-only logistic log loss {robustness[0]:.4f} "
            f"vs class prior {robustness[1]:.4f} on {int(complete.sum())} matches with full form."
        )

    if "2021" in set(test_df["season"].astype(str)):
        print("\nNote: the test split includes 2020-21, a season played mostly without crowds.")
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
