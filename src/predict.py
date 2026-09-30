"""Fit the final models and attach their predictions to the held-out matches.

Used by the Streamlit demo so it reports the same models as ``python -m src.train``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train import (
    ODDS_COLUMNS,
    PROJECT_ROOT,
    impute_with_train_medians,
    make_xgboost,
    prepare_training_data,
    train_logistic_regression,
)

XGB_CONFIG_PATH = PROJECT_ROOT / "results" / "xgb_config.json"
FALLBACK_XGB = {
    "params": {"learning_rate": 0.05, "max_depth": 3, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    "n_estimators": 200,
}


def load_xgb_config(path: Path = XGB_CONFIG_PATH) -> dict:
    """Tuned settings written by ``src.train``; falls back to fixed defaults if it has not been run."""
    if path.exists():
        return json.loads(path.read_text())["both"]
    return FALLBACK_XGB


def predict_holdout(data_dir: str | Path) -> pd.DataFrame:
    """Test-set rows with logistic, XGBoost and bookmaker probabilities (columns ordered away/draw/home)."""
    X_train, X_test, y_train, _, _, test_df = prepare_training_data(data_dir)

    X_train_filled, X_test_filled = impute_with_train_medians(X_train, X_test)
    logreg = train_logistic_regression(X_train_filled, y_train)

    config = load_xgb_config()
    xgb = make_xgboost(config["n_estimators"], **config["params"]).fit(X_train, y_train)

    out = test_df.copy()
    out["logreg_prob"] = list(logreg.predict_proba(X_test_filled))
    out["xgb_prob"] = list(xgb.predict_proba(X_test))
    out["logreg_pred"] = logreg.predict(X_test_filled)
    out["xgb_pred"] = xgb.predict(X_test)
    out["bookmaker_best"] = out[ODDS_COLUMNS].to_numpy(dtype=float).argmax(axis=1)
    return out


def probability_matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
    return np.vstack(frame[column].to_numpy())
