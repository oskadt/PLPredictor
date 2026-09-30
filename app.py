from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.predict import predict_holdout, probability_matrix
from src.train import ODDS_COLUMNS

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULT_MAP = {0: "Away", 1: "Draw", 2: "Home"}


@st.cache_data
def load_demo_data() -> tuple[pd.DataFrame, pd.Series]:
    test_df = predict_holdout(DATA_DIR)
    return test_df, test_df["result"].astype(int)


def render_summary(test_df: pd.DataFrame, y_true: pd.Series) -> None:
    logreg_acc = accuracy_score(y_true, test_df["logreg_pred"])
    xgb_acc = accuracy_score(y_true, test_df["xgb_pred"])
    bookmaker_acc = accuracy_score(y_true, test_df["bookmaker_best"])

    logreg_f1 = f1_score(y_true, test_df["logreg_pred"], average="macro")
    xgb_f1 = f1_score(y_true, test_df["xgb_pred"], average="macro")
    bookmaker_f1 = f1_score(y_true, test_df["bookmaker_best"], average="macro")

    logreg_loss = log_loss(y_true, probability_matrix(test_df, "logreg_prob"), labels=[0, 1, 2])
    xgb_loss = log_loss(y_true, probability_matrix(test_df, "xgb_prob"), labels=[0, 1, 2])
    bookmaker_loss = log_loss(
        y_true,
        test_df[ODDS_COLUMNS].to_numpy(dtype=float),
        labels=[0, 1, 2],
    )

    st.subheader("Model benchmark")
    summary = pd.DataFrame(
        {
            "Model": ["Logistic regression", "XGBoost", "Bookmaker"],
            "Accuracy": [logreg_acc, xgb_acc, bookmaker_acc],
            "Macro F1": [logreg_f1, xgb_f1, bookmaker_f1],
            "Log loss": [logreg_loss, xgb_loss, bookmaker_loss],
        }
    )
    st.dataframe(summary.round(4), hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Premier League Predictor", page_icon="⚽", layout="wide")
    st.title("Premier League Predictor Demo")
    st.caption("Time-aware match forecasting with a logistic regression baseline, XGBoost comparator, and bookmaker benchmark.")

    test_df, y_true = load_demo_data()

    selected_season = st.sidebar.selectbox("Choose season", options=sorted(test_df["season"].unique()))
    sample_size = st.sidebar.slider("Sample matches", min_value=5, max_value=20, value=10)

    season_rows = test_df[test_df["season"] == selected_season].copy().reset_index(drop=True)
    sample_df = season_rows.sample(n=min(sample_size, len(season_rows)), random_state=42).copy()

    sample_df["Actual"] = sample_df["result"].map(RESULT_MAP)
    sample_df["Logistic"] = sample_df["logreg_pred"].map(RESULT_MAP)
    sample_df["XGBoost"] = sample_df["xgb_pred"].map(RESULT_MAP)
    sample_df["Bookmaker"] = sample_df["bookmaker_best"].map(RESULT_MAP)

    st.subheader(f"Prediction sample for {selected_season}")
    st.dataframe(sample_df[["Date", "HomeTeam", "AwayTeam", "Actual", "Logistic", "XGBoost", "Bookmaker"]], hide_index=True, use_container_width=True)

    render_summary(test_df, y_true)

    selected_match = st.selectbox(
        "Inspect a single match",
        options=[f"{row['HomeTeam']} vs {row['AwayTeam']} ({row['Date'].strftime('%Y-%m-%d')})" for _, row in sample_df.iterrows()],
    )

    if selected_match:
        chosen_row = sample_df[
            sample_df.apply(
                lambda row: f"{row['HomeTeam']} vs {row['AwayTeam']} ({row['Date'].strftime('%Y-%m-%d')})",
                axis=1,
            )
            == selected_match
        ].iloc[0]

        prob_df = pd.DataFrame(
            {
                "Outcome": ["Away", "Draw", "Home"],
                "Logistic": [*chosen_row["logreg_prob"]],
                "XGBoost": [*chosen_row["xgb_prob"]],
                "Bookmaker": [
                    chosen_row["away_implied_prob"],
                    chosen_row["draw_implied_prob"],
                    chosen_row["home_implied_prob"],
                ],
            }
        )

        st.subheader(f"Probability view: {selected_match}")
        st.dataframe(prob_df.round(4), hide_index=True, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Actual", RESULT_MAP[int(chosen_row["result"])])
        col2.metric("Logistic", RESULT_MAP[int(chosen_row["logreg_pred"])])
        col3.metric("XGBoost", RESULT_MAP[int(chosen_row["xgb_pred"])])

        st.bar_chart(
            prob_df.set_index("Outcome").loc[:, ["Logistic", "XGBoost", "Bookmaker"]],
            use_container_width=True,
        )

    with st.expander("Project notes"):
        st.markdown(
            """
            This demo is designed to show a realistic forecasting workflow:

            - feature engineering is based on prior team form only
            - the model is evaluated on a time-based holdout season
            - the bookmaker is treated as a realistic external benchmark
            - the strongest result is not the claim of a perfect predictive system, but a transparent and defensible workflow
            """
        )


if __name__ == "__main__":
    main()
