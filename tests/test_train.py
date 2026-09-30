import numpy as np
import pandas as pd
import pytest

from src.features import _add_bookmaker_features
from src.train import (
    class_prior_probabilities,
    impute_with_train_medians,
    paired_bootstrap,
    score_probabilities,
)


def test_implied_probabilities_are_normalised_and_preserve_ordering():
    odds = pd.DataFrame({"B365H": [2.0, 1.5], "B365D": [3.5, 4.0], "B365A": [4.0, 8.0]})
    out = _add_bookmaker_features(odds)
    probs = out[["home_implied_prob", "draw_implied_prob", "away_implied_prob"]]

    assert np.allclose(probs.sum(axis=1), 1.0)
    # The overround is removed proportionally: 1/2.0 / (1/2.0 + 1/3.5 + 1/4.0)
    assert probs.loc[0, "home_implied_prob"] == pytest.approx(0.5 / (0.5 + 1 / 3.5 + 0.25))
    assert probs.loc[1, "home_implied_prob"] > probs.loc[1, "away_implied_prob"]


def test_zero_odds_give_missing_probabilities_not_infinities():
    odds = pd.DataFrame({"B365H": [0.0], "B365D": [3.0], "B365A": [4.0]})
    out = _add_bookmaker_features(odds)
    assert not np.isinf(out.select_dtypes("number")).any().any()


def test_paired_bootstrap_identical_models_have_zero_difference():
    y = np.array([0, 1, 2, 2, 1, 0, 2, 1])
    p = np.tile([0.3, 0.3, 0.4], (len(y), 1))
    loss_a, loss_b, diff, ci = paired_bootstrap(y, p, p, n=200)

    assert loss_a == pytest.approx(loss_b)
    assert diff == 0
    assert ci[0] == 0 and ci[1] == 0


def test_paired_bootstrap_detects_a_clearly_better_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=500)
    good = np.full((500, 3), 0.1)
    good[np.arange(500), y] = 0.8
    uniform = np.full((500, 3), 1 / 3)

    loss_a, loss_b, diff, ci = paired_bootstrap(y, good, uniform, n=300)

    assert loss_a < loss_b
    assert diff < 0
    assert ci[1] < 0  # whole interval below zero


def test_paired_bootstrap_rejects_invalid_probabilities():
    y = np.array([0, 1])
    bad = np.array([[0.5, 0.5, 0.5], [0.2, 0.3, 0.5]])
    good = np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]])
    with pytest.raises(AssertionError):
        paired_bootstrap(y, bad, good, n=10)


def test_score_probabilities_matches_hand_computation():
    y = pd.Series([0, 1])
    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    scores = score_probabilities(y, probs)

    assert scores["accuracy"] == 1.0
    assert scores["log_loss"] == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2)


def test_class_prior_probabilities_use_training_frequencies():
    y_train = pd.Series([2, 2, 2, 1, 0, 2, 2, 1, 0, 2])
    probs = class_prior_probabilities(y_train, 3)

    assert probs.shape == (3, 3)
    assert np.allclose(probs[0], [0.2, 0.2, 0.6])


def test_imputation_uses_training_medians_only():
    train = pd.DataFrame({"x": [1.0, 3.0, np.nan]})
    test = pd.DataFrame({"x": [np.nan, 100.0]})
    _, test_filled = impute_with_train_medians(train, test)

    assert test_filled["x"].tolist() == [2.0, 100.0]
