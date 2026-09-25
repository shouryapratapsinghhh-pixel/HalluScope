import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from halluscope.eval.metrics import (
    aurc,
    auroc,
    ece,
    risk_coverage_curve,
    selective_accuracy,
    summarize,
)


def test_auroc_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_auroc_hand_computed():
    # pairs (wrong, right): (0.4 vs 0.1) win, (0.4 vs 0.5) loss,
    # (0.9 vs 0.1) win, (0.9 vs 0.5) win -> 3/4
    y = np.array([0, 1, 0, 1])
    risk = np.array([0.1, 0.4, 0.5, 0.9])
    assert auroc(y, risk) == 0.75


def test_auroc_ties_count_half():
    y = np.array([0, 1])
    assert auroc(y, np.array([0.5, 0.5])) == 0.5


@pytest.mark.parametrize("seed", range(5))
def test_auroc_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=200)
    risk = rng.normal(size=200) + y * 0.7
    risk[::7] = risk[0]  # force some ties
    assert auroc(y, risk) == pytest.approx(roc_auc_score(y, risk), abs=1e-12)


def test_auroc_single_class_is_nan():
    assert np.isnan(auroc(np.zeros(5), np.random.rand(5)))


def test_ece_perfectly_calibrated_is_zero():
    # in each bin, predicted p equals the observed error rate exactly
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
    p = np.array([0.0] * 4 + [1.0] * 4 + [0.5, 0.5])
    assert ece(y, p) == pytest.approx(0.0)


def test_ece_overconfident_is_large():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.99, 0.99, 0.99, 0.99])  # says "wrong" confidently, half are right
    assert ece(y, p) == pytest.approx(0.49)


def test_selective_accuracy_keeps_lowest_risk():
    y = np.array([1, 0, 1, 0])  # wrong, right, wrong, right
    risk = np.array([0.9, 0.1, 0.8, 0.2])
    assert selective_accuracy(y, risk, 0.5) == 1.0  # keeps the two right ones
    assert selective_accuracy(y, risk, 1.0) == 0.5


def test_selective_accuracy_bad_coverage():
    with pytest.raises(ValueError):
        selective_accuracy(np.array([0, 1]), np.array([0.1, 0.2]), 0.0)


def test_risk_coverage_and_aurc():
    y = np.array([0, 1])
    cov, err = risk_coverage_curve(y, np.array([0.1, 0.9]))
    np.testing.assert_allclose(cov, [0.5, 1.0])
    np.testing.assert_allclose(err, [0.0, 0.5])
    assert aurc(y, np.array([0.1, 0.9])) == 0.25
    assert aurc(y, np.array([0.9, 0.1])) == 0.75  # wrong ordering is worse


def test_summarize_keys():
    y = np.array([0, 1, 0, 1])
    out = summarize(y, np.array([0.1, 0.9, 0.2, 0.8]), is_probability=True)
    assert {"auroc", "aurc", "sel_acc@50", "sel_acc@80", "base_accuracy", "ece"} <= set(out)
    assert np.isnan(summarize(y, np.array([1, 2, 3, 4.0]))["ece"])
