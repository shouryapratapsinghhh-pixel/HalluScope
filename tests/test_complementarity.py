import numpy as np
import pytest

from halluscope.analyze import analyze, output_features
from halluscope.collect import Collected
from halluscope.eval.baselines import EMPTY_ANSWER_RISK
from halluscope.models.lm import RunResult

pytestmark = pytest.mark.slow  # several full analyze() runs


def _world(n=800, seed=0, redundant=False):
    """Two independent, continuous causes of error, za and zb, that ADD on
    the log-odds scale (the assumption logistic stacking makes), with
    balanced classes. The activation sees za; the answer's log-probs see
    zb -- or also za, if redundant.

    (An earlier version used y = a OR b: that's not additive in log-odds, so
    no linear combiner can reach the ceiling, and it left only ~40 negatives
    in the test split -- too few for a significant paired difference.)
    """
    rng = np.random.default_rng(seed)
    za, zb = rng.normal(size=n), rng.normal(size=n)
    latent = za if redundant else za + zb
    y = (latent + 0.3 * rng.normal(size=n) > 0).astype(int)
    hidden = rng.normal(size=(n, 3, 8)).astype(np.float32)
    hidden[:, 1, 0] += 2.0 * za
    seen_by_logprob = za if redundant else zb
    runs = []
    for i in range(n):
        lp = list(-(2.0 + 0.8 * seen_by_logprob[i]) + rng.normal(0, 0.3, 3))
        runs.append(RunResult(hidden[i], "x", lp, list(rng.random(3))))
    return Collected(hidden, y, ["x"] * n, runs, None)


def test_complementary_signals_combine():
    _, _, h = analyze(_world(redundant=False), n_boot=300)
    assert h["probe_adds_beyond_output"]["significant"]
    assert h["output_adds_beyond_probe"]["significant"]


def test_redundant_signals_do_not_combine():
    _, _, h = analyze(_world(redundant=True), n_boot=300)
    assert not h["probe_adds_beyond_output"]["significant"]
    assert h["spearman_probe_vs_output"] > 0.5  # both rank errors the same way


def test_output_features_impute_empty_answers_from_train_only():
    runs = [
        RunResult(np.zeros((1, 2)), "a", [-1.0], [0.5]),
        RunResult(np.zeros((1, 2)), "b", [-2.0], [0.7]),
        RunResult(np.zeros((1, 2)), "", [], []),  # empty answer, in TEST
        RunResult(np.zeros((1, 2)), "c", [-9.0], [3.0]),  # large value, in TEST
    ]
    X = output_features(runs, tr=np.array([0, 1]))
    assert (X < EMPTY_ANSWER_RISK).all()
    # empty answer gets the TRAIN max (2.0 for neg_mean_logprob), not the test max (9.0)
    assert X[2, 0] == pytest.approx(2.0)


# -- stacking must not leak ----------------------------------------------------


def test_stratified_kfold_partitions_and_stratifies():
    from halluscope.analyze import stratified_kfold

    y = np.array([0] * 40 + [1] * 10)
    folds = stratified_kfold(y, k=5, seed=0)
    all_idx = np.concatenate(folds)
    assert sorted(all_idx.tolist()) == list(range(50))  # every item exactly once
    assert all(y[f].sum() == 2 for f in folds)  # each fold keeps the 20% positive rate


def test_oof_scores_are_honest_on_pure_noise():
    """High-dim noise: an in-sample probe memorizes (AUROC >> 0.5), but
    out-of-fold scores must stay near chance -- otherwise the stacking
    meta-model would be trained on leaked, overconfident scores."""
    from halluscope.analyze import oof_probe_scores
    from halluscope.eval.metrics import auroc
    from halluscope.probes.linear import LinearProbe

    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(200, 150)), rng.integers(0, 2, 200)
    in_sample = LinearProbe(l2=1e-3).fit(X, y).predict_proba(X)
    assert auroc(y, in_sample) > 0.9  # memorized
    assert 0.35 < auroc(y, oof_probe_scores(X, y, k=5)) < 0.65  # honest
