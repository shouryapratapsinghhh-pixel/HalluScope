import json

import numpy as np
import pytest

from halluscope.analyze import analyze
from halluscope.collect import Collected, load_collected, save_collected
from halluscope.eval.controls import QUESTION_WORDS, question_features
from halluscope.eval.stats import bootstrap_ci, paired_bootstrap_diff
from halluscope.models.lm import RunResult

# -- stats -------------------------------------------------------------------------


def test_bootstrap_ci_contains_estimate_and_is_seeded():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    risk = rng.normal(size=200) + y
    est, lo, hi = bootstrap_ci(y, risk, n_boot=300, seed=1)
    assert lo <= est <= hi
    assert (est, lo, hi) == bootstrap_ci(y, risk, n_boot=300, seed=1)


def test_bootstrap_ci_perfect_separation():
    y = np.array([0] * 20 + [1] * 20)
    est, lo, hi = bootstrap_ci(y, y.astype(float), n_boot=200)
    assert est == lo == hi == 1.0


def test_paired_diff_identical_scorers_is_zero():
    rng = np.random.default_rng(2)
    y, risk = rng.integers(0, 2, 150), rng.random(150)
    assert paired_bootstrap_diff(y, risk, risk, n_boot=200) == (0.0, 0.0, 0.0)


def test_paired_diff_detects_real_gap():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    good, noise = y + rng.normal(0, 0.5, 300), rng.random(300)
    diff, lo, _hi = paired_bootstrap_diff(y, good, noise, n_boot=300)
    assert diff > 0.3 and lo > 0  # CI excludes zero -> significant


def test_bootstrap_one_class_raises():
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_ci(np.zeros(10), np.random.rand(10), n_boot=10)


# -- controls ------------------------------------------------------------------------


def test_question_features():
    F = question_features(["Who wrote Hamlet?", "In 1066, which battle happened?", ""])
    assert F.shape[0] == 3
    who_col = 6 + QUESTION_WORDS.index("who")
    assert F[0, who_col] == 1 and F[1, who_col] == 0
    assert F[1, 3] == 4  # four digits in "1066"


# -- analyze on a planted signal ----------------------------------------------------


def _planted(n=400, n_layers=7, d=12, signal_layer=3, seed=0, questions=True):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    hidden = rng.normal(size=(n, n_layers, d)).astype(np.float32)
    hidden[:, signal_layer, 0] += 2.5 * y
    runs = [RunResult(hidden[i], "x", list(rng.normal(-1, 0.3, 3)), list(rng.random(3)))
            for i in range(n)]
    qs = [f"Question number {i}?" for i in range(n)] if questions else None
    return Collected(hidden, y, ["x"] * n, runs, qs), y


def test_analyze_finds_planted_layer_on_train_and_is_significant():
    c, _ = _planted(signal_layer=3)
    sweep, summary, headline = analyze(c, c.questions, n_boot=200)
    assert headline["best_layer"] == 3  # selected on TRAIN validation, not test
    assert sweep.loc[3, "test_auroc"] == sweep["test_auroc"].max()
    assert headline["significant"]
    # questions carry no information here -> control stays near chance
    assert summary.loc["control:question_features", "auroc"] < 0.65


def test_cross_model_control_near_chance_without_shared_signal():
    c, _ = _planted(seed=0)
    other, _ = _planted(seed=99)  # different model: no information about c's errors
    other.questions = c.questions
    _, summary, _ = analyze(c, c.questions, control=other, n_boot=200)
    assert summary.loc["control:other_model_probe", "auroc"] < 0.65


def test_cross_model_control_mismatch_raises():
    c, _ = _planted(seed=0)
    other, _ = _planted(seed=1)
    other.questions = [q + "!" for q in c.questions]
    with pytest.raises(ValueError, match="don't match"):
        analyze(c, c.questions, control=other, n_boot=50)


# -- backward compatibility + plot ---------------------------------------------------


def test_old_cache_without_questions_still_loads(tmp_path):
    c, _ = _planted(n=20, questions=True)
    save_collected(c, tmp_path / "c.npz")
    meta = json.loads((tmp_path / "c.json").read_text())
    del meta["questions"]  # simulate a Phase 0 cache
    (tmp_path / "c.json").write_text(json.dumps(meta))
    assert load_collected(tmp_path / "c.npz").questions is None


def test_plot_layer_sweep_writes_png(tmp_path):
    from halluscope.viz import plot_layer_sweep

    c, _ = _planted()
    sweep, summary, headline = analyze(c, c.questions, n_boot=50)
    out = tmp_path / "sweep.png"
    plot_layer_sweep(sweep, summary, headline, out)
    assert out.exists() and out.stat().st_size > 0


# -- separating self-knowledge from question difficulty --------------------------


def _difficulty_world(n=500, seed=0, activation_has_extra_signal=False):
    """Labels driven by question LENGTH (a surface feature). The activation
    copies that length signal; optionally it also carries independent info
    about the error that the question text can't see."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    extra = rng.integers(0, 2, n)  # hidden "self-knowledge" component
    if activation_has_extra_signal:
        y = ((y + extra) >= 1).astype(int)  # errors depend on something beyond the question
    lengths = np.where(rng.random(n) < 0.8, y, 1 - y) * 6 + rng.integers(1, 4, n)
    questions = [" ".join(["word"] * int(k)) + "?" for k in lengths]
    hidden = rng.normal(size=(n, 4, 8)).astype(np.float32)
    hidden[:, 2, 0] += 0.4 * lengths  # activation encodes question length
    if activation_has_extra_signal:
        hidden[:, 2, 1] += 2.5 * extra
    runs = [RunResult(hidden[i], "x", list(rng.normal(-1, 0.3, 3)), list(rng.random(3)))
            for i in range(n)]
    return Collected(hidden, y, ["x"] * n, runs, questions)


def test_activations_add_nothing_when_errors_are_pure_difficulty():
    c = _difficulty_world(activation_has_extra_signal=False)
    _, _, headline = analyze(c, c.questions, n_boot=300)
    assert not headline["activations_add_beyond_questions"]["significant"]


def test_activations_add_signal_when_model_knows_more_than_the_question():
    c = _difficulty_world(activation_has_extra_signal=True)
    _, _, headline = analyze(c, c.questions, n_boot=300)
    assert headline["activations_add_beyond_questions"]["significant"]
