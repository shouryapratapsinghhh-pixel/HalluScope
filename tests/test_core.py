import json

import numpy as np
import pytest

from halluscope.collect import collect_qa, load_collected, save_collected
from halluscope.data.datasets import (
    is_correct,
    load_qa_jsonl,
    load_truth_csv,
    make_toy_qa,
    normalize_answer,
)
from halluscope.eval.baselines import BASELINES, neg_mean_logprob, random_risk
from halluscope.eval.metrics import auroc
from halluscope.models.lm import prompt_hidden_states, run_prompt, tiny_random_model
from halluscope.probes.linear import LinearProbe, select_l2

# -- grading ------------------------------------------------------------------


def test_normalize_answer():
    assert normalize_answer("The Beatles.") == "beatles"
    assert normalize_answer("  An   Apple!! ") == "apple"


def test_is_correct_containment_and_aliases():
    assert is_correct("It was Paris, France", ["Paris"])
    assert is_correct("eight", ["8", "eight"])
    assert not is_correct("London", ["Paris"])
    assert not is_correct("", ["Paris"])


def test_loaders(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(json.dumps({"question": "Q?", "answers": ["A"]}) + "\n\n")
    assert load_qa_jsonl(qa)[0].answers == ["A"]

    csv = tmp_path / "t.csv"
    csv.write_text("statement,label\nParis is in France.,1\nParis is in Peru.,0\n")
    items = load_truth_csv(csv)
    assert [i.label for i in items] == [1, 0]

    with pytest.raises(FileNotFoundError):
        load_qa_jsonl(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.csv"
    bad.write_text("text,y\na,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_truth_csv(bad)


# -- probe -----------------------------------------------------------------------


def _separable(n=300, d=20, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, d))
    X[:, 0] += 3.0 * y  # the "error signal" lives on one direction
    return X, y


def test_probe_learns_linear_signal():
    X, y = _separable()
    probe = LinearProbe().fit(X[:200], y[:200])
    assert auroc(y[200:], probe.predict_proba(X[200:])) > 0.95
    # the learned direction should point mostly along feature 0
    assert np.argmax(np.abs(probe.direction())) == 0


def test_probe_chance_on_noise():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(400, 10)), rng.integers(0, 2, size=400)
    probe = LinearProbe().fit(X[:300], y[:300])
    assert 0.35 < auroc(y[300:], probe.predict_proba(X[300:])) < 0.65


def test_probe_standardization_uses_train_only():
    X, y = _separable()
    probe = LinearProbe().fit(X[:200], y[:200])
    np.testing.assert_allclose(probe.mean_, X[:200].mean(axis=0))
    assert not np.allclose(probe.mean_, X.mean(axis=0))


def test_probe_outputs_probabilities_and_is_deterministic():
    X, y = _separable()
    p1 = LinearProbe(seed=3).fit(X, y).predict_proba(X)
    p2 = LinearProbe(seed=3).fit(X, y).predict_proba(X)
    assert np.all((p1 >= 0) & (p1 <= 1))
    np.testing.assert_array_equal(p1, p2)


def test_probe_single_class_raises():
    with pytest.raises(ValueError, match="both"):
        LinearProbe().fit(np.random.rand(10, 3), np.zeros(10))


def test_select_l2_returns_grid_value():
    X, y = _separable()
    from halluscope.probes.linear import L2_GRID

    assert select_l2(X, y) in L2_GRID


# -- baselines -------------------------------------------------------------------


def test_baselines():
    assert random_risk(5, seed=1).shape == (5,)
    risk = neg_mean_logprob([[-0.1, -0.2], [-3.0], []])
    assert risk[1] > risk[0]  # less likely answer = riskier
    assert risk[2] == max(risk)  # empty answer = maximally risky


# -- real (tiny) transformer ------------------------------------------------------


@pytest.fixture(scope="module")
def tiny():
    return tiny_random_model(seed=0, n_layer=3, n_embd=32)


@pytest.mark.slow
def test_run_prompt_shapes(tiny):
    model, tok = tiny
    r = run_prompt(model, tok, "Question: What is 2+2?\nAnswer:", max_new_tokens=5)
    assert r.hidden.shape == (4, 32)  # 3 layers + embedding layer
    assert len(r.token_logprobs) == len(r.token_entropies) <= 5
    assert all(lp <= 0 for lp in r.token_logprobs)
    assert all(e >= 0 for e in r.token_entropies)


@pytest.mark.slow
def test_hidden_state_is_fixed_before_generation(tiny):
    """The probe's input must exist BEFORE any answer token is generated --
    so it cannot depend on how many tokens we go on to generate."""
    model, tok = tiny
    prompt = "Question: Capital of Peru?\nAnswer:"
    h1 = run_prompt(model, tok, prompt, max_new_tokens=1).hidden
    h2 = run_prompt(model, tok, prompt, max_new_tokens=12).hidden
    np.testing.assert_allclose(h1, h2, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(h1, prompt_hidden_states(model, tok, prompt), rtol=1e-5, atol=1e-6)


@pytest.mark.slow
def test_run_prompt_deterministic(tiny):
    model, tok = tiny
    a = run_prompt(model, tok, "Question: Hi?\nAnswer:", max_new_tokens=6)
    b = run_prompt(model, tok, "Question: Hi?\nAnswer:", max_new_tokens=6)
    assert a.answer == b.answer
    np.testing.assert_array_equal(a.hidden, b.hidden)


@pytest.mark.slow
def test_collect_and_cache_round_trip(tiny, tmp_path):
    model, tok = tiny
    items = make_toy_qa()
    c = collect_qa(model, tok, items, max_new_tokens=6)
    assert c.hidden.shape == (len(items), 4, 32)
    assert set(np.unique(c.y_wrong)) <= {0, 1}

    save_collected(c, tmp_path / "toy.npz")
    c2 = load_collected(tmp_path / "toy.npz")
    np.testing.assert_array_equal(c.hidden, c2.hidden)
    np.testing.assert_array_equal(c.y_wrong, c2.y_wrong)
    for name, fn in BASELINES.items():
        np.testing.assert_allclose(fn(c.runs), fn(c2.runs), err_msg=name)


# -- full evaluate() pipeline on a planted signal ----------------------------


def test_evaluate_probe_beats_baselines_on_planted_signal():
    """Plant an error signal in ONE layer's activations, leave the baselines'
    log-probs uninformative. The pipeline must find it: probe AUROC high,
    baselines near chance. Proves split/probe/metrics are wired correctly."""
    from halluscope.collect import Collected
    from halluscope.models.lm import RunResult
    from halluscope.run import evaluate

    rng = np.random.default_rng(0)
    n, n_layers, d = 400, 5, 16
    y = rng.integers(0, 2, size=n)
    hidden = rng.normal(size=(n, n_layers, d)).astype(np.float32)
    hidden[:, 2, 0] += 3.0 * y  # signal lives in layer 2 only
    runs = [
        RunResult(hidden[i], "x", list(rng.normal(-1, 0.3, 3)), list(rng.random(3)))
        for i in range(n)
    ]
    table = evaluate(Collected(hidden, y, ["x"] * n, runs), layer=2)
    probe_auc = table.loc["linear_probe@L2", "auroc"]
    baseline_auc = table.loc["neg_mean_logprob", "auroc"]
    assert probe_auc > 0.9
    # uninformative baseline: chance, within sampling noise for ~120 test items
    assert 0.3 < baseline_auc < 0.7
    assert probe_auc - baseline_auc > 0.25

    # and the signal is layer-specific: probing a different layer finds ~nothing
    other = evaluate(Collected(hidden, y, ["x"] * n, runs), layer=4)
    assert other.loc["linear_probe@L4", "auroc"] < 0.65


# -- precision ------------------------------------------------------------------


def test_resolve_dtype():
    import torch

    from halluscope.models.lm import resolve_dtype

    assert resolve_dtype("auto", torch.device("mps")) == torch.float16
    assert resolve_dtype("auto", torch.device("cpu")) == torch.float32
    assert resolve_dtype("bfloat16", torch.device("cpu")) == torch.bfloat16
    with pytest.raises(ValueError, match="dtype"):
        resolve_dtype("int8", torch.device("cpu"))


@pytest.mark.slow
def test_half_precision_model_still_saves_float32_hidden():
    """Caches must be comparable whatever precision the model ran in."""
    import torch

    model, tok = tiny_random_model(seed=0)
    model = model.to(torch.bfloat16)
    r = run_prompt(model, tok, "Question: Hi?\nAnswer:", max_new_tokens=4)
    assert r.hidden.dtype == np.float32
    assert np.isfinite(r.hidden).all()
    assert all(np.isfinite(r.token_logprobs))
