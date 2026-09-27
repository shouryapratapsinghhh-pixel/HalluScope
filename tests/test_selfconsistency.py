import json
import sys

import numpy as np
import pytest

from halluscope.collect import collect_qa, save_collected
from halluscope.data.datasets import QAItem, make_toy_qa
from halluscope.models.lm import run_prompt, tiny_random_model
from halluscope.selfconsistency import disagreement, run_with_samples


def test_disagreement():
    assert disagreement("Paris", ["paris", "Paris.", "The Paris"]) == 0.0  # normalized: all agree
    assert disagreement("Paris", ["London", "Paris", "Rome", "paris"]) == 0.5
    assert disagreement("x", []) == 0.0


@pytest.fixture(scope="module")
def tiny():
    return tiny_random_model(seed=0)


@pytest.mark.slow
def test_greedy_matches_run_prompt_and_cache_copy_is_safe(tiny):
    model, tok = tiny
    prompt = "Question: Capital of Peru?\nAnswer:"
    r = run_with_samples(model, tok, prompt, k=3, temperature=1.0, max_new_tokens=8)
    assert r["greedy"] == run_prompt(model, tok, prompt, max_new_tokens=8).answer
    assert len(r["samples"]) == 3
    assert r["steps_greedy"] == len(r["greedy_logprobs"])  # one forward step per generated token


@pytest.mark.slow
def test_near_zero_temperature_samples_equal_greedy(tiny):
    model, tok = tiny
    r = run_with_samples(model, tok, "Question: Hi?\nAnswer:", k=3, temperature=1e-4, max_new_tokens=6)
    assert all(s == r["greedy"] for s in r["samples"])
    assert disagreement(r["greedy"], r["samples"]) == 0.0


@pytest.mark.slow
def test_sampling_is_seeded(tiny):
    model, tok = tiny
    a = run_with_samples(model, tok, "Question: Hi?\nAnswer:", k=4, temperature=1.5, seed=7)["samples"]
    b = run_with_samples(model, tok, "Question: Hi?\nAnswer:", k=4, temperature=1.5, seed=7)["samples"]
    assert a == b


@pytest.mark.slow
def test_cli_end_to_end_and_resume(tiny, tmp_path, monkeypatch):
    model, tok = tiny
    items = [QAItem(f"{q.question} ({i})", q.answers) for i in range(3) for q in make_toy_qa()]  # 24
    data = tmp_path / "qa.jsonl"
    data.write_text("".join(json.dumps({"question": q.question, "answers": q.answers}) + "\n" for q in items))
    c = collect_qa(model, tok, items, max_new_tokens=6)
    c.y_wrong = np.array([i % 2 for i in range(len(items))])  # planted labels: both classes present
    save_collected(c, tmp_path / "c.npz")

    from halluscope import selfconsistency

    args = ["x", "--model", "tiny", "--data", str(data), "--qa-cache", str(tmp_path / "c.npz"),
            "--n", "8", "--k", "2", "--max-new-tokens", "6", "--n-boot", "50", "--out", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", args)
    selfconsistency.main()
    log = tmp_path / "out" / "samples.jsonl"
    assert len(log.read_text().splitlines()) == 8
    assert (tmp_path / "out" / "cost_vs_accuracy.csv").exists()

    selfconsistency.main()  # resume: nothing left to do, no duplicate rows
    assert len(log.read_text().splitlines()) == 8
