import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from halluscope.collect import collect_qa, save_collected
from halluscope.data.datasets import QAItem, make_toy_qa, qa_prompt
from halluscope.export import export
from halluscope.models.lm import prompt_hidden_states, run_prompt, tiny_random_model
from halluscope.probes.linear import LinearProbe, load_probe, save_probe
from halluscope.serve import create_app

pytestmark = pytest.mark.slow  # runs a (tiny) transformer


def test_probe_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(100, 6)), rng.integers(0, 2, 100)
    p = LinearProbe().fit(X, y)
    save_probe(p, tmp_path / "p.npz")
    np.testing.assert_allclose(load_probe(tmp_path / "p.npz").predict_proba(X), p.predict_proba(X), rtol=1e-6)


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("art")
    model, tok = tiny_random_model(seed=0)
    items = [QAItem(f"{q.question} ({i})", q.answers) for i in range(3) for q in make_toy_qa()]
    c = collect_qa(model, tok, items, max_new_tokens=6)
    c.y_wrong = np.array([i % 2 for i in range(len(items))])  # planted: both classes
    save_collected(c, tmp / "c.npz")
    meta = export("tiny", str(tmp / "c.npz"), str(tmp / "art"))
    probe = load_probe(tmp / "art" / "probe.npz")
    return TestClient(create_app(model, tok, probe, meta)), model, tok, probe, meta, tmp


def test_export_writes_artifact(served):
    *_, meta, tmp = served
    saved = json.loads((tmp / "art" / "meta.json").read_text())
    assert saved == meta  # what export returned is exactly what it wrote
    assert saved["model"] == "tiny" and 0 <= saved["layer"] <= 3
    assert "heldout_auroc_estimate" in saved


def test_health_and_info(served):
    client, *_ = served
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/info").json()["model"] == "tiny"


def test_assess_matches_offline_computation(served):
    client, model, tok, probe, meta, _ = served
    q = "What is the capital of France?"
    r = client.post("/assess", json={"question": q}).json()
    h = prompt_hidden_states(model, tok, qa_prompt(q))[meta["layer"]]
    assert r["risk"] == pytest.approx(float(probe.predict_proba(h[None, :])[0]), rel=1e-5)
    assert r["answer_generated"] is False


def test_answer_matches_run_prompt(served):
    client, model, tok, *_ = served
    q = "Who wrote Romeo and Juliet?"
    r = client.post("/answer", json={"question": q, "max_new_tokens": 6}).json()
    assert r["answer"] == run_prompt(model, tok, qa_prompt(q), max_new_tokens=6).answer
    assert r["abstained"] is False and 0 <= r["risk"] <= 1


def test_abstain_skips_generation(served):
    client, *_ = served
    r = client.post("/answer", json={"question": "Who wrote Hamlet?", "abstain_above": 0.0}).json()
    assert r["abstained"] is True and r["answer"] is None and r["ms_generate"] == 0.0
    r = client.post("/answer", json={"question": "Who wrote Hamlet?", "abstain_above": 1.0}).json()
    assert r["abstained"] is False


@pytest.mark.parametrize("body", [{"question": "   "}, {"question": "x" * 1001},
                                  {"question": "ok?", "abstain_above": 1.5}, {}])
def test_bad_requests_are_422(served, body):
    client, *_ = served
    assert client.post("/answer", json=body).status_code == 422
