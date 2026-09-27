import numpy as np
import pandas as pd
import pytest

from halluscope.collect import Collected
from halluscope.data.datasets import QAItem
from halluscope.demo_data import build_demo_frame, cross_fitted_risk
from halluscope.eval.metrics import auroc
from halluscope.models.lm import RunResult

pytestmark = pytest.mark.slow  # 5 folds x layer selection


def _collected(n=300, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    hidden = rng.normal(size=(n, 3, 20)).astype(np.float32)
    if signal:
        hidden[:, 1, 0] += 2.5 * y
    runs = [RunResult(hidden[i], "ans", list(rng.normal(-1, 0.3, 3)), list(rng.random(3))) for i in range(n)]
    return Collected(hidden, y, ["ans"] * n, runs, [f"q{i}?" for i in range(n)])


def test_cross_fitted_risk_is_honest_on_noise():
    c = _collected(signal=False)
    risk, _ = cross_fitted_risk(c.hidden, c.y_wrong)
    assert 0.35 < auroc(c.y_wrong, risk) < 0.65  # no leakage -> chance on noise


def test_cross_fitted_risk_finds_real_signal():
    c = _collected(signal=True)
    risk, layers = cross_fitted_risk(c.hidden, c.y_wrong)
    assert auroc(c.y_wrong, risk) > 0.85
    assert (layers == 1).all()  # every fold independently finds the signal layer


def test_demo_frame_columns():
    c = _collected(n=100)
    items = [QAItem(q, ["a", "b"]) for q in c.questions]
    df = build_demo_frame(c, items)
    assert list(df.columns) == ["question", "model_answer", "gold_answers", "correct",
                                "probe_risk", "token_risk", "probe_layer"]
    assert df["correct"].tolist() == (1 - c.y_wrong).tolist()
    assert df["probe_risk"].between(0, 1).all()


def test_streamlit_app_runs(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    rng = np.random.default_rng(0)
    n = 200
    correct = rng.integers(0, 2, n)
    pd.DataFrame({"question": [f"Question {i}?" for i in range(n)], "model_answer": "x",
                  "gold_answers": "y", "correct": correct,
                  "probe_risk": np.clip(0.5 - 0.3 * correct + rng.normal(0, 0.1, n), 0, 1),
                  "token_risk": rng.random(n), "probe_layer": 3}).to_csv(tmp_path / "m.csv", index=False)
    (tmp_path / "m.name").write_text("Test model")
    monkeypatch.setenv("HALLUSCOPE_DEMO_DIR", str(tmp_path))

    at = AppTest.from_file("../demo/streamlit_app.py", default_timeout=60).run()
    assert not at.exception
    assert at.selectbox[0].value == "Test model"
    at.slider[0].set_value(50).run()  # interact: skip the riskiest half
    assert not at.exception
    answered = next(m for m in at.metric if m.label == "Questions answered")
    assert answered.value == "100 / 200"
