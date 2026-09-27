import numpy as np
import pytest

from halluscope.data.datasets import StatementItem
from halluscope.transfer import collect_statements, transfer_sweep

pytestmark = pytest.mark.slow  # layer sweeps with L2 selection


def _world(shared: bool, n_qa=500, n_st=400, n_layers=3, d=10, seed=0):
    rng = np.random.default_rng(seed)
    y_wrong, y_false = rng.integers(0, 2, n_qa), rng.integers(0, 2, n_st)
    qa = rng.normal(size=(n_qa, n_layers, d)).astype(np.float32)
    st = rng.normal(size=(n_st, n_layers, d)).astype(np.float32)
    qa[:, 1, 0] += 2.5 * y_wrong
    st[:, 1, 0 if shared else 5] += 2.5 * y_false  # same axis, or a different one
    return qa, y_wrong, st, y_false


def test_shared_direction_transfers():
    sweep = transfer_sweep(*_world(shared=True), n_boot=200).set_index("layer")
    assert sweep.loc[1, "transfer_auroc"] > 0.85
    assert sweep.loc[1, "cosine"] > 0.5


def test_different_directions_do_not_transfer_but_are_learnable_in_domain():
    sweep = transfer_sweep(*_world(shared=False), n_boot=200).set_index("layer")
    assert 0.35 < sweep.loc[1, "transfer_auroc"] < 0.65  # QA probe is blind to statements
    assert sweep.loc[1, "in_domain_auroc"] > 0.85  # but the truth signal IS there
    assert abs(sweep.loc[1, "cosine"]) < 0.4


def test_mismatched_models_raise():
    qa, y, st, yf = _world(shared=True)
    with pytest.raises(ValueError, match="same model"):
        transfer_sweep(qa, y, st[:, :, :5], yf, n_boot=50)


def test_collect_statements_labels_and_shapes():
    from halluscope.models.lm import tiny_random_model

    model, tok = tiny_random_model(seed=0)
    items = [StatementItem("Paris is in France.", 1), StatementItem("Paris is in Peru.", 0)]
    hidden, y_false = collect_statements(model, tok, items)
    assert hidden.shape == (2, 4, 32) and hidden.dtype == np.float32
    assert y_false.tolist() == [0, 1]  # 1 = FALSE statement, like 1 = wrong answer
