import pytest

from halluscope.robustness import run_seeds
from tests.test_phase1 import _difficulty_world

pytestmark = pytest.mark.slow  # 3 seeds x full layer selection each


def _verdict(agg, label):
    return agg.set_index("comparison").loc[label]


def test_real_signal_is_robust_across_seeds():
    c = _difficulty_world(activation_has_extra_signal=True)
    per_seed, agg = run_seeds(c, c.questions, None, seeds=[0, 1, 2], n_boot=200)
    assert len(per_seed) == 3
    v = _verdict(agg, "activations+questions vs questions alone")
    assert v["robust"] and v["significant_seeds"] == "3/3"


def test_pure_difficulty_is_not_robust():
    c = _difficulty_world(activation_has_extra_signal=False)
    _, agg = run_seeds(c, c.questions, None, seeds=[0, 1, 2], n_boot=200)
    assert not _verdict(agg, "activations+questions vs questions alone")["robust"]


def test_seeds_use_different_splits():
    c = _difficulty_world(activation_has_extra_signal=True)
    per_seed, _ = run_seeds(c, c.questions, None, seeds=[0, 1], n_boot=100)
    # different splits -> the probe's test AUROC differs between seeds
    assert per_seed["probe_auroc"].nunique() == 2
