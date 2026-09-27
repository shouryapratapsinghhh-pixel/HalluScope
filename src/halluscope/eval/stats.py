"""Bootstrap uncertainty, from scratch.

Phase 0's first real run had only ~34 correct answers in a 300-item test
set, so a single AUROC number is noisy. Two tools:

- bootstrap_ci: 95% percentile CI for one scorer's metric.
- paired_bootstrap_diff: CI for (scorer A - scorer B), resampling the SAME
  items for both each time. This is the right test for "does the probe
  beat the baseline?": both scorers see identical resamples, so
  per-question difficulty cancels out instead of adding noise.

Resamples that happen to contain only one class (AUROC undefined) are
redrawn, not dropped silently.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from halluscope.eval.metrics import auroc


def _resample_indices(y: np.ndarray, rng: np.random.Generator, max_tries: int = 100) -> np.ndarray:
    n = len(y)
    for _ in range(max_tries):
        idx = rng.integers(0, n, size=n)
        if 0 < y[idx].sum() < n:
            return idx
    raise ValueError("could not draw a resample containing both classes -- too few of one class")


def bootstrap_ci(
    y: np.ndarray,
    risk: np.ndarray,
    metric: Callable = auroc,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point estimate, CI low, CI high)."""
    y, risk = np.asarray(y), np.asarray(risk)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = _resample_indices(y, rng)
        stats[i] = metric(y[idx], risk[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y, risk)), float(lo), float(hi)


def paired_bootstrap_diff(
    y: np.ndarray,
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    metric: Callable = auroc,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(metric(A) - metric(B), CI low, CI high). If the CI excludes 0,
    A genuinely beats B on this data, not just by sampling luck."""
    y, a, b = np.asarray(y), np.asarray(risk_a), np.asarray(risk_b)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = _resample_indices(y, rng)
        diffs[i] = metric(y[idx], a[idx]) - metric(y[idx], b[idx])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y, a) - metric(y, b)), float(lo), float(hi)
