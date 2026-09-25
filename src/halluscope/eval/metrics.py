"""Metrics, implemented from scratch (sklearn is used only in tests to
verify AUROC).

Convention for this whole repo:
  y_wrong[i] = 1 if the model's answer i was WRONG
  risk[i]    = a score where HIGHER means "more likely wrong"

- auroc: how well risk ranks wrong answers above right ones. 0.5 = random.
- ece: calibration of a risk that is also a probability (the probe's
  output). Does "0.8 risk" really mean wrong 80% of the time?
- selective_accuracy: if the model only answers the `coverage` fraction of
  questions it is most confident on, how accurate is it? This is the
  metric a deployed system actually cares about.
- aurc: area under the risk-coverage curve, summarising selective accuracy
  over ALL coverage levels in one number. Lower is better.
"""

from __future__ import annotations

import numpy as np


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, ties get the average of their ranks."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1  # average rank of the tie group
        i = j + 1
    return ranks


def auroc(y_wrong: np.ndarray, risk: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic: the probability that a
    randomly chosen wrong answer gets a higher risk than a randomly chosen
    right one (ties count as half).
    """
    y = np.asarray(y_wrong).astype(int)
    r = np.asarray(risk, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")  # undefined with only one class
    ranks = _rankdata(r)
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def ece(y_wrong: np.ndarray, p_wrong: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    y = np.asarray(y_wrong, dtype=np.float64)
    p = np.clip(np.asarray(p_wrong, dtype=np.float64), 0.0, 1.0)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def selective_accuracy(y_wrong: np.ndarray, risk: np.ndarray, coverage: float) -> float:
    """Accuracy on the `coverage` fraction of items with the LOWEST risk."""
    if not 0 < coverage <= 1:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    y = np.asarray(y_wrong).astype(int)
    order = np.argsort(np.asarray(risk), kind="mergesort")
    k = max(1, round(coverage * len(y)))
    return float(1.0 - y[order[:k]].mean())


def risk_coverage_curve(y_wrong: np.ndarray, risk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(coverage, error_rate) after answering the k most confident items,
    for k = 1..n."""
    y = np.asarray(y_wrong, dtype=np.float64)
    order = np.argsort(np.asarray(risk), kind="mergesort")
    cum_errors = np.cumsum(y[order])
    k = np.arange(1, len(y) + 1)
    return k / len(y), cum_errors / k


def aurc(y_wrong: np.ndarray, risk: np.ndarray) -> float:
    """Area under the risk-coverage curve (mean error rate over coverages)."""
    _, err = risk_coverage_curve(y_wrong, risk)
    return float(err.mean())


def summarize(y_wrong: np.ndarray, risk: np.ndarray, is_probability: bool = False) -> dict:
    """All metrics for one scorer. ECE only when risk is a real probability."""
    out = {
        "auroc": auroc(y_wrong, risk),
        "aurc": aurc(y_wrong, risk),
        "sel_acc@50": selective_accuracy(y_wrong, risk, 0.5),
        "sel_acc@80": selective_accuracy(y_wrong, risk, 0.8),
        "base_accuracy": float(1.0 - np.mean(y_wrong)),
    }
    out["ece"] = ece(y_wrong, risk) if is_probability else float("nan")
    return out
