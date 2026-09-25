"""Output-confidence baselines: the numbers you can get WITHOUT looking
inside the model. The probe has to beat these to be worth anything.

Every function returns a risk array (higher = more likely wrong).
"""

from __future__ import annotations

import numpy as np

EMPTY_ANSWER_RISK = 1e3  # an empty answer is maximally suspicious


def random_risk(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def neg_mean_logprob(token_logprobs: list[list[float]]) -> np.ndarray:
    """Average surprise of the answer tokens. Low average log-prob = risky."""
    return np.array([-np.mean(t) if t else EMPTY_ANSWER_RISK for t in token_logprobs])


def neg_min_logprob(token_logprobs: list[list[float]]) -> np.ndarray:
    """Surprise of the single least likely token -- one shaky token can be
    enough to make a factual answer wrong."""
    return np.array([-np.min(t) if t else EMPTY_ANSWER_RISK for t in token_logprobs])


def mean_entropy(token_entropies: list[list[float]]) -> np.ndarray:
    """How spread out the next-token distribution was, on average."""
    return np.array([np.mean(t) if t else EMPTY_ANSWER_RISK for t in token_entropies])


BASELINES = {
    "neg_mean_logprob": lambda runs: neg_mean_logprob([r.token_logprobs for r in runs]),
    "neg_min_logprob": lambda runs: neg_min_logprob([r.token_logprobs for r in runs]),
    "mean_entropy": lambda runs: mean_entropy([r.token_entropies for r in runs]),
}
