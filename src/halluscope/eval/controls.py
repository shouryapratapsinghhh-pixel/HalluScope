"""Control baselines for the key objection: "is the probe reading the
model's self-knowledge, or just detecting that some QUESTIONS are hard?"

1. question_features: surface features of the question text alone --
   no model internals at all. If a probe on these matches the activation
   probe, the activation probe may just be detecting question type.

2. Cross-model control (in analyze.py): a probe on a DIFFERENT model's
   activations predicting THIS model's errors. Another model can see how
   hard a question is but has no access to this model's knowledge. If it
   predicts this model's errors as well as this model's own activations do,
   the signal is mostly difficulty, not self-knowledge.

The activation probe has to beat BOTH controls for the self-knowledge
claim to hold.
"""

from __future__ import annotations

import re

import numpy as np

QUESTION_WORDS = ("what", "who", "which", "where", "when", "how", "in")


def question_features(questions: list[str]) -> np.ndarray:
    """(n, n_features) array of cheap, model-free difficulty proxies."""
    rows = []
    for q in questions:
        words = q.split()
        first = words[0].lower().strip("?,.") if words else ""
        capitalized = sum(1 for w in words[1:] if w[:1].isupper())  # proper-noun proxy
        rows.append(
            [
                len(q),
                len(words),
                np.mean([len(w) for w in words]) if words else 0.0,
                sum(ch.isdigit() for ch in q),
                capitalized,
                int('"' in q or "'" in q),
                *[int(first == qw) for qw in QUESTION_WORDS],
                int(first not in QUESTION_WORDS),
                len(re.findall(r"[,;:]", q)),  # clause count proxy
            ]
        )
    return np.asarray(rows, dtype=np.float64)
