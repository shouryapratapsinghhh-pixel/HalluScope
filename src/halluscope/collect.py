"""Run a model over a QA dataset ONCE and save everything to disk:
hidden states (all layers), answers, correctness labels, and per-token
log-probs/entropies.

Running the LLM is the slow part; fitting probes is fast. Caching here
means every later experiment (layer sweeps, probe variants, transfer
tests) reuses the same activations instead of re-running the model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from halluscope.data.datasets import QAItem, is_correct, qa_prompt
from halluscope.models.lm import RunResult, run_prompt

logger = logging.getLogger(__name__)


@dataclass
class Collected:
    hidden: np.ndarray  # (n, n_layers + 1, d_model)
    y_wrong: np.ndarray  # (n,) 1 = model answered wrong
    answers: list[str]
    runs: list[RunResult]  # kept for baselines (log-probs, entropies)


def collect_qa(model, tokenizer, items: list[QAItem], max_new_tokens: int = 16) -> Collected:
    runs, labels = [], []
    for i, item in enumerate(items):
        r = run_prompt(model, tokenizer, qa_prompt(item.question), max_new_tokens=max_new_tokens)
        runs.append(r)
        labels.append(0 if is_correct(r.answer, item.answers) else 1)
        if (i + 1) % 50 == 0:
            logger.info("collected %d / %d", i + 1, len(items))
    return Collected(
        hidden=np.stack([r.hidden for r in runs]).astype(np.float32),
        y_wrong=np.array(labels, dtype=np.int64),
        answers=[r.answer for r in runs],
        runs=runs,
    )


def save_collected(c: Collected, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, hidden=c.hidden, y_wrong=c.y_wrong)
    meta = {
        "answers": c.answers,
        "token_logprobs": [r.token_logprobs for r in c.runs],
        "token_entropies": [r.token_entropies for r in c.runs],
    }
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f)


def load_collected(path: str | Path) -> Collected:
    path = Path(path)
    arrays = np.load(path)
    with open(path.with_suffix(".json")) as f:
        meta = json.load(f)
    runs = [
        RunResult(hidden=h, answer=a, token_logprobs=lp, token_entropies=te)
        for h, a, lp, te in zip(
            arrays["hidden"], meta["answers"], meta["token_logprobs"], meta["token_entropies"]
        )
    ]
    return Collected(arrays["hidden"], arrays["y_wrong"], meta["answers"], runs)
