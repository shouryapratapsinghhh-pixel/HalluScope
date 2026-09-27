"""Export per-question predictions for the demo app -- held out, no leakage.

  python -m halluscope.demo_data --cache cache/qwen15_tqa3k.npz \
      --data data/raw/triviaqa3k.jsonl --name "Qwen2.5-1.5B" --out reports/demo/qwen15.csv

5-fold cross-fitting: every question's probe risk comes from a probe that
never saw that question. Each fold also re-chooses its layer and L2 using
only its own training questions, so the demo meets the same no-leakage bar
as the research results.

Output is a small CSV (a few hundred KB) the demo can load without torch or
the model: question, model answer, gold answers, correct, probe risk, and
token-confidence risk.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from halluscope.analyze import output_features, stratified_kfold
from halluscope.collect import load_collected
from halluscope.data.datasets import load_qa_jsonl
from halluscope.probes.linear import LinearProbe
from halluscope.selfconsistency import choose_layer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def cross_fitted_risk(hidden: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0):
    """(risk (n,), layer chosen for each item's fold (n,))."""
    risk = np.empty(len(y))
    layers = np.empty(len(y), dtype=int)
    for f, test in enumerate(stratified_kfold(y, k, seed)):
        train = np.setdiff1d(np.arange(len(y)), test)
        layer, l2 = choose_layer(hidden, y, train, seed)
        probe = LinearProbe(l2=l2, seed=seed).fit(hidden[train, layer, :], y[train])
        risk[test] = probe.predict_proba(hidden[test, layer, :])
        layers[test] = layer
        logger.info("fold %d/%d: layer %d", f + 1, k, layer)
    return risk, layers


def build_demo_frame(c, items, k: int = 5, seed: int = 0) -> pd.DataFrame:
    risk, layers = cross_fitted_risk(c.hidden, c.y_wrong, k, seed)
    all_idx = np.arange(len(c.y_wrong))
    token_risk = output_features(c.runs, all_idx)[:, 1]  # neg_min_logprob, empty answers imputed
    return pd.DataFrame({
        "question": [i.question for i in items],
        "model_answer": c.answers,
        "gold_answers": [" | ".join(i.answers[:5]) for i in items],
        "correct": (1 - c.y_wrong).astype(int),
        "probe_risk": np.round(risk, 4),
        "token_risk": np.round(token_risk, 4),
        "probe_layer": layers,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Export held-out per-question predictions for the demo.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--data", required=True, help="the QA jsonl the cache was collected from")
    parser.add_argument("--name", required=True, help="model name shown in the demo")
    parser.add_argument("--out", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    c = load_collected(args.cache)
    items = load_qa_jsonl(args.data, limit=len(c.y_wrong))
    if len(items) != len(c.y_wrong) or (c.questions and [i.question for i in items] != c.questions):
        raise ValueError("--data doesn't match the cache -- different file, order, or length")

    df = build_demo_frame(c, items, args.folds, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    (out.with_suffix(".name")).write_text(args.name)
    logger.info("wrote %d questions to %s (%.0f KB)", len(df), out, out.stat().st_size / 1024)


if __name__ == "__main__":
    main()
