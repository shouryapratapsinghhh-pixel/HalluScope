"""End-to-end run: collect activations -> train probe -> compare to baselines.

  python -m halluscope.run --model EleutherAI/pythia-410m --data data/raw/triviaqa.jsonl --limit 1000
  python -m halluscope.run --model tiny --data toy          # offline smoke test

Split: 70/30 train/test, seeded, stratified by label. The probe's L2
strength is chosen on a validation slice of TRAIN only.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from halluscope.collect import collect_qa, load_collected, save_collected
from halluscope.data.datasets import load_qa_jsonl, make_toy_qa
from halluscope.eval.baselines import BASELINES, random_risk
from halluscope.eval.metrics import summarize
from halluscope.models.lm import load_model, tiny_random_model
from halluscope.probes.linear import LinearProbe, select_l2

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def stratified_split(y: np.ndarray, test_frac: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        n_test = round(len(idx) * test_frac)
        test.extend(idx[:n_test])
        train.extend(idx[n_test:])
    return np.array(sorted(train)), np.array(sorted(test))


def evaluate(collected, layer: int | None, seed: int = 0) -> pd.DataFrame:
    y = collected.y_wrong
    n_layers = collected.hidden.shape[1]
    layer = n_layers // 2 if layer is None else layer  # middle layer by default (the hypothesis)

    tr, te = stratified_split(y, seed=seed)
    rows = {"random": summarize(y[te], random_risk(len(te), seed))}
    test_runs = [collected.runs[i] for i in te]
    for name, fn in BASELINES.items():
        rows[name] = summarize(y[te], fn(test_runs))

    X = collected.hidden[:, layer, :]
    l2 = select_l2(X[tr], y[tr], seed=seed)
    probe = LinearProbe(l2=l2, seed=seed).fit(X[tr], y[tr])
    rows[f"linear_probe@L{layer}"] = summarize(y[te], probe.predict_proba(X[te]), is_probability=True)

    logger.info("layer=%d of %d, l2=%.0e, train=%d, test=%d", layer, n_layers - 1, l2, len(tr), len(te))
    return pd.DataFrame(rows).T


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict LLM errors from hidden states.")
    parser.add_argument("--model", default="tiny", help="HF model name, or 'tiny' for offline test")
    parser.add_argument("--data", default="toy", help="path to QA jsonl, or 'toy'")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--layer", type=int, default=None, help="default: middle layer")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"],
                        help="auto = float16 on Apple Silicon/GPU (half the memory), float32 on CPU")
    parser.add_argument("--cache", default=None, help="npz path to save/load activations")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.cache and Path(args.cache).exists():
        logger.info("loading cached activations from %s", args.cache)
        collected = load_collected(args.cache)
    else:
        model, tok = tiny_random_model() if args.model == "tiny" else load_model(args.model, dtype=args.dtype)
        items = make_toy_qa() if args.data == "toy" else load_qa_jsonl(args.data, limit=args.limit)
        logger.info("running %s on %d questions", args.model, len(items))
        collected = collect_qa(model, tok, items, max_new_tokens=args.max_new_tokens)
        if args.cache:
            save_collected(collected, args.cache)

    acc = 1 - collected.y_wrong.mean()
    logger.info("model accuracy: %.3f (%d questions)", acc, len(collected.y_wrong))
    if len(np.unique(collected.y_wrong)) < 2:
        logger.warning(
            "model got every question %s -- nothing to predict. Use a real model and more data.",
            "wrong" if acc == 0 else "right",
        )
        return

    table = evaluate(collected, args.layer, seed=args.seed)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
