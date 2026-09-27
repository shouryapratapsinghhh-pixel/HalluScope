"""Export a servable probe artifact from a cached model run.

  python -m halluscope.export --model Qwen/Qwen2.5-1.5B --qa-cache cache/qwen15_tqa3k.npz \
      --out artifacts/qwen15_probe

The final probe is trained on ALL cached questions (more data = a better
deployed probe). The layer and L2 are still chosen on an inner validation
split, and that validation AUROC is saved as an honest held-out estimate --
the multi-seed test-set numbers in the README remain the reported results.

Writes <out>/probe.npz (weights + train standardization) and <out>/meta.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from halluscope.collect import load_collected
from halluscope.probes.linear import LinearProbe, save_probe
from halluscope.run import stratified_split
from halluscope.selfconsistency import choose_layer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def export(model_name: str, qa_cache: str, out: str, seed: int = 0, dtype: str = "auto") -> dict:
    c = load_collected(qa_cache)
    if len(np.unique(c.y_wrong)) < 2:
        raise ValueError("cache has only one class of answers -- nothing to learn")
    all_idx = np.arange(len(c.y_wrong))
    layer, l2 = choose_layer(c.hidden, c.y_wrong, all_idx, seed)

    # honest estimate: refit on a train split, score the held-out part
    tr, va = stratified_split(c.y_wrong, test_frac=0.2, seed=seed)
    from halluscope.eval.metrics import auroc

    check = LinearProbe(l2=l2, seed=seed).fit(c.hidden[tr, layer, :], c.y_wrong[tr])
    val_auroc = auroc(c.y_wrong[va], check.predict_proba(c.hidden[va, layer, :]))

    probe = LinearProbe(l2=l2, seed=seed).fit(c.hidden[:, layer, :], c.y_wrong)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_probe(probe, out_dir / "probe.npz")
    meta = {
        "model": model_name, "dtype": dtype, "layer": int(layer), "l2": float(l2),
        "n_train_questions": len(c.y_wrong),
        "model_accuracy_on_cache": float(1 - c.y_wrong.mean()),
        "heldout_auroc_estimate": float(val_auroc),
        "hidden_size": int(c.hidden.shape[2]),
        "prompt_format": "Question: {question}\\nAnswer:",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("exported layer %d probe to %s (held-out AUROC estimate %.3f)", layer, out_dir, val_auroc)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a servable probe artifact.")
    parser.add_argument("--model", required=True, help="the model the cache was collected with, or 'tiny'")
    parser.add_argument("--qa-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(export(args.model, args.qa_cache, args.out, args.seed, args.dtype), indent=2))


if __name__ == "__main__":
    main()
