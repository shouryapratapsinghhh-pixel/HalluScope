"""Robustness: does the result survive different train/test splits?

  python -m halluscope.robustness --cache cache/qwen05_tqa.npz \
      --control-cache cache/pythia410m_tqa.npz --seeds 0 1 2 3 4 --out reports/qwen05

Each seed is a completely fresh stratified split, with the layer and L2
re-chosen on that seed's train data. The bootstrap CIs in analyze.py
cover test-set sampling noise; THIS covers split noise (which items landed
in train vs test, and which layer got picked as a result).

A verdict counts as robust only if it is significant on most seeds AND
its mean effect is clearly positive. "Significant on 1 of 5" is noise.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from halluscope.analyze import analyze
from halluscope.collect import Collected, load_collected
from halluscope.data.datasets import load_qa_jsonl

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

VERDICTS = {
    "probe_vs_best_baseline": "probe vs best output-confidence baseline",
    "probe_vs_question_control": "probe vs question-features control",
    "activations_add_beyond_questions": "activations+questions vs questions alone",
    "probe_vs_other_model_control": "probe vs other model's activations",
    "probe_adds_beyond_output": "probe+output vs output alone",
    "output_adds_beyond_probe": "probe+output vs probe alone",
}


def run_seeds(
    c: Collected,
    questions: list[str] | None,
    control: Collected | None,
    seeds: list[int],
    n_boot: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per_seed rows, aggregate verdict table)."""
    logging.getLogger("halluscope.analyze").setLevel(logging.WARNING)  # mute per-layer lines
    rows = []
    for seed in seeds:
        _, summary, h = analyze(c, questions, control, seed=seed, n_boot=n_boot, sweep_n_boot=0)
        row = {
            "seed": seed,
            "best_layer": h["best_layer"],
            "probe_auroc": summary.loc[f"linear_probe@L{h['best_layer']}", "auroc"],
            "best_baseline": h["best_baseline"],
            "probe_vs_best_baseline": h["probe_minus_best_baseline_auroc"],
            "probe_vs_best_baseline_sig": h["significant"],
        }
        for key in list(VERDICTS)[1:]:
            if key in h:
                row[key] = h[key]["diff"]
                row[f"{key}_sig"] = h[key]["significant"]
        row["spearman_probe_vs_output"] = h["spearman_probe_vs_output"]
        rows.append(row)
        logger.info("seed %d: layer %d, probe AUROC %.3f", seed, h["best_layer"], row["probe_auroc"])

    per_seed = pd.DataFrame(rows)
    agg = []
    for key, label in VERDICTS.items():
        if key not in per_seed:
            continue
        diffs = per_seed[key].to_numpy(dtype=float)
        n_sig = int(per_seed[f"{key}_sig"].sum())
        agg.append(
            {
                "comparison": label,
                "mean_diff": diffs.mean(),
                "std_diff": diffs.std(ddof=1) if len(diffs) > 1 else 0.0,
                "min_diff": diffs.min(),
                "significant_seeds": f"{n_sig}/{len(seeds)}",
                # robust = significant on a majority of splits AND never flips negative on average
                "robust": bool(n_sig > len(seeds) / 2 and diffs.mean() > 0),
            }
        )
    return per_seed, pd.DataFrame(agg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat the analysis over several splits.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--control-cache", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    c = load_collected(args.cache)
    questions = c.questions
    if questions is None and args.data:
        questions = [item.question for item in load_qa_jsonl(args.data, limit=args.limit)]
        if len(questions) != len(c.y_wrong):
            raise ValueError("--data/--limit don't match the cache; use the same --limit as collection")
    control = load_collected(args.control_cache) if args.control_cache else None

    per_seed, agg = run_seeds(c, questions, control, args.seeds, n_boot=args.n_boot)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "robustness_per_seed.csv", index=False)
    agg.to_csv(out / "robustness.csv", index=False)

    print(per_seed[["seed", "best_layer", "probe_auroc", "best_baseline"]].round(3).to_string(index=False))
    print()
    print(agg.round(3).to_string(index=False))
    print(f"\nrank correlation probe vs output confidence: "
          f"{per_seed['spearman_probe_vs_output'].mean():.2f} (mean over seeds)")
    print(f"layers chosen across seeds: {sorted(per_seed['best_layer'].tolist())}"
          f"  (spread {int(np.ptp(per_seed['best_layer']))})")
    print(f"wrote {out}/robustness.csv")


if __name__ == "__main__":
    main()
