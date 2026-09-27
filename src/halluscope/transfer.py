"""Cross-domain transfer: does the "I'm about to be wrong" direction learned
on trivia QA also flag FALSE statements?

  python -m halluscope.transfer --model Qwen/Qwen2.5-1.5B \
      --qa-cache cache/qwen15_tqa3k.npz --statements data/raw/cities.csv \
      --stmt-cache cache/qwen15_cities.npz --out reports/transfer_qwen15

For every layer:
  - transfer_auroc: probe trained on ALL QA items (label = model answered
    wrong), tested on statements (label = statement is false). No
    statement label is used to train it.
  - in_domain_auroc: probe trained and tested on statements (70/30 split),
    as a reference ceiling -- how much truth signal exists at that layer.
  - cosine: angle between the QA-error direction and the statement-falsity
    direction (in raw activation space). Near 0 means the two probes use
    unrelated directions, whatever their AUROCs say.

The headline layer is chosen on QA validation data (source domain only),
never by looking at statement results.

Honest caveat: QA activations are read at the end of "Question: ...
Answer:"; statement activations at the end of the statement. A failed
transfer could mean "no shared signal" OR "different token position" --
the in-domain reference and the cosine help tell these apart, but can't
fully separate them.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from halluscope.analyze import _fit_select
from halluscope.collect import load_collected
from halluscope.data.datasets import StatementItem, load_truth_csv
from halluscope.eval.metrics import auroc
from halluscope.eval.stats import bootstrap_ci
from halluscope.probes.linear import LinearProbe
from halluscope.run import stratified_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def collect_statements(model, tokenizer, items: list[StatementItem]) -> tuple[np.ndarray, np.ndarray]:
    """(hidden (n, n_layers + 1, d), y_false (n,)) -- 1 = the statement is FALSE,
    matching the QA convention that 1 = 'wrong'."""
    from halluscope.models.lm import prompt_hidden_states

    hidden = []
    for i, item in enumerate(items):
        hidden.append(prompt_hidden_states(model, tokenizer, item.statement))
        if (i + 1) % 100 == 0:
            logger.info("statements %d / %d", i + 1, len(items))
    y_false = np.array([1 - item.label for item in items], dtype=np.int64)
    return np.stack(hidden).astype(np.float32), y_false


def _raw_direction(probe: LinearProbe) -> np.ndarray:
    """Probe weights mapped back to raw activation space (w / std), so
    directions from probes with different standardizations are comparable."""
    w = probe.direction() / probe.std_
    return w / (np.linalg.norm(w) + 1e-12)


def transfer_sweep(qa_hidden, y_wrong, st_hidden, y_false, seed=0, n_boot=500) -> pd.DataFrame:
    if qa_hidden.shape[1:] != st_hidden.shape[1:]:
        raise ValueError(
            f"QA activations {qa_hidden.shape[1:]} and statement activations "
            f"{st_hidden.shape[1:]} differ -- were they collected with the same model?"
        )
    st_tr, st_te = stratified_split(y_false, seed=seed)
    rows = []
    for layer in range(qa_hidden.shape[1]):
        Xq, Xs = qa_hidden[:, layer, :], st_hidden[:, layer, :]
        if np.allclose(Xq.std(axis=0), 0) or np.allclose(Xs.std(axis=0), 0):
            continue  # e.g. layer 0 when every prompt ends on the same token

        l2_q, val_q = _fit_select(Xq, y_wrong, seed)  # source-domain validation only
        qa_probe = LinearProbe(l2=l2_q, seed=seed).fit(Xq, y_wrong)
        t_auc, t_lo, t_hi = bootstrap_ci(y_false, qa_probe.predict_proba(Xs), n_boot=n_boot, seed=seed)

        l2_s, _ = _fit_select(Xs[st_tr], y_false[st_tr], seed)
        st_probe = LinearProbe(l2=l2_s, seed=seed).fit(Xs[st_tr], y_false[st_tr])
        in_auc = auroc(y_false[st_te], st_probe.predict_proba(Xs[st_te]))

        cos = float(_raw_direction(qa_probe) @ _raw_direction(st_probe))
        rows.append({"layer": layer, "qa_val_auroc": val_q, "transfer_auroc": t_auc,
                     "transfer_ci_low": t_lo, "transfer_ci_high": t_hi,
                     "in_domain_auroc": in_auc, "cosine": cos})
        logger.info("layer %2d: transfer %.3f [%.3f, %.3f]  in-domain %.3f  cos %+.2f",
                    layer, t_auc, t_lo, t_hi, in_auc, cos)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="QA error probe -> true/false statements.")
    parser.add_argument("--model", required=True, help="same model the QA cache was collected with")
    parser.add_argument("--qa-cache", required=True)
    parser.add_argument("--statements", required=True, help="Geometry-of-Truth style CSV")
    parser.add_argument("--stmt-cache", default=None, help="npz to save/load statement activations")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()

    qa = load_collected(args.qa_cache)
    if args.stmt_cache and Path(args.stmt_cache).exists():
        data = np.load(args.stmt_cache)
        st_hidden, y_false = data["hidden"], data["y_false"]
    else:
        from halluscope.models.lm import load_model, tiny_random_model

        model, tok = tiny_random_model() if args.model == "tiny" else load_model(args.model, dtype=args.dtype)
        st_hidden, y_false = collect_statements(model, tok, load_truth_csv(args.statements))
        if args.stmt_cache:
            Path(args.stmt_cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.stmt_cache, hidden=st_hidden, y_false=y_false)

    sweep = transfer_sweep(qa.hidden, qa.y_wrong, st_hidden, y_false, seed=args.seed, n_boot=args.n_boot)
    best = sweep.loc[sweep["qa_val_auroc"].idxmax()]  # chosen on QA (source) data only

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out / "transfer.csv", index=False)
    print(f"\nlayer chosen on QA validation: L{int(best['layer'])}")
    print(f"  transfer AUROC (QA probe -> statements): {best['transfer_auroc']:.3f} "
          f"[{best['transfer_ci_low']:.3f}, {best['transfer_ci_high']:.3f}]")
    print(f"  in-domain statement probe (reference):  {best['in_domain_auroc']:.3f}")
    print(f"  cosine(QA-error dir, falsity dir):      {best['cosine']:+.2f}")
    print(f"best transfer at any layer (descriptive only): "
          f"L{int(sweep.loc[sweep['transfer_auroc'].idxmax(), 'layer'])} "
          f"= {sweep['transfer_auroc'].max():.3f}")
    print(f"wrote {out}/transfer.csv")


if __name__ == "__main__":
    main()
