"""Phase 1 analysis, run entirely from cached activations (no model re-runs).

  python -m halluscope.analyze --cache cache/pythia410m_tqa.npz \
      --data data/raw/triviaqa.jsonl --limit 1000 --out reports/pythia410m

  # optional cross-model control: another model's cache on the SAME questions
  python -m halluscope.analyze --cache cache/pythia410m_tqa.npz \
      --control-cache cache/qwen05_tqa.npz --out reports/pythia410m

Leakage rules:
  - One stratified train/test split, seeded, shared by everything.
  - The headline layer is chosen by AUROC on an inner validation slice of
    TRAIN. The test-set layer sweep is reported as a descriptive plot, but
    it never picks the headline number (that would be test-set tuning).
  - L2 is chosen per layer on the same inner validation slice.
  - "Best baseline" IS picked on test, which favours the baseline -- a
    conservative choice: the probe has to beat the baseline's best case.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from halluscope.collect import Collected, load_collected
from halluscope.data.datasets import load_qa_jsonl
from halluscope.eval.baselines import BASELINES, EMPTY_ANSWER_RISK, random_risk
from halluscope.eval.controls import question_features
from halluscope.eval.metrics import auroc, summarize
from halluscope.eval.stats import bootstrap_ci, paired_bootstrap_diff
from halluscope.probes.linear import LinearProbe, select_l2
from halluscope.run import stratified_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def output_features(runs, tr) -> np.ndarray:
    """(n, 3) output-confidence features for ALL items. Empty answers get a
    sentinel risk; replace it with the TRAIN max of real values so one
    outlier doesn't wreck standardization (train-only, so no test leakage).
    Needs the generated answer, so anything using these is POST-generation.
    """
    X = np.column_stack([fn(runs) for fn in BASELINES.values()]).astype(np.float64)
    for j in range(X.shape[1]):
        real = X[tr, j][X[tr, j] < EMPTY_ANSWER_RISK]
        fill = real.max() if len(real) else 0.0
        X[X[:, j] >= EMPTY_ANSWER_RISK, j] = fill
    return X


def stratified_kfold(y: np.ndarray, k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """k folds (positions into y), each with both classes in proportion."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return [np.array(sorted(f)) for f in folds]


def oof_probe_scores(X_tr: np.ndarray, y_tr: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold probe p(wrong) for every TRAIN row: each row is scored
    by a probe that never saw it (L2 re-chosen inside each fold). These are
    honest stand-ins for how the probe behaves on unseen data, which is
    what the stacking meta-model must learn from."""
    oof = np.empty(len(y_tr))
    for val in stratified_kfold(y_tr, k, seed):
        fit = np.setdiff1d(np.arange(len(y_tr)), val)
        l2 = select_l2(X_tr[fit], y_tr[fit], seed=seed)
        oof[val] = LinearProbe(l2=l2, seed=seed).fit(X_tr[fit], y_tr[fit]).predict_proba(X_tr[val])
    return oof


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stacked_risk(X_act, p_probe_test, X_side, y, tr, te, seed=0, k=5) -> np.ndarray:
    """Stacking: combine the probe (as ONE out-of-fold score) with a small
    set of side features (output confidence, or question features) in a
    low-dimensional logistic regression.

    Why not just concatenate activations + side features into one model?
    With hundreds of activation dims and one shared L2 penalty, the few
    side features get regularized as hard as the activations and drown --
    an earlier version of this repo did exactly that and gave unstable,
    uninterpretable results. Here both sides enter as a handful of
    comparable columns. Nothing from the test split is used to fit anything.
    """
    oof = oof_probe_scores(X_act[tr], y[tr], k=k, seed=seed)
    meta_tr = np.column_stack([_logit(oof), X_side[tr]])
    meta_te = np.column_stack([_logit(p_probe_test), X_side[te]])
    l2 = select_l2(meta_tr, y[tr], seed=seed)
    return LinearProbe(l2=l2, seed=seed).fit(meta_tr, y[tr]).predict_proba(meta_te)


def _fit_select(X_tr, y_tr, seed):
    """Choose L2 on an inner val slice of train, return (l2, inner-val AUROC)."""
    inner_tr, inner_val = stratified_split(y_tr, test_frac=0.25, seed=seed + 1)
    l2 = select_l2(X_tr[inner_tr], y_tr[inner_tr], seed=seed)
    probe = LinearProbe(l2=l2, seed=seed).fit(X_tr[inner_tr], y_tr[inner_tr])
    return l2, auroc(y_tr[inner_val], probe.predict_proba(X_tr[inner_val]))


def layer_sweep(hidden, y, tr, te, seed=0, n_boot=1000) -> pd.DataFrame:
    rows = []
    for layer in range(hidden.shape[1]):
        X = hidden[:, layer, :]
        l2, val_auc = _fit_select(X[tr], y[tr], seed)
        probe = LinearProbe(l2=l2, seed=seed).fit(X[tr], y[tr])
        if n_boot > 0:
            test_auc, lo, hi = bootstrap_ci(y[te], probe.predict_proba(X[te]), n_boot=n_boot, seed=seed)
        else:  # robustness mode: only the train-side val AUROC is needed per layer
            test_auc, lo, hi = auroc(y[te], probe.predict_proba(X[te])), float("nan"), float("nan")
        rows.append(
            {"layer": layer, "l2": l2, "val_auroc": val_auc, "test_auroc": test_auc,
             "ci_low": lo, "ci_high": hi}
        )
        logger.info("layer %2d: val=%.3f test=%.3f [%.3f, %.3f] l2=%g",
                    layer, val_auc, test_auc, lo, hi, l2)
    return pd.DataFrame(rows)


def _probe_risk(X, y, tr, te, seed):
    l2, _ = _fit_select(X[tr], y[tr], seed)
    probe = LinearProbe(l2=l2, seed=seed).fit(X[tr], y[tr])
    return probe.predict_proba(X[te])


def analyze(
    c: Collected,
    questions: list[str] | None = None,
    control: Collected | None = None,
    seed: int = 0,
    n_boot: int = 1000,
    sweep_n_boot: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    y = c.y_wrong
    tr, te = stratified_split(y, seed=seed)
    sweep_boot = n_boot if sweep_n_boot is None else sweep_n_boot
    sweep = layer_sweep(c.hidden, y, tr, te, seed=seed, n_boot=sweep_boot)
    best_layer = int(sweep.loc[sweep["val_auroc"].idxmax(), "layer"])  # chosen on TRAIN

    test_runs = [c.runs[i] for i in te]
    risks: dict[str, tuple[np.ndarray, bool]] = {
        "random": (random_risk(len(te), seed), False),
        **{name: (fn(test_runs), False) for name, fn in BASELINES.items()},
    }
    if questions is not None:
        risks["control:question_features"] = (
            _probe_risk(question_features(questions), y, tr, te, seed), True
        )
    if control is not None:
        if len(control.y_wrong) != len(y):
            raise ValueError("control cache has a different number of questions")
        if control.questions and c.questions and control.questions != c.questions:
            raise ValueError("control cache questions don't match -- not the same dataset/order")
        mid = control.hidden.shape[1] // 2
        # another model's activations, predicting THIS model's errors
        risks["control:other_model_probe"] = (_probe_risk(control.hidden[:, mid, :], y, tr, te, seed), True)
    X_act = c.hidden[:, best_layer, :]
    p_probe = _probe_risk(X_act, y, tr, te, seed)
    risks[f"linear_probe@L{best_layer}"] = (p_probe, True)

    # complementarity: do activations and output confidence catch DIFFERENT errors?
    # Both sides are small learned models, so the comparison is fair.
    X_out = output_features(c.runs, tr)
    risks["output_features_combined"] = (_probe_risk(X_out, y, tr, te, seed), True)
    risks["probe+output_features"] = (stacked_risk(X_act, p_probe, X_out, y, tr, te, seed), True)

    if questions is not None:
        # does the model's activation add anything BEYOND question difficulty?
        X_q = question_features(questions)
        risks["probe+question_features"] = (stacked_risk(X_act, p_probe, X_q, y, tr, te, seed), True)

    rows = {}
    for name, (risk, is_prob) in risks.items():
        m = summarize(y[te], risk, is_probability=is_prob)
        _, m["auroc_ci_low"], m["auroc_ci_high"] = bootstrap_ci(y[te], risk, n_boot=n_boot, seed=seed)
        rows[name] = m
    summary = pd.DataFrame(rows).T

    baseline_names = list(BASELINES)
    best_baseline = max(baseline_names, key=lambda n: summary.loc[n, "auroc"])
    probe_name = f"linear_probe@L{best_layer}"
    diff, lo, hi = paired_bootstrap_diff(
        y[te], risks[probe_name][0], risks[best_baseline][0], n_boot=n_boot, seed=seed
    )
    headline = {
        "best_layer": best_layer,
        "n_layers": c.hidden.shape[1] - 1,
        "model_accuracy": float(1 - y.mean()),
        "n_test": len(te),
        "n_test_correct": int((y[te] == 0).sum()),
        "best_baseline": best_baseline,
        "probe_minus_best_baseline_auroc": diff,
        "diff_ci_low": lo,
        "diff_ci_high": hi,
        "significant": bool(lo > 0),
    }

    def _paired(a: str, b: str, key: str) -> None:
        if a in risks and b in risks:
            d, dlo, dhi = paired_bootstrap_diff(y[te], risks[a][0], risks[b][0], n_boot=n_boot, seed=seed)
            headline[key] = {"diff": d, "ci_low": dlo, "ci_high": dhi, "significant": bool(dlo > 0)}

    # the comparisons that separate self-knowledge from question difficulty
    _paired(probe_name, "control:question_features", "probe_vs_question_control")
    _paired("probe+question_features", "control:question_features", "activations_add_beyond_questions")
    _paired(probe_name, "control:other_model_probe", "probe_vs_other_model_control")
    # complementarity (fair: output side is also a learned combination, not one score)
    _paired("probe+output_features", "output_features_combined", "probe_adds_beyond_output")
    _paired("probe+output_features", probe_name, "output_adds_beyond_probe")
    # do the two scorers rank errors the same way? (Spearman on test)
    r_probe = pd.Series(risks[probe_name][0]).rank()
    r_out = pd.Series(risks["output_features_combined"][0]).rank()
    headline["spearman_probe_vs_output"] = float(r_probe.corr(r_out))
    return sweep, summary, headline


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer sweep, CIs and difficulty controls.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--data", default=None, help="QA jsonl, to recover questions for old caches")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--control-cache", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    c = load_collected(args.cache)
    questions = c.questions
    if questions is None and args.data:
        questions = [item.question for item in load_qa_jsonl(args.data, limit=args.limit)]
        if len(questions) != len(c.y_wrong):
            raise ValueError(
                f"{args.data} gave {len(questions)} questions but the cache has "
                f"{len(c.y_wrong)} -- pass the same --limit used when collecting"
            )
    if questions is None:
        logger.warning("no questions available -- skipping the question-features control")
    control = load_collected(args.control_cache) if args.control_cache else None

    sweep, summary, headline = analyze(c, questions, control, seed=args.seed, n_boot=args.n_boot)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out / "layer_sweep.csv", index=False)
    summary.to_csv(out / "summary.csv")
    import json

    (out / "headline.json").write_text(json.dumps(headline, indent=2))
    from halluscope.viz import plot_layer_sweep

    plot_layer_sweep(sweep, summary, headline, out / "layer_sweep.png")

    print(summary[["auroc", "auroc_ci_low", "auroc_ci_high", "aurc", "sel_acc@50", "ece"]]
          .round(3).to_string())
    print(f"\nheadline layer {headline['best_layer']} (chosen on train). "
          f"probe - {headline['best_baseline']} = {headline['probe_minus_best_baseline_auroc']:+.3f} "
          f"AUROC, 95% CI [{headline['diff_ci_low']:+.3f}, {headline['diff_ci_high']:+.3f}] "
          f"-> {'SIGNIFICANT' if headline['significant'] else 'not significant'}")
    labels = {
        "probe_vs_question_control": "probe vs question-features control",
        "activations_add_beyond_questions": "activations+questions vs questions alone",
        "probe_vs_other_model_control": "probe vs other model's activations",
        "probe_adds_beyond_output": "probe+output vs output alone",
        "output_adds_beyond_probe": "probe+output vs probe alone",
    }
    for key, label in labels.items():
        if key in headline:
            h = headline[key]
            print(f"{label}: {h['diff']:+.3f} AUROC, 95% CI [{h['ci_low']:+.3f}, {h['ci_high']:+.3f}] "
                  f"-> {'SIGNIFICANT' if h['significant'] else 'not significant'}")
    print(f"rank correlation, probe vs output confidence: {headline['spearman_probe_vs_output']:.2f}")
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
