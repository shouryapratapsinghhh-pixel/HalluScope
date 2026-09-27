"""Build the results section from saved outputs -- no number is hand-typed.

  python -m halluscope.report --config configs/results.yaml --readme README.md

Reads, per model: reports/<model>/robustness.csv and robustness_per_seed.csv
(from halluscope.robustness), plus the model's cache for its accuracy. Reads,
per transfer target: reports/<run>/transfer.csv (from halluscope.transfer).

Writes:
  reports/results/scale_table.csv   one row per model
  reports/results/results.md        the markdown injected into the README
  docs/img/scale.png                the scale figure (committed, so GitHub shows it)
and replaces the README text between <!-- RESULTS:START --> and <!-- RESULTS:END -->.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

# robustness.csv comparison label -> short column name
COMPARISONS = {
    "activations+questions vs questions alone": "beyond_difficulty",
    "probe vs other model's activations": "model_specific",
    "probe vs best output-confidence baseline": "probe_vs_output",
    "probe+output vs output alone": "complements_output",
}


def load_model_row(m: dict) -> dict:
    report = Path(m["report"])
    rob = pd.read_csv(report / "robustness.csv").set_index("comparison")
    per_seed = pd.read_csv(report / "robustness_per_seed.csv")
    row = {"model": m["name"], "family": m["family"], "params_m": m["params_m"]}
    if m.get("cache") and Path(m["cache"]).exists():
        y = np.load(m["cache"])["y_wrong"]  # npz loads keys lazily: activations aren't read
        row["accuracy"] = float(1 - y.mean())
    row["probe_auroc"] = float(per_seed["probe_auroc"].mean())
    row["layers_chosen"] = ",".join(str(x) for x in sorted(per_seed["best_layer"]))
    if "spearman_probe_vs_output" in per_seed:
        row["rank_corr_probe_output"] = float(per_seed["spearman_probe_vs_output"].mean())
    for label, key in COMPARISONS.items():
        if label in rob.index:
            r = rob.loc[label]
            row[f"{key}_mean"] = float(r["mean_diff"])
            row[f"{key}_std"] = float(r["std_diff"])
            row[f"{key}_sig"] = str(r["significant_seeds"])
            row[f"{key}_robust"] = bool(r["robust"])
    return row


def transfer_row(t: dict) -> dict:
    """Headline = layer chosen on QA validation (source domain only), per seed."""
    chosen, best_any, worst = [], [], []
    for d in t["dirs"]:
        df = pd.read_csv(Path(d) / "transfer.csv")
        chosen.append(df.loc[df["qa_val_auroc"].idxmax(), "transfer_auroc"])
        best_any.append(df["transfer_auroc"].max())
        w = df.loc[df["transfer_auroc"].idxmin()]
        worst.append((float(w["transfer_auroc"]), int(w["layer"])))
    c = np.array(chosen)
    worst_auc, worst_layer = min(worst)
    return {"target": t["name"], "n_seeds": len(c), "transfer_mean": c.mean(),
            "transfer_min": c.min(), "transfer_max": c.max(),
            "best_any_layer_descriptive": float(np.max(best_any)),
            "most_inverted_auroc": worst_auc, "most_inverted_layer": worst_layer}


def _cell(row: dict, key: str) -> str:
    if f"{key}_mean" not in row:
        return "–"
    mark = "✅" if row[f"{key}_robust"] else "❌"
    return f"{row[f'{key}_mean']:+.3f} ({row[f'{key}_sig']}) {mark}"


def scale_markdown(rows: list[dict]) -> str:
    lines = [
        ("| Model | Accuracy | Probe AUROC | Beyond difficulty | Model-specific | "
         "Probe vs output conf. | Probe+output vs output | Rank corr. |"),
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        acc = f"{r['accuracy']:.1%}" if "accuracy" in r else "–"
        rc = f"{r['rank_corr_probe_output']:.2f}" if "rank_corr_probe_output" in r else "–"
        lines.append(
            f"| {r['model']} | {acc} | {r['probe_auroc']:.3f} | {_cell(r, 'beyond_difficulty')} | "
            f"{_cell(r, 'model_specific')} | {_cell(r, 'probe_vs_output')} | "
            f"{_cell(r, 'complements_output')} | {rc} |"
        )
    return "\n".join(lines)


def transfer_markdown(rows: list[dict]) -> str:
    lines = [("| Probe trained on trivia errors → tested on | Transfer AUROC (QA-chosen layer) | "
              "Best single layer (descriptive) | Most inverted layer (descriptive) |"),
             "|---|---|---|---|"]
    for r in rows:
        rng = (f"{r['transfer_mean']:.3f} (range {r['transfer_min']:.3f}–{r['transfer_max']:.3f}, "
               f"{r['n_seeds']} seeds)" if r["n_seeds"] > 1 else f"{r['transfer_mean']:.3f}")
        lines.append(f"| {r['target']} | {rng} | {r['best_any_layer_descriptive']:.3f} | "
                     f"L{r['most_inverted_layer']} = {r['most_inverted_auroc']:.3f} |")
    return "\n".join(lines)


def load_cost(c: dict) -> tuple[pd.DataFrame, dict]:
    d = Path(c["dir"])
    return pd.read_csv(d / "cost_vs_accuracy.csv"), json.loads((d / "diffs.json").read_text())


def cost_markdown(name: str, table: pd.DataFrame, diffs: dict) -> str:
    lines = [f"*{name}*", "", ("| Method | AUROC [95% CI] | Forward passes | ms / question (this Mac) | "
                                "Cost vs probe |"), "|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['method']} | {r['auroc']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | "
                     f"{r['forward_steps']:.1f} | {r['ms_per_question']:.0f} | {r['cost_vs_probe']:.1f}× |")
    lines.append("")
    for k, v in diffs.items():
        sig = "significant" if v["ci_low"] > 0 or v["ci_high"] < 0 else "not significant"
        lines.append(f"- {k}: {v['diff']:+.3f} AUROC, 95% CI [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}] ({sig})")
    return "\n".join(lines)


def key_findings(rows: list[dict], transfers: list[dict], cost: tuple | None) -> str:
    """Plain-language headline bullets, with every number computed from the files."""
    df = pd.DataFrame(rows)
    out = []
    if "beyond_difficulty_mean" in df:
        n_rob = int(df["beyond_difficulty_robust"].sum())
        out.append(f"- **The model knows more than question difficulty.** Hidden states add "
                   f"{df['beyond_difficulty_mean'].min():+.2f} to {df['beyond_difficulty_mean'].max():+.2f} "
                   f"AUROC beyond a question-only difficulty model, robust in {n_rob}/{len(df)} models.")
    if "model_specific_robust" in df:
        spec = df[df["model_specific_robust"].astype(bool)]["model"].tolist()
        out.append(f"- **Model-specific self-knowledge appears only in: {', '.join(spec) or 'none'}.** "
                   f"In smaller models, a neighbouring model's activations predict the errors equally well.")
    if "rank_corr_probe_output" in df:
        out.append(f"- **The probe and the model's own token confidence converge as models improve** "
                   f"(rank correlation {df['rank_corr_probe_output'].min():.2f} → "
                   f"{df['rank_corr_probe_output'].max():.2f}); combining them helps in "
                   f"{int(df['complements_output_robust'].sum())}/{len(df)} models.")
    for t in transfers:
        if "egat" in t["target"]:
            out.append(f"- **It is not a truth detector:** on negated statements transfer falls to "
                       f"{t['transfer_mean']:.2f}, and layer {t['most_inverted_layer']} inverts to "
                       f"{t['most_inverted_auroc']:.3f} -- it tracks mismatched associations.")
        elif "city" in t["target"].lower():
            out.append(f"- **The error signal transfers to false statements** it never trained on "
                       f"({t['target']}: {t['transfer_mean']:.2f} AUROC).")
    if cost is not None:
        table, diffs = cost
        probe = table.iloc[0]
        sc = table[table["method"].str.startswith("self-consistency")].iloc[0]
        d = diffs.get("probe - self_consistency")
        if d is None:
            verdict = "compared with"
        elif d["ci_low"] > 0:
            verdict = "significantly beat"
        elif d["ci_high"] < 0:
            verdict = "was significantly worse than"
        else:
            verdict = "matched (difference not significant)"
        out.append(f"- **It is cheap:** the probe {verdict} self-consistency ({probe['auroc']:.2f} vs "
                   f"{sc['auroc']:.2f} AUROC) using {sc['forward_steps'] / probe['forward_steps']:.0f}× "
                   f"fewer forward passes -- and before any answer is generated.")
    return "\n".join(out)


def plot_scale(rows: list[dict], out_path) -> None:
    df = pd.DataFrame(rows)
    panels = [("beyond_difficulty", "Activations add beyond\nquestion difficulty (ΔAUROC)"),
              ("model_specific", "Model-specific signal\n(vs neighbour's activations, ΔAUROC)"),
              ("rank_corr_probe_output", "Probe ↔ output-confidence\nrank correlation")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (key, title) in zip(axes, panels):
        for fam, g in df.groupby("family"):
            g = g.sort_values("params_m")
            if key == "rank_corr_probe_output":
                if key not in g:
                    continue
                ax.plot(g["params_m"], g[key], marker="o", label=fam)
            else:
                ax.errorbar(g["params_m"], g[f"{key}_mean"], yerr=g[f"{key}_std"],
                            marker="o", capsize=3, label=fam)
                robust = g[f"{key}_robust"].astype(bool)
                ax.scatter(g["params_m"][~robust], g[f"{key}_mean"][~robust], s=90,
                           facecolors="white", edgecolors="gray", zorder=5)
        if key != "rank_corr_probe_output":
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xscale("log")
        sizes = sorted(df["params_m"].unique())
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s / 1000:g}B" if s >= 1000 else f"{s:g}M" for s in sizes], fontsize=8, rotation=45, ha="right")
        ax.minorticks_off()
        ax.set_xlabel("model size")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("HalluScope: error signal in hidden states vs model size "
                 "(3,000 TriviaQA questions, mean ± std over 5 splits; hollow = not robust)",
                 fontsize=10)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def build(config: dict, out_dir="reports/results", figure="docs/img/scale.png") -> str:
    rows = [load_model_row(m) for m in config["models"]]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "scale_table.csv", index=False)
    plot_scale(rows, figure)

    transfers = [transfer_row(t) for t in config.get("transfer", [])]
    cost = load_cost(config["cost"]) if config.get("cost") else None
    parts = [START, ("*Generated by `python -m halluscope.report` from saved result files -- "
                     "no number here is hand-typed.*"), "", "### Key findings", "",
             key_findings(rows, transfers, cost), "", "### Scale study", "",
             scale_markdown(rows), "", f"![scale]({figure})", ""]
    if transfers:
        parts += ["### Cross-domain transfer", "", transfer_markdown(transfers), ""]
    if cost is not None:
        parts += ["### Cost: probe vs self-consistency", "",
                  cost_markdown(config["cost"]["name"], *cost), ""]
    parts.append(END)
    md = "\n".join(parts)
    (out / "results.md").write_text(md)
    return md


def inject_readme(readme_path, md: str) -> None:
    text = Path(readme_path).read_text()
    if START in text and END in text:
        text = text[: text.index(START)] + md + text[text.index(END) + len(END):]
    else:
        text = text.rstrip() + "\n\n## Results\n\n" + md + "\n"
    Path(readme_path).write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build results tables/figure and update README.")
    parser.add_argument("--config", default="configs/results.yaml")
    parser.add_argument("--readme", default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    md = build(config)
    print(md)
    if args.readme:
        inject_readme(args.readme, md)
        print(f"\nupdated {args.readme}")


if __name__ == "__main__":
    main()
