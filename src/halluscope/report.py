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
    chosen, best_any = [], []
    for d in t["dirs"]:
        df = pd.read_csv(Path(d) / "transfer.csv")
        chosen.append(df.loc[df["qa_val_auroc"].idxmax(), "transfer_auroc"])
        best_any.append(df["transfer_auroc"].max())
    c = np.array(chosen)
    return {"target": t["name"], "n_seeds": len(c), "transfer_mean": c.mean(),
            "transfer_min": c.min(), "transfer_max": c.max(),
            "best_any_layer_descriptive": float(np.max(best_any))}


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
              "Best single layer (descriptive only) |"), "|---|---|---|"]
    for r in rows:
        rng = (f"{r['transfer_mean']:.3f} (range {r['transfer_min']:.3f}–{r['transfer_max']:.3f}, "
               f"{r['n_seeds']} seeds)" if r["n_seeds"] > 1 else f"{r['transfer_mean']:.3f}")
        lines.append(f"| {r['target']} | {rng} | {r['best_any_layer_descriptive']:.3f} |")
    return "\n".join(lines)


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

    parts = [START, ("*Generated by `python -m halluscope.report` from saved result files -- "
                     "no number here is hand-typed.*"), "", "### Scale study", "",
             scale_markdown(rows), "", f"![scale]({figure})", ""]
    if config.get("transfer"):
        parts += ["### Cross-domain transfer", "", transfer_markdown(
            [transfer_row(t) for t in config["transfer"]]), ""]
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
