"""Figures. Headless (Agg), always savefig, never show."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_layer_sweep(sweep: pd.DataFrame, summary: pd.DataFrame, headline: dict, out_path) -> str:
    """Test AUROC per layer with 95% CI band, against the best output-
    confidence baseline and the question-features control."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep["layer"], sweep["test_auroc"], marker="o", color="steelblue", label="linear probe")
    ax.fill_between(sweep["layer"], sweep["ci_low"], sweep["ci_high"], color="steelblue", alpha=0.2,
                    label="95% bootstrap CI")

    best = headline["best_baseline"]
    ax.axhline(summary.loc[best, "auroc"], color="firebrick", linestyle="--",
               label=f"best output-confidence baseline ({best})")
    if "control:question_features" in summary.index:
        ax.axhline(summary.loc["control:question_features", "auroc"], color="gray", linestyle=":",
                   label="control: question features only")
    if "control:other_model_probe" in summary.index:
        ax.axhline(summary.loc["control:other_model_probe", "auroc"], color="darkorange",
                   linestyle="-.", label="control: other model's activations")
    ax.axhline(0.5, color="black", linewidth=0.8, alpha=0.4)
    ax.axvline(headline["best_layer"], color="green", alpha=0.4,
               label=f"layer chosen on train (L{headline['best_layer']})")

    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("AUROC for predicting a wrong answer")
    ax.set_title("Where does the 'I'm about to be wrong' signal live?")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return str(out_path)
