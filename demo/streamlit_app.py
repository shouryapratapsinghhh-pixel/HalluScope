"""HalluScope demo -- precomputed, held-out predictions (no live model).

Run locally from the repo root:   streamlit run demo/streamlit_app.py
Deploy: Streamlit Community Cloud, main file path = demo/streamlit_app.py
(demo/requirements.txt is used, so the free tier never installs PyTorch).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = Path(os.environ.get("HALLUSCOPE_DEMO_DIR", ROOT / "reports" / "demo"))
RESULTS_MD = ROOT / "reports" / "results" / "results.md"
SCALE_PNG = ROOT / "docs" / "img" / "scale.png"


def auroc(y_wrong: np.ndarray, risk: np.ndarray) -> float:
    """Probability a random wrong answer gets higher risk than a random right one."""
    ranks = pd.Series(risk).rank().to_numpy()
    n_pos, n_neg = y_wrong.sum(), len(y_wrong) - y_wrong.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y_wrong == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def keep_safest(df: pd.DataFrame, col: str, coverage: float) -> pd.DataFrame:
    """The `coverage` fraction of questions with the LOWEST risk (the ones answered)."""
    n = max(1, round(coverage * len(df)))
    return df.sort_values(col, kind="mergesort").head(n)


@st.cache_data
def load_models() -> dict[str, pd.DataFrame]:
    models = {}
    for csv in sorted(DEMO_DIR.glob("*.csv")):
        name_file = csv.with_suffix(".name")
        name = name_file.read_text().strip() if name_file.exists() else csv.stem
        models[name] = pd.read_csv(csv)
    return models


st.set_page_config(page_title="HalluScope", layout="wide")
st.title("HalluScope")
st.markdown(
    "**Can a language model know it's about to be wrong — before it answers?** HalluScope reads "
    "the model's internal activations right after it reads a question and predicts whether its "
    "answer will be wrong."
)
st.caption(
    "Precomputed demo: free hosting can't run a 1.5B-parameter model, so these are real "
    "predictions saved from runs on a laptop. Every risk score is held out: each question was "
    "scored by a probe that never saw it."
)

models = load_models()
if not models:
    st.error(f"No prediction files found in {DEMO_DIR}. Run `python -m halluscope.demo_data` first.")
    st.stop()

name = st.selectbox("Model", list(models))
df = models[name].copy()
df["wrong"] = 1 - df["correct"]
base_acc = df["correct"].mean()

tab_sim, tab_browse, tab_results = st.tabs(["Abstention simulator", "Explore questions", "Research results"])

with tab_sim:
    st.subheader("Let the model skip the questions it's most likely to get wrong")
    skip = st.slider("Skip the riskiest questions (%)", 0, 90, 30, step=5)
    coverage = 1 - skip / 100
    answered = keep_safest(df, "probe_risk", coverage)
    answered_tok = keep_safest(df, "token_risk", coverage)
    wrong_total = int(df["wrong"].sum())
    wrong_avoided = wrong_total - int(answered["wrong"].sum())
    right_lost = int(df["correct"].sum() - answered["correct"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Questions answered", f"{len(answered)} / {len(df)}")
    c2.metric("Accuracy on answered", f"{answered['correct'].mean():.1%}",
              delta=f"{answered['correct'].mean() - base_acc:+.1%} vs answering all")
    c3.metric("Wrong answers avoided", f"{wrong_avoided} of {wrong_total}")
    c4.metric("Right answers skipped", f"{right_lost}")
    st.caption(
        f"Answering everything: {base_acc:.1%} accuracy. At the same coverage, skipping by the "
        f"model's own token confidence (which needs the answer generated first) gives "
        f"{answered_tok['correct'].mean():.1%}. The probe decides before generating anything."
    )

    grid = np.linspace(0.05, 1.0, 20)
    curve = pd.DataFrame({
        "share of questions answered": grid,
        "probe (before answering)": [keep_safest(df, "probe_risk", g)["correct"].mean() for g in grid],
        "token confidence (after answering)": [keep_safest(df, "token_risk", g)["correct"].mean()
                                               for g in grid],
        "no filter": base_acc,
    }).set_index("share of questions answered")
    st.line_chart(curve)
    a1, a2 = st.columns(2)
    a1.metric("Probe AUROC (held out)", f"{auroc(df['wrong'].to_numpy(), df['probe_risk'].to_numpy()):.3f}")
    a2.metric("Token-confidence AUROC", f"{auroc(df['wrong'].to_numpy(), df['token_risk'].to_numpy()):.3f}")

with tab_browse:
    st.subheader("Every question, its answer, and the risk score")
    q = st.text_input("Search questions")
    show = st.radio("Show", ["All", "Model was right", "Model was wrong"], horizontal=True)
    view = df
    if q:
        view = view[view["question"].str.contains(q, case=False, regex=False)]
    if show != "All":
        view = view[view["correct"] == (1 if show == "Model was right" else 0)]
    view = view.sort_values("probe_risk", ascending=False)
    st.dataframe(
        view[["question", "model_answer", "gold_answers", "correct", "probe_risk"]].rename(columns={
            "model_answer": "model's answer", "gold_answers": "accepted answers",
            "correct": "right?", "probe_risk": "risk"}),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"{len(view)} questions. Sorted riskiest first.")

with tab_results:
    if SCALE_PNG.exists():
        st.image(str(SCALE_PNG), use_container_width=True)
    if RESULTS_MD.exists():
        md = "\n".join(line for line in RESULTS_MD.read_text().splitlines()
                       if not line.startswith("<!--") and not line.startswith("![scale]"))
        st.markdown(md)
    else:
        st.info("Run `python -m halluscope.report` to generate the research results.")
