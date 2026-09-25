"""Dataset loaders and answer grading.

Two kinds of data:
  - QA (TriviaQA / NQ-style): a question plus a list of accepted answer
    aliases. The model generates an answer; we grade it automatically.
    Label = 1 if the model's answer is WRONG (that's what we predict).
  - True/false statements (Geometry of Truth, Marks & Tegmark 2023): a
    statement plus a 0/1 truth label. Used to test whether a probe finds a
    general "truth direction", independent of generation.

make_toy_qa() needs no external files, so tests and CI run offline.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class QAItem:
    question: str
    answers: list[str]  # all accepted aliases


@dataclass
class StatementItem:
    statement: str
    label: int  # 1 = true statement, 0 = false


def normalize_answer(text: str) -> str:
    """Standard SQuAD/TriviaQA normalization: lowercase, strip punctuation,
    articles, and extra whitespace. Without this, "The Beatles." and
    "beatles" would be graded as different answers.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_correct(prediction: str, answers: list[str]) -> bool:
    """Correct if any normalized alias appears in the normalized
    prediction. Containment (not exact match) because small models often
    wrap the answer: "It was Paris, France" should count for "Paris".
    """
    pred = normalize_answer(prediction)
    if not pred:
        return False
    return any(normalize_answer(a) and normalize_answer(a) in pred for a in answers)


def load_qa_jsonl(path: str | Path, limit: int | None = None) -> list[QAItem]:
    """One JSON object per line: {"question": str, "answers": [str, ...]}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"QA file not found: {path}. See data/README.md.")
    items = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(QAItem(question=row["question"], answers=list(row["answers"])))
            if limit and len(items) >= limit:
                break
    return items


def load_truth_csv(path: str | Path, limit: int | None = None) -> list[StatementItem]:
    """Geometry-of-Truth CSV format: columns `statement`, `label`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Statement file not found: {path}. See data/README.md.")
    df = pd.read_csv(path)
    missing = {"statement", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if limit:
        df = df.head(limit)
    return [StatementItem(str(s), int(y)) for s, y in zip(df["statement"], df["label"])]


def make_toy_qa() -> list[QAItem]:
    """Tiny offline QA set for tests and smoke runs. Not for real results."""
    return [
        QAItem("What is the capital of France?", ["Paris"]),
        QAItem("How many legs does a spider have?", ["8", "eight"]),
        QAItem("What planet is known as the Red Planet?", ["Mars"]),
        QAItem("Who wrote Romeo and Juliet?", ["Shakespeare", "William Shakespeare"]),
        QAItem("What is the chemical symbol for gold?", ["Au"]),
        QAItem("What is the largest ocean on Earth?", ["Pacific", "Pacific Ocean"]),
        QAItem("In what year did World War II end?", ["1945"]),
        QAItem("What gas do plants absorb from the air?", ["carbon dioxide", "CO2"]),
    ]


def qa_prompt(question: str) -> str:
    """Few-shot-free prompt. Kept deliberately simple and fixed so the
    probe learns about the model, not about prompt variation.
    """
    return f"Question: {question}\nAnswer:"
