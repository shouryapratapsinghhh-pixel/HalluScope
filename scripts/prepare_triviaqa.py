"""Download TriviaQA and write the jsonl format halluscope expects.

  pip install datasets
  python scripts/prepare_triviaqa.py --n 2000 --out data/raw/triviaqa.jsonl

Uses the "rc.nocontext" validation split (closed-book: the model only sees
the question). Every accepted alias is kept, so grading isn't unfairly strict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--out", default="data/raw/triviaqa.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset  # optional dependency, only needed here

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in ds:
            answers = sorted(set(row["answer"]["aliases"] + [row["answer"]["value"]]))
            f.write(json.dumps({"question": row["question"], "answers": answers}) + "\n")
    print(f"wrote {len(ds)} questions to {out}")


if __name__ == "__main__":
    main()
