# Data

Nothing here is committed (`data/raw/` is gitignored). Tests use `make_toy_qa()` and need no downloads.

## TriviaQA (closed-book QA)
```bash
pip install datasets
python scripts/prepare_triviaqa.py --n 2000 --out data/raw/triviaqa.jsonl
```

## Geometry of Truth (true/false statements, Marks & Tegmark 2023)
```bash
mkdir -p data/raw
curl -o data/raw/cities.csv https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/cities.csv
```
(verified: 1,496 statements, 748 true / 748 false). Other files in that repo's `datasets/` folder use the same `statement,label` format.
