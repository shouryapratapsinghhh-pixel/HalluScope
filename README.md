# HalluScope

**Can an LLM know it's about to be wrong?** HalluScope reads a language model's internal
activations at the end of the question — before a single answer token is generated — and
predicts whether its answer will be wrong.

## Hypothesis
A linear probe on middle-layer activations predicts wrong answers better than output
confidence (token log-probs, entropy), at a fraction of the cost of self-consistency sampling
(1 forward pass vs. k generations).

## Status
**Phase 0 done:** QA + true/false loaders, automatic answer grading, hand-written greedy
generation that records hidden states (all layers) + per-token log-probs/entropy, output-
confidence baselines, linear probe (logistic regression from scratch, train-only
standardization), and from-scratch metrics (AUROC, ECE, selective accuracy, AURC).
30 tests, all offline.

**Next:** layer sweep, verbalized-confidence + self-consistency baselines, MLP probe,
cross-domain + cross-model transfer, FastAPI answer + risk endpoint, latency benchmark.

## Quickstart
```bash
pip install -e . -r requirements.txt
pytest -q
python -m halluscope.run --model tiny --data toy        # offline smoke test

# real run (needs network once, to download model + data):
pip install datasets
python scripts/prepare_triviaqa.py --n 1000 --out data/raw/triviaqa.jsonl
python -m halluscope.run --model EleutherAI/pythia-410m --data data/raw/triviaqa.jsonl \
    --cache cache/pythia410m_tqa.npz
```

No results are reported here yet — they'll come from real model runs, never typed by hand.
