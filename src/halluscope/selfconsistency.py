"""Self-consistency baseline and cost benchmark.

  python -m halluscope.selfconsistency --model Qwen/Qwen2.5-1.5B \
      --data data/raw/triviaqa3k.jsonl --qa-cache cache/qwen15_tqa3k.npz \
      --n 500 --k 10 --out reports/sc_qwen15

Self-consistency: sample k extra answers with temperature, and call the
question risky when the samples DISAGREE with the model's normal (greedy)
answer. It's a strong error detector -- and expensive: k extra generations.

Compared on the same held-out questions:
  probe            reads the prompt's hidden state only (no answer generated)
  neg_min_logprob  needs the greedy answer generated
  self-consistency needs the greedy answer + k sampled answers

Cost is reported two ways:
  - forward steps per question: how many times the model runs (prompt pass
    = 1, each generated token = 1). Hardware-independent -> the fair number.
  - wall-clock ms per question on THIS machine (implementation-dependent:
    samples run sequentially here; a GPU server would batch them).

Leakage: the probe (and its layer) is trained only on questions OUTSIDE the
benchmark subset. Progress is appended to samples.jsonl after every
question, so an interrupted run resumes instead of restarting.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from halluscope.analyze import _fit_select
from halluscope.collect import load_collected
from halluscope.data.datasets import load_qa_jsonl, normalize_answer, qa_prompt
from halluscope.eval.baselines import neg_min_logprob
from halluscope.eval.stats import bootstrap_ci, paired_bootstrap_diff
from halluscope.probes.linear import LinearProbe

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation with cost accounting
# ---------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    """GPU/MPS work is asynchronous; wait for it before reading the clock."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def _decode(model, tokenizer, logits, past, max_new_tokens, temperature, generator):
    """Greedy (temperature=0) or sampled decoding from an existing prompt cache.
    Returns (answer, token_logprobs, n_forward_steps)."""
    device = logits.device
    out_ids, logprobs, steps = [], [], 0
    for _ in range(max_new_tokens):
        log_p = F.log_softmax(logits.float(), dim=-1)
        if temperature <= 0:
            next_id = int(torch.argmax(log_p))
        else:
            probs = F.softmax(logits.float() / temperature, dim=-1)
            next_id = int(torch.multinomial(probs.cpu(), 1, generator=generator))
        if next_id == tokenizer.eos_token_id or "\n" in tokenizer.decode([next_id]):
            break
        out_ids.append(next_id)
        logprobs.append(float(log_p[next_id]))
        step = model(torch.tensor([[next_id]], device=device), past_key_values=past, use_cache=True)
        steps += 1
        logits, past = step.logits[0, -1, :], step.past_key_values
    return tokenizer.decode(out_ids).strip(), logprobs, steps


@torch.inference_mode()
def run_with_samples(model, tokenizer, prompt, k=10, temperature=0.7, max_new_tokens=16, seed=0):
    """One prompt pass, then greedy + k sampled answers. The prompt's KV cache
    is COPIED for every decode, so the prompt is only processed once -- the
    fair way to charge self-consistency (it still pays for k extra decodes)."""
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)

    _sync(device)
    t0 = time.perf_counter()
    out = model(torch.tensor([tokenizer.encode(prompt)], device=device), use_cache=True)
    logits, past = out.logits[0, -1, :], out.past_key_values
    _sync(device)
    t_prompt = time.perf_counter() - t0

    t0 = time.perf_counter()
    greedy, lp, g_steps = _decode(model, tokenizer, logits, copy.deepcopy(past),
                                  max_new_tokens, 0.0, gen)
    _sync(device)
    t_greedy = time.perf_counter() - t0

    t0 = time.perf_counter()
    samples, s_steps = [], 0
    for _ in range(k):
        ans, _, n = _decode(model, tokenizer, logits, copy.deepcopy(past), max_new_tokens, temperature, gen)
        samples.append(ans)
        s_steps += n
    _sync(device)
    t_samples = time.perf_counter() - t0

    return {
        "greedy": greedy, "greedy_logprobs": lp, "samples": samples,
        "ms_prompt": 1000 * t_prompt, "ms_greedy": 1000 * t_greedy, "ms_samples": 1000 * t_samples,
        "steps_greedy": g_steps, "steps_samples": s_steps,
    }


def disagreement(greedy: str, samples: list[str]) -> float:
    """Fraction of samples whose normalized answer differs from the greedy one."""
    if not samples:
        return 0.0
    g = normalize_answer(greedy)
    return float(np.mean([normalize_answer(s) != g for s in samples]))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def choose_layer(hidden, y, tr, seed=0) -> tuple[int, float]:
    """Best layer by validation AUROC inside the TRAIN questions only."""
    best, best_auc, best_l2 = 0, -1.0, 1.0
    for layer in range(hidden.shape[1]):
        X = hidden[tr, layer, :]
        if np.allclose(X.std(axis=0), 0):
            continue
        l2, val_auc = _fit_select(X, y[tr], seed)
        if val_auc > best_auc:
            best, best_auc, best_l2 = layer, val_auc, l2
    return best, best_l2


def score_table(rows: list[dict], y: np.ndarray, probe_risk: np.ndarray, k: int, n_boot: int, seed: int):
    df = pd.DataFrame(rows)
    risks = {
        "probe (prompt only)": probe_risk,
        "neg_min_logprob (greedy answer)": neg_min_logprob(df["greedy_logprobs"].tolist()),
        f"self-consistency (greedy + {k} samples)": np.array(
            [disagreement(g, s) for g, s in zip(df["greedy"], df["samples"])]
        ),
    }
    prompt_ms, greedy_ms, sample_ms = df["ms_prompt"].mean(), df["ms_greedy"].mean(), df["ms_samples"].mean()
    g_steps, s_steps = df["steps_greedy"].mean(), df["steps_samples"].mean()
    cost = {  # (forward steps, wall ms) per question
        "probe (prompt only)": (1.0, prompt_ms),
        "neg_min_logprob (greedy answer)": (1.0 + g_steps, prompt_ms + greedy_ms),
        f"self-consistency (greedy + {k} samples)": (1.0 + g_steps + s_steps,
                                                     prompt_ms + greedy_ms + sample_ms),
    }
    table = []
    for name, risk in risks.items():
        est, lo, hi = bootstrap_ci(y, risk, n_boot=n_boot, seed=seed)
        steps, ms = cost[name]
        table.append({"method": name, "auroc": est, "ci_low": lo, "ci_high": hi,
                      "forward_steps": steps, "ms_per_question": ms,
                      "cost_vs_probe": steps / cost["probe (prompt only)"][0]})
    names = list(risks)
    diffs = {
        "probe - self_consistency": paired_bootstrap_diff(y, risks[names[0]], risks[names[2]], n_boot=n_boot, seed=seed),
        "self_consistency - neg_min_logprob": paired_bootstrap_diff(y, risks[names[2]], risks[names[1]], n_boot=n_boot, seed=seed),
    }
    return pd.DataFrame(table), diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-consistency baseline + cost benchmark.")
    parser.add_argument("--model", required=True, help="same model the QA cache was collected with, or 'tiny'")
    parser.add_argument("--data", required=True, help="the QA jsonl the cache was collected from")
    parser.add_argument("--qa-cache", required=True)
    parser.add_argument("--n", type=int, default=500, help="benchmark questions (held out from the probe)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    c = load_collected(args.qa_cache)
    items = load_qa_jsonl(args.data, limit=len(c.y_wrong))
    if len(items) != len(c.y_wrong):
        raise ValueError("--data doesn't match the cache's question count")
    if c.questions and [i.question for i in items] != c.questions:
        raise ValueError("--data questions don't match the cache -- different file or order")

    rng = np.random.default_rng(args.seed)
    bench = np.sort(rng.choice(len(items), size=min(args.n, len(items)), replace=False))
    train = np.setdiff1d(np.arange(len(items)), bench)

    layer, l2 = choose_layer(c.hidden, c.y_wrong, train, args.seed)
    probe = LinearProbe(l2=l2, seed=args.seed).fit(c.hidden[train, layer, :], c.y_wrong[train])
    probe_risk = probe.predict_proba(c.hidden[bench, layer, :])
    logger.info("probe: layer %d chosen on %d training questions", layer, len(train))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "samples.jsonl"
    done = {}
    if log_path.exists():  # resume
        for line in log_path.read_text().splitlines():
            row = json.loads(line)
            done[row["idx"]] = row
        logger.info("resuming: %d / %d questions already done", len(done), len(bench))

    todo = [i for i in bench if int(i) not in done]
    if todo:
        from halluscope.models.lm import load_model, tiny_random_model

        model, tok = tiny_random_model() if args.model == "tiny" else load_model(args.model, dtype=args.dtype)
        start = time.perf_counter()
        with open(log_path, "a") as f:
            for j, i in enumerate(todo):
                r = run_with_samples(model, tok, qa_prompt(items[i].question), k=args.k,
                                     temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                                     seed=args.seed + int(i))
                r["idx"] = int(i)
                f.write(json.dumps(r) + "\n")
                f.flush()
                done[int(i)] = r
                if (j + 1) % 10 == 0 or j + 1 == len(todo):
                    per_q = (time.perf_counter() - start) / (j + 1)
                    logger.info("sampled %d / %d  (%.1f s/question, ~%.0f min left)",
                                len(done), len(bench), per_q, per_q * (len(todo) - j - 1) / 60)

    rows = [done[int(i)] for i in bench]
    # cached greedy answers vs this run's: should agree (a sanity check on determinism)
    match = np.mean([normalize_answer(r["greedy"]) == normalize_answer(c.answers[i])
                     for r, i in zip(rows, bench)])
    table, diffs = score_table(rows, c.y_wrong[bench], probe_risk, args.k, args.n_boot, args.seed)
    table.to_csv(out / "cost_vs_accuracy.csv", index=False)
    (out / "diffs.json").write_text(json.dumps(
        {k: {"diff": d, "ci_low": lo, "ci_high": hi} for k, (d, lo, hi) in diffs.items()}, indent=2))

    print(table.round(3).to_string(index=False))
    for name, (d, lo, hi) in diffs.items():
        print(f"{name}: {d:+.3f} AUROC, 95% CI [{lo:+.3f}, {hi:+.3f}] -> "
              f"{'SIGNIFICANT' if lo > 0 or hi < 0 else 'not significant'}")
    print(f"greedy answers matching the cache: {match:.1%} (should be ~100%)")
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
