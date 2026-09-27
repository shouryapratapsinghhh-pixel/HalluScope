"""HalluScope API: answer questions with a risk score computed BEFORE the answer.

  ARTIFACT_DIR=artifacts/qwen15_probe uvicorn halluscope.serve:app --port 8000

  GET  /health
  GET  /info                model, layer, held-out AUROC estimate
  POST /assess  {"question": "..."}
       -> risk only, from the question's forward pass. No answer is generated.
  POST /answer  {"question": "...", "abstain_above": 0.8}
       -> risk first; if risk > abstain_above, abstain WITHOUT generating;
          otherwise generate, reusing the same forward pass (the probe is free).

For tests, use create_app(model, tokenizer, probe, meta) directly.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from halluscope.data.datasets import qa_prompt
from halluscope.probes.linear import load_probe
from halluscope.selfconsistency import _decode, _sync

MAX_QUESTION_CHARS = 1000


class AssessRequest(BaseModel):
    question: str = Field(..., description="A closed-book factual question")


class AnswerRequest(AssessRequest):
    abstain_above: float | None = Field(None, ge=0.0, le=1.0,
                                        description="Skip generation when risk exceeds this")
    max_new_tokens: int = Field(16, ge=1, le=64)


def _check(question: str) -> str:
    q = question.strip()
    if not q:
        raise HTTPException(422, "question is empty")
    if len(q) > MAX_QUESTION_CHARS:
        raise HTTPException(422, f"question longer than {MAX_QUESTION_CHARS} characters")
    return q


def create_app(model, tokenizer, probe, meta: dict, warmup: bool = True) -> FastAPI:
    app = FastAPI(title="HalluScope", description="Predicts an LLM's errors before it answers.")
    layer = meta["layer"]
    lock = threading.Lock()  # one model, one request at a time (MPS/CUDA models aren't thread-safe)
    device = next(model.parameters()).device

    @torch.inference_mode()
    def forward(question: str):
        ids = torch.tensor([tokenizer.encode(qa_prompt(question))], device=device)
        _sync(device)
        t0 = time.perf_counter()
        out = model(ids, output_hidden_states=True, use_cache=True)
        h = out.hidden_states[layer][0, -1, :].float().cpu().numpy()
        risk = float(probe.predict_proba(h[None, :])[0])
        _sync(device)
        return risk, out.logits[0, -1, :], out.past_key_values, 1000 * (time.perf_counter() - t0)

    if warmup:
        # the first forward pass on a GPU/MPS compiles kernels (~4 s on a Mac); pay it at
        # startup, not on the first user's request
        forward("warm-up question")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/info")
    def info():
        return meta

    @app.post("/assess")
    def assess(req: AssessRequest):
        q = _check(req.question)
        with lock:
            risk, _, _, ms = forward(q)
        return {"question": q, "risk": risk, "ms_assess": ms, "answer_generated": False}

    @app.post("/answer")
    def answer(req: AnswerRequest):
        q = _check(req.question)
        with lock:
            risk, logits, past, ms_assess = forward(q)
            if req.abstain_above is not None and risk > req.abstain_above:
                return {"question": q, "risk": risk, "abstained": True, "answer": None,
                        "ms_assess": ms_assess, "ms_generate": 0.0}
            t0 = time.perf_counter()
            text, logprobs, _ = _decode(model, tokenizer, logits, past, req.max_new_tokens, 0.0, None)
            _sync(device)
            ms_gen = 1000 * (time.perf_counter() - t0)
        return {"question": q, "risk": risk, "abstained": False, "answer": text,
                "min_token_logprob": min(logprobs) if logprobs else None,
                "ms_assess": ms_assess, "ms_generate": ms_gen}

    return app


def _app_from_env() -> FastAPI:
    art = Path(os.environ["ARTIFACT_DIR"])
    meta = json.loads((art / "meta.json").read_text())
    from halluscope.models.lm import load_model, tiny_random_model

    if meta["model"] == "tiny":
        model, tok = tiny_random_model()
    else:
        model, tok = load_model(meta["model"], dtype=meta.get("dtype", "auto"))
    return create_app(model, tok, load_probe(art / "probe.npz"), meta)


# built only when run as a server (ARTIFACT_DIR set); importing for tests never loads a model
app = _app_from_env() if os.environ.get("ARTIFACT_DIR") else None
