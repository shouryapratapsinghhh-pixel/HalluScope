"""Language-model wrapper: load a model, run a prompt, and record everything
the rest of the project needs from ONE prompt forward pass plus a greedy
generation loop:

  - hidden states at the LAST PROMPT TOKEN for every layer. This is the
    probe's input. It exists before a single answer token is generated,
    which is the whole point: predict the error before it happens.
  - the generated answer text
  - per-generated-token log-probability and entropy (the output-confidence
    baselines the probe has to beat)

The generation loop is written out by hand (greedy + KV cache) instead of
calling model.generate(), so every recorded number is visible in this file.
"""

from __future__ import annotations

import logging
import string
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer adapter
# ---------------------------------------------------------------------------


class HFTokenizer:
    """Thin adapter so the rest of the code only needs encode/decode/eos."""

    def __init__(self, hf_tokenizer):
        self.tok = hf_tokenizer
        self.eos_token_id = hf_tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)


class CharTokenizer:
    """Character-level tokenizer for OFFLINE TESTS ONLY. Id 0 is EOS."""

    def __init__(self):
        self.chars = list(string.printable)
        self.stoi = {c: i + 1 for i, c in enumerate(self.chars)}
        self.itos = {i + 1: c for i, c in enumerate(self.chars)}
        self.eos_token_id = 0
        self.vocab_size = len(self.chars) + 1

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def load_model(name: str, device: torch.device | None = None):
    """Load a real Hugging Face causal LM, e.g. "EleutherAI/pythia-410m",
    "Qwen/Qwen2.5-0.5B", "meta-llama/Llama-3.2-1B". Needs network the first
    time (weights are cached by Hugging Face afterwards).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or get_device()
    hf_tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    model.to(device).eval()
    return model, HFTokenizer(hf_tok)


def tiny_random_model(seed: int = 0, n_layer: int = 3, n_embd: int = 32):
    """A randomly initialised GPT-2 with a char tokenizer. No download.
    Same code path as a real model, so tests exercise real extraction logic.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    tok = CharTokenizer()
    config = GPT2Config(
        vocab_size=tok.vocab_size, n_positions=512, n_embd=n_embd, n_layer=n_layer, n_head=2,
        bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
    )
    model = GPT2LMHeadModel(config).eval()
    return model, tok


# ---------------------------------------------------------------------------
# Running one prompt
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    hidden: np.ndarray  # (n_layers + 1, d_model) at the last prompt token
    answer: str
    token_logprobs: list[float]  # log p(chosen token), one per generated token
    token_entropies: list[float]  # entropy of the next-token distribution, per step


@torch.inference_mode()
def run_prompt(model, tokenizer, prompt: str, max_new_tokens: int = 16) -> RunResult:
    """Prompt forward pass + greedy generation. Stops at EOS or newline
    (small models ramble past the answer otherwise).
    """
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("prompt encoded to zero tokens")
    input_ids = torch.tensor([prompt_ids], device=device)

    out = model(input_ids, output_hidden_states=True, use_cache=True)
    # hidden_states: tuple of (n_layers + 1) tensors, each (1, seq, d). Index 0
    # is the embedding layer. Take the last prompt position from every layer.
    hidden = torch.stack([h[0, -1, :] for h in out.hidden_states]).float().cpu().numpy()

    logits = out.logits[0, -1, :]
    past = out.past_key_values
    generated: list[int] = []
    logprobs: list[float] = []
    entropies: list[float] = []

    for _ in range(max_new_tokens):
        log_p = F.log_softmax(logits.float(), dim=-1)
        entropy = float(-(log_p.exp() * log_p).sum())
        next_id = int(torch.argmax(log_p))

        if next_id == tokenizer.eos_token_id or "\n" in tokenizer.decode([next_id]):
            break

        generated.append(next_id)
        logprobs.append(float(log_p[next_id]))
        entropies.append(entropy)

        step = model(torch.tensor([[next_id]], device=device), past_key_values=past, use_cache=True)
        logits = step.logits[0, -1, :]
        past = step.past_key_values

    return RunResult(
        hidden=hidden,
        answer=tokenizer.decode(generated).strip(),
        token_logprobs=logprobs,
        token_entropies=entropies,
    )


@torch.inference_mode()
def prompt_hidden_states(model, tokenizer, text: str) -> np.ndarray:
    """Hidden states at the last token, no generation. Used for true/false
    statements, where there is nothing to generate.
    """
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(text)], device=device)
    out = model(ids, output_hidden_states=True)
    return torch.stack([h[0, -1, :] for h in out.hidden_states]).float().cpu().numpy()
