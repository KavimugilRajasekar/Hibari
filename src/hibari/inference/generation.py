"""Autoregressive text generation routines for Hibari.

Decoding uses the model's KV cache: the prompt is processed once (prefill),
then each new token is fed alone. This is numerically equivalent to
re-running the full sequence every step (verified in tests/test_inference.py)
but O(n) instead of O(n^2) per reply.

Context handling: the model was trained with ``max_seq_len`` positions. The
prompt is left-truncated so that ``prompt + max_new_tokens`` never exceeds that
window; generation also stops when the window is full. Multi-turn sessions
should trim history at turn boundaries *before* calling this (see
``HibariChatSession``) so that truncation never cuts a turn header.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from hibari.data.tokenizer import HibariTokenizer
from hibari.inference.sampling import sample_next_token
from hibari.model.llm import HibariKotoLM


def _stop_ids(tokenizer: HibariTokenizer) -> set[int]:
    return {
        tokenizer.eos_id,
        tokenizer.user_id,
        tokenizer.system_id,
        tokenizer.assistant_id,
        tokenizer.pad_id,
        tokenizer.bos_id,
    }


def generate_tokens(
    model: HibariKotoLM,
    tokenizer: HibariTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.5,
    top_k: int = 25,
    top_p: float = 0.85,
    repetition_penalty: float = 1.15,
    device: str | torch.device = "cpu",
    seed: Optional[int] = None,
    prompt_ids: Optional[List[int]] = None,
) -> List[int]:
    """Generate new token IDs (excluding the prompt and the stop token).

    Args:
        prompt: Prompt text (already chat-formatted for conversations).
        temperature: 0 selects greedy decoding (deterministic).
        seed: Optional RNG seed for reproducible sampling.
        prompt_ids: Pre-tokenized prompt; overrides ``prompt`` when given.
    """
    model.eval()
    token_ids = list(prompt_ids) if prompt_ids is not None else tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.bos_id]

    max_ctx = model.config.max_seq_len
    max_new_tokens = max(0, max_new_tokens)
    # Reserve room for the reply (at most half the window) and left-truncate
    # only a prompt that would not leave that room. Generation then stops
    # when the window is full.
    reserve = min(max_new_tokens, max_ctx // 2)
    budget = max_ctx - reserve
    if len(token_ids) > budget:
        token_ids = token_ids[-budget:]

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    stop_ids = _stop_ids(tokenizer)
    new_tokens: List[int] = []

    with torch.no_grad():
        inp = torch.tensor([token_ids], dtype=torch.long, device=device)
        logits, _, caches = model(inp, use_cache=True)
        next_logits = logits[0, -1, :]

        for _ in range(max_new_tokens):
            next_token = sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                past_token_ids=new_tokens,
                generator=generator,
            )
            if next_token in stop_ids:
                break
            new_tokens.append(next_token)
            if len(token_ids) + len(new_tokens) >= max_ctx:
                break
            inp = torch.tensor([[next_token]], dtype=torch.long, device=device)
            logits, _, caches = model(inp, kv_caches=caches, use_cache=True)
            next_logits = logits[0, -1, :]

    return new_tokens


def generate_text(
    model: HibariKotoLM,
    tokenizer: HibariTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.5,
    top_k: int = 25,
    top_p: float = 0.85,
    repetition_penalty: float = 1.15,
    device: str | torch.device = "cpu",
    seed: Optional[int] = None,
) -> str:
    """Generate output text given a prompt string."""
    new_tokens = generate_tokens(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        device=device,
        seed=seed,
    )
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
