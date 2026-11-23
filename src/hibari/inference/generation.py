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

from typing import Dict, Iterator, List, Optional, Tuple

import torch

from hibari.data.tokenizer import HibariTokenizer
from hibari.inference.sampling import banned_ngram_tokens, sample_next_token
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


def reply_reserve(max_ctx: int, max_new_tokens: int) -> int:
    """Tokens of the window kept free for the reply: ``max_new_tokens``, at most half the window.
    Shared by generation and ``HibariChatSession.build_prompt_ids`` so both trim the same way."""
    return max(0, min(max_new_tokens, max_ctx // 2))


def _truncate_prompt(token_ids: List[int], max_ctx: int, max_new_tokens: int) -> List[int]:
    # Reserve room for the reply and left-truncate only a prompt that would
    # not leave that room. Generation then stops when the window is full.
    budget = max_ctx - reply_reserve(max_ctx, max_new_tokens)
    return token_ids[-budget:] if len(token_ids) > budget else token_ids


@torch.no_grad()
def generate_batch(
    model: HibariKotoLM,
    tokenizer: HibariTokenizer,
    prompt_ids: List[int],
    decodings: List[Dict],
    seeds: List[Optional[int]],
    max_new_tokens: int = 64,
    device: str | torch.device = "cpu",
) -> List[Tuple[List[int], str]]:
    """Sample ``len(decodings)`` continuations of ONE prompt in a single batch.

    The prompt is processed once; its KV cache is copied to every row (the
    "shared context" sampling of GPT-2's ``sample_sequence`` / ``expand_tile``),
    then all rows decode together, each with its own decoding settings and its
    own RNG seed. Rows that hit a stop token leave the batch. Row ``i`` follows
    exactly the procedure of ``generate_tokens(..., seed=seeds[i], **decodings[i])``;
    results agree up to floating-point differences of batched matrix products.

    Each ``decodings[i]`` may contain ``temperature``, ``top_k``, ``top_p``,
    ``repetition_penalty`` and ``no_repeat_ngram_size``.

    Returns ``[(new_token_ids, stop_reason), ...]`` with stop_reason
    ``stop_token`` / ``max_new_tokens`` / ``context_full``.
    """
    if len(decodings) != len(seeds):
        raise ValueError("decodings and seeds must have the same length")
    n = len(decodings)
    if n == 0:
        return []
    model.eval()
    token_ids = list(prompt_ids) or [tokenizer.bos_id]
    max_ctx = model.config.max_seq_len
    max_new_tokens = max(0, max_new_tokens)
    token_ids = _truncate_prompt(token_ids, max_ctx, max_new_tokens)
    stop_ids = _stop_ids(tokenizer)
    gens = [torch.Generator(device="cpu").manual_seed(s) if s is not None else None for s in seeds]
    outs: List[List[int]] = [[] for _ in range(n)]
    reasons = ["max_new_tokens"] * n

    inp = torch.tensor([token_ids], dtype=torch.long, device=device)
    logits, _, caches = model(inp, use_cache=True)
    last = logits[0, -1, :]
    if max_new_tokens == 0:
        return [([], "max_new_tokens") for _ in range(n)]
    # expand the prompt cache to n rows (no recomputation of the prompt)
    caches = [(k.expand(n, -1, -1, -1).contiguous(), v.expand(n, -1, -1, -1).contiguous()) for k, v in caches]
    next_logits = last.unsqueeze(0).expand(n, -1)
    active = list(range(n))           # row index of each batch position

    for _ in range(max_new_tokens):
        chosen: List[int] = []
        keep: List[int] = []
        for pos, row in enumerate(active):
            d = decodings[row]
            tok = sample_next_token(
                next_logits[pos],
                temperature=d.get("temperature", 0.5), top_k=d.get("top_k", 25), top_p=d.get("top_p", 0.85),
                repetition_penalty=d.get("repetition_penalty", 1.15), past_token_ids=outs[row], generator=gens[row],
                banned_token_ids=banned_ngram_tokens(outs[row], d.get("no_repeat_ngram_size", 0)),
            )
            if tok in stop_ids:
                reasons[row] = "stop_token"
                continue
            outs[row].append(tok)
            if len(token_ids) + len(outs[row]) >= max_ctx:
                reasons[row] = "max_new_tokens" if len(outs[row]) >= max_new_tokens else "context_full"
                continue
            if len(outs[row]) >= max_new_tokens:
                continue
            keep.append(pos)
            chosen.append(tok)
        if not keep:
            break
        if len(keep) < len(active):
            idx = torch.tensor(keep, dtype=torch.long, device=device)
            caches = [(k.index_select(0, idx), v.index_select(0, idx)) for k, v in caches]
            active = [active[p] for p in keep]
        inp = torch.tensor(chosen, dtype=torch.long, device=device).unsqueeze(1)
        logits, _, caches = model(inp, kv_caches=caches, use_cache=True)
        next_logits = logits[:, -1, :]
    return list(zip(outs, reasons))


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
    no_repeat_ngram_size: int = 0,
    stats: Optional[dict] = None,
) -> List[int]:
    """Generate new token IDs (excluding the prompt and the stop token).

    Args:
        prompt: Prompt text (already chat-formatted for conversations).
        temperature: 0 selects greedy decoding (deterministic).
        seed: Optional RNG seed for reproducible sampling.
        prompt_ids: Pre-tokenized prompt; overrides ``prompt`` when given.
        no_repeat_ngram_size: if > 1, never repeat an n-gram of this size
            within the generated reply (the prompt is not counted).
        stats: optional dict; ``stats["stop_reason"]`` is set to ``"stop_token"``,
            ``"max_new_tokens"`` or ``"context_full"``.
    """
    ids = list(prompt_ids) if prompt_ids is not None else tokenizer.encode(prompt)
    decoding = dict(temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size)
    (new_tokens, stop_reason), = generate_batch(model, tokenizer, ids, [decoding], [seed], max_new_tokens, device)
    if stats is not None:
        stats["stop_reason"] = stop_reason
    return new_tokens


def stream_text(
    model: HibariKotoLM,
    tokenizer: HibariTokenizer,
    prompt_ids: List[int],
    max_new_tokens: int = 64,
    device: str | torch.device = "cpu",
    seed: Optional[int] = None,
    **decoding,
) -> Iterator[str]:
    """Yield the reply as text pieces while it is generated.

    Byte-level BPE tokens can end in the middle of a multi-byte UTF-8 character
    (GPT-2's encoder decodes with ``errors="replace"`` for this reason). Here
    the decoded prefix is only emitted up to the last complete character, so a
    stream never shows a replacement character that the final text would not
    contain; ``"".join(stream_text(...))`` equals the decoded full reply.
    """
    model.eval()
    ids = _truncate_prompt(list(prompt_ids) or [tokenizer.bos_id], model.config.max_seq_len, max(0, max_new_tokens))
    stop_ids = _stop_ids(tokenizer)
    gen = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    out: List[int] = []
    shown = 0
    with torch.no_grad():
        logits, _, caches = model(torch.tensor([ids], dtype=torch.long, device=device), use_cache=True)
        nxt = logits[0, -1, :]
        for _ in range(max(0, max_new_tokens)):
            tok = sample_next_token(
                nxt, temperature=decoding.get("temperature", 0.5), top_k=decoding.get("top_k", 25),
                top_p=decoding.get("top_p", 0.85), repetition_penalty=decoding.get("repetition_penalty", 1.15),
                past_token_ids=out, generator=gen,
                banned_token_ids=banned_ngram_tokens(out, decoding.get("no_repeat_ngram_size", 0)),
            )
            if tok in stop_ids:
                break
            out.append(tok)
            text = tokenizer.decode(out, skip_special_tokens=True)
            safe = text[:-1] if text.endswith("\ufffd") else text
            if len(safe) > shown:
                yield safe[shown:]
                shown = len(safe)
            if len(ids) + len(out) >= model.config.max_seq_len:
                break
            logits, _, caches = model(torch.tensor([[tok]], dtype=torch.long, device=device), kv_caches=caches, use_cache=True)
            nxt = logits[0, -1, :]
    final = tokenizer.decode(out, skip_special_tokens=True)
    if len(final) > shown:
        yield final[shown:]


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
    no_repeat_ngram_size: int = 0,
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
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
