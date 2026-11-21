"""Sampling strategies for text generation."""

from typing import Dict, Iterable, List, Optional, Set, Tuple
import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    past_token_ids: Optional[List[int]] = None,
    generator: Optional[torch.Generator] = None,
    banned_token_ids: Optional[Iterable[int]] = None,
) -> int:
    """Sample next token ID from logits with temperature, top-k, top-p and repetition penalty.

    Args:
        logits: 1D tensor of shape [vocab_size]
        temperature: sampling temperature (higher = more random)
        top_k: filter to top k tokens
        top_p: nucleus sampling cumulative probability
        repetition_penalty: penalty for previously generated tokens
        past_token_ids: list of previous token IDs
        generator: optional torch.Generator for reproducible sampling
        banned_token_ids: tokens that must not be produced (e.g. tokens that
            would repeat an n-gram, see ``banned_ngram_tokens``)
    """
    logits = logits.clone().float()

    if banned_token_ids:
        banned = [tid for tid in set(banned_token_ids) if 0 <= tid < logits.size(0)]
        if banned and len(banned) < logits.size(0):
            logits[banned] = -float("Inf")

    # Apply repetition penalty
    if repetition_penalty != 1.0 and past_token_ids:
        for tid in set(past_token_ids):
            if tid < logits.size(0):
                if logits[tid] > 0:
                    logits[tid] /= repetition_penalty
                else:
                    logits[tid] *= repetition_penalty

    # Temperature scaling
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    logits = logits / max(1e-5, temperature)

    # Top-K filtering
    if top_k > 0 and top_k < logits.size(0):
        val, _ = torch.topk(logits, top_k)
        min_val = val[-1]
        logits[logits < min_val] = -float("Inf")

    # Top-P (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

    probs = F.softmax(logits, dim=-1)
    # Check for NaN / zeros
    if torch.isnan(probs).any() or probs.sum() == 0:
        return int(torch.argmax(logits).item())

    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(next_token.item())


def banned_ngram_tokens(token_ids: List[int], n: int) -> Set[int]:
    """Tokens that would complete an n-gram already present in ``token_ids``.

    Standard "no repeat n-gram" blocking: if the last ``n - 1`` tokens equal
    the first ``n - 1`` tokens of an earlier n-gram, the token that followed
    them is banned. ``n <= 1`` disables blocking.
    """
    if n <= 1 or len(token_ids) < n:
        return set()
    seen: Dict[Tuple[int, ...], Set[int]] = {}
    for i in range(len(token_ids) - n + 1):
        seen.setdefault(tuple(token_ids[i : i + n - 1]), set()).add(token_ids[i + n - 1])
    return seen.get(tuple(token_ids[len(token_ids) - n + 1 :]), set())
