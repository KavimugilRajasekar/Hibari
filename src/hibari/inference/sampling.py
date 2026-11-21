"""Sampling strategies for text generation."""

from typing import List, Optional
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
    """
    logits = logits.clone().float()

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
