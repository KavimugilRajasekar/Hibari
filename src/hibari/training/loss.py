"""Loss functions and perplexity metric computation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalLMLoss(nn.Module):
    """Cross-entropy loss for causal language modeling."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)
        return F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            ignore_index=self.ignore_index,
        )


def compute_perplexity(loss_value: float) -> float:
    """Calculate perplexity from cross-entropy loss."""
    try:
        return math.exp(loss_value)
    except OverflowError:
        return float("inf")
