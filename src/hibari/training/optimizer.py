"""Optimizer configuration for Hibari models."""

import torch
import torch.nn as nn
from torch.optim import AdamW


def configure_optimizers(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Configure AdamW with weight decay only applied to 2D matrix parameters (weights),

    and not to 1D vectors (biases, normalization scale).
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)
