"""Token embedding and Rotary Position Embedding (RoPE) for Hibari."""

from typing import Tuple
import torch
import torch.nn as nn


def precompute_rope_freqs(
    head_dim: int, max_seq_len: int, theta: float = 10000.0, device: torch.device | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine frequency tables for RoPE."""
    # head_dim must be even
    assert head_dim % 2 == 0, f"head_dim ({head_dim}) must be even for RoPE"
    
    # freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (theta ** (dim_indices / head_dim))
    
    # Position indices: [0, 1, ..., max_seq_len - 1]
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    
    # Outer product -> [max_seq_len, head_dim // 2]
    freqs = torch.outer(positions, inv_freq)
    
    # Repeat along last dim to match head_dim: [max_seq_len, head_dim]
    freqs = torch.repeat_interleave(freqs, 2, dim=-1)
    
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embedding to input tensor.

    Args:
        x: [batch_size, n_heads, seq_len, head_dim]
        cos: [seq_len, head_dim] or broadcastable
        sin: [seq_len, head_dim] or broadcastable
    """
    seq_len = x.shape[2]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(1)

    # Rotate half
    # x = [x1, x2, x3, x4, ...] -> [-x2, x1, -x4, x3, ...]
    x_pairs = x.view(*x.shape[:-1], -1, 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]
    x_rot = torch.stack([-x2, x1], dim=-1).flatten(-2)

    return (x * cos) + (x_rot * sin)


class TokenEmbedding(nn.Module):
    """Token embedding layer mapping token IDs to dense vectors."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)
