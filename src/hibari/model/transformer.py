"""Transformer block and SwiGLU Feed-Forward Network for Hibari."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from hibari.model.attention import MultiHeadAttention
from hibari.model.normalization import RMSNorm


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network (Shazeer, 2020)."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer Decoder Block with SwiGLU."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm Self-Attention with residual connection
        normed_attn = self.attn_norm(x)
        attn_out, new_cache = self.attn(
            normed_attn,
            cos=cos,
            sin=sin,
            mask=mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        x = x + attn_out

        # Pre-norm SwiGLU FFN with residual connection
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache
