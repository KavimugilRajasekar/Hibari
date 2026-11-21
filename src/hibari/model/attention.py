"""Multi-Head Causal Attention with RoPE and KV Cache support for Hibari."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from hibari.model.embedding import apply_rotary_emb


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with Rotary Position Embeddings and causal masking."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for causal self-attention.

        Args:
            x: [batch_size, seq_len, d_model]
            cos: [seq_len, head_dim]
            sin: [seq_len, head_dim]
            mask: Optional attention mask [batch_size, 1, seq_len, total_seq_len]
            kv_cache: Optional past (key, value) tuple
            use_cache: Whether to return updated kv_cache
        """
        bsz, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [batch_size, n_heads, seq_len, head_dim]
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        # If caching, cos/sin should match position offset
        if kv_cache is not None:
            past_len = kv_cache[0].shape[2]
            cos_q = cos[past_len : past_len + seq_len]
            sin_q = sin[past_len : past_len + seq_len]
            q = apply_rotary_emb(q, cos_q, sin_q)
            k = apply_rotary_emb(k, cos_q, sin_q)
        else:
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # Update KV cache if provided
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v) if use_cache else None
        total_k_len = k.shape[2]

        # Use FlashAttention / Scaled Dot Product Attention if available
        if mask is None and seq_len == total_k_len:
            # Standard causal training
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True, dropout_p=self.dropout.p if isinstance(self.dropout, nn.Dropout) and self.training else 0.0
            )
        else:
            # Scaled dot-product with explicit causal/padding mask.
            # Query i sits at absolute position (past_len + i) and may attend to
            # keys 0..past_len+i. This is correct both for prefill (past_len=0)
            # and for multi-token chunks appended to an existing KV cache.
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            past_len = total_k_len - seq_len
            causal_mask = torch.triu(
                torch.full((seq_len, total_k_len), float("-inf"), device=x.device),
                diagonal=1 + past_len,
            )
            scores = scores + causal_mask
            if mask is not None:
                scores = scores + mask

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, v)

        # Reshape back to [batch_size, seq_len, d_model]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.out_proj(out)
        return out, new_cache
