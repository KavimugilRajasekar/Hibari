"""Minimal NumPy reference runtime for Hibari GGUF files (F32 / F16 only).

Purpose: prove that an exported GGUF is self-contained and numerically
equivalent to the PyTorch model, independently of llama.cpp. Every
hyper-parameter is read from the GGUF metadata (not from the checkpoint).
It implements the ``llama`` architecture as llama.cpp does for this model:
RMSNorm -> MHA with RoPE (adjacent-pair / "NORM" rotation) -> residual ->
RMSNorm -> SwiGLU -> residual, final RMSNorm, output = token_embd if
``output.weight`` is absent (tied embeddings).

Quantized files (Q8_0, Q4_K_M) are verified with llama.cpp instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import gguf
import numpy as np


class GGUFReferenceModel:
    def __init__(self, path: str | Path):
        r = gguf.GGUFReader(str(path))

        def kv(name):
            return r.fields[name].contents()

        arch = kv("general.architecture")
        if arch != "llama":
            raise ValueError(f"Unsupported architecture {arch!r}")
        self.n_layer = int(kv("llama.block_count"))
        self.d_model = int(kv("llama.embedding_length"))
        self.n_head = int(kv("llama.attention.head_count"))
        self.n_head_kv = int(kv("llama.attention.head_count_kv"))
        self.eps = float(kv("llama.attention.layer_norm_rms_epsilon"))
        self.theta = float(kv("llama.rope.freq_base"))
        self.n_ctx = int(kv("llama.context_length"))
        self.head_dim = self.d_model // self.n_head
        self.tokens = list(kv("tokenizer.ggml.tokens")) if "tokenizer.ggml.tokens" in r.fields else []

        self.w: Dict[str, np.ndarray] = {}
        for t in r.tensors:
            if t.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
                raise ValueError(f"{t.name}: reference runtime supports F32/F16 only, got {t.tensor_type.name}")
            # gguf stores dims fastest-first; numpy view is already row-major [rows, cols]
            self.w[t.name] = np.asarray(t.data, dtype=np.float32).reshape(
                [int(x) for x in reversed(t.shape)]
            )
        self.output = self.w.get("output.weight", self.w["token_embd.weight"])

    def _rms(self, x, w):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps) * w

    def _rope(self, x, pos):
        # x: [T, H, D]; rotate adjacent pairs (llama.cpp ROPE_TYPE_NORM)
        d = self.head_dim
        inv = 1.0 / (self.theta ** (np.arange(0, d, 2, dtype=np.float32) / d))
        ang = pos[:, None].astype(np.float32) * inv[None, :]
        cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
        x0, x1 = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = x0 * cos - x1 * sin
        out[..., 1::2] = x0 * sin + x1 * cos
        return out

    def forward(self, ids) -> np.ndarray:
        """Return logits [T, vocab] for a 1-D sequence of token IDs."""
        ids = np.asarray(ids, dtype=np.int64)
        T = len(ids)
        pos = np.arange(T)
        x = self.w["token_embd.weight"][ids]
        mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
        H, D = self.n_head, self.head_dim
        for i in range(self.n_layer):
            p = f"blk.{i}."
            h = self._rms(x, self.w[p + "attn_norm.weight"])
            q = (h @ self.w[p + "attn_q.weight"].T).reshape(T, H, D)
            k = (h @ self.w[p + "attn_k.weight"].T).reshape(T, H, D)
            v = (h @ self.w[p + "attn_v.weight"].T).reshape(T, H, D)
            q, k = self._rope(q, pos), self._rope(k, pos)
            att = np.einsum("thd,shd->hts", q, k) / np.sqrt(D) + mask
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            o = np.einsum("hts,shd->thd", att, v).reshape(T, H * D)
            x = x + o @ self.w[p + "attn_output.weight"].T
            h = self._rms(x, self.w[p + "ffn_norm.weight"])
            g = h @ self.w[p + "ffn_gate.weight"].T
            u = h @ self.w[p + "ffn_up.weight"].T
            x = x + ((g / (1.0 + np.exp(-g))) * u) @ self.w[p + "ffn_down.weight"].T
        x = self._rms(x, self.w["output_norm.weight"])
        return x @ self.output.T

    def greedy(self, ids, max_new_tokens: int, stop_ids=()) -> list:
        ids = list(ids)
        out = []
        for _ in range(max_new_tokens):
            nxt = int(np.argmax(self.forward(ids)[-1]))
            if nxt in stop_ids:
                break
            out.append(nxt)
            ids.append(nxt)
        return out
