"""Export Hibari models to GGUF (llama.cpp ``llama`` architecture).

Why ``llama``: Hibari Koto is architecturally a LLaMA-style decoder —
pre-RMSNorm, RoPE, SwiGLU FFN, no biases, full multi-head attention
(n_head_kv == n_head), untied output norm and a tied output head.

RoPE layout: ``hibari.model.embedding.apply_rotary_emb`` rotates *adjacent*
pairs (x0,x1), (x2,x3)... This is exactly llama.cpp's ``LLAMA_ROPE_TYPE_NORM``
used by the ``llama`` architecture, so Q/K weights are written unpermuted.
(HF LLaMA checkpoints need a permutation because HF uses rotate-half;
Hibari does not.) Parity is verified numerically in the test suite and in
``scripts/verify_gguf.py``.

Tokenizer: byte-level BPE with the GPT-2 split regex and no prefix space,
stored as ``tokenizer.ggml.model = "gpt2"`` / ``tokenizer.ggml.pre = "gpt-2"``
with the full vocabulary, every merge rule, token types and special IDs, so
the GGUF is self-contained (no Python tokenizer needed at inference time).
The Python tokenizer additionally applies NFKC normalisation, which llama.cpp
does not; this only matters for compatibility characters (e.g. full-width
letters, ligatures) and is documented as a known difference.

Tensor types: 2-D weight matrices are written as F32 or F16. 1-D tensors
(RMSNorm weights) are always F32 — ggml's CPU kernels reject F16 norm
weights (the original exporter wrote them as F16, which crashed llama.cpp
with "binary_op: unsupported types"). Q8_0 / Q4_K_M are produced from the
F32 file with llama.cpp's ``llama-quantize`` (see ``scripts/export.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import gguf
import numpy as np

from hibari.data.formatting import DEFAULT_SYSTEM_PROMPT
from hibari.data.tokenizer import SPECIAL_TOKENS, HibariTokenizer
from hibari.model.llm import HibariKotoLM

# Jinja chat template (llama.cpp / HF style). Mirrors
# HibariTokenizer.apply_chat_template exactly.
CHAT_TEMPLATE = (
    "{%- if messages[0]['role'] != 'system' %}"
    "{{- '<|system|>\\n" + DEFAULT_SYSTEM_PROMPT + "<|eos|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{{- '<|' + message['role'] + '|>\\n' + message['content'] | trim + '<|eos|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|assistant|>\\n' }}{%- endif %}"
)

SUPPORTED_OUTTYPES = {"f32", "f16"}


def _get_bpe_merges(tokenizer: HibariTokenizer) -> List[str]:
    """Return BPE merges as ``"left right"`` strings in training order."""
    raw_json = json.loads(tokenizer._tokenizer.to_str())
    merges_raw = raw_json.get("model", {}).get("merges", [])
    out = []
    for m in merges_raw:
        if isinstance(m, str):  # older tokenizers serialisation
            out.append(m)
        else:
            left, right = m
            out.append(f"{left} {right}")
    return out


def _build_token_types(sorted_tokens: List[str], special_tokens: List[str]) -> List[int]:
    special_set = set(special_tokens)
    return [
        int(gguf.TokenType.CONTROL) if tok in special_set else int(gguf.TokenType.NORMAL)
        for tok in sorted_tokens
    ]


def tensor_name_map(n_layers: int, include_output: bool) -> Dict[str, str]:
    """PyTorch state-dict name -> GGUF tensor name."""
    m = {
        "embed_tokens.embedding.weight": "token_embd.weight",
        "norm.weight": "output_norm.weight",
    }
    if include_output:
        m["lm_head.weight"] = "output.weight"
    for i in range(n_layers):
        m.update(
            {
                f"layers.{i}.attn_norm.weight": f"blk.{i}.attn_norm.weight",
                f"layers.{i}.attn.q_proj.weight": f"blk.{i}.attn_q.weight",
                f"layers.{i}.attn.k_proj.weight": f"blk.{i}.attn_k.weight",
                f"layers.{i}.attn.v_proj.weight": f"blk.{i}.attn_v.weight",
                f"layers.{i}.attn.out_proj.weight": f"blk.{i}.attn_output.weight",
                f"layers.{i}.ffn_norm.weight": f"blk.{i}.ffn_norm.weight",
                f"layers.{i}.ffn.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
                f"layers.{i}.ffn.up_proj.weight": f"blk.{i}.ffn_up.weight",
                f"layers.{i}.ffn.down_proj.weight": f"blk.{i}.ffn_down.weight",
            }
        )
    return m


def export_to_gguf(
    model: HibariKotoLM,
    tokenizer: Optional[HibariTokenizer],
    output_path: str | Path,
    quantization_type: str = "f32",
    version: str = "2.0",
    author: str = "Kavimugil Rajasekar",
    description: Optional[str] = None,
) -> Path:
    """Export a trained Hibari model to a self-contained GGUF file.

    Args:
        model: HibariKotoLM instance.
        tokenizer: Tokenizer to embed (required for a usable file; ``None``
            only for weight-only test files).
        output_path: Target ``.gguf`` path.
        quantization_type: ``'f32'`` or ``'f16'`` (for Q8_0/Q4_K_M use
            llama-quantize on the F32 output).
    """
    if quantization_type not in SUPPORTED_OUTTYPES:
        raise ValueError(
            f"quantization_type must be one of {sorted(SUPPORTED_OUTTYPES)}, got {quantization_type!r}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = model.config
    if config.d_model % config.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")

    writer = gguf.GGUFWriter(str(output_path), "llama")

    # ---------------- general metadata ----------------
    name = config.model_name or "Hibari-Koto"
    n_params = sum(p.numel() for p in model.parameters())
    writer.add_name(name)
    writer.add_basename(name)
    writer.add_version(version)
    writer.add_author(author)
    writer.add_organization("Hibari")
    writer.add_license("mit")
    writer.add_size_label(f"{n_params / 1e6:.1f}M")
    writer.add_description(
        description
        or f"{name}: a {n_params:,}-parameter conversational decoder trained from scratch."
    )
    writer.add_file_type(
        gguf.LlamaFileType.MOSTLY_F16 if quantization_type == "f16" else gguf.LlamaFileType.ALL_F32
    )

    # ---------------- architecture metadata ----------------
    writer.add_vocab_size(config.vocab_size)
    writer.add_context_length(config.max_seq_len)
    writer.add_embedding_length(config.d_model)
    writer.add_block_count(config.n_layers)
    writer.add_feed_forward_length(config.d_ff)
    writer.add_head_count(config.n_heads)
    writer.add_head_count_kv(config.n_heads)
    writer.add_rope_dimension_count(config.head_dim)
    writer.add_rope_freq_base(config.rope_theta)
    writer.add_layer_norm_rms_eps(config.norm_eps)

    # ---------------- tokenizer ----------------
    if tokenizer is not None:
        merges = _get_bpe_merges(tokenizer)
        if not merges:
            raise RuntimeError("The tokenizer contains no BPE merge rules. Train it before exporting.")
        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError(
                f"Tokenizer vocab ({tokenizer.vocab_size}) != model vocab ({config.vocab_size})"
            )
        vocab = tokenizer.get_vocab()
        sorted_tokens = [tok for tok, _ in sorted(vocab.items(), key=lambda x: x[1])]
        if [vocab[t] for t in sorted_tokens] != list(range(len(sorted_tokens))):
            raise RuntimeError("Tokenizer IDs are not contiguous from 0")

        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("gpt-2")
        writer.add_token_list(sorted_tokens)
        writer.add_token_merges(merges)
        writer.add_token_scores([0.0] * len(sorted_tokens))  # unused by BPE; kept for tool compatibility
        writer.add_token_types(_build_token_types(sorted_tokens, SPECIAL_TOKENS))
        writer.add_bos_token_id(tokenizer.bos_id)
        writer.add_eos_token_id(tokenizer.eos_id)
        writer.add_eot_token_id(tokenizer.eos_id)
        writer.add_pad_token_id(tokenizer.pad_id)
        writer.add_unk_token_id(tokenizer.unk_id)
        # Training sequences start with <|system|>, never with <|bos|>.
        writer.add_add_bos_token(False)
        writer.add_add_eos_token(False)
        writer.add_chat_template(CHAT_TEMPLATE)

    # ---------------- tensors ----------------
    state_dict = model.state_dict()
    tied = config.tie_embeddings and (
        model.lm_head.weight.data_ptr() == model.embed_tokens.embedding.weight.data_ptr()
    )
    # With tied embeddings llama.cpp (and Ollama) fall back to token_embd for
    # the output projection when output.weight is absent: no duplicate copy.
    name_map = tensor_name_map(config.n_layers, include_output=not tied)
    missing = [k for k in name_map if k not in state_dict]
    if missing:
        raise RuntimeError(f"State dict is missing tensors: {missing}")

    wdtype = np.float16 if quantization_type == "f16" else np.float32
    for src_name, gguf_name in name_map.items():
        arr = state_dict[src_name].detach().cpu().float().numpy()
        arr = arr.astype(np.float32 if arr.ndim == 1 else wdtype)
        writer.add_tensor(gguf_name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return output_path
