# 34 — GPT-2 as an Engineering Reference (engine 0.4.0)

Reference: OpenAI GPT-2, <https://github.com/openai/gpt-2> (`src/model.py`, `src/encoder.py`,
`src/sample.py`, `src/interactive_conditional_samples.py`), "Modified MIT License",
Copyright (c) 2019 OpenAI. Studied 2026-11-22 (files read, not executed).

**No GPT-2 code, weights or vocabulary were copied.** Everything below is an independent
PyTorch implementation informed by reading the reference. GPT-2 is TensorFlow 1.x; TensorFlow
was not added as a dependency. The GPT-2 repository has no training code, and none was invented.

## 1. What was studied, and the decision for each

| GPT-2 concept (file) | Hibari before | Decision | Where |
|---|---|---|---|
| `HParams` model config (`model.py`) | `ModelConfig` dataclass, no validation | **Adapted**: `ModelConfig.validate()` on construction (positive sizes, `d_model % n_heads`, even head dim for RoPE, dropout / eps / theta ranges) | `src/hibari/model/config.py` |
| Causal mask "counting from the lower right corner" (`attention_mask(nd, ns)`) for queries appended to a cache | `torch.triu(..., diagonal=1 + past_len)` | **Already equivalent**; kept | `src/hibari/model/attention.py` |
| `positions_for(tokens, past_length)` | RoPE angles sliced at `past_len` | **Already equivalent** (with RoPE instead of learned positions); kept | `attention.py` |
| `past` / `presents` KV cache | per-layer `(k, v)` cache | **Already equivalent**; kept | `attention.py`, `llm.py` |
| Tied output projection `h @ wte.T` | tied `lm_head` | **Already equivalent**; kept | `llm.py` |
| `sample_sequence(batch_size, context)` — many samples of one context in one batch | candidates generated one after another, prompt re-run each time | **Adopted**: `generate_batch` — prompt prefilled once, cache copied to N rows (`expand_tile` idea), per-row decoding settings and seeds, finished rows leave the batch | `src/hibari/inference/generation.py` |
| `top_k_logits`, `top_p_logits` | top-k, top-p, repetition penalty, n-gram blocking | **Kept** (Hibari already has a superset) | `src/hibari/inference/sampling.py` |
| Incremental decoding with `past` | yes | **Kept**; streaming added on top | `stream_text` |
| `bytes_to_unicode`, rank-ordered BPE merges, word cache (`encoder.py`) | HF `tokenizers` BPE + ByteLevel (production, unchanged) | **Adopted as a verification tool**: `ReferenceBPE`, an independent pure-Python implementation of the same algorithm reading Hibari's `tokenizer.json` | `src/hibari/data/bpe_reference.py` |
| GPT-2 pre-tokenisation pattern `'s\|'t\|…\| ?\p{L}+\| ?\p{N}+\| ?[^\s\p{L}\p{N}]+\|\s+(?!\S)\|\s+` | used inside HF ByteLevel | **Re-implemented** as a Unicode-category scanner (no `regex` dependency) | `pretokenize()` |
| `decode(..., errors="replace")` for byte sequences ending mid-character | HF decode | **Adapted for streaming**: emitted text is held back until a character is complete | `stream_text` |
| `interactive_conditional_samples` (`nsamples`, `batch_size`, `seed`) | `generate.py` chat | **Adapted**: `--stream`, `--candidates`, seeds | `scripts/engine/generate.py` |

## 2. Deliberately rejected (Hibari architecture preserved)

| GPT-2 choice | Reason it is not used |
|---|---|
| Learned absolute position embeddings (`wpe`) | Hibari uses RoPE; changing it breaks every checkpoint and GGUF and llama.cpp's `llama` architecture mapping |
| LayerNorm with bias (`norm`) | Hibari uses pre-RMSNorm (no bias, cheaper); same reason |
| GELU MLP with 4× width | Hibari uses SwiGLU (d_ff 768 for d_model 256); same reason |
| `conv1d` fused QKV with biases | Separate bias-free Q/K/V projections match the GGUF tensor layout `attn_q/k/v` |
| `-1e10` additive mask | Hibari uses `-inf` / SDPA `is_causal`, which is exact |
| Residual-projection init scaling (GPT-2 paper, 1/√N) | Changes training dynamics of from-scratch runs; would need an ablation the owner runs. Not adopted; candidate for a future experiment |
| GPT-2's 50,257-token vocabulary / weights | Koto keeps its own 4,096-token tokenizer and IDs; no GPT-2 weights are imported |

Koto's configuration is unchanged and still supported: vocabulary 4,096, d_model 256,
6 layers, 8 heads, d_ff 768, context 256, **6,163,712 parameters** (verified by
`ModelConfig.estimate_parameters()` and `tests/test_model.py`). Existing checkpoints load
unchanged; no migration is needed.

## 3. Measured results

| Check | Result |
|---|---|
| `ReferenceBPE` vs production tokenizer, 26 edge cases (Unicode, emoji, full-width, ligatures, `\r\n`, special tokens) | identical, token for token |
| Same, on 19,552 real validation messages (`processed_1.2.0`) | **19,552 / 19,552 identical** |
| `ReferenceBPE(nfkc=False)` vs `llama-tokenize` on NFKC-sensitive text ("ﬁ", "ＡＢＣ", "x² ½") | identical — and the production tokenizer differs (see §4) |
| `generate_batch` vs sequential `generate_tokens`, real 1.1.1 model, 16 candidates (greedy + 15 sampled seeds) | **16 / 16 identical**; 0.89 s → 0.20 s (4.4×) |
| Chat session with candidate reranking (8–16 candidates) | ~0.38 s per turn (was ~0.8–1.1 s) |
| `"".join(stream_text(...))` vs full reply | identical; a split multi-byte character is never shown half-done |

## 4. Finding: NFKC makes Python and llama.cpp tokenise differently

Hibari's tokenizer applies NFKC normalisation before BPE; GPT-2, llama.cpp and Ollama do not.
For compatibility characters (ligatures, full-width letters, superscripts, vulgar fractions)
the Python engine and the GGUF runtimes therefore produce different token IDs. Ordinary text,
accents and emoji are identical. Removing NFKC would change `tokenizer.json` and its SHA-256,
which every existing release profile records, so it was **not changed**. Recommended for the
next MAJOR version (2.0): drop the NFKC normaliser when the tokenizer is retrained.
