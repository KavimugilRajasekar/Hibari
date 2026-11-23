# Docs/assets — diagrams

Images used by the documentation. Names are lower-case kebab-case with no spaces, so links
work unescaped on GitHub and on every OS. Add each new file to this table and reference it with a
relative path (`Docs/assets/<file>` from the repository root, `../assets/<file>` from a
`Docs/<section>/` page).

All four diagrams are drawn for **Koto 1.3.0** (29.4M parameters: d_model 512, 8 layers,
8 heads × 64, d_ff 1536, 4,096-token BPE, 256-token context, RoPE θ = 10,000, tied embeddings).
Koto 1.1.x has the same structure at 6.16M parameters (d_model 256, 6 layers, 8 heads × 32, d_ff 768).
Checked against `src/hibari/model/` and `models/Koto/configs/model/koto_1.3.0_large.yaml` on 2026-11-23.

| File | Shows | Used in |
|---|---|---|
| [hibari-koto-overall-architecture.jpeg](hibari-koto-overall-architecture.jpeg) | End to end: chat turns → tokenizer → embedding → 8 pre-norm blocks → final RMSNorm → tied LM head → sampling → autoregressive loop | `README.md` (Architectural Principles), `architecture/04_MODEL_DESIGN.md` |
| [hibari-koto-transformer-block.jpeg](hibari-koto-transformer-block.jpeg) | One block: `h = x + Attn(RMSNorm(x))`, `y = h + SwiGLU(RMSNorm(h))`, with attention and SwiGLU detail | `README.md` §4, `04_MODEL_DESIGN.md` §2 |
| [hibari-koto-attention.jpeg](hibari-koto-attention.jpeg) | Multi-head causal self-attention: bias-free Q/K/V, RoPE on Q and K, KV cache, scaled dot-product with causal mask | `README.md` §3, `04_MODEL_DESIGN.md` §2 |
| [hibari-koto-embedding-and-output.jpeg](hibari-koto-embedding-and-output.jpeg) | Token embedding lookup, tied output projection (`logits = h Eᵀ`), sampling and detokenisation | `README.md` §5, `04_MODEL_DESIGN.md` §1 |
