# ADR 0001 — Use PyTorch as the Core Framework

**Status**: Accepted  
**Date**: 2026-11

## Context

Hibari needs a deep learning framework for implementing the model, training loop, and inference engine from scratch. The main options were PyTorch and JAX.

## Decision

Use **PyTorch** as the primary framework.

## Reasons

- Imperative, Pythonic — easy to debug and understand step by step
- Dominant in LLM research (nanoGPT, LLaMA, Mistral are all PyTorch)
- Rich ecosystem: `torch.nn`, `torch.optim`, `torch.utils.data`
- Strong community and documentation
- Runs on CPU, CUDA, and MPS (Apple Silicon)

## Consequences

- **Positive**: Readable code, easy experimentation, wide reference material
- **Negative**: Slightly more verbose than JAX for certain functional patterns
- **Neutral**: JAX / XLA optimization not available without a rewrite

---
