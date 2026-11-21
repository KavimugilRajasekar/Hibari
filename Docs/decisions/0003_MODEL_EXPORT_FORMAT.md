# ADR 0003 — Model Export Format

**Status**: Accepted  
**Date**: 2026-11

## Context

After training, Hibari needs to be runnable locally without requiring a full Python/PyTorch setup. The options considered were:

- ONNX
- TorchScript
- safetensors + custom loader
- **GGUF** (llama.cpp format)

## Decision

Export to **GGUF** for local inference via llama.cpp and Ollama.

## Reasons

- GGUF is the standard format for local LLM inference
- Ollama wraps llama.cpp and makes model serving trivial
- Supports quantization (Q4, Q8) for running on consumer hardware
- Large ecosystem of tooling and community support

## Consequences

- **Positive**: Can run Hibari locally with one command (`ollama run hibari`)
- **Negative**: Requires conversion step (PyTorch → safetensors → GGUF)
- **Neutral**: GGUF format is primarily for inference, not continued training

---
