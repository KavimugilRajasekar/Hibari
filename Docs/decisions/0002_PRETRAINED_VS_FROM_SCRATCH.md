# ADR 0002 — Build from Scratch vs. Fine-Tuning a Pretrained Model

**Status**: Accepted  
**Date**: 2026-11

## Context

The fastest path to a working LLM-Brain would be to take an existing pretrained model (e.g., LLaMA, Mistral) and fine-tune it. The alternative is implementing and training the full model from scratch.

## Decision

**Build from scratch.**

## Reasons

- Hibari's primary goal is deep understanding, not deployment speed
- Fine-tuning a pretrained model skips the most educational parts: architecture, tokenizer, pretraining loop
- Building from scratch forces engagement with every component
- The result is a model that is fully owned and understood

## Consequences

- **Positive**: Complete understanding of every layer and decision
- **Negative**: Much slower path to a usable model; requires significant compute
- **Neutral**: Fine-tuning on top of a custom pretrained base is still planned in later phases

---
