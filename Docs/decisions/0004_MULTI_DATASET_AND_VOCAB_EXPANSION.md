# ADR 0004 — Multi-Dataset Blending, Vocabulary Expansion & SFT Loss Masking

**Status**: Accepted  
**Date**: 2026-11-19

## Context

A small BPE vocabulary (around 1,024 tokens) trained on a single corpus splits everyday English words into 2 to 5 byte-level chunks, which leads to ungrammatical, fragmented dialogue. A single dataset also gives too narrow a conversational range for an empathetic companion. Furthermore, training without prompt loss masking forces the model to expend parameter capacity memorizing the static system prompt rather than learning to generate assistant replies.

## Decision

1. **Multi-Dataset Ingestion**: Blend 4 conversational corpora:
   - `anezatra/blended-skill-talk`: Persona consistency, empathy, knowledge.
   - `roskoN/dailydialog`: Natural human daily dialogue.
   - `allenai/prosocial-dialog`: Empathetic, prosocial, and safe social interactions.
   - `OpenAssistant/oasst1`: High-quality human multi-turn conversation and reasoning.
2. **Vocabulary Size**: Use a 4,096-token BPE vocabulary, ensuring common English words, dialogue contractions ("I'm", "don't", "you're"), and conversational markers are single tokens.
3. **Model Capacity**: Size the model at 6.16M parameters (6 layers, 8 heads, $d_{model} = 256$, $d_{ff} = 768$).
4. **SFT Target Masking**: Set targets for system prompt and user input tokens to `-100`, computing loss strictly on assistant response tokens and `<|eos|>`.

## Consequences

- **Positive**:
  - Full words and dialogue patterns are learned cleanly without byte fragmentation.
  - Generates coherent, emotionally warm, and contextually relevant conversational responses.
  - SFT loss masking improves gradient efficiency by 3–5x.
  - The model remains lightweight (~6M parameters) and easily trainable and runnable on standard CPU hardware.
- **Negative**:
  - Full training checkpoints are ~74MB.
- **Neutral**:
  - GGUF export mappings and tokenizer serialization format remain fully backward-compatible.

---
