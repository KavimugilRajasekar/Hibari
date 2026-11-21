# 01 — Project Overview

## What is Hibari?

Hibari is an LLM-Brain — a large language model cognitive architecture designed and built from scratch. The goal is not just to use an LLM, but to deeply understand and construct one.

## Motivation

Most developers use LLMs as black boxes. Hibari exists to break that pattern — to study, implement, and reason about every layer: from tokenization and embeddings to attention, training, and inference.

## Scope

Hibari covers:
- A decoder-only Transformer architecture (GPT-style)
- A custom tokenizer (BPE)
- A full pretraining and fine-tuning pipeline
- Personality and conversational behaviour design
- Local inference and Ollama integration

## Model — Hibari Koto

| Property | Hibari Koto |
|----------|-------------|
| Model | **Hibari Koto (`Hibari-Koto`)** |
| Parameters | **~6.16M (6,163,712)** |
| Vocabulary | **4,096 BPE tokens** |
| Hidden Dim / Layers / Heads | **d_model=256, 6 layers, 8 heads, d_ff=768** |
| Training Datasets | **7 conversation datasets + hand-written curated set; from v5 also CoEdIT, ParaDetox, smol-rewrite and a polite / email set** |
| Train examples | **71,715 (v2) → 73,428 (v4) → 107,634 (v6)** |
| Loss Objective | **Loss on assistant tokens only (`targets = -100` on system and user tokens)** |
| Best validation perplexity | **20.86 (v4)** |
| Export Formats | **GGUF F32 (24.8 MB), F16 (12.5 MB), Q8_0 (6.7 MB), Q4_K_M (4.0 MB)** |

Hibari Koto is the initial step of the project: whole words are single tokens,
replies are short and mostly grammatical, and the whole model trains
and runs on a CPU. It remains a 6M-parameter model; see
`18_KNOWN_LIMITATIONS.md` for what that means in practice.

## Status

> **Active (2026-11-21)** — Hibari Koto v4 (full 12k-step train, val
> perplexity 20.86) is exported to GGUF and runs in llama.cpp and Ollama with
> output identical to PyTorch. v5 (fine-tune of v4 adding writing help:
> grammar fixes, polite rewrites, short emails) and v6 (full train including
> writing help) are training. Test suite: 186 passed. See
> `07_TRAINING_PIPELINE.md` for all runs.

---

*See [02_PROJECT_GOALS.md](02_PROJECT_GOALS.md) for goals and milestones.*
