# 03 — System Architecture

## Overview

Hibari follows a modular pipeline architecture where each stage is independently testable and swappable.

```
Raw Text
   │
   ▼
[ Data Pipeline ]  ──▶  tokenizer, dataset, preprocessing, collator
   │
   ▼
[ Model ]          ──▶  embedding → attention → transformer blocks → LLM head
   │
   ▼
[ Training ]       ──▶  trainer, optimizer, scheduler, loss, checkpoint
   │
   ▼
[ Fine-tuning ]    ──▶  SFT, instruction tuning, LoRA adapters
   │
   ▼
[ Evaluation ]     ──▶  perplexity, benchmarks, custom metrics
   │
   ▼
[ Export ]         ──▶  safetensors → GGUF → Ollama
   │
   ▼
[ Inference ]      ──▶  generation engine, sampling, conversation
```

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Tokenizer | `src/hibari/data/tokenizer.py` | BPE tokenization |
| Model | `src/hibari/model/llm.py` | Full model assembly |
| Trainer | `src/hibari/training/trainer.py` | Training loop |
| Inference Engine | `src/hibari/inference/engine.py` | Text generation |
| Export | `src/hibari/export/` | Model conversion |

---

*See [04_MODEL_DESIGN.md](04_MODEL_DESIGN.md) for the model architecture.*
