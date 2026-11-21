# 02 — Project Goals

## Primary Goals

- Build a working decoder-only LLM from scratch in PyTorch
- Understand every component deeply — no magic, no skipped steps
- Train on curated conversational data to give Hibari a distinct personality
- Export and run the model locally via Ollama / llama.cpp

## Non-Goals

- Competing with GPT-4, LLaMA, or any frontier model on benchmarks
- Building a production-grade API service (at this stage)
- Using high-level trainer frameworks (HuggingFace Trainer, Lightning) during the core build phase

## Milestones

| Phase | Goal |
|-------|------|
| 0 | Study — understand all foundations (current) |
| 1 | Implement tokenizer, dataset pipeline, and model skeleton |
| 2 | Pretraining on small corpus |
| 3 | Fine-tuning on conversational data |
| 4 | Personality shaping and safety alignment |
| 5 | Export to GGUF and run via Ollama |

---

*See [03_SYSTEM_ARCHITECTURE.md](03_SYSTEM_ARCHITECTURE.md) for the architecture.*
