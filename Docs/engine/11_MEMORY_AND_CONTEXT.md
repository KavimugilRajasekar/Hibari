# 11 — Memory and Context

## Context Window

Hibari's context window defines the maximum number of tokens the model can attend to at once. Beyond this limit, older tokens are dropped.

## Memory Systems

### In-Context Memory
- The model's attention over the current token sequence
- Limited by context window size

### Conversation History
- Managed by `inference/conversation.py`
- Sliding window or summarization for long conversations

### External Memory (future)
- Retrieval-Augmented Generation (RAG)
- Vector store for long-term facts and knowledge

## Files

| File | Purpose |
|------|---------|
| `memory/store.py` | Memory storage abstraction |
| `memory/retrieval.py` | Retrieval logic |
| `memory/context.py` | Context window management |

---
