# ADR 0005 — Writing-Help Data and Curated-Data Weighting

**Status**: Accepted  
**Date**: 2026-11-21

## Context

Testing Hibari Koto in Ollama showed two gaps:

1. **Behaviour.** Hostile messages, identity questions with unusual
   phrasing, and friend or family conflicts got off-topic replies. The
   curated examples that teach these behaviours were about 2 % (v2) and 4 %
   (v4) of the training mix. A fine-tune with about 10 % curated data (v3)
   fixed most of the reported conversation, while the more fluent full train
   (v4) lost part of it.
2. **Skill.** Users asked Hibari to correct grammar and to make a message or
   email polite. No training data covered this; v4 scored 0/7 on held-out
   rewrite cases.

## Decision

1. **Add writing-help data** through `hibari/src/hibari/data/rewrite.py`:
   * `grammarly/coedit` (apache-2.0) for grammar, paraphrase, simplification
     and clarity, with instructions reworded into informal user requests.
   * `s-nlp/paradetox` (openrail++) for rude → neutral rewrites.
   * `HuggingFaceTB/smoltalk` smol-rewrite for professional / concise /
     friendly rewrites, keeping only examples that fit 256 tokens.
   * A hand-written polite / email set, because no public dataset covers
     short polite workplace messages.
   * Replies use a **fixed lead line** ("Here's the corrected sentence:")
     followed by the rewritten text, which is easy for a 6M model to learn.
2. **Keep curated data at about 10 % of a mix** and general conversation at
   50 % or more, using `build_finetune_mix.py` repeat factors.
3. **Prefer fine-tuning a well-trained base** (v4 → v5) over only retraining
   from scratch when adding behaviour, since it is about 3× cheaper and keeps
   the curated share high.
4. **Hold out evaluation prompts**: rewrite cases are written separately and
   tests assert they are absent from training data.

## Consequences

- **Positive**: Hibari gains a practical skill that fits its size (short
  edits), and behaviour fixes no longer get diluted by general data.
- **Negative**: Writing help works only for short text. Long-email rewriting
  is barely trained (2,299 usable smol-rewrite examples). Some training
  sources (`roskoN/dailydialog`, smol-rewrite) have no licence in their
  Hugging Face metadata. A 6M model can still change the meaning of rewritten
  text.
- **Neutral**: The architecture, tokenizer, GGUF format and chat template are
  unchanged.
