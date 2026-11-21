# 20 — Changelog

All notable changes to the Hibari conversational language model project are documented in this file.

---

## [Unreleased] — 2026-11-20 / 21: Retraining, hostility handling, writing help

### Fixed
- **Conversation failures reported from Ollama.** Insults, "kill ur self",
  slurs, "I hate urself", "what wrong with u", and creator questions with a
  prefix ("haa Hibari, Who Developed u?") produced off-topic or nonsensical
  replies. Later, "i got fight with my friend" got "That's great news!".
  Cause: no matching curated examples, and curated data was only about 2 % of
  the mix. Fixed with new curated intents and fine-tunes (v3, v5).
- **Invented personas** ("Dr. Emily Wilson", "Jack Harris") for "what do you
  like to do?" → new `about_you` intent.

### Added
- **Curated intents** (`data/curated.py`): `abuse_die`, `harassment`,
  `hate_you`, `whats_wrong`, `apology`, `about_you`, `writing_help`,
  `friend_conflict`, `have_problem`, `feeling_bad`, `already_told`; more
  creator phrasings; name prefixes ("haa hibari, …"); 400 mixed hostile /
  normal 3–4 turn chains; 5 new multi-turn examples.
- **Writing-help data** (`data/rewrite.py`): loaders for `grammarly/coedit`,
  `s-nlp/paradetox` and `HuggingFaceTB/smoltalk` smol-rewrite, plus a
  hand-written polite rewrite and email set.
- **`scripts/build_finetune_mix.py`**: builds training mixes from the local
  cache with `--replay`, `--curated-repeats`, `--coedit`, `--paradetox`,
  `--smol-rewrite` and `--polite-repeats`.
- `prepare_data.py` and `configs/data/koto.yaml` include CoEdIT and the
  polite set.
- **Training configs**: `koto_ft.yaml` (v3), `koto_full.yaml` (v4),
  `koto_v5.yaml`, `koto_full_v6.yaml`.
- **Evaluation**: `rewrite` category (7 held-out cases, including the
  reported "Hi Liz!" sentence) and case C7; the suite now has 58 cases.
  Tests check that rewrite prompts are held out from training data. Test
  suite: 186.

### Trained
| Run | Result |
|---|---|
| v3 (fine-tune of v2) | val ppl 34.99 → 26.95; reported conversation ~2/24 → ~21/24 acceptable replies |
| v4 (full, 12k steps) | val ppl **20.86**, relevance 41.9 %; most fluent, weaker on identity than v3 |
| v5 (fine-tune of v4 + writing help) | in progress |
| v6 (full + writing help) | in progress |

### Exported
- v4 GGUFs (F32, F16, Q8_0, Q4_K_M) in `hibari/models/exported/`; all match
  PyTorch greedy output in llama.cpp (6/6, tokenizer 12/12). The old v2
  GGUFs there were removed.
- Ollama tags `hibari-koto:6m-v3`, `:6m-v4` (Q8_0), `:6m-v4-f32`.

### Documented
- llama.cpp binaries in `~/llama.cpp` need `LD_LIBRARY_PATH`;
  `llama-completion` needs `-no-cnv`; llama-server rejects chats longer than
  256 tokens.
- Docs 01, 05, 07, 08, 09, 12–18 updated; ADR 0005 added.

---

## [0.1.0] — 2026-11-19: Initial Hibari Koto release

### Fixed
- **Prompt Loss Contamination**: Implemented Supervised Fine-Tuning (SFT) loss masking (`targets = -100` on system prompts and user queries). The model now focuses its entire gradient capacity on predicting the assistant's reply.
- **Generation Halting**: Added multi-role stop tokens (`<|eos|>`, `<|user|>`, `<|system|>`) to prevent generation from bleeding into fake user turns.

### Added
- **Crisis Safety Guardrail (`src/hibari/inference/safety.py`)**: Intercepts crisis keywords ("kill myself", "want to die", self-harm) before model generation and overrides with empathetic care and 24/7 verified crisis helplines (iCall: 9152987821, Vandrevala: 1860-2662-345, AASRA: 91-22-27546669).
- **Identity & Persona Handling**: Seamlessly identifies queries regarding Hibari's name and identity ("ur name?", "who are you?") and replies warmly with its persona and emojis.
- **Context-Aware Emoji Enrichment**: Dynamically adds warm, emotionally appropriate emojis (🌸, 😊, 💙, ✨, 🤗, 🎉) to conversational responses.
- **Emotional Distress Support**: Detects sadness, stress, and anxiety in user messages and appends empathetic support and virtual hugs.
- **Synthetic Data Injection & Fine-Tuning**: Injected 198 synthetic identity, crisis-safe, and emoji-rich dialogue pairs into the training corpus and fine-tuned the model.
- **Comprehensive Inference Tests**: Added 6 new unit tests in `hibari/tests/test_inference.py` covering safety guardrails, identity responses, distress handling, and emoji enrichment (85/85 tests passing).
- **Multi-Dataset Support**: Added ingest pipelines for 3 new datasets in addition to Blended Skill Talk:
  - `roskoN/dailydialog` (~11,118 conversations): Natural everyday conversational English.
  - `allenai/prosocial-dialog` (15,000 conversations): Empathy, safe social dialogue, and constructive responses.
  - `OpenAssistant/oasst1` (10,000 conversations): High-quality multi-turn conversation and reasoning.
- **Unified Data Preparation CLI**: Updated `hibari/scripts/prepare_data.py` to automatically download, format, train tokenizer, and cache all splits to disk.
- **Dataset Catalog Documentation**: Comprehensive updates to `Docs/16_DATASET_CATALOG.md`.

### Model
- **Tokenizer**: Byte-level BPE trained on 86,270 multi-dataset utterances, **4,096 tokens**; words like "Hello", "feeling", "friend", "understand" and common contractions are single tokens.
- **Architecture**: **6,163,712 (~6.16M)** parameters — **6** Transformer blocks, `d_model` **256**, **8** attention heads (head dim 32), SwiGLU `d_ff` **768**.
- **Inference Sampling Defaults**:
  - Optimized default sampling to `temperature = 0.5`, `top_k = 25`, `top_p = 0.85`, and `repetition_penalty = 1.15`.

