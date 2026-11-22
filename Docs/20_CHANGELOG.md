# 20 — Changelog

All notable changes to the Hibari conversational language model project are documented in this file.
Model-specific changes to Koto are recorded in `models/Koto/CHANGELOG.md`.

---

## [0.4.0] — 2026-11-22: GPT-2-informed engineering (branch `feature/koto-1.3.0`)

GPT-2 studied as an engineering reference: `architecture/34_GPT2_REFERENCE.md`. Architecture,
tokenizer IDs and checkpoints unchanged (Koto: 6,163,712 parameters).

### Added
- `generate_batch` — N continuations of one prompt in a single batch (prompt prefilled once,
  per-row decoding and seeds); identical to sequential generation on the 1.1.1 model (16/16),
  4.4× faster. `HibariChatSession` candidate reranking uses it (~0.4 s per turn).
- `stream_text` and `generate.py --stream` — streaming that never shows half a UTF-8 character.
- `hibari.data.bpe_reference.ReferenceBPE` — independent GPT-2-algorithm BPE; agrees with the
  production tokenizer on 19,552/19,552 validation messages. Found that NFKC makes Python and
  llama.cpp tokenise compatibility characters differently (documented, not changed).
- `hibari.data.pipeline` — `validate_conversation`, `deduplicate`, `QualityFilter`,
  `stable_split` (identical to Koto's `split_of`), `conversation_stats`.
- `hibari.evaluation.evaluate_records` — token-weighted loss / perplexity overall and per source
  (was a stub; Koto's runner now uses it and reproduces the published 1.1.0 losses exactly).
- `ModelConfig.validate()` (runs on construction) and `ModelConfig.from_dict_strict`.
- `Trainer(resume_from=...)` / Koto `train.py --resume`: restores weights, optimizer,
  scheduler, step, best result and history, and fast-forwards the data stream.
- Message analysis: `DISAGREEMENT`, `REPAIR`, `TOPIC_CHANGE` intents; past-tense bad days,
  lost objects, "don't know what to do"; critic rules for them, for condolences without a
  loss and for cheerful replies to insults. Router: "rewrite this in a polite way".

### Fixed
- `pyproject.toml` did not declare `tokenizers` and `gguf` although `import hibari` needs both
  (`tests/test_packaging.py` checks every third-party import is declared).
- `TrainingArgs.output_dir` default pointed at the removed `hibari/models/checkpoints`.

### Kept on purpose
- `HibariKotoLM.generate` (uncached, sliding window) remains as the reference implementation that
  the KV-cache tests compare against; it has a tested contract the cached path does not have.
- Placeholder modules `hibari.memory.{store,retrieval,context}`, `hibari.personality.*`,
  `hibari.utils.*`, `evaluation.benchmarks`, `export.checkpoint_export`, `data.preprocessing`,
  `training.fine_tuning` (LoRA) are still stubs; nothing imports them. Model-specific
  conversation configuration is provided by `hibari.release.ModelProfile` + `InferenceSettings`.

## [0.3.1] — 2026-11-22: Analyzer and critic fixes from the owner's Ollama chat

### Fixed (regression tests `tests/test_conversational_layer.py::test_o1…o5`)
- "ur really a bad guy", "you're the worst bot" were not recognised as hostile.
- Short low-mood replies ("bad thats all", "meh") were not recognised as feelings.
- "say my name?" when no name was given: replies that answer with Koto's own identity
  are now penalised; "you haven't told me" is preferred (`MessageAnalysis.asks_user_fact`).
- A reply identical to one of Koto's previous replies is penalised (`repeats_previous_reply`).
- Third-party sympathy ("someone close to you", "for them") is penalised when the
  user mentioned no third person.
No false positives on "not bad thanks", "bad weather today", "that movie was bad".
- `Trainer(path_root=...)` and Koto `train.py` record repo-relative paths in
  `training_history.json` and the run metadata (they used to contain absolute local paths).
  Test: `tests/test_regressions.py::test_t4_history_paths_relative_to_root`.

## [0.3.0] — 2026-11-22: Conversational layer, release profiles, development template

### Added
- `hibari.inference.intent` (message analysis), `hibari.inference.normalization`
  (informal input), `hibari.inference.reranking` (candidate scoring: fluency, PMI
  relevance, named critic rules). `HibariChatSession` options `normalize_input`,
  `num_candidates`, `intent_candidates`, `candidate_sampling`, `identity_terms`,
  `last_reply_info`; defaults leave the old path unchanged. See `engine/26_CONVERSATIONAL_LAYER.md`.
- `hibari.release`: `ModelProfile`, `InferenceSettings`, `validate_profile`,
  `render_modelfile`; `InferenceEngine.from_profile`. See `engine/27_RELEASE_PROFILES_AND_PROVENANCE.md`.
- GGUF export: `system_prompt` and `hibari.*` provenance metadata; `build_chat_template`.
- Scripts: `export.py --profile/--record/--force` (no silent replacement),
  `validate_release.py`, `validate_experiments.py`, `compare_backends.py`,
  `generate.py --profile/--raw/--explain`.
- More rewrite phrasings ("rewrite this properly", "improve this", "make this better").
- Development template: `template/21–25`, validated config templates in `hibari/configs/`
  (model, data, training, sft, inference, release, experiment).
- Tests: `tests/test_conversational_layer.py`, `tests/test_release.py` (311 tests in total).

### Fixed
- Engine config templates used keys no code reads (`context_length`, `embedding_dim`, `use_lora`, …).
- `generate.py` / `export.py` defaulted to the 1.0.0 checkpoint and to `general.version 2.0`.

## [0.2.0] — 2026-11-21: Rewriting tasks, n-gram blocking, gradient accumulation (branch `feature/koto-1.1.0`)

### Added
- **`hibari.inference.tasks`** — one task representation for training and
  inference (`Task: <label>\nInput: <text>`, reply = rewritten text only) for
  `rephrase`, `correct_grammar`, `formal`, `casual`, `shorten`, `simplify`,
  `polite`; `parse_task_request` for explicit requests typed in chat;
  `TASK_DECODING`; protected-term (numbers, times, names, technical tokens)
  preservation checks.
- **`HibariChatSession.run_task` / `InferenceEngine.rewrite`** returning a
  `TaskResult`; `reply()` routes recognised rewrite requests (`route_tasks`).
  Rewrites use only the system prompt + request, deterministic decoding,
  seeded retries when a protected term is dropped, sentence chunking for
  input longer than the context window, fixed replies for empty / too-long
  input; the safety layer still checks rewrite input.
- **`no_repeat_ngram_size`** in `sample_next_token` / `generate_tokens` /
  `generate_text` / `InferenceEngine.generate` (default 0 = unchanged).
- `HibariChatSession.last_stopped_on_eos`.
- **`hibari.evaluation.metrics`** (was a stub): repetition, repeated sentence,
  distinct-n, grammar heuristic, echo ratio, chrF-style character F-score,
  length ratio, lead / self-introduction / AI-disclaimer detectors.
- **`TrainingArgs.gradient_accumulation_steps`** (default 1 = unchanged);
  verified to match a full-batch step exactly.
- Tests: `tests/test_tasks.py`, `tests/test_metrics.py` (237 tests in total).

### Fixed (2026-11-22, code review; regression tests in `tests/test_regressions.py`)
- **Special-token injection**: message text containing `<|assistant|>`,
  `<|eos|>` etc. was encoded as real control tokens, letting a user forge
  turns. `escape_special_tokens()` now neutralises them in
  `apply_chat_template` (training data contains none, so training is unchanged).
- `HibariChatSession.reset()` keeps the constructor's system prompt
  (`base_system_prompt`) instead of reverting to the default.
- `generate_tokens(..., stats=)` reports `stop_reason` (`stop_token`,
  `max_new_tokens`, `context_full`); `last_stopped_on_eos` no longer claims
  EOS when the context window filled.
- The over-long-message fallback keeps the `<|user|>` and `<|assistant|>` headers.
- Gradient accumulation is token-weighted (equal to a full-batch step for
  any token counts); skipped non-finite micro-batches are re-weighted;
  `max_nonfinite_steps` (default 10) stops a run that would otherwise loop forever.
- `ConversationDataset` truncates over-long conversations after the last
  complete assistant turn, so no kept reply loses its `<|eos|>`.
- Tests: 247 (10 new).

### Documentation and layout (2026-11-21)
- **Root `README.md` rewritten** to match the repository: engine / model
  split and the one-way Koto → Hibari dependency, package responsibilities,
  data-flow diagrams, a file-by-file project structure (placeholders and
  local-only folders marked), a structuring plan (phases A–E), a
  documentation map, and a Quick Start with working paths. Headline metrics
  corrected to the 1.0.0 checkpoint (val loss 3.038 / ppl 20.86); test count
  corrected (237 engine + 58 Koto).
- `Docs/README.md` index links fixed (files are in `engine/`,
  `architecture/` and the Koto submodule); `hibari/README.md` now describes
  only the base configs and where everything else moved;
  `Docs/15_API_AND_CLI_REFERENCE.md` rewritten for `scripts/engine/` and the
  Koto scripts; stale `hibari/scripts/…`, `hibari/models/…` paths replaced
  in the engine and Koto docs (changelogs and ADRs are left as historical
  records).
- **`scripts/engine/*` defaults** now point at `models/Koto/checkpoints/…`,
  `models/Koto/tokenizer` and `models/Koto/releases/exported` (the old
  `hibari/models/…` defaults no longer existed).

### Fixed
- `tests/test_model.py` looked for `configs/model/koto.yaml` in the engine
  repository; the config moved to the Koto submodule. The tests now read
  `models/Koto/configs/model/koto.yaml` and are skipped without the submodule.
- A rewrite request longer than the context window was left-truncated
  through its `Task:` header; it is now split at sentence boundaries.

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

