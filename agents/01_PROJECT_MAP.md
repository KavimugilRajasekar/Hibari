# 01 — Project Map

Two git repositories:

| Repository | Path | Role | Default branch for work |
|---|---|---|---|
| **Hibari** (engine) | `.` | LLM architecture, tokenizer, training loop, inference, export, release tooling | a `feature/*` branch, never `main` directly |
| **Koto** (model, submodule) | `models/Koto` | the first conversational model: configs, data, evaluation, experiments, releases | `release/<version>` |

A change that touches both is committed in Koto first, then the submodule pointer in Hibari.

## Where things are

| Need | Location |
|---|---|
| Model architecture | `src/hibari/model/` |
| Tokenizer, chat template, special-token escaping | `src/hibari/data/tokenizer.py` |
| Chat session (history, context trimming, routing, candidates) | `src/hibari/inference/conversation.py` |
| Intent / emotion analysis | `src/hibari/inference/intent.py` |
| Informal-text normalisation | `src/hibari/inference/normalization.py` |
| Reply critic and candidate reranking | `src/hibari/inference/reranking.py` |
| Rewriting tasks and request parsing | `src/hibari/inference/tasks.py` |
| Safety layer | `src/hibari/inference/safety.py` |
| GGUF export + reference runtime | `src/hibari/export/` |
| Batched / streaming generation | `src/hibari/inference/generation.py` (`generate_batch`, `stream_text`) |
| Data pipeline steps, reference BPE | `src/hibari/data/pipeline.py`, `src/hibari/data/bpe_reference.py` |
| Loss / perplexity evaluator | `src/hibari/evaluation/evaluator.py` |
| Release profiles, validation, Modelfile | `src/hibari/release/` |
| Engine tools | `scripts/engine/` (export, validate_release, validate_experiments, compare_backends, generate, verify_gguf) |
| Config templates | `hibari/configs/` |
| Engine docs / development template | `Docs/`, `Docs/template/21–25` |
| Koto configs (model, data, training, inference, release) | `models/Koto/configs/` |
| Koto data builders | `models/Koto/koto/data/` (curated, conversation, dialogue_skills, tasks, rewrite, behaviour) |
| Koto evaluation suites | `models/Koto/koto/evaluation/` (upgrade_suite, suite, behaviour_suite) |
| Koto runners | `models/Koto/evaluation/`, `models/Koto/scripts/`, `models/Koto/training/train.py` |
| Experiment records | `models/Koto/experiments/EXP-NNNN.yaml` |
| Reports | `models/Koto/outputs/evaluation/`, dataset reports `models/Koto/outputs/data/` |
| Koto docs | `models/Koto/docs/` (behaviour spec 28, data standards 29, model template 30, reports 19, 31, 32, 33) |

Git-ignored, local only: `checkpoints/`, `data/processed*`, `*.gguf`, `archive/**/data`.

## Current state (keep this section updated)

| Item | Value |
|---|---|
| Engine version | 0.4.1 (branch `feature/koto-1.3.1`; `feature/koto-1.3.0` = same commits up to the 1.3.1 release) |
| Current Koto release | **1.3.1** `Hibari-Koto1.3.1:29M` (29.4M, EXP-0007) — `configs/release/koto_1.3.1.yaml`; released 2026-11-23 by the owner's decision although it failed the EXP-0007 rule (behaviour 0.628 vs 0.660, rewriting 0.333; conversation 0.733 vs 0.645) — Koto docs/34. Published: HF `kavimugil-r/Hibari-Koto`, Ollama `kavimugil-r/Hibari-Koto1.3.1:29M` |
| Previous release | 1.1.1 (weights 1.1.0) — `configs/release/koto_1.1.1.yaml` |
| Trained, not released | 1.2.0 — `configs/release/koto_1.2.0.yaml` (EXP-0004): rewriting regression, see Koto docs/32 |
| Trained, not accepted | 1.3.0 — 29.4M (EXP-0006, `checkpoints/koto_1.3.0_large/`): trained on other data than its config named; behaviour 0.558 vs 0.660, reformulation 0.333; GGUFs in `releases/exported/1.3.0/` for testing only |
| Next Koto version | not prepared; the measured weakness is rewriting (1.3.1: 0.333) — raising the rewriting share in the data is the owner's decision |
| Hubs | HF model `kavimugil-r/Hibari-Koto` (public), HF dataset `kavimugil-r/koto-data` (private, `processed_1.3.0`); Ollama `kavimugil-r/Hibari-Koto1.3.1:29M`; cards in Koto `releases/hub/` |
| Branches / tags | Koto `release/1.3.1` (current work; `release/1.3.0` kept), `release/1.2.0`, `release/1.1.0`, tags `v1.1.0` `v1.1.1` `v1.3.1`; Hibari `feature/koto-1.1.0`, tag `engine-v0.3.0`; both `main` branches are behind |
| Legacy trap | `models/Koto/releases/exported/*.gguf` hold **1.0.0** weights; use `exported/1.3.1/` |
| Environment | Python venv `.venv`; llama.cpp binaries in `~/llama.cpp` (need `LD_LIBRARY_PATH=~/llama.cpp`); Ollama CLI installed (the system `ollama.service` is broken: missing `/usr/share/ollama` and `llama-server`; for `ollama create/push` run a user server: `OLLAMA_HOST=127.0.0.1:11435 ollama serve`) |
