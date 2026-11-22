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
| Release profiles, validation, Modelfile | `src/hibari/release/` |
| Engine tools | `scripts/engine/` (export, validate_release, validate_experiments, compare_backends, generate, verify_gguf) |
| Config templates | `hibari/configs/` |
| Engine docs / development template | `Docs/`, `Docs/template/21–25` |
| Koto configs (model, data, training, inference, release) | `models/Koto/configs/` |
| Koto data builders | `models/Koto/koto/data/` (curated, conversation, dialogue_skills, tasks, rewrite, behaviour) |
| Koto evaluation suites | `models/Koto/koto/evaluation/` (upgrade_suite, suite, behaviour_suite) |
| Koto runners | `models/Koto/evaluation/`, `models/Koto/scripts/`, `models/Koto/training/train.py` |
| Experiment records | `models/Koto/experiments/EXP-NNNN.yaml` |
| Reports | `models/Koto/outputs/evaluation/` |
| Koto docs | `models/Koto/docs/` (behaviour spec 28, data standards 29, model template 30, reports 19, 31, 32) |

Git-ignored, local only: `checkpoints/`, `data/processed*`, `*.gguf`, `archive/**/data`.

## Current state (keep this section updated)

| Item | Value |
|---|---|
| Engine version | 0.3.1 |
| Current Koto release | 1.1.1 (weights 1.1.0) — `configs/release/koto_1.1.1.yaml` |
| Trained, not released | 1.2.0 — `configs/release/koto_1.2.0.yaml` (EXP-0004): rewriting regression, see Koto docs/32 |
| Next Koto version | 1.2.1 — data built (`data/processed_1.2.1`), training NOT STARTED (EXP-0005) |
| Branches / tags | Koto `release/1.2.0` (work), `release/1.1.0`, tags `v1.1.0` `v1.1.1`; Hibari `feature/koto-1.1.0`, tag `engine-v0.3.0`; both `main` branches are behind |
| Legacy trap | `models/Koto/releases/exported/*.gguf` hold **1.0.0** weights; use `exported/1.1.1/` |
| Environment | Python venv `.venv`; llama.cpp binaries in `~/llama.cpp` (need `LD_LIBRARY_PATH=~/llama.cpp`); Ollama CLI installed (the system `ollama.service` was broken on 2026-11-22: missing `/usr/share/ollama`) |
