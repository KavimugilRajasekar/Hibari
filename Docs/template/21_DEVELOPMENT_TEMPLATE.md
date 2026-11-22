# 21 — Official Hibari Model Development Template

This is the engineering blueprint for every model built on Hibari. It is not
only documentation: each rule below is backed by a file you copy, a script
that checks it, or a test that fails when it is broken.

| Part | Document | Enforced by |
|---|---|---|
| Repository responsibilities and directory standard | this file | code review; `tests/test_release.py::test_p9` |
| Lifecycle (17 stages, controlled training gate) | [22_DEVELOPMENT_LIFECYCLE.md](22_DEVELOPMENT_LIFECYCLE.md) | `validate_release.py`, `validate_experiments.py` |
| Standards (testing, evaluation, data, versioning, configuration, code) | [23_ENGINEERING_STANDARDS.md](23_ENGINEERING_STANDARDS.md) | test suites, leak filter, validators |
| Ten task templates | [24_DEVELOPMENT_TEMPLATES.md](24_DEVELOPMENT_TEMPLATES.md) | acceptance criteria per template |
| Official checklist | [25_DEVELOPMENT_CHECKLIST.md](25_DEVELOPMENT_CHECKLIST.md) | must be completed in the PR / experiment record |
| Configuration templates | [`hibari/configs/`](../../hibari/configs) | `tests/test_release.py::test_p9` (keys = code fields) |

Koto's instance of this template: `models/Koto/docs/30_KOTO_MODEL_TEMPLATE.md`.

## 1. Who owns what

**Hibari (engine, this repository)** — reusable by any model; contains nothing model-specific:

| Area | Location |
|---|---|
| Transformer, attention, RoPE, RMSNorm, SwiGLU, KV cache, config | `src/hibari/model/` |
| Tokenizer, chat template, special-token escaping, datasets, collator | `src/hibari/data/` |
| Training loop, checkpoints, optimiser, scheduler | `src/hibari/training/` |
| Generation, sampling, chat session, task router, safety layer | `src/hibari/inference/` |
| Message analysis, informal normalisation, candidate reranking | `src/hibari/inference/{intent,normalization,reranking}.py` |
| GGUF export (with provenance), NumPy reference runtime | `src/hibari/export/` |
| Release profiles, validation, Modelfile generation | `src/hibari/release/` |
| Shared metrics | `src/hibari/evaluation/metrics.py` |
| Engine tools (export, validate, compare backends, chat) | `scripts/engine/` |
| Configuration and experiment templates | `hibari/configs/` |

**A model repository (e.g. Koto, `models/<Model>` submodule)** — everything that makes a particular model:

| Area | Location |
|---|---|
| Model / data / training / inference / release configs | `configs/{model,data,training,inference,release}/` |
| Tokenizer files | `tokenizer/` |
| Data builders (dialogue, empathy, rewriting, identity) | `<model>/data/` |
| Behavioural evaluation suites | `<model>/evaluation/`, runners in `evaluation/` |
| Training entry point | `training/train.py` |
| Experiment records | `experiments/EXP-NNNN.yaml` |
| Checkpoints, processed data, exports (git-ignored) | `checkpoints/`, `data/`, `releases/exported/X.Y.Z/` |
| Reports | `outputs/evaluation/` |
| Model card, changelog, behaviour specification, limitations | `MODEL_CARD.md`, `CHANGELOG.md`, `docs/` |

Rules
- The engine never imports a model package. A model's name, identity, system
  prompt and defaults reach the engine only through a **release profile**.
  (`hibari.data.formatting.DEFAULT_SYSTEM_PROMPT` is a legacy default kept for
  API compatibility; releases always pass their own prompt.)
- A model repository never copies engine code; it imports `hibari`.
- A new model is a new repository with the same layout, not a branch of Koto.

## 2. Directory standard for a model repository

```
<model>/
├── <model>/                 package: data builders, evaluation suites, paths
├── configs/
│   ├── model/<model>.yaml           ← hibari/configs/model/base.yaml
│   ├── data/<model>_X.Y.Z.yaml      ← hibari/configs/data/base.yaml
│   ├── training/<model>_X.Y.Z.yaml  ← hibari/configs/training/{base,sft}.yaml
│   ├── inference/<model>.yaml       ← hibari/configs/inference/base.yaml (only source of decoding defaults)
│   └── release/<model>_X.Y.Z.yaml   ← hibari/configs/release/template.yaml
├── experiments/EXP-NNNN.yaml        ← hibari/configs/experiment/template.yaml
├── tokenizer/  checkpoints/  data/  releases/exported/X.Y.Z/  outputs/evaluation/
├── training/train.py  evaluation/*.py  scripts/*.py  tests/
└── docs/  MODEL_CARD.md  CHANGELOG.md
```

## 3. Documentation map (Phase 13.3)

Existing files are reused; new ones were added only where nothing covered the topic.

| Required document | File |
|---|---|
| Model Architecture Specification | `Docs/architecture/04_MODEL_DESIGN.md` |
| Model Configuration Reference | `hibari/configs/model/base.yaml`, Koto `docs/14_CONFIGURATION_REFERENCE.md` |
| Tokenizer Specification | `Docs/architecture/06_TOKENIZER_DESIGN.md` |
| Inference Architecture | `Docs/engine/10_INFERENCE_AND_GENERATION.md`, `Docs/engine/26_CONVERSATIONAL_LAYER.md` |
| Context Management Specification | `Docs/engine/11_MEMORY_AND_CONTEXT.md` |
| GGUF Export Specification | `Docs/engine/12_MODEL_EXPORT_AND_QUANTIZATION.md`, `Docs/engine/27_RELEASE_PROFILES_AND_PROVENANCE.md` |
| Personality, Conversational, Empathy, Reformulation, Safety, Identity / System Prompt specifications | Koto `docs/28_BEHAVIOUR_SPECIFICATION.md` |
| Dataset Catalog | Koto `docs/16_DATASET_CATALOG.md` |
| Dataset Quality, Preprocessing, Validation Checklist, Leakage Prevention | Koto `docs/29_DATA_STANDARDS.md`, `docs/05_DATA_PIPELINE.md` |
| Unit / Integration Testing, Conversational Evaluation, Regression Suite, Inference Compatibility, Metrics & Reporting | `23_ENGINEERING_STANDARDS.md` §1–3, Koto `docs/09_EVALUATION_AND_TESTING.md` |
| Version Management | `23_ENGINEERING_STANDARDS.md` §5 |
| Experiment Tracking | `hibari/configs/experiment/template.yaml`, Koto `experiments/` |
| Change Logs | `Docs/20_CHANGELOG.md`, Koto `CHANGELOG.md` |
| Known Limitations | Koto `docs/18_KNOWN_LIMITATIONS.md` |
| Release Readiness Checklist | `25_DEVELOPMENT_CHECKLIST.md` §B |
| Future Development Roadmap | `Docs/engine/02_PROJECT_GOALS.md`, Koto `docs/31_KOTO_1.1.1_REPAIR_REPORT.md` §11 |

## 4. Tools

| Command | Purpose |
|---|---|
| `scripts/engine/validate_release.py --profile P` | config ↔ checkpoint ↔ tokenizer ↔ GGUF ↔ Modelfile consistency (read-only) |
| `scripts/engine/validate_experiments.py <repo>/experiments` | experiment records follow the template; training has an owner |
| `scripts/engine/export.py --profile P --record` | hash-checked export with provenance; never silently replaces a file |
| `scripts/engine/compare_backends.py --profile P` | tokenizer, chat-template and greedy parity: PyTorch / llama.cpp / Ollama |
| `scripts/engine/generate.py --profile P [--explain]` | chat with exactly the release settings |
| `pytest` (both repositories) | unit, integration and regression tests |
