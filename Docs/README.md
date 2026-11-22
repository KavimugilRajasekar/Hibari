# Hibari — Docs

Documentation for the **Hibari** engine lives here. Documentation for the
**Koto** model lives in the Koto submodule, `models/Koto/docs/`. Both use one
shared number sequence (01–32), so each number always means the same topic.

## Engine documentation (this folder)

| File | Description |
|------|-------------|
| [engine/01_PROJECT_OVERVIEW.md](engine/01_PROJECT_OVERVIEW.md) | What Hibari is and why it exists |
| [engine/02_PROJECT_GOALS.md](engine/02_PROJECT_GOALS.md) | Goals, non-goals, and milestones |
| [architecture/03_SYSTEM_ARCHITECTURE.md](architecture/03_SYSTEM_ARCHITECTURE.md) | High-level system design |
| [architecture/04_MODEL_DESIGN.md](architecture/04_MODEL_DESIGN.md) | Model architecture decisions |
| [architecture/06_TOKENIZER_DESIGN.md](architecture/06_TOKENIZER_DESIGN.md) | Tokenizer approach and vocabulary |
| [engine/10_INFERENCE_AND_GENERATION.md](engine/10_INFERENCE_AND_GENERATION.md) | Inference engine, sampling, chat session, rewriting tasks |
| [engine/11_MEMORY_AND_CONTEXT.md](engine/11_MEMORY_AND_CONTEXT.md) | Memory systems and context handling |
| [engine/12_MODEL_EXPORT_AND_QUANTIZATION.md](engine/12_MODEL_EXPORT_AND_QUANTIZATION.md) | GGUF export and quantization |
| [engine/26_CONVERSATIONAL_LAYER.md](engine/26_CONVERSATIONAL_LAYER.md) | Message analysis, informal normalisation, candidate reranking |
| [engine/27_RELEASE_PROFILES_AND_PROVENANCE.md](engine/27_RELEASE_PROFILES_AND_PROVENANCE.md) | Release profiles, GGUF provenance, validation, backend parity |
| [15_API_AND_CLI_REFERENCE.md](15_API_AND_CLI_REFERENCE.md) | API and CLI usage |
| [20_CHANGELOG.md](20_CHANGELOG.md) | Engine changelog |
| [decisions/](decisions/README.md) | Architecture Decision Records (ADRs 0001–0005) |

## Official development template (`template/`)

| File | Description |
|------|-------------|
| [template/21_DEVELOPMENT_TEMPLATE.md](template/21_DEVELOPMENT_TEMPLATE.md) | Responsibilities, directory standard, documentation map, tools |
| [template/22_DEVELOPMENT_LIFECYCLE.md](template/22_DEVELOPMENT_LIFECYCLE.md) | 17-stage lifecycle and the controlled training gate |
| [template/23_ENGINEERING_STANDARDS.md](template/23_ENGINEERING_STANDARDS.md) | Testing, evaluation, compatibility, data, versioning, configuration, code |
| [template/24_DEVELOPMENT_TEMPLATES.md](template/24_DEVELOPMENT_TEMPLATES.md) | Ten task templates with acceptance criteria |
| [template/25_DEVELOPMENT_CHECKLIST.md](template/25_DEVELOPMENT_CHECKLIST.md) | Official checklist and release readiness |

## AI developer instructions

[`agents/`](../agents/README.md) — ground rules, project map, workflow and role guides for AI coding agents
(loaded via `AGENTS.md` / `CLAUDE.md` in both repositories).

## Koto model documentation (`models/Koto/docs/`)

| File | Description |
|------|-------------|
| [05_DATA_PIPELINE.md](../models/Koto/docs/05_DATA_PIPELINE.md) | Data ingestion, the 1.1.0 data mix and its builders |
| [07_TRAINING_PIPELINE.md](../models/Koto/docs/07_TRAINING_PIPELINE.md) | Training workflow and runs |
| [08_FINE_TUNING.md](../models/Koto/docs/08_FINE_TUNING.md) | Fine-tuning runs |
| [09_EVALUATION_AND_TESTING.md](../models/Koto/docs/09_EVALUATION_AND_TESTING.md) | Evaluation suites, metrics and tests |
| [13_OLLAMA_INTEGRATION.md](../models/Koto/docs/13_OLLAMA_INTEGRATION.md) | Running Koto via Ollama |
| [14_CONFIGURATION_REFERENCE.md](../models/Koto/docs/14_CONFIGURATION_REFERENCE.md) | All config options |
| [16_DATASET_CATALOG.md](../models/Koto/docs/16_DATASET_CATALOG.md) | Datasets used for training, with licences |
| [17 Model Card](../models/Koto/MODEL_CARD.md) | Model card for Hibari Koto |
| [18_KNOWN_LIMITATIONS.md](../models/Koto/docs/18_KNOWN_LIMITATIONS.md) | Known issues and limitations |
| [19_KOTO_1.1.0_DEVELOPMENT_REPORT.md](../models/Koto/docs/19_KOTO_1.1.0_DEVELOPMENT_REPORT.md) | 1.1.0 review, fixes, training run and 1.0.0 vs 1.1.0 evaluation |
| [28_BEHAVIOUR_SPECIFICATION.md](../models/Koto/docs/28_BEHAVIOUR_SPECIFICATION.md) | Identity, personality, empathy, hostility, reformulation, safety |
| [29_DATA_STANDARDS.md](../models/Koto/docs/29_DATA_STANDARDS.md) | Data quality, preprocessing, leakage prevention, validation checklist |
| [30_KOTO_MODEL_TEMPLATE.md](../models/Koto/docs/30_KOTO_MODEL_TEMPLATE.md) | Koto's instance of the template; 1.2.0 workflow |
| [31_KOTO_1.1.1_REPAIR_REPORT.md](../models/Koto/docs/31_KOTO_1.1.1_REPAIR_REPORT.md) | 1.1.1 audit, root causes, changes, results, readiness |
| [32_KOTO_1.2.0_EVALUATION.md](../models/Koto/docs/32_KOTO_1.2.0_EVALUATION.md) | 1.2.0 training result vs 1.1.1; why it is not released; 1.2.1 plan |
| [Koto CHANGELOG](../models/Koto/CHANGELOG.md) | Model changelog (1.0.0, 1.1.0) |

## Where new documents go

- A change to the `hibari` package → `Docs/engine/` or `Docs/architecture/`.
- A design decision with trade-offs → a new ADR in `Docs/decisions/`.
- Koto data, training runs, evaluation results, releases → `models/Koto/docs/`.
- Engine changes are logged in `Docs/20_CHANGELOG.md`; model changes in `models/Koto/CHANGELOG.md`.
