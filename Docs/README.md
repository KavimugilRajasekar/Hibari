# Hibari — Docs

Documentation for the **Hibari** engine lives here. Documentation for the
**Koto** model lives in the Koto submodule, `models/Koto/docs/`. Both use one
shared number sequence (01–20), so each number always means the same topic.

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
| [15_API_AND_CLI_REFERENCE.md](15_API_AND_CLI_REFERENCE.md) | API and CLI usage |
| [20_CHANGELOG.md](20_CHANGELOG.md) | Engine changelog |
| [decisions/](decisions/README.md) | Architecture Decision Records (ADRs 0001–0005) |

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
| [Koto CHANGELOG](../models/Koto/CHANGELOG.md) | Model changelog (1.0.0, 1.1.0) |

## Where new documents go

- A change to the `hibari` package → `Docs/engine/` or `Docs/architecture/`.
- A design decision with trade-offs → a new ADR in `Docs/decisions/`.
- Koto data, training runs, evaluation results, releases → `models/Koto/docs/`.
- Engine changes are logged in `Docs/20_CHANGELOG.md`; model changes in `models/Koto/CHANGELOG.md`.
