# Architecture Decision Records (ADRs)

This directory records significant architectural and design decisions made during the development of Hibari.

## Format

Each ADR follows this structure:
- **Status**: Proposed / Accepted / Deprecated / Superseded
- **Context**: Why this decision was needed
- **Decision**: What was decided
- **Consequences**: Trade-offs and implications

## Index

| ADR | Title |
|-----|-------|
| [0001](0001_USE_PYTORCH.md) | Use PyTorch as the core framework |
| [0002](0002_PRETRAINED_VS_FROM_SCRATCH.md) | Build from scratch vs. fine-tuning a pretrained model |
| [0003](0003_MODEL_EXPORT_FORMAT.md) | Model export format (GGUF for Ollama) |
| [0004](0004_MULTI_DATASET_AND_VOCAB_EXPANSION.md) | Multi-dataset blending, vocabulary expansion & SFT loss masking |
| [0005](0005_WRITING_HELP_AND_CURATED_WEIGHTING.md) | Writing-help data and curated-data weighting |
