<div align="center">

<img src="./hibari.jpeg" alt="Hibari Logo" width="180" />

# Hibari

**A Purpose-Built Cognitive LLM Architecture & Language Model Built from Scratch**

Author: **[Kavimugil Rajasekar](https://kavimugilrajasekar.github.io)**  
[GitHub](https://github.com/KavimugilRajasekar) | [Portfolio](https://kavimugilrajasekar.github.io) | [Repository](https://github.com/KavimugilRajasekar/Hibari)

</div>

---

## Overview

**Hibari** is the name of the overarching cognitive Large Language Model (LLM) architecture and research project being engineered entirely from the ground up in pure PyTorch.

Rather than fine-tuning pre-existing weights or relying on high-level prebuilt model wrappers, Hibari is implemented from foundational mathematical principles. Every core component—from Rotary Position Embeddings (RoPE), Root Mean Square Normalization (RMSNorm), and SwiGLU feed-forward networks to custom Byte-Pair Encoding (BPE), autoregressive inference engines, and native binary GGUF export—is written directly.

**Hibari Koto** (`Hibari-Koto`) is the first conversational model developed and trained under the Hibari architecture. Built specifically for human-like empathetic dialogue, it has 6.16M parameters and blends multiple conversation corpora with a 4,096 BPE vocabulary and supervised fine-tuning (SFT) loss masking.

---

## Model Specifications: Hibari Koto (`Hibari-Koto`)

| Specification | Hibari Koto | Technical Context |
|---|---|---|
| Parent Architecture | Hibari | Decoder-only causal Transformer |
| Model Identifier | `Hibari-Koto` | Conversational empathy specialist |
| Trainable Parameters | **6,163,712 (~6.16 Million)** | Unconstrained CPU efficiency |
| Hidden Dimension (d_model) | **256** | Activation width per layer |
| Transformer Layers | **6** | Stacked causal Transformer blocks |
| Attention Heads | **8** | Head dimension = 32 |
| Feed-Forward Network | **SwiGLU (d_ff = 768)** | 3x expansion with Swish gating |
| Position Encoding | RoPE (theta = 10,000.0) | Rotary embeddings across all heads |
| Normalization | Pre-RMSNorm (eps = 1e-5) | Scale-invariant layer normalization |
| Vocabulary Size | **4,096 tokens** | Byte-level BPE, eliminates token fragmentation |
| Max Context Window | 256 tokens | Dynamic causal triangular mask |
| Weight Tying | Enabled | token_embeddings.weight tied to lm_head |
| Training Objective | **SFT Target Masking (`targets = -100`)** | Masks prompt/user turns to focus solely on assistant |
| Training Corpus | **4 Datasets Blended (40,214 convs)** | BST + DailyDialog + Prosocial-Dialog + OASST1 |
| Validation Loss | **4.2939** | Evaluated on multi-turn validation split |
| Validation Perplexity | **73.25** | Stable convergence across expanded 4k vocab |
| Export Formats | **GGUF (FP32 & FP16)** | Native llama.cpp and Ollama compatibility |
| Binary File Size | **27.65 MB (FP32) / 13.90 MB (FP16)** | Compact, immediate edge deployment |

---

## Directory Structure

```
Hibari/
├── README.md                           # Main project documentation and overview
├── hibari.jpeg                         # Official Hibari project logo
├── CONTRIBUTING.md                     # Contribution workflow and guidelines
├── LICENSE                             # MIT License (Copyright 2026 Kavimugil Rajasekar)
├── pyproject.toml                      # Build metadata, dependencies, and test paths
├── .gitignore                          # Git tracking exclusions
│
├── Docs/                               # Project technical documentation and ADRs
│   ├── README.md                       # Documentation catalog
│   ├── 01_PROJECT_OVERVIEW.md          # Project vision and scope
│   ├── 02_PROJECT_GOALS.md             # Concrete goals and milestones
│   ├── 03_SYSTEM_ARCHITECTURE.md       # High-level architecture specification
│   ├── 04_MODEL_DESIGN.md              # Mathematical formulation of Transformer blocks
│   ├── 05_DATA_PIPELINE.md             # Dataset acquisition and pipeline
│   ├── 06_TOKENIZER_DESIGN.md          # Custom Byte-level BPE tokenizer design
│   ├── 07_TRAINING_PIPELINE.md         # Optimization and training loop mechanics
│   ├── 08_FINE_TUNING.md               # SFT and LoRA fine-tuning roadmap
│   ├── 09_EVALUATION_AND_TESTING.md    # Automated testing and validation metrics
│   ├── 10_INFERENCE_AND_GENERATION.md  # Sampling algorithms and generation engine
│   ├── 11_MEMORY_AND_CONTEXT.md        # Long-term memory and context design
│   ├── 12_MODEL_EXPORT_AND_QUANTIZATION.md # GGUF serialization and quantization
│   ├── 13_OLLAMA_INTEGRATION.md        # Local Ollama deployment guide
│   ├── 14_CONFIGURATION_REFERENCE.md   # Configuration schema reference
│   ├── 15_API_AND_CLI_REFERENCE.md     # Command-line interface reference
│   ├── 16_DATASET_CATALOG.md           # Training data provenance and analysis
│   ├── 17_MODEL_CARD.md                # Hibari Koto model card
│   ├── 18_KNOWN_LIMITATIONS.md         # Documented limitations and planned mitigations
│   └── decisions/                      # Architecture Decision Records (ADRs)
│       ├── 0001_USE_PYTORCH.md         # ADR 1: Selection of PyTorch
│       ├── 0002_PRETRAINED_VS_FROM_SCRATCH.md # ADR 2: Building from mathematical scratch
│       └── 0003_MODEL_EXPORT_FORMAT.md # ADR 3: Standardizing on GGUF format
│
├── Studies/                            # Machine learning & transformer study curriculum
│   ├── README.md                       # Curriculum index and learning tracks
│   ├── 01_MATHEMATICS/                 # Linear algebra, calculus, probability
│   ├── 02_NEURAL_NETWORKS/             # Perceptrons, backprop, optimizers
│   ├── 03_DEEP_LEARNING/               # PyTorch mechanics, autograd, dataloaders
│   ├── 04_LANGUAGE_MODELS/             # Language modeling, BPE, perplexity
│   ├── 05_TRANSFORMERS/                # Attention, RoPE, RMSNorm, SwiGLU
│   ├── 06_LARGE_LANGUAGE_MODELS/       # Pretraining, SFT, LoRA, quantization
│   ├── 07_DATA_AND_PERSONALITY/        # Persona design, conversational tone, empathy
│   ├── 08_MODEL_DEPLOYMENT/            # Checkpoints, GGUF, llama.cpp, Ollama
│   ├── 09_EXPERIMENTS/                 # Experiment tracking logs and templates
│   ├── REFERENCES/                     # Annotated papers, books, glossary
│   └── assets/                         # SVG technical architectural diagrams
│
└── hibari/                             # Hibari core package, configs, models, and CLI
    ├── README.md                       # Source package documentation
    ├── configs/                        # YAML configuration files
    │   ├── data/                       # Data configs (base, koto)
    │   ├── model/                      # Model configs (base, koto)
    │   ├── training/                   # Trainer hyperparameters (base, koto, sft)
    │   └── inference/                  # Sampling parameters (base, koto)
    ├── data/                           # Training data partitions
    │   ├── raw/                        # Raw downloaded dataset partition (train.parquet)
    │   ├── interim/                    # Cleaned corpora (placeholder)
    │   ├── processed/                  # Preprocessed tensors (placeholder)
    │   └── evaluation/                 # Evaluation benchmarks (placeholder)
    ├── models/                         # Model weights and artifacts
    │   ├── checkpoints/                # PyTorch checkpoints (best_model.pt, final_model.pt)
    │   ├── tokenizer/                  # BPE files (tokenizer.json, tokenizer_config.json)
    │   ├── exported/                   # GGUF binaries (Hibari-Koto-*.gguf)
    │   ├── pretrained/                 # Base weights (placeholder)
    │   └── adapters/                   # LoRA adapters (placeholder)
    ├── notebooks/                      # Exploratory notebooks (scaffolding)
    ├── outputs/                        # Output logs, metrics, and generated samples
    ├── scripts/                        # Production CLI scripts
    │   ├── prepare_data.py             # Download dataset and train BPE tokenizer
    │   ├── train.py                    # Pretrain Hibari Koto model
    │   ├── generate.py                 # Interactive terminal chat & generation
    │   ├── evaluate.py                 # Validation loss and perplexity evaluation
    │   ├── export.py                   # PyTorch checkpoint to GGUF converter
    │   └── fine_tune.py                # SFT CLI stub (future roadmap)
    ├── src/hibari/                     # Reusable Python library
    │   ├── model/                      # Attention, RoPE, Pre-RMSNorm, SwiGLU, LLM
    │   ├── data/                       # Tokenizer wrapper, dataset, collator, formatter
    │   ├── training/                   # Trainer, loss functions, optimizer, lr scheduler
    │   ├── inference/                  # Autoregressive engine, sampling, chat session
    │   ├── export/                     # GGUF binary writer and tensor mapping
    │   ├── utils/                      # Configuration, seed, device helpers
    │   ├── memory/                     # Context management stubs (future roadmap)
    │   └── personality/                # Persona and safety stubs (future roadmap)
    └── tests/                          # Automated Pytest suite (13 passing tests)
        ├── test_model.py               # Config, parameter count, forward pass, generation
        ├── test_tokenizer.py           # BPE training, encoding, chat formatting
        ├── test_dataset.py             # Data collator and dataset formatting
        ├── test_inference.py           # Greedy, temperature sampling, chat session
        ├── test_export.py              # GGUF binary structure and header tests
        └── test_training.py            # Trainer instantiation tests
```

---

## Architectural Principles

### 1. Pre-RMSNorm
Standard LayerNorm calculates both the mean and variance across activations, adding unnecessary compute operations. Root Mean Square Normalization preserves scale invariance without mean subtraction:

```
RMSNorm(x) = (x / sqrt(mean(x^2) + epsilon)) * gamma
```

### 2. Rotary Position Embeddings (RoPE)
Instead of adding absolute positional embeddings to token representations, RoPE applies a rotation to pairs of dimensions in query and key vectors proportional to token position. Relative distances between tokens are naturally injected into attention dot products without learned position tables.

### 3. Multi-Head Causal Self-Attention
Queries, keys, and values are linearly projected across 4 attention heads ($d_k = 32$). An upper-triangular causal mask prevents tokens from attending to subsequent tokens in the sequence:

```
Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k) + M_causal) V
```

### 4. SwiGLU Feed-Forward Network
SwiGLU replaces traditional two-layer feed-forward networks with gated linear units driven by the Swish (SiLU) activation function:

```
SwiGLU(x) = (x W_gate * SiLU(x W_up)) W_down
```

### 5. Weight Tying
To minimize memory usage and parameter count on edge devices, the output projection matrix (`lm_head.weight`) shares parameters directly with the input embedding table (`token_embeddings.weight`).

---

## Quick Start Guide

### 1. Environment Setup

Create and activate an isolated Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install datasets tokenizers gguf pytest
```

### 2. Prepare Dataset and Train Tokenizer

Download the conversation datasets and prepare the Byte-level BPE tokenizer (vocabulary size: 4,096):

```bash
python hibari/scripts/prepare_data.py --config hibari/configs/data/koto.yaml
```

Artifacts produced:
- `hibari/models/tokenizer/tokenizer.json`
- `hibari/models/tokenizer/tokenizer_config.json`

### 3. Train Hibari Koto

Train the model on CPU using AdamW with cosine learning rate decay and linear warmup:

```bash
python hibari/scripts/train.py \
  --model-config hibari/configs/model/koto.yaml \
  --train-config hibari/configs/training/koto.yaml
```

Saved checkpoints:
- `hibari/models/checkpoints/best_model.pt`
- `hibari/models/checkpoints/final_model.pt`

### 4. Interactive Chat & Text Generation

Run multi-turn interactive conversation in the terminal:

```bash
python hibari/scripts/generate.py \
  --checkpoint hibari/models/checkpoints/best_model.pt \
  --tokenizer hibari/models/tokenizer
```

Or generate a single response:

```bash
python hibari/scripts/generate.py \
  --checkpoint hibari/models/checkpoints/best_model.pt \
  --prompt "Hello Hibari, how are you today?"
```

### 5. Evaluate Model Perplexity

Evaluate validation loss and perplexity across the held-out test split:

```bash
python hibari/scripts/evaluate.py \
  --checkpoint hibari/models/checkpoints/best_model.pt \
  --tokenizer hibari/models/tokenizer
```

Current evaluation results:
- **Validation Loss**: `3.5354`
- **Validation Perplexity**: `34.31`

### 6. Export to GGUF (for llama.cpp and Ollama)

Convert PyTorch weights into binary GGUF format:

```bash
# Export F32, F16, Q8_0 and Q4_K_M (Q8_0/Q4_K_M need llama.cpp's llama-quantize)
python hibari/scripts/export.py \
  --checkpoint hibari/models/checkpoints/koto_v2/best_model.pt \
  --outdir release \
  --variants f32 f16 q8_0 q4_k_m \
  --llama-quantize /path/to/llama.cpp/build/bin/llama-quantize
```

Deploying with Ollama (the Modelfile lives in `release/`, next to the GGUF files):

```bash
cd release
ollama create hibari-koto -f Modelfile
ollama run hibari-koto
```

See [Docs/13_OLLAMA_INTEGRATION.md](Docs/13_OLLAMA_INTEGRATION.md) for the template and parameters.

### 7. Run Test Suite

Run the automated test suite verifying all modules:

```bash
pytest
```

Output:
```
13 passed in 2.60s
```

---

## Detailed Technical Documentation

For an in-depth breakdown of every stage of implementation, refer to:
- [Docs/03_SYSTEM_ARCHITECTURE.md](Docs/03_SYSTEM_ARCHITECTURE.md): System modularity and data flow.
- [Docs/04_MODEL_DESIGN.md](Docs/04_MODEL_DESIGN.md): Detailed mathematical formulation.
- [Docs/12_MODEL_EXPORT_AND_QUANTIZATION.md](Docs/12_MODEL_EXPORT_AND_QUANTIZATION.md): GGUF serialization internals.
- [Docs/17_MODEL_CARD.md](Docs/17_MODEL_CARD.md): Hibari Koto model card and intended use.
- [Docs/18_KNOWN_LIMITATIONS.md](Docs/18_KNOWN_LIMITATIONS.md): Limitations and planned mitigations.

---

## Project Roadmap

- [x] Phase 1: Machine learning and transformer foundational studies (`Studies/`)
- [x] Phase 2: From-scratch PyTorch architecture (RoPE, Pre-RMSNorm, SwiGLU, Causal MHA)
- [x] Phase 3: Byte-level BPE tokenizer training pipeline
- [x] Phase 4: Training pipeline on Blended Skill Talk dataset
- [x] Phase 5: Hibari Koto model training and validation
- [x] Phase 6: Native GGUF export (FP32, FP16) and verification
- [x] Phase 7: Automated unit and integration testing suite (13/13 passing)
- [ ] Phase 8: LoRA and Supervised Fine-Tuning (SFT) implementation
- [ ] Phase 9: External memory store and retrieval integration
- [ ] Phase 10: Multi-modal sensory extensions

---

## Author & Contact

- Author: **Kavimugil Rajasekar**
- GitHub: [@KavimugilRajasekar](https://github.com/KavimugilRajasekar)
- Portfolio: [https://kavimugilrajasekar.github.io](https://kavimugilrajasekar.github.io)
- Repository: [https://github.com/KavimugilRajasekar/Hibari](https://github.com/KavimugilRajasekar/Hibari)
