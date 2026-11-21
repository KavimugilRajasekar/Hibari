# hibari — Source Package

This directory contains the full source code for the Hibari LLM-Brain and the **Hibari Koto ('Hibari-Koto')** conversational model.

## Model Summary: Hibari Koto (Hibari-Koto)

- **Architecture**: Decoder-only Transformer (RoPE, RMSNorm, SwiGLU, tied embeddings)
- **Parameter Count**: **6,163,712 parameters (~6.16M parameters)**
- **Vocabulary**: 4,096-token Byte-level BPE Tokenizer (eliminates token fragmentation)
- **Sequence Length**: 256 tokens
- **Layers**: 6 Transformer Blocks
- **Attention**: 8 Multi-Head Attention heads with Rotary Position Embeddings (RoPE)
- **FFN**: SwiGLU activation ($d_{ff} = 768$)
- **Normalization**: Pre-RMSNorm ($\epsilon = 10^{-5}$)
- **Datasets**: Blended 4 datasets (`blended-skill-talk`, `dailydialog`, `prosocial-dialog`, `oasst1`) totaling 40,214 conversations
- **Training Strategy**: Supervised Fine-Tuning (SFT) target loss masking (`targets = -100` for system & user tokens)
- **Export Format**: **GGUF** (`Hibari-Koto.gguf` [27.65 MB] & `Hibari-Koto-f16.gguf` [13.90 MB]) compatible with llama.cpp and Ollama

## Layout

```
hibari/
├── src/hibari/       # Python source package
│   ├── model/        # Architecture: attention, embedding (RoPE), norm (RMSNorm), SwiGLU, llm
│   ├── data/         # Tokenizer (BPE), dataset, collator, dialogue formatting (4 datasets)
│   ├── training/     # Trainer, AdamW optimizer, cosine warmup scheduler, checkpoints
│   ├── inference/    # Sampling, generation engine, multi-turn chat session
│   └── export/       # GGUF export and quantization utilities
├── configs/          # YAML configuration files (model, training, data, inference)
├── data/             # Raw and processed datasets (cached train/val conversations)
├── models/           # Checkpoints, tokenizer, and GGUF exports
│   ├── checkpoints/  # best_model.pt, final_model.pt (step 1500)
│   ├── tokenizer/    # tokenizer.json (4,096 vocab), tokenizer_config.json
│   └── exported/     # Hibari-Koto.gguf (f32 & f16)
├── scripts/          # CLI entry points: prepare_data, train, generate, evaluate, export
├── tests/            # Pytest test suite (79 passing tests)
└── outputs/          # Logs, metrics, and generated samples
```

## Quick Start

### 1. Prepare Data & Train Tokenizer
```bash
python hibari/scripts/prepare_data.py --config hibari/configs/data/koto.yaml
```

### 2. Train Hibari Koto
```bash
python hibari/scripts/train.py --model-config hibari/configs/model/koto.yaml --train-config hibari/configs/training/koto.yaml
```

### 3. Interactive Friendly Chat with Hibari Koto
```bash
python hibari/scripts/generate.py --checkpoint hibari/models/checkpoints/best_model.pt
```

### 4. Evaluate Validation Perplexity
```bash
python hibari/scripts/evaluate.py --checkpoint hibari/models/checkpoints/best_model.pt
```

### 5. Export to GGUF for Ollama
```bash
python hibari/scripts/export.py --quantization f16 --output hibari/models/exported/Hibari-Koto-f16.gguf
```

### 5. Export to GGUF
```bash
python hibari/scripts/export.py --checkpoint hibari/models/checkpoints/best_model.pt --output hibari/models/exported/Hibari-Koto.gguf
```
