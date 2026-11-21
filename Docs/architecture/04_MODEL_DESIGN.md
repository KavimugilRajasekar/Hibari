# 04 — Model Design

## Architecture

Hibari uses a state-of-the-art **decoder-only Transformer** (LLaMA/Mistral-style) architecture.

```
Input Tokens
     │
     ▼
Token Embedding ──▶ [Embedding Table (tied with LM Head)]
     │
     ▼
Transformer Block 1..N:
     ├── RMSNorm ──▶ Multi-Head Causal Attention (with RoPE) ──▶ Add (Residual)
     └── RMSNorm ──▶ SwiGLU Feed-Forward Network ──────────────▶ Add (Residual)
     │
     ▼
Final RMSNorm
     │
     ▼
LM Head (Linear Projection / Tied Weights) ──▶ Vocabulary Logits
```

## Architectural Components

### 1. Embeddings & Positional Encoding
- **Token Embeddings**: Maps discrete token IDs to dense representations ($d_{model} = 256$).
- **Rotary Position Embeddings (RoPE)**: Encodes relative position directly within self-attention queries and keys ($\theta = 10,000$), rotating **adjacent dimension pairs** $(x_0,x_1),(x_2,x_3),\dots$ (llama.cpp's `ROPE_TYPE_NORM`). The model is trained on 256 positions only; quality beyond 256 tokens is not supported.

### 2. Transformer Blocks
Each block implements:
- **Pre-Normalization**: RMSNorm applied prior to attention and FFN sub-layers for numerical stability.
- **Multi-Head Self-Attention**: Causal masking to maintain autoregressive order.
- **SwiGLU Activation**: Gated feed-forward network ($FFN_{SwiGLU}(x) = (\text{SiLU}(xW_{gate}) \otimes xW_{up})W_{down}$, no biases) providing richer representation capacity than classic ReLU or GELU.

### 3. Output Head & Weight Tying
- Linear output layer mapping from $d_{model}$ to vocabulary size.
- **Weight Tying**: Output projection matrix shares weights with the token embedding table, saving memory while accelerating convergence.

---

## Architectural Configuration

| Parameter | Hibari Koto |
|---|---|
| **Total Parameters** | **6,163,712 (~6.16M)** |
| **Vocabulary Size** | **4,096 tokens (Byte-level BPE)** |
| **Layers (`n_layers`)** | **6 Transformer Blocks** |
| **Embedding Dim (`d_model`)** | **256** |
| **Attention Heads (`n_heads`)** | **8 (head dim: 32)** |
| **FFN Hidden Dim (`d_ff`)** | **768 (SwiGLU)** |
| **Max Context Length** | **256 tokens** |
| **Weight Tying** | Yes |
| **Positional Encoding** | RoPE ($\theta = 10,000$) |
| **Layer Normalization** | RMSNorm ($\epsilon = 10^{-5}$) |

## Design Rationale

1. **Vocabulary Size (4,096)**:
   A small vocabulary would split common English conversational words (such as "Hello", "feeling", "friend", "understand") into 2-5 byte chunks, degrading fluency. A 4,096-token vocabulary ensures  common English vocabulary, punctuation, and colloquial expressions are single tokens.
2. **Capacity (6.16M)**:
   A 6-layer, 256-dimension model possesses the representational capacity necessary to model nuanced dialogue flow, emotional tone, and empathetic response phrasing while remaining fast and lightweight on CPU-only hardware.
3. **Loss Masking Support**:
   Supervised Fine-Tuning (SFT) target masking ignores system prompt and user input tokens during loss computation (`ignore_index = -100`), ensuring all parameter updates directly optimize assistant response quality.

---

## Verified parameter breakdown (Koto)

| Component | Parameters |
|---|---|
| Token embedding 4096 × 256 (shared with LM head) | 1,048,576 |
| Per layer: attention 4 × 256² = 262,144; SwiGLU 3 × 256 × 768 = 589,824; 2 RMSNorm = 512 | 852,480 |
| 6 layers | 5,114,880 |
| Final RMSNorm | 256 |
| **Total** | **6,163,712** |

`ModelConfig.estimate_parameters()`, the instantiated model and every v1/v2
checkpoint agree on 6,163,712 (asserted in `tests/test_model.py`).
