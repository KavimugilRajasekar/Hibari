# 06 — Tokenizer Design

## Approach

Hibari uses **Byte-Pair Encoding (BPE)** tokenization with byte-level fallback, guaranteeing zero out-of-vocabulary (`<|unk|>`) tokens while preserving word-level representations for common conversational vocabulary.

## Special Tokens

| Token | ID | Role |
|---|---|---|
| `<\|pad\|>` | 0 | Padding token for batch collation |
| `<\|unk\|>` | 1 | Fallback unknown token |
| `<\|bos\|>` | 2 | Beginning-of-sequence token |
| `<\|eos\|>` | 3 | End-of-sequence token (signals turn completion) |
| `<\|system\|>` | 4 | Role marker initiating system instructions and personas |
| `<\|user\|>` | 5 | Role marker initiating user input |
| `<\|assistant\|>` | 6 | Role marker initiating Hibari's response |

---

## Tokenizer Characteristics

| Metric | Hibari Koto |
|---|---|
| **Vocabulary Size** | **4,096 tokens** |
| **Training Corpus** | 4 corpora (`blended-skill-talk`, `dailydialog`, `prosocial`, `oasst1`) |
| **Sample Count** | **86,270 diverse utterances** |
| **Word Segmentation** | Single-token representation for full words & common phrases |
| **Punctuation & Contractions** | Cleanly tokenized ("I'm", "don't", "you're", "can't") |

### Tokenization Sanity Test

```python
# Input string:
"Hello Hibari Koto! How are you doing today? I feel a little sad."

# Tokenizer output (4,096 vocab):
['Hello', ' Hibari', ' Koto', '!', ' How', ' are', ' you', ' doing', ' today', '?', ' I', ' feel', ' a', ' little', ' sad', '.']
# -> 16 clean, coherent tokens
```

---

## Chat Template

The tokenizer provides `apply_chat_template(messages, add_generation_prompt=False)`:

```
<|system|>
You are Hibari Koto, a friendly and calm AI companion. Keep replies short, kind and honest.<|eos|>
<|user|>
How was your day?<|eos|>
<|assistant|>
It's been good, thanks for asking. How about yours?<|eos|>
```

When generating, `add_generation_prompt=True` appends `<|assistant|>\n`, prompting the model to generate the next response.

---

## Verified implementation details (v2 audit)

| Item | Value |
|---|---|
| Library | Hugging Face `tokenizers` BPE (`hibari/models/tokenizer/tokenizer.json`) |
| Normaliser | NFKC |
| Pre-tokenizer | ByteLevel, `add_prefix_space=false`, GPT-2 split regex |
| Decoder | ByteLevel |
| Vocabulary / merges | 4,096 tokens (IDs contiguous 0–4095) / 3,833 merges |
| Special tokens | `<\|pad\|>`=0, `<\|unk\|>`=1, `<\|bos\|>`=2, `<\|eos\|>`=3, `<\|system\|>`=4, `<\|user\|>`=5, `<\|assistant\|>`=6 |
| BOS | never used in training; GGUF sets `add_bos_token=false` |

Round-trip `text → ids → text` is exact for ASCII, punctuation, numbers,
contractions, newlines and indentation, emoji, accented Latin, CJK, special
tokens and random strings. NFKC intentionally folds compatibility characters
(`ｆｕｌｌ` → `full`, `ﬁ` → `fi`).

GGUF parity: `llama-tokenize` on the exported GGUF produces identical IDs
for all 12 cases in `scripts/verify_gguf.py` (the v1 export, which declared
`tokenizer.ggml.pre = "default"`, did not — e.g. 18 vs 14 tokens for
`"I'm fine, don't worry. It's 3.14!"`). llama.cpp does not apply NFKC, so
only compatibility characters can tokenize differently there.

The tokenizer was **not** retrained for v2–v6, so token IDs are unchanged
and every Koto checkpoint shares `hibari/models/tokenizer/`.
