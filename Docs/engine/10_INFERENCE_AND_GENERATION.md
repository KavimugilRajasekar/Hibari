# 10 — Inference and Generation

## Command line

```bash
# interactive multi-turn chat (commands: /reset, /exit)
python hibari/scripts/generate.py

# one prompt, deterministic
python hibari/scripts/generate.py --prompt "Hi! How are you?" --temperature 0

# options
--checkpoint PATH  --system TEXT  --temperature 0.5  --top-k 25  --top-p 0.85
--repetition-penalty 1.15  --max-new-tokens 80  --seed N  --no-safety
```

## Engine (`hibari/src/hibari/inference/`)

| Piece | Behaviour |
|---|---|
| `generation.generate_tokens` | Prefill the prompt once, then decode one token at a time with the **KV cache** (verified equal to full recomputation to 1e-5). |
| Positions | RoPE positions continue from the cache length; the RoPE table is extended if needed. |
| Context | 256 tokens. Room for the reply (≤ half the window) is reserved; an over-long prompt is left-truncated; generation stops when the window is full. |
| Stop tokens | `<|eos|>`, `<|user|>`, `<|system|>`, `<|assistant|>`, `<|pad|>`, `<|bos|>`. |
| Sampling (`sampling.sample_next_token`) | repetition penalty on tokens of the current reply → temperature → top-k → top-p. `temperature 0` = greedy (deterministic, verified over repeated runs). `seed` makes sampling reproducible. |
| `conversation.HibariChatSession` | Keeps history; before each reply drops the oldest turns (never the system prompt) until prompt + reply fit; history always restarts on a user turn. |
| `safety.clean_response` | Removes chat markers, caps emojis at 1, fixes spacing before punctuation, drops a dangling unfinished final sentence. Never adds text. |

## Removed behaviour

The v1 session returned hardcoded strings for identity questions
("I'm Hibari Koto! 🌸 …"), appended emoji to every reply and appended
canned empathy suffixes whenever a distress keyword appeared. These masked
what the model actually generated and were removed. Identity, tone and
emotional replies now come from the model (learned from the curated data).

## Safety layer (application only)

See `09_EVALUATION_AND_TESTING.md`. The session intercepts self-harm,
violent intent and harmful-instruction requests with fixed reviewed replies
before the model is called. This layer is **not** inside the GGUF.
