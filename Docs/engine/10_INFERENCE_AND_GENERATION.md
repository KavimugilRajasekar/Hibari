# 10 — Inference and Generation

## Command line

```bash
# interactive multi-turn chat (commands: /reset, /exit)
python scripts/engine/generate.py --checkpoint models/Koto/checkpoints/koto_start/best_model.pt \
    --tokenizer-dir models/Koto/tokenizer

# one prompt, deterministic
python scripts/engine/generate.py --checkpoint ... --tokenizer-dir ... --prompt "Hi! How are you?" --temperature 0

# options
--checkpoint PATH  --system TEXT  --temperature 0.5  --top-k 25  --top-p 0.85
--repetition-penalty 1.15  --max-new-tokens 80  --seed N  --no-safety
```

## Engine (`src/hibari/inference/`)

| Piece | Behaviour |
|---|---|
| `generation.generate_tokens` | Prefill the prompt once, then decode one token at a time with the **KV cache** (verified equal to full recomputation to 1e-5). |
| Positions | RoPE positions continue from the cache length; the RoPE table is extended if needed. |
| Context | 256 tokens. Room for the reply (≤ half the window) is reserved; an over-long prompt is left-truncated; generation stops when the window is full. |
| Stop tokens | `<|eos|>`, `<|user|>`, `<|system|>`, `<|assistant|>`, `<|pad|>`, `<|bos|>`. |
| Sampling (`sampling.sample_next_token`) | banned tokens (n-gram block) → repetition penalty on tokens of the current reply → temperature → top-k → top-p. `temperature 0` = greedy (deterministic, verified over repeated runs). `seed` makes sampling reproducible. |
| N-gram blocking (`no_repeat_ngram_size`) | Optional (default 0 = off). With n > 1 a token that would repeat an n-gram already in the reply is banned. Used by rewriting tasks (n = 4) to stop copy loops. |
| `conversation.HibariChatSession` | Keeps history; before each reply drops the oldest turns (never the system prompt) until prompt + reply fit; history always restarts on a user turn. |
| `safety.clean_response` | Removes chat markers, caps emojis at 1, fixes spacing before punctuation, drops a dangling unfinished final sentence. Never adds text. |

## Rewriting tasks (`inference/tasks.py`, engine 0.2.0)

One task representation is shared by training data and inference. A rewrite
is an ordinary user turn in a fixed form, and the model answers with the
rewritten text only:

```
Task: Rephrase
Input: i am learning python from yesterday.
```

| Task (`TASKS`) | `Task:` label | Purpose |
|---|---|---|
| `rephrase` | Rephrase | clearer, natural wording; fixes mistakes; same meaning |
| `correct_grammar` | Correct grammar | grammar / spelling / punctuation only |
| `formal` | Make formal | professional tone |
| `casual` | Make casual | relaxed tone |
| `shorten` | Make shorter | remove unnecessary words, keep every fact |
| `simplify` | Simplify | simpler words and structure |
| `polite` | Make polite | remove rudeness |

* `format_task_request(task, text)` builds the request; model training data
  must use it so that training and inference prompts are identical.
* `parse_task_request(message)` recognises the canonical block and explicit
  instructions typed in chat — `Rephrase: "..."`, `Rephrase this sentence: ...`,
  `Make this professional: ...`, `Make this casual: ...`, `Make this shorter: ...`,
  `Correct grammar: ...`, `fix my grammar pls: ...`, `"..." make it polite`.
  An instruction must be followed by a separator (`:`, newline, quote) or
  nothing, so "rewrite the story about dragons" stays normal chat.
* `HibariChatSession.reply()` routes recognised requests to `run_task()`
  (disable with `route_tasks=False`). `run_task(task, text)` and
  `InferenceEngine.rewrite(task, text)` are the direct APIs and return a
  `TaskResult` (`output`, `missing_terms`, `candidates_tried`, `chunks`).
* A rewrite prompt contains only the system prompt and the request — a
  rewrite does not depend on earlier chat — but the request and result are
  appended to the history, so later turns can refer to them.
* Decoding (`TASK_DECODING`): greedy, no repetition penalty (rewrites must be
  free to repeat input words), 4-gram block; reply budget scales with the
  input (`task_max_new_tokens`: 1.6x input + 16, 1.0x for shorten / simplify,
  at most 160 tokens and never beyond the context window).
* Meaning preservation: `protected_terms` (numbers incl. number words,
  times, weekdays / months, mid-sentence names, technical tokens such as
  `iOS`, `HTTP`, `3.11`). If the greedy output drops one, up to three seeded
  samples (T = 0.3) are tried and the candidate that keeps the most terms is
  returned; anything still missing is reported in `missing_terms`.
* Long input: if the request plus room for the reply does not fit the
  256-token window, the text is rewritten in consecutive groups of sentences
  (each with the full `Task:` header) and the results are joined. A single
  sentence that is too long gets a fixed message; the prompt is never
  truncated through the header.
* Empty input (`Rephrase: ""`) gets a fixed request for the text; the safety
  layer also checks rewrite input.

These are inference mechanics only. Whether a model actually rewrites well
depends on its training data (Koto 1.1.0, see `models/Koto/docs`).

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
