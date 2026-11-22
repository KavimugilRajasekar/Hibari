# 26 — Conversational Layer (engine 0.3.0)

Three modules around the model in `HibariChatSession`. All are off by
default (`num_candidates=1`, `normalize_input=False`); a release profile
turns them on. None writes replies: every reply comes from the model, except
the reviewed fixed messages of the safety layer and input-validation messages.

```
user message ─┬─► safety layer (original text) ──flagged──► fixed reviewed response
              ├─► task router (original text) ──rewrite──► canonical "Task/Input" prompt, greedy
              └─► analyze_message ─► normalize_informal ─► model generates N candidates
                                                         └─► reranker picks one ─► clean_response
```

## `hibari.inference.normalization`
Expands unambiguous chat shorthand for the **model input only** (`wat r u` →
`what are you`, `thx` → `thanks`, `ur` → `your` / `you're` by the next word,
`!!!` → `!`, `soooo` → `so`, lone `i` → `I`). Words that are also English
(`ill`, `its`, `wed`) are never changed. Rewriting requests never pass
through it: correcting the user's text is the model's job.

## `hibari.inference.intent`
`analyze_message(text) -> MessageAnalysis`: intent (greeting, farewell,
thanks, apology, compliment, hostile, identity, rewrite request, e-mail
request, emotional disclosure, question, request, statement, empty, noise),
emotion (grief, sadness, loneliness, frustration, anger, anxiety,
disappointment, confusion, excitement, happiness), subject (`self` /
`other`: "my friend is struggling"), `needs_live_data` (weather, scores,
"right now"), with negation handling ("not sad") and an `evidence` list of
the rules that fired. English heuristics, not a trained classifier.

## `hibari.inference.reranking`
For each reply the model generates `num_candidates` (per-intent override
`intent_candidates`) replies: the first with the caller's decoding, the
others with `candidate_sampling` and fixed seeds (so a session is
reproducible). Each candidate is scored:

`score = fluency + 0.4 · clip(relevance, ±1.5) + Σ critic adjustments`

- **fluency** — mean log p(reply + EOS | conversation);
- **relevance** — PMI: (log p(reply | conversation) − log p(reply | empty user turn)) / tokens.
  Generic replies ("I'm glad I could help.") score ≈ 0; clipping stops
  parroting the user's words from winning;
- **critic** (`ReplyCritic.judge`, named rules, weights in `CriticWeights`):
  affect mismatch (cheerful reply to grief, condolences for good news), empathy
  present / absent for negative disclosures, thanking / agreeing with /
  insulting back after an insult, de-escalation, greeting / thanks / farewell
  returned, identity stated, live-data honesty, e-mail structure (subject,
  greeting, body, closing), human-experience claims, topics or capitalised
  entities that appear from nowhere, yes/no questions about what the user
  just said, echo, repetition, truncation, length per intent.

`session.last_reply_info` holds the analysis, the model input and every
candidate with its score breakdown (`generate.py --explain` prints it).

**Limit:** the reranker can only choose among what the model produces. On
the 1.1.0 weights some prompts have no acceptable candidate in 16 samples
(insults such as "You're so pathetic!", live weather); those need training
data (Koto 1.2.0). Cost: about 1.4 s per reply on CPU with 8–16 candidates
for the 6.2M model, against about 0.1 s for one.

## Not available in GGUF runtimes
llama.cpp and Ollama run the raw model: no safety layer, router,
normalisation or reranking. The generated Modelfile says so in its header.
Applications that need these behaviours should call the Python engine.
