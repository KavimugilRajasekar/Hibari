# 23 — Engineering Standards

## 1. Unit and integration testing

- Every bug fix ships with a regression test that **fails on the old code**
  (e.g. `tests/test_regressions.py`, `tests/test_conversational_layer.py`).
- Tests use tiny models (d_model 32, one layer) and tiny tokenizers so the
  suites run in seconds; no test depends on downloaded data or real weights.
  Tests that need real artefacts are `skipif` when the file is absent.
- Integration tests run the real pipelines end to end on tiny inputs:
  profile → export → validate (`tests/test_release.py`), data build and the
  training script (Koto `tests/test_upgrade.py`).
- A passing test suite proves the code does what it claims. **It does not
  prove conversational quality** — that is §2.

## 2. Conversational / behavioural evaluation

- Suites live in the model repository (`<model>/evaluation/*_suite.py`); each
  case has turns, the scored turn, a human-readable `expected`, and checks.
- **Held out**: evaluation prompts are written separately from training
  templates *and* from the prompts used to tune decoding or the reply critic.
  The data builder removes every evaluation prompt of three or more words
  from training data, and a test checks the behaviour data directly.
- **Several kinds of evidence**, never regex alone:
  exact match (grammar corrections), protected-term preservation (numbers,
  names), intent / affect checks (empathy present, no cheerful mismatch,
  de-escalation), context checks (a fact from an earlier turn is recalled),
  repetition and echo detection, completeness (stopped on EOS), lexical
  similarity to references (chrF — reported, never decisive), and the
  **transcript, read by a person**.
- Run the raw model and the application pipeline separately, and several
  sampling seeds; report pass-rate agreement across seeds.
- A report states its limits: automatic checks are heuristics.

## 3. Inference compatibility

`scripts/engine/compare_backends.py` per GGUF variant: tokenizer parity
(`llama-tokenize` vs Python, edge cases + every rendered prompt), chat-template
parity (Ollama prompt-token count vs Python), greedy parity (PyTorch vs
llama.cpp vs Ollama chat vs Ollama raw). F32 / F16 / Q8_0 must match PyTorch
greedy exactly; Q4_K_M differences are reported, not hidden.

## 4. Data quality and leakage

See Koto `docs/29_DATA_STANDARDS.md` (model-specific) — the general rules:
replies are written for the model's identity and never claim human
experiences; one source text never appears in two splits (stable hash
split); every example fits the context window without cutting a reply;
contradictory targets for the same source are dropped; output directories
are never overwritten silently; `data_stats.json` records hashes, licences,
leak drops and per-source counts.

## 5. Versioning

Semantic versioning per model: **MAJOR** architecture / tokenizer / chat
format, **MINOR** new training data or capabilities, **PATCH** inference,
export, evaluation or compatibility repairs. A PATCH may ship the previous
weights (`weights_version` < `version`).

Every version has: a release profile (config references, checkpoint and
tokenizer SHA-256), an experiment record, an evaluation report, a changelog
entry, a compatibility record (the backend parity report) and a GGUF
directory `releases/exported/X.Y.Z/` whose files carry `hibari.*`
provenance. A GGUF without provenance, or with provenance that does not
match the profile, fails validation — this is what makes it impossible to
ship the wrong weights under a version label again.

The engine has its own version (`hibari.__version__`, `pyproject.toml`),
recorded in each GGUF as `hibari.engine.version`.

## 6. Configuration

- Each config type has a template in `hibari/configs/` whose keys are tested
  against the dataclass it feeds (`ModelConfig`, `TrainingArgs`,
  `InferenceSettings`, `ModelProfile`). Unknown inference keys are an error.
- Decoding defaults exist in exactly one file per model
  (`configs/inference/<model>.yaml`). The Ollama Modelfile is generated from
  it; hand edits fail validation.
- Relative paths in model configs resolve against the model repository root.

## 7. Code

- Engine / model separation (doc 21 §1). No hard-coded model replies:
  rule-based components may route, normalise, check safety and *rank*
  model-generated candidates; text shown to the user comes from the model,
  except the reviewed fixed messages of the safety layer and input-validation
  messages (empty input, text too long).
- Type hints on public functions; docstrings state behaviour and limits.
- No new heavy dependencies (the engine needs `torch`, `tokenizers`,
  `gguf`, `numpy`, `pyyaml`).
- Dead code is removed only after confirming nothing imports it.
