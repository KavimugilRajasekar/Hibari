# 24 — Development Templates

Copy the relevant section into the experiment record (`objective`,
`source_changes`, `evaluation`) when starting a task. Every template ends
with acceptance criteria; the general checklist (doc 25) applies too.

## T1. New model architecture (engine)
- **Objective:** a new architectural feature (e.g. grouped-query attention) usable by any model.
- **Files:** `src/hibari/model/*`, `model/config.py`, `export/gguf_export.py` (+ tensor map), `export/gguf_reference.py`, `hibari/configs/model/base.yaml`, `Docs/architecture/04_MODEL_DESIGN.md`, an ADR.
- **Steps:** add config fields with backward-compatible defaults → implement → KV-cache path → GGUF metadata/tensors → reference runtime.
- **Tests:** shape tests; cached vs uncached decoding equality; NumPy reference vs PyTorch logits; llama.cpp parity on a tiny exported model.
- **Accept:** old checkpoints load and produce identical logits; new option exports and runs in llama.cpp.

## T2. New conversational model (model repository)
- **Objective:** a new model on Hibari (not a Koto branch).
- **Files:** the directory standard of doc 21 §2; release profile; behaviour spec; model card.
- **Steps:** copy templates → tokenizer → data builders → training (gate) → evaluation suites → profile → export.
- **Tests:** data well-formedness, leak test, profile validation, experiment validation.
- **Accept:** `validate_release.py` 0 errors; evaluation report with transcripts; checklist done.

## T3. Dataset integration
- **Objective:** add a source or a hand-written family.
- **Files:** `<model>/data/<family>.py`, the data config (new key, default off so old configs rebuild unchanged), dataset catalog (licence), `data_stats.json`.
- **Steps:** load offline → normalise → identity rewrite if needed → leak filter → hash split → length filter.
- **Tests:** well-formed turns; no placeholder left; no human claims; eval prompts absent; counts in stats.
- **Accept:** rebuilding the previous version's config gives the same output hash; the new family appears in the stats with its licence.

## T4. Tokenizer development
- **Objective:** new vocabulary or special tokens (MAJOR version).
- **Files:** `src/hibari/data/tokenizer.py`, `Docs/architecture/06_TOKENIZER_DESIGN.md`, the model's `tokenizer/`.
- **Tests:** round trip; special-token IDs stable; escape of special-token strings; `llama-tokenize` parity on the edge-case list.
- **Accept:** new tokenizer hash recorded; old profiles still validate against the old tokenizer.

## T5. Inference engine development
- **Objective:** decoding, session or routing behaviour.
- **Files:** `src/hibari/inference/*`, `InferenceSettings` if a new setting, `Docs/engine/10_*` or `26_*`.
- **Steps:** off by default (`num_candidates=1` style) → profile switch → dev set → held-out suite.
- **Tests:** default path byte-identical to before; new path deterministic with a seed; regression test per fixed failure.
- **Accept:** suite improves on held-out prompts for ≥ 2 seeds; latency reported.

## T6. Behavioural improvement
- **Objective:** a behaviour from the behaviour spec (e.g. empathy).
- **Decide first:** can decoding fix it (does the model ever produce a good candidate)? If yes → T5. If no → data (T3) + training gate.
- **Tests / Accept:** the behaviour suite category improves on 3 seeds without lowering other categories; transcripts reviewed.

## T7. Sentence-rewriting capability
- **Objective:** a rewriting task or a better one.
- **Files:** `hibari/inference/tasks.py` (task spec + request patterns), `<model>/data/tasks.py`, rewrite data, upgrade / behaviour suite cases.
- **Tests:** routing positives *and* negatives ("rewrite the story about dragons" stays chat); protected-term preservation; exact-match grammar cases.
- **Accept:** no chat message newly routed as a task in the suite; grammar exact-match and preservation do not regress.

## T8. Evaluation-suite development
- **Objective:** measure a behaviour.
- **Steps:** write prompts **before** looking at model output; independent checks (not imported from the critic); expected behaviour in words; add to the data builder's leak list.
- **Tests:** suite structure test; scoring examples (a good and a bad reply per check type); leak test.
- **Accept:** a known-bad reply fails and a known-good reply passes for every check type.

## T9. Model export
- **Files:** release profile, `releases/exported/X.Y.Z/`.
- **Steps:** `validate_release.py --no-gguf` → `export.py --profile --record` → `validate_release.py` → `compare_backends.py`.
- **Accept:** 0 validation errors; F32/F16/Q8_0 greedy parity 100 %; Q4_K_M parity reported.

## T10. Release preparation
- **Steps:** checklist §B; changelog; model card; limitations; readiness assessment in the report; owner tags and publishes.
- **Accept:** every checklist item answered with evidence or an explicit "not met" and its reason.
