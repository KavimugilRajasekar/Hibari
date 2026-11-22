# 25 — Official Development Checklist

Copy into the PR description or the experiment record and answer every line
with evidence (command output, report path) or "not met: <reason>".
**A model is not ready because its Python code runs.**

## A. Before a change is complete

**Architecture**
- [ ] Cached and uncached decoding agree; NumPy reference matches PyTorch (if the model code changed).
- [ ] Old checkpoints still load (or the change is a MAJOR version).

**Configuration**
- [ ] `pytest tests/test_release.py` (templates ↔ dataclasses) passes.
- [ ] Decoding defaults changed only in `configs/inference/<model>.yaml`.
- [ ] `validate_release.py --profile P --no-gguf` has 0 errors.

**Tokenizer**
- [ ] Tokenizer SHA-256 equals the profile's; special-token IDs unchanged.

**Data**
- [ ] Data built with a new `out_dir`; `data_stats.json` reviewed (counts, leak drops, too-long drops, licences).
- [ ] Evaluation prompts absent from training data (leak test passes; `eval_prompts_in_train_user_turns` empty).
- [ ] Assistant replies match the identity and claim no human experiences.

**Inference**
- [ ] The default path is unchanged unless the change is meant to alter it.
- [ ] Spot check with `generate.py --profile P --explain` on the motivating prompts.

**Conversational quality**
- [ ] Behaviour suite run on ≥ 3 seeds; pass rate and per-category rates compared with the previous version.
- [ ] Transcripts read; failures listed with examples in the report.

**Context**
- [ ] Multi-turn cases (recall, follow-up, topic change) checked in the transcripts.

**Safety**
- [ ] Safety category tests pass; harmless sentences with sensitive words are not flagged.

**Regression**
- [ ] `pytest` in both repositories; one new regression test per fixed bug.

**Export**
- [ ] `export.py --profile --record`; `validate_release.py` 0 errors; `compare_backends.py` report saved.

**Documentation**
- [ ] Changelogs, model card, limitations, behaviour spec updated.

**Version integrity**
- [ ] Profile `version` / `weights_version` correct; checkpoint metadata `model_version` = `weights_version`.
- [ ] Experiment record validates (`validate_experiments.py`).

## B. Release readiness

- [ ] Every item of A answered.
- [ ] Remaining limitations documented honestly (no claim of human-level understanding).
- [ ] The weights the GGUF holds are proven by provenance, not by file name.
- [ ] The owner has decided to publish. (Assistants and scripts never publish or push.)
