# Role: Engine developer (Hibari)

**Scope:** `src/hibari/`, `scripts/engine/`, `hibari/configs/`, `tests/`, `Docs/`.

**Must**
- Keep the engine model-agnostic; model specifics come from a `ModelProfile`.
- Keep the default path of every changed function byte-identical unless the change is the fix.
- Architecture changes: cached vs uncached decoding equal; `gguf_reference.py` and llama.cpp
  parity on a tiny export; ADR in `Docs/decisions/`; MAJOR version if old checkpoints break.
- Inference changes: deterministic with a seed; `last_reply_info` explains decisions;
  measure latency.
- Add settings to `InferenceSettings` and `hibari/configs/inference/base.yaml` together
  (`tests/test_release.py::test_p9` checks the templates).
- Bump `hibari.__version__` and `pyproject.toml` together; log in `Docs/20_CHANGELOG.md`.

**Done when:** engine and Koto tests pass, the release still validates, docs updated.
