# 03 — Coding Standards

Full rules: `Docs/template/23_ENGINEERING_STANDARDS.md` §6–7 and `CONTRIBUTING.md`.

- Python ≥ 3.10, PEP 8, type hints on public functions, a docstring stating behaviour **and limits**.
- No emojis in code, docs or logs.
- Engine (`src/hibari/`) must not import `koto` or mention model-specific replies. Model specifics
  arrive through a `ModelProfile` (identity terms, system prompt, inference settings).
- Backward compatibility: keep existing signatures; add keyword arguments with defaults that
  preserve the old behaviour (e.g. `num_candidates=1`).
- Decoding defaults live only in `models/Koto/configs/inference/koto.yaml`. Do not add new
  hard-coded defaults elsewhere.
- Prefer small, explainable rule sets with names (`ReplyCritic` style: every adjustment has a
  rule name) over opaque heuristics.
- Errors: raise specific exceptions with a message that says what to do; scripts exit non-zero
  on failure and never leave half-written outputs (write `*.tmp`, then move).
- Files written for publication use repo-relative paths.
- Keep tests fast: tiny models (`d_model` 32, 1 layer) and tiny tokenizers; real artefacts only
  behind `pytest.mark.skipif(not path.exists())`.
