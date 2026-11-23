# Role: Model developer (Koto)

**Scope:** `models/Koto/` — `koto/data/`, `configs/`, `scripts/build_mix.py`, `experiments/`, `docs/`.

**Must**
- Follow the behaviour specification (`docs/28_BEHAVIOUR_SPECIFICATION.md`) and data standards (29).
- Each new version: new data config, new training config (with `model_version`), draft release
  profile (`training_status: NOT_STARTED`), new experiment record. Configs must agree on system
  prompt and version (`tests/test_release_and_behaviour.py::test_1_2_0_configs_agree` is the pattern).
- Identity changes need both the system prompt and the assistant turns (see `identity_v2`).
- Build data into a new directory and inspect it; **never train**.
- Bump `koto.__version__` only when releasing that version; log in `CHANGELOG.md`.

**Done when:** data builds, leak test passes, `data_stats.json` reviewed, the owner has the training command.
