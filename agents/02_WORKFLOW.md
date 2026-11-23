# 02 — How to Work on a Task

This is the agent-facing short form of the lifecycle (`Docs/template/22_DEVELOPMENT_LIFECYCLE.md`).

## 1. Understand
- Restate the goal in one sentence. Find the matching task template
  (`Docs/template/24_DEVELOPMENT_TEMPLATES.md`, T1–T10).
- Reproduce the problem with real commands before changing code, e.g.
  `python scripts/engine/generate.py --profile models/Koto/configs/release/koto_1.1.1.yaml --prompt "..." --explain`.
- Decide whether it is a **code/decoding** problem (the model can produce a good reply
  among candidates) or a **weights/data** problem (it never does in ~16 samples).
  Weights/data problems are prepared as data + configs and handed to the owner.

## 2. Plan
- List the files you will touch in both repositories and the tests you will add.
- For large or risky changes, present the plan before editing.

## 3. Implement
- Small, focused changes; new behaviour behind a setting whose default keeps the old path.
- New settings go into `InferenceSettings` / config templates, and the one Koto inference YAML.

## 4. Verify
```bash
source .venv/bin/activate
pytest -q                                              # engine
(cd models/Koto && python -m pytest -q tests)          # Koto
PYTHONPATH=src python scripts/engine/validate_release.py --profile models/Koto/configs/release/koto_1.1.1.yaml --quiet
PYTHONPATH=src python scripts/engine/validate_experiments.py models/Koto/experiments
```
For behaviour changes also run (from `models/Koto`, takes ~15 min on CPU):
```bash
python evaluation/evaluate_behaviour.py --profile configs/release/koto_1.1.1.yaml --name <report_name> --seeds 0 1 2
```
For export changes: `scripts/engine/compare_backends.py` (see 06).

## 5. Record
- Update or create the experiment record (`experiments/EXP-NNNN.yaml`).
- Update changelogs (`Docs/20_CHANGELOG.md`, `models/Koto/CHANGELOG.md`), limitations,
  and `agents/01_PROJECT_MAP.md` "Current state" if it changed.
- Complete the checklist (`Docs/template/25_DEVELOPMENT_CHECKLIST.md`) in your report.

## 6. Hand over
Report to the owner (format in 07): what changed, evidence, what failed, what needs training,
and the exact commands the owner may want to run (training, push). Do not run them yourself.

## Git (only when asked to commit)
```bash
cd models/Koto && git add -A && git commit -m "<what and why>"     # model repo first
cd ../.. && git add -A && git commit -m "<what and why>"           # engine + submodule pointer
```
Never `git push`, never commit to `main`, never commit weights, datasets, `.env` or local paths.
