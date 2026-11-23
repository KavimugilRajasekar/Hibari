# AGENTS.md — Hibari

You are working on **Hibari**, an LLM engine built from scratch, and its first model **Koto**
(git submodule `models/Koto`). Full agent instructions: [`agents/`](agents/README.md).

## Non-negotiable
- **Do not start any training** (pretraining, fine-tuning, LoRA, extra epochs). The owner trains.
- **Do not `git push`**, open PRs, tag, or publish anything. The owner pushes.
- **Do not overwrite or delete** checkpoints, datasets, archives, GGUFs or reports.
- **Do not hard-code chat replies**; replies come from the model (see `agents/00_GROUND_RULES.md`).
- Report only numbers you measured in this session; state failures honestly.

## Quick commands
```bash
source .venv/bin/activate
pytest -q                                                    # engine tests
(cd models/Koto && python -m pytest -q tests)                # Koto tests
PYTHONPATH=src python scripts/engine/validate_release.py --profile models/Koto/configs/release/koto_1.1.1.yaml --quiet
PYTHONPATH=src python scripts/engine/generate.py --profile models/Koto/configs/release/koto_1.1.1.yaml --explain
```

## Read next
`agents/00_GROUND_RULES.md` → `agents/01_PROJECT_MAP.md` → `agents/02_WORKFLOW.md`, then the role
guide in `agents/roles/`. Process and standards: `Docs/template/21–25`.
