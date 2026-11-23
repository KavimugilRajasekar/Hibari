# agents/ — Instructions for AI Developers

This folder tells an AI coding agent (Claude Code, Codex, Copilot agents, Cursor, …) how to
work on Hibari and Koto. Humans may find it a useful summary too. The authoritative engineering
process is the Official Development Template in `Docs/template/`; these files are the
agent-facing rules and shortcuts on top of it.

**Read in this order before doing anything:**

| # | File | What it covers |
|---|---|---|
| 00 | [00_GROUND_RULES.md](00_GROUND_RULES.md) | **Mandatory.** Things you must never do (train, push, overwrite weights…) and must always do |
| 01 | [01_PROJECT_MAP.md](01_PROJECT_MAP.md) | Repositories, where code lives, current versions, known traps |
| 02 | [02_WORKFLOW.md](02_WORKFLOW.md) | Understand → plan → implement → verify → record → hand over |
| 03 | [03_CODING_STANDARDS.md](03_CODING_STANDARDS.md) | Code style, engine/model separation, defaults |
| 04 | [04_TESTING_AND_EVALUATION.md](04_TESTING_AND_EVALUATION.md) | Test suites, behaviour evaluation rules, baselines |
| 05 | [05_DATA_AND_TRAINING.md](05_DATA_AND_TRAINING.md) | Data building (allowed) vs training (owner only) |
| 06 | [06_RELEASE_AND_EXPORT.md](06_RELEASE_AND_EXPORT.md) | Release profiles, GGUF export, Ollama / llama.cpp parity |
| 07 | [07_REPORTING.md](07_REPORTING.md) | How to report results and hand commands to the owner |
| — | [roles/](roles/README.md) | Role guides: engine developer, model developer, evaluator, release engineer |

Entry points that load these automatically: `AGENTS.md` (most agents) and `CLAUDE.md`
(Claude Code) in the Hibari root and in `models/Koto/`.

Keep `01_PROJECT_MAP.md` → "Current state" up to date when versions change.
