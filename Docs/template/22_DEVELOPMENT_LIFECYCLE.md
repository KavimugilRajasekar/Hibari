# 22 — Model Development Lifecycle

Every model version passes these stages in order. A stage is done when its
**exit check** passes; the evidence goes into the experiment record
(`experiments/EXP-NNNN.yaml`).

| # | Stage | What happens | Exit check |
|---|---|---|---|
| 1 | Requirement definition | Write the objective and the observed failures (with reproducible prompts) | experiment record `objective`, `problem` filled |
| 2 | Architecture inspection | Read the model/engine code that the change touches; note risks | listed in `source_changes` |
| 3 | Dataset inspection | Sample the data the behaviour depends on; check what the model has actually seen | notes in the record; `data_stats.json` reviewed |
| 4 | Tokenizer compatibility | Same tokenizer hash as the checkpoint; special tokens unchanged | `validate_release.py` `sha256:tokenizer`, `vocab` |
| 5 | Configuration validation | Configs load into their dataclasses; one inference YAML | `pytest tests/test_release.py`, `validate_release.py --no-gguf` |
| 6 | Model implementation | Code / data changes | code review |
| 7 | Development and debugging | Iterate on a **development set** that is not the evaluation suite | dev prompts kept separate |
| 8 | Unit testing | One regression test per bug fixed; tests for new modules | `pytest` green in both repositories |
| 9 | Integration testing | End-to-end: data build, chat session, export on tiny models | `tests/test_release.py`, Koto `tests/test_upgrade.py` |
| 10 | Behavioural evaluation | Behaviour + upgrade suites, several seeds, transcripts **read by a person** | report in `outputs/evaluation/` |
| 11 | Inference validation | Python chat CLI with the profile; raw vs application layer | `generate.py --profile ... --explain` spot checks |
| 12 | Export validation | GGUF with provenance; Modelfile generated | `validate_release.py` 0 errors |
| 13 | Quantization validation | Q8_0 / Q4_K_M parity measured | `compare_backends.py` report |
| 14 | Regression testing | Compare with the previous version on the same suites and settings | report shows both versions |
| 15 | Model documentation | Model card, changelog, limitations, behaviour spec | files updated |
| 16 | Version readiness assessment | Checklist §B completed honestly | `25_DEVELOPMENT_CHECKLIST.md` |
| 17 | Release preparation | Tag and publish — **only by the owner** | owner decision |

## The training gate (between stages 6 and 8)

Training is a separate, controlled step and is never started by automation,
by a script that "includes" a training stage, or by an assistant.

1. The experiment record says `training.status: NOT_STARTED` (or
   `NOT_REQUIRED` for inference-only versions such as Koto 1.1.1).
2. The data is built and inspected (`build_mix.py` refuses to overwrite).
3. The owner sets `training.status: MANUALLY_AUTHORIZED`, fills
   `training.authorized_by` and `training.config`, and runs the training
   command personally. The trainer refuses to overwrite an existing
   `best_model.pt` and records `model_version`, git commits and data hashes in
   the checkpoint.
4. After the run: `training.status: COMPLETED`, the checkpoint SHA-256 goes
   into the release profile, and stages 8–17 run on the new weights.

`validate_experiments.py` rejects a record that claims a run without an
owner or a training config.

## Version types and which stages they need

| Change | Version | Stages |
|---|---|---|
| Inference / export / evaluation repair, same weights | PATCH (e.g. 1.1.0 → 1.1.1) | 1–2, 4–17; training `NOT_REQUIRED` |
| New data or training, same architecture | MINOR (1.1 → 1.2) | all, with the training gate |
| Architecture, tokenizer or chat-format change | MAJOR (1.x → 2.0) | all; new tokenizer hash; old GGUFs incompatible |
