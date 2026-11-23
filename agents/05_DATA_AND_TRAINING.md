# 05 — Data and Training

Standards: `models/Koto/docs/29_DATA_STANDARDS.md`. Training gate: `Docs/template/22_DEVELOPMENT_LIFECYCLE.md`.

## What an AI developer may do
- Write or change data builders in `models/Koto/koto/data/` (new families behind a config key
  that defaults to off, so older configs rebuild unchanged).
- Write data / training / model configs for a **new** version.
- Build data into a **new** directory:
  `HF_HUB_OFFLINE=1 python scripts/build_mix.py --config configs/data/koto_X.Y.Z.yaml`
  (never `--overwrite` an existing mix without the owner's permission).
- Inspect `data_stats.json` and sample examples.

## What only the owner does
- Run `training/train.py` (any config, any number of steps).
- Set `training.status: MANUALLY_AUTHORIZED` / `COMPLETED` and `authorized_by` in the experiment record.

## Data rules (short)
- Assistant replies: Koto's identity for that version, no claimed human experiences, empathy =
  acknowledge + one relevant question or step, rewrites = output only.
- Every example uses the version's system prompt (`system_prompt` in the data config) and fits
  the context window; no cut replies.
- Slot templates with several phrasings per slot; never one fixed sentence per input.
- Evaluation prompts never enter training (leak filter; `eval_prompts_in_train_user_turns` must be empty).
- New sources need a licence entry in `docs/16_DATASET_CATALOG.md`; no downloads in the builder.

## When a problem needs training
Write: the failure with examples → why decoding cannot fix it (candidate evidence) → the data
family that teaches it → the config → the evaluation that will show it worked. Put it in a new
experiment record with `training.status: NOT_STARTED`, and give the owner the command.
