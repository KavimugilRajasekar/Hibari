# 15 — API and CLI Reference

There are two sets of scripts:

- **Engine tools** in `scripts/engine/` — run from the **Hibari root**.
- **Koto tools** in `models/Koto/scripts/`, `models/Koto/training/` and
  `models/Koto/evaluation/` — run from **`models/Koto`**. Relative paths in
  Koto configs and flags are resolved against the Koto root, so they also
  work from any other directory.

Activate the virtual environment first: `source .venv/bin/activate`.

## Data (Koto)

### scripts/prepare_data.py
Download every source, clean, window, de-duplicate, train the tokenizer and
cache the general corpus in `data/processed/`.
```bash
python scripts/prepare_data.py --config configs/data/koto.yaml
```
| Flag | Default |
|---|---|
| `--config` | `configs/data/koto.yaml` |

### scripts/build_mix.py
Build a versioned training mix from the local cache (`HF_HUB_OFFLINE=1`, no
downloads): general conversation, curated identity / safety data, hand-written
conversations, dialogue skills, emails and writing tasks. Evaluation prompts
are filtered out; tasks and hand-written conversations are split into
train / val / test by a stable hash.
```bash
python scripts/build_mix.py --config configs/data/koto_1.1.0.yaml
```
| Flag | Default | Meaning |
|---|---|---|
| `--config` | `configs/data/koto_1.1.0.yaml` | data mix config (`out_dir` names the output folder) |
| `--overwrite` | off | replace an existing, non-empty `out_dir` |

### scripts/build_finetune_mix.py
Older (v3–v6) fine-tune mix builder from the local cache plus curated and
writing-help data.
```bash
python scripts/build_finetune_mix.py --dst data/processed_v5 \
  --replay 30000 --curated-repeats 4 --coedit 12000 --paradetox 6000 --smol-rewrite 5000 --polite-repeats 2
```
| Flag | Default | Meaning |
|---|---|---|
| `--src` | `data/processed` | base cache |
| `--dst` | `data/processed_ft` | output folder |
| `--tokenizer-dir` | `tokenizer` | |
| `--replay` | 30000 | general conversations to sample |
| `--curated-repeats` | 3 | curated set ×K |
| `--coedit` / `--paradetox` / `--smol-rewrite` | 0 | writing-help examples to add |
| `--polite-repeats` | 0 | hand-written polite / email set ×K |
| `--max-seq-len` | 256 | examples longer than this + 1 are dropped |
| `--seed` | 7 | |

## Training (Koto)

### training/train.py
Full training, or fine-tuning from existing weights (`--checkpoint` or
`init_checkpoint` in the training config).
```bash
python training/train.py --train-config configs/training/koto_1.1.0.yaml 2>&1 | tee outputs/logs/train_1.1.0.log
```
| Flag | Default |
|---|---|
| `--model-config` | `configs/model/koto.yaml` |
| `--train-config` | `configs/training/koto.yaml` |
| `--tokenizer-dir` | `tokenizer` |
| `--data-cache-dir` | `data_cache_dir` from the training config, else `data/processed` |
| `--max-steps` | from config |
| `--checkpoint` | none (random init); a `.pt` path continues from those weights |

The run refuses to start if `output_dir` already contains `best_model.pt`
(unless `overwrite_output_dir: true`).

### training/fine_tune.py
Placeholder. Use `train.py --checkpoint` instead.

## Evaluation (Koto)

### evaluation/evaluate_upgrade.py
The 1.1.0 suite (`koto.evaluation.upgrade_suite`): conversation, context,
rephrase, formal / casual, grammar, meaning preservation and robustness, in
`unrouted` and `routed` modes, greedy and sampled, plus a stability check.
```bash
python evaluation/evaluate_upgrade.py \
  --checkpoint checkpoints/koto_1.1.0/best_model.pt --name koto_1.1.0 \
  --data-dir data/processed_1.1.0 --train-file data/processed_1.1.0/train_conversations.json
```
| Flag | Default |
|---|---|
| `--checkpoint`, `--name` | required |
| `--tokenizer-dir` | `tokenizer` |
| `--data-dir` | none (adds a held-out loss table per source) |
| `--train-file` | none (flags eval prompts seen in training) |
| `--modes` | `unrouted routed` |
| `--no-stability` | off |
| `--out-dir` | `outputs/evaluation` |

### evaluation/evaluate.py
Validation perplexity plus the legacy held-out conversation suite
(`koto.evaluation.suite`, 58 cases).
```bash
python evaluation/evaluate.py --checkpoint checkpoints/koto_start/best_model.pt --out outputs/metrics/eval_start
```
| Flag | Default |
|---|---|
| `--checkpoint` | required |
| `--compare` | other checkpoints to evaluate side by side |
| `--tokenizer-dir` | `tokenizer` |
| `--val` | `data/processed/val_conversations.json` |
| `--val-limit` | 4000 |
| `--out` | `outputs/evaluation/legacy_report` (writes `.json` and `.md`) |

### scripts/inject_synthetic_data.py
Deprecated (v1 emoji data). Do not use.

## Inference (engine)

### scripts/engine/generate.py
Interactive multi-turn chat, or a single reply with `--prompt`. Includes the
deterministic safety layer and rewriting-task routing.
```bash
python scripts/engine/generate.py
python scripts/engine/generate.py --prompt "Fix the grammar: she dont like tea"
```
| Flag | Default |
|---|---|
| `--checkpoint` | `models/Koto/checkpoints/koto_start/best_model.pt` |
| `--tokenizer-dir` | `models/Koto/tokenizer` |
| `--prompt` | none (interactive) |
| `--system` | the training system prompt |
| `--temperature` | 0.5 (0 = greedy) |
| `--top-k` / `--top-p` | 25 / 0.85 |
| `--repetition-penalty` | 1.15 |
| `--max-new-tokens` | 80 |
| `--seed` | none |
| `--no-safety` | off |

### scripts/engine/benchmark.py
CPU speed of the PyTorch checkpoint and of GGUF files in llama.cpp.
| Flag | Default |
|---|---|
| `--checkpoint`, `--gguf …`, `--llama-bin` | required |
| `--tokenizer-dir` | `models/Koto/tokenizer` |
| `--threads` / `--repeats` | 4 / 3 |
| `--out` | `release/benchmarks.json` |

## Export and verification (engine)

### scripts/engine/export.py
```bash
python scripts/engine/export.py --variants f32 f16 q8_0 q4_k_m --llama-quantize ~/llama.cpp/build/bin/llama-quantize
```
| Flag | Default |
|---|---|
| `--checkpoint` | `models/Koto/checkpoints/koto_start/best_model.pt` |
| `--tokenizer-dir` | `models/Koto/tokenizer` |
| `--outdir` | `models/Koto/releases/exported` |
| `--variants` | `f32 f16 q8_0 q4_k_m` |
| `--llama-quantize` | none (Q8_0 / Q4_K_M are skipped without it) |
| `--version` | `2.0` (GGUF `general.version`) |

### scripts/engine/verify_gguf.py
Header, metadata, tensors, tokenizer parity and greedy parity against PyTorch.
| Flag | Default |
|---|---|
| `--checkpoint`, `--gguf …`, `--llama-bin` | required (`--llama-bin ~/llama.cpp/build/bin`) |
| `--tokenizer-dir` | `models/Koto/tokenizer` |
| `--max-new-tokens` | 48 |
| `--report` | `release/inference_parity.json` |

### scripts/engine/verify_llama_cpp_python.py
Re-runs the parity prompts through llama-cpp-python (`pip install llama-cpp-python`).
| Flag | Default |
|---|---|
| `--parity` (report from `verify_gguf.py`), `--gguf …`, `--report` | required |
| `--max-tokens` | 48 |

## Python API

Engine (`hibari`):

```python
from hibari.training.checkpoint import load_checkpoint
from hibari.data.tokenizer import HibariTokenizer
from hibari.inference import (
    InferenceEngine,       # from_checkpoint(), create_chat_session(), rewrite(), generate()
    HibariChatSession,     # multi-turn chat: history trimming, safety layer, task routing
    TaskResult,
    generate_text,         # single prompt
    parse_task_request,    # detect "Rephrase: ...", "Make this formal: ..." requests
    format_task_request,   # canonical "Task: ...\nInput: ..." form
    TASKS,
)
from hibari.evaluation.metrics import *   # repetition, distinct-n, chrF, echo, grammar heuristic
from hibari.export.gguf_export import export_to_gguf
```

Koto (`koto`, importable from `models/Koto`; importing it makes `hibari`
importable from the submodule checkout):

```python
from koto.data.curated import build_curated_conversations
from koto.data.conversation import build_conversation_conversations
from koto.data.dialogue_skills import build_dialogue_skill_conversations
from koto.data.rewrite import load_coedit_rows, load_paradetox_examples, load_smol_rewrite_examples
from koto.data.tasks import handwritten_task_examples
from koto.evaluation.suite import CASES, score_case
from koto.evaluation.upgrade_suite import eval_user_texts
```
