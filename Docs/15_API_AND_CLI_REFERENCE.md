# 15 — API and CLI Reference

All scripts live in `hibari/scripts/`. Run them from the repository root with
the virtual environment active (`source .venv/bin/activate`).

## Data

### prepare_data.py
Download every source, clean, window, de-duplicate and cache the base corpus.
```bash
python hibari/scripts/prepare_data.py --config hibari/configs/data/koto.yaml
```
| Flag | Default |
|---|---|
| `--config` | `hibari/configs/data/koto.yaml` |

### build_finetune_mix.py
Build a training mix from the local cache plus the current curated and
writing-help data.
```bash
python hibari/scripts/build_finetune_mix.py --dst hibari/data/processed_v5 \
  --replay 30000 --curated-repeats 4 --coedit 12000 --paradetox 6000 --smol-rewrite 5000 --polite-repeats 2
```
| Flag | Default | Meaning |
|---|---|---|
| `--src` | `hibari/data/processed` | base cache |
| `--dst` | `hibari/data/processed_ft` | output folder |
| `--tokenizer-dir` | `hibari/models/tokenizer` | |
| `--replay` | 30000 | general conversations to sample (70179 = all) |
| `--curated-repeats` | 3 | curated set ×K |
| `--coedit` / `--paradetox` / `--smol-rewrite` | 0 | writing-help examples to add |
| `--polite-repeats` | 0 | hand-written polite / email set ×K |
| `--max-seq-len` | 256 | examples longer than this + 1 are dropped |
| `--seed` | 7 | |

## Training

### train.py
Full training, or fine-tuning with `--checkpoint`.
```bash
python hibari/scripts/train.py --model-config hibari/configs/model/koto.yaml \
  --train-config hibari/configs/training/koto.yaml --data-cache-dir hibari/data/processed
```
| Flag | Default |
|---|---|
| `--model-config` | `hibari/configs/model/koto.yaml` |
| `--train-config` | `hibari/configs/training/koto.yaml` |
| `--tokenizer-dir` | `hibari/models/tokenizer` |
| `--data-cache-dir` | `hibari/data/processed` |
| `--max-steps` | from config |
| `--checkpoint` | none (random init); a `.pt` path continues from those weights |

### fine_tune.py
Placeholder (prints a TODO). Use `train.py --checkpoint` instead.

## Evaluation

### evaluate.py
Validation perplexity plus the held-out conversation suite.
```bash
python hibari/scripts/evaluate.py --checkpoint hibari/models/checkpoints/koto_v4/best_model.pt \
  --out hibari/outputs/metrics/eval_v4
```
| Flag | Default |
|---|---|
| `--checkpoint` | required |
| `--compare` | other checkpoints to evaluate side by side |
| `--tokenizer-dir` | `hibari/models/tokenizer` |
| `--val` | `hibari/data/processed/val_conversations.json` |
| `--val-limit` | 4000 |
| `--out` | `release/evaluation_report` (writes `.json` and `.md`) |

### benchmark.py
CPU speed of the PyTorch checkpoint and of GGUF files in llama.cpp.
| Flag | Default |
|---|---|
| `--checkpoint`, `--gguf …`, `--llama-bin` | required |
| `--threads` / `--repeats` | 4 / 3 |
| `--out` | `release/benchmarks.json` |

## Inference

### generate.py
Interactive multi-turn chat, or a single reply with `--prompt`. Includes the
deterministic safety layer.
```bash
python hibari/scripts/generate.py --checkpoint hibari/models/checkpoints/koto_v4/best_model.pt
python hibari/scripts/generate.py --checkpoint … --prompt "Fix the grammar: she dont like tea"
```
| Flag | Default |
|---|---|
| `--checkpoint` | `hibari/models/checkpoints/koto_v2/best_model.pt` |
| `--tokenizer-dir` | `hibari/models/tokenizer` |
| `--prompt` | none (interactive) |
| `--system` | the training system prompt |
| `--temperature` | 0.5 (0 = greedy) |
| `--top-k` / `--top-p` | 25 / 0.85 |
| `--repetition-penalty` | 1.15 |
| `--max-new-tokens` | 80 |
| `--seed` | none |
| `--no-safety` | off |

## Export and verification

### export.py
```bash
python hibari/scripts/export.py --checkpoint hibari/models/checkpoints/koto_v4/best_model.pt \
  --outdir hibari/models/exported --variants f32 f16 q8_0 q4_k_m --llama-quantize ~/llama.cpp/llama-quantize
```
| Flag | Default |
|---|---|
| `--checkpoint` | `hibari/models/checkpoints/koto_v2/best_model.pt` |
| `--tokenizer-dir` | `hibari/models/tokenizer` |
| `--outdir` | `hibari/models/exported` |
| `--variants` | `f32 f16 q8_0 q4_k_m` |
| `--llama-quantize` | none (Q8_0 / Q4_K_M are skipped without it) |
| `--version` | `2.0` (GGUF `general.version`) |

### verify_gguf.py
Header, metadata, tensors, tokenizer parity and greedy parity against PyTorch.
| Flag | Default |
|---|---|
| `--checkpoint`, `--gguf …`, `--llama-bin` | required (`--llama-bin ~/llama.cpp`) |
| `--max-new-tokens` | 48 |
| `--report` | `release/inference_parity.json` |

### verify_llama_cpp_python.py
Re-runs the parity prompts through llama-cpp-python (`pip install llama-cpp-python`).
| Flag | Default |
|---|---|
| `--parity` (report from `verify_gguf.py`), `--gguf …`, `--report` | required |
| `--max-tokens` | 48 |

### inject_synthetic_data.py
Deprecated (v1 emoji data). Do not use.

## Python API

```python
from hibari.training.checkpoint import load_checkpoint
from hibari.data.tokenizer import HibariTokenizer
from hibari.inference.conversation import HibariChatSession   # multi-turn chat with history trimming + safety
from hibari.inference.generation import generate_text         # single prompt
from hibari.export.gguf_export import export_to_gguf
from hibari.data.curated import build_curated_conversations
from hibari.data.rewrite import load_coedit, load_paradetox, load_smol_rewrite, build_polite_conversations
from hibari.evaluation.suite import CASES, score_case
```
