# hibari/ — Base Configurations

This folder holds the engine's **configuration templates** (Docs/template/21). Their keys are
tested against the code (`tests/test_release.py::test_p9`); nothing reads them at run time —
models copy them into their own `configs/`.

```
hibari/configs/
├── model/base.yaml          # TEMPLATE: ModelConfig fields
├── training/base.yaml       # TEMPLATE: training from scratch (TrainingArgs + repo keys)
├── training/sft.yaml        # TEMPLATE: continued training / full fine-tuning (no LoRA yet)
├── inference/base.yaml      # TEMPLATE: InferenceSettings (decoding + application layer)
├── data/base.yaml           # TEMPLATE: keys every model data config defines
├── release/template.yaml    # TEMPLATE: release profile (hibari.release.ModelProfile)
└── experiment/template.yaml # TEMPLATE: experiment record
```

Everything else that used to live here has moved:

| Was | Now |
|---|---|
| `hibari/src/hibari/` | `src/hibari/` (the engine package) |
| `hibari/scripts/` (engine tools) | `scripts/engine/` |
| `hibari/scripts/` (Koto data / training / evaluation) | `models/Koto/scripts/`, `models/Koto/training/`, `models/Koto/evaluation/` |
| `hibari/tests/` | `tests/` (engine) and `models/Koto/tests/` (Koto) |
| `hibari/models/` (checkpoints, tokenizer, GGUFs) | `models/Koto/checkpoints/`, `models/Koto/tokenizer/`, `models/Koto/releases/exported/` |
| `hibari/data/`, `hibari/outputs/` | `models/Koto/data/`, `models/Koto/outputs/` |
| Koto configs (`koto*.yaml`) | `models/Koto/configs/` |

See the [root README](../README.md) for the full layout and the Quick Start.
