# hibari/ — Base Configurations

This folder holds only the engine's **base YAML configurations**:

```
hibari/configs/
├── model/base.yaml        # Default ModelConfig values
├── training/base.yaml     # Default TrainingArgs values
├── training/sft.yaml      # SFT defaults
├── inference/base.yaml    # Default sampling settings
└── data/base.yaml         # Default data settings
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
