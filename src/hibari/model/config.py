"""Model configuration for Hibari LLM architectures."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


@dataclass
class ModelConfig:
    """Hyperparameters for Hibari decoder-only transformer models."""

    vocab_size: int = 1024
    max_seq_len: int = 256
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 384
    norm_eps: float = 1e-5
    dropout: float = 0.0          # on attention weights
    resid_dropout: float = 0.0    # on each sub-layer output before the residual add (GPT-2 style); 0 = off
    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    model_name: str = "Hibari-Koto"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "ModelConfig":
        """Raise ``ValueError`` listing every inconsistent hyper-parameter.

        Checked: positive sizes, ``d_model`` divisible by ``n_heads``, an even
        head dimension (RoPE rotates pairs), dropout and resid_dropout in [0, 1), positive
        ``norm_eps`` and ``rope_theta``. Called on construction, so a bad YAML
        fails when it is loaded instead of deep inside the first forward pass.
        """
        problems = []
        for name in ("vocab_size", "max_seq_len", "d_model", "n_layers", "n_heads", "d_ff"):
            v = getattr(self, name)
            low = 0 if name == "n_layers" else 1   # 0 layers = embedding-only (bigram) model, used in tests
            if not isinstance(v, int) or isinstance(v, bool) or v < low:
                problems.append(f"{name} must be an integer >= {low}, got {v!r}")
        if not problems:
            if self.d_model % self.n_heads:
                problems.append(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
            elif (self.d_model // self.n_heads) % 2:
                problems.append(f"head_dim ({self.d_model // self.n_heads}) must be even for RoPE")
        for name in ("dropout", "resid_dropout"):
            if not 0.0 <= getattr(self, name) < 1.0:
                problems.append(f"{name} must be in [0, 1), got {getattr(self, name)}")
        if self.norm_eps <= 0:
            problems.append(f"norm_eps must be > 0, got {self.norm_eps}")
        if self.rope_theta <= 0:
            problems.append(f"rope_theta must be > 0, got {self.rope_theta}")
        if problems:
            raise ValueError("Invalid ModelConfig: " + "; ".join(problems))
        return self

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self.d_model // self.n_heads

    def estimate_parameters(self) -> int:
        """Calculate approximate parameter count for this architecture."""
        # Token embedding
        emb_params = self.vocab_size * self.d_model
        
        # Transformer blocks
        # Attention: Wq, Wk, Wv, Wo
        attn_params = 4 * (self.d_model * self.d_model)
        # 2x RMSNorm per layer
        norm_params = 2 * self.d_model
        # SwiGLU FFN: W_gate, W_up, W_down
        ffn_params = 3 * (self.d_model * self.d_ff)
        layer_params = attn_params + norm_params + ffn_params
        total_layers = self.n_layers * layer_params
        
        # Final RMSNorm
        final_norm = self.d_model
        
        # Output head
        head_params = 0 if self.tie_embeddings else (self.d_model * self.vocab_size)
        
        return emb_params + total_layers + final_norm + head_params

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_dict_strict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Like ``from_dict`` but unknown keys are an error (for hand-written YAML)."""
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown ModelConfig keys: {unknown}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
