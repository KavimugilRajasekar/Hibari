"""Checkpointing utilities for saving and restoring model states."""

from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from hibari.model.config import ModelConfig
from hibari.model.llm import HibariKotoLM


def save_checkpoint(
    path: str | Path,
    model: HibariKotoLM,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    loss: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model checkpoint and training state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
        "step": step,
        "loss": loss,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    load_training_state: bool = False,
) -> Dict[str, Any]:
    """Load model checkpoint and optionally return training state.

    The model is returned in eval mode: with ``resid_dropout`` > 0 (engine 0.4.1) a model left in
    training mode gives a different answer on every call. ``Trainer.train`` switches back to train mode.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    # TorchVersion is a str subclass written into run metadata by early v2
    # runs; allow-list it so weights_only (safe) loading keeps working.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = ModelConfig.from_dict(checkpoint["config"])
    model = HibariKotoLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    result = {
        "model": model,
        "config": config,
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", 0.0),
    }
    if load_training_state:
        result["optimizer_state_dict"] = checkpoint.get("optimizer_state_dict")
        result["scheduler_state_dict"] = checkpoint.get("scheduler_state_dict")

    return result
