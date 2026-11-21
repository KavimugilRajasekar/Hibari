"""Training and evaluation loop for Hibari models.

Fixes over the original loop (see Docs/07_TRAINING_PIPELINE.md):

* Validation loss is token-weighted (sum of token losses / number of
  supervised tokens) instead of a mean of per-batch means.
* Periodic checkpoints record the real recent training loss (the original
  stored ``running_loss`` right after resetting it, i.e. always 0.0).
* ``best_model.pt`` is only replaced when validation improves *within this
  run*, and the run refuses to start in a directory that already contains a
  ``best_model.pt`` unless ``overwrite_output_dir`` is set. (Previously a
  250-step continuation run silently overwrote the 1500-step run's best.)
* Training history and the run configuration are written to
  ``<output_dir>/training_history.json``.
* Wall-clock time limit (``max_hours``) for CPU budgets.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from hibari.model.llm import HibariKotoLM
from hibari.training.checkpoint import save_checkpoint
from hibari.training.loss import compute_perplexity


@dataclass
class TrainingArgs:
    """Hyperparameters for model training."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_steps: int = 1000
    batch_size: int = 16
    gradient_clip: float = 1.0
    checkpoint_every: int = 200
    eval_every: int = 100
    log_every: int = 20
    output_dir: str = "hibari/models/checkpoints"
    save_best: bool = True
    seed: int = 42
    min_lr_ratio: float = 0.1
    max_hours: float = 0.0
    overwrite_output_dir: bool = False
    early_stopping_patience: int = 0  # evals without improvement; 0 = disabled


def evaluate_loss(model: HibariKotoLM, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Token-weighted cross-entropy over all supervised (target != -100) tokens."""
    model.eval()
    total, n_tokens = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            logits, _, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-100, reduction="sum",
            )
            total += loss.item()
            n_tokens += int((targets != -100).sum().item())
    avg = total / max(1, n_tokens)
    return {"eval_loss": avg, "eval_perplexity": compute_perplexity(avg), "eval_tokens": n_tokens}


class Trainer:
    """Engine executing the training loop with metric tracking and checkpointing."""

    def __init__(
        self,
        model: HibariKotoLM,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        args: Optional[TrainingArgs] = None,
        device: str | torch.device = "cpu",
        run_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.args = args or TrainingArgs()
        self.device = torch.device(device)
        self.output_dir = Path(self.args.output_dir)
        self.run_metadata = run_metadata or {}
        if (
            self.args.save_best
            and (self.output_dir / "best_model.pt").exists()
            and not self.args.overwrite_output_dir
        ):
            raise FileExistsError(
                f"{self.output_dir}/best_model.pt already exists. Use a new output_dir "
                "or set overwrite_output_dir: true to replace it."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self) -> Dict[str, float]:
        if self.val_dataloader is None:
            return {}
        return evaluate_loss(self.model, self.val_dataloader, self.device)

    def _write_history(self, history: List[Dict[str, Any]], best: Dict[str, Any]) -> None:
        with open(self.output_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(
                {"args": asdict(self.args), "metadata": self.run_metadata, "best": best, "history": history},
                f, indent=2,
            )

    def train(self) -> Dict[str, Any]:
        self.model.train()
        history: List[Dict[str, Any]] = []
        best: Dict[str, Any] = {"eval_loss": float("inf"), "step": 0}
        step = 0
        window_loss, window_steps = 0.0, 0
        last_train_loss = float("nan")
        evals_without_improvement = 0
        t0 = time.time()

        model_desc = getattr(self.model.config, "model_name", "Hibari-Koto")
        pbar = tqdm(total=self.args.max_steps, desc=f"Training {model_desc}", mininterval=5)
        train_iter = iter(self.train_dataloader)
        stop_reason = "max_steps"

        while step < self.args.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            _, loss, _ = self.model(input_ids, targets=targets)
            if loss is None or not torch.isfinite(loss):
                pbar.write(f"Skipping non-finite loss at step {step}")
                continue
            loss.backward()
            grad_norm = None
            if self.args.gradient_clip > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip))
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            step += 1
            window_loss += loss.item()
            window_steps += 1
            pbar.update(1)

            if step % self.args.log_every == 0 or step == self.args.max_steps:
                last_train_loss = window_loss / max(1, window_steps)
                window_loss, window_steps = 0.0, 0
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": f"{last_train_loss:.4f}", "lr": f"{lr:.2e}"})
                history.append({
                    "step": step, "train_loss": last_train_loss,
                    "train_perplexity": compute_perplexity(last_train_loss),
                    "learning_rate": lr, "grad_norm": grad_norm,
                    "elapsed_s": round(time.time() - t0, 1),
                })

            out_of_time = self.args.max_hours > 0 and (time.time() - t0) > self.args.max_hours * 3600
            is_last = step == self.args.max_steps or out_of_time

            if self.val_dataloader is not None and (step % self.args.eval_every == 0 or is_last):
                metrics = self.evaluate()
                self.model.train()
                eval_loss = metrics["eval_loss"]
                pbar.write(
                    f"Step {step} | train {last_train_loss:.4f} | eval {eval_loss:.4f} "
                    f"| ppl {metrics['eval_perplexity']:.2f} | {time.time() - t0:.0f}s"
                )
                history.append({"step": step, **metrics, "elapsed_s": round(time.time() - t0, 1)})
                if eval_loss < best["eval_loss"]:
                    best = {"eval_loss": eval_loss, "eval_perplexity": metrics["eval_perplexity"], "step": step}
                    evals_without_improvement = 0
                    if self.args.save_best:
                        save_checkpoint(
                            self.output_dir / "best_model.pt", self.model,
                            step=step, loss=eval_loss,
                            extra={"best_val_loss": eval_loss, "run_metadata": self.run_metadata},
                        )
                else:
                    evals_without_improvement += 1
                self._write_history(history, best)
                if (
                    self.args.early_stopping_patience > 0
                    and evals_without_improvement >= self.args.early_stopping_patience
                ):
                    stop_reason = "early_stopping"
                    break

            if step % self.args.checkpoint_every == 0 or is_last:
                save_checkpoint(
                    self.output_dir / "last_checkpoint.pt", self.model,
                    optimizer=self.optimizer, scheduler=self.scheduler,
                    step=step, loss=last_train_loss,
                    extra={"run_metadata": self.run_metadata},
                )

            if out_of_time:
                stop_reason = "max_hours"
                break

        pbar.close()
        save_checkpoint(
            self.output_dir / "final_model.pt", self.model, step=step, loss=last_train_loss,
            extra={"run_metadata": self.run_metadata, "stop_reason": stop_reason},
        )
        best["stop_reason"] = stop_reason
        best["final_step"] = step
        self._write_history(history, best)
        return {"history": history, "best": best}
