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
* Optional gradient accumulation (``gradient_accumulation_steps``): each
  optimizer step averages the gradients of N micro-batches, so the effective
  batch is ``batch_size * N`` without the memory cost. ``max_steps`` counts
  optimizer steps. The default of 1 is the original behaviour. Micro-batches
  are weighted by their number of supervised tokens, so the step equals a
  full-batch step even when micro-batches have different token counts.
* A micro-batch with a non-finite loss is skipped and the remaining ones are
  re-weighted; ``max_nonfinite_steps`` consecutive steps with no usable
  micro-batch stop the run with an error instead of looping forever.
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
    output_dir: str = "checkpoints/run"
    save_best: bool = True
    seed: int = 42
    min_lr_ratio: float = 0.1
    max_hours: float = 0.0
    overwrite_output_dir: bool = False
    early_stopping_patience: int = 0  # evals without improvement; 0 = disabled
    gradient_accumulation_steps: int = 1
    max_nonfinite_steps: int = 10  # consecutive all-non-finite steps before aborting


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
        path_root: Optional[str | Path] = None,
        resume_from: Optional[str | Path] = None,
    ):
        """``path_root``: when given, paths under it are written relative to it in
        ``training_history.json`` (the file is committed; it must not contain local paths)."""
        self.model = model.to(device)
        self.path_root = Path(path_root).resolve() if path_root else None
        # resume_from: a last_checkpoint.pt written by this trainer; model, optimizer,
        # scheduler, step, best result and history are restored and training continues
        # to max_steps. The data stream is fast-forwarded past the batches already
        # used, so a resumed run sees the same batches as an uninterrupted one
        # (given the same loader / sampler seed).
        self.resume_from = Path(resume_from) if resume_from else None
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
            and self.resume_from is None
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
        args = asdict(self.args)
        if self.path_root is not None:
            try:
                args["output_dir"] = str(Path(args["output_dir"]).resolve().relative_to(self.path_root))
            except ValueError:
                pass
        with open(self.output_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(
                {"args": args, "metadata": self.run_metadata, "best": best, "history": history},
                f, indent=2,
            )

    def _restore(self, path: Path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if "optimizer_state_dict" not in ck:
            raise ValueError(f"{path} has no optimizer state; resume needs a last_checkpoint.pt")
        self.model.load_state_dict(ck["model_state_dict"])
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        if self.scheduler is not None and ck.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(ck["scheduler_state_dict"])
        step = int(ck.get("step", 0))
        history, best = [], {"eval_loss": float("inf"), "step": 0}
        hist_file = self.output_dir / "training_history.json"
        if hist_file.exists():
            h = json.loads(hist_file.read_text(encoding="utf-8"))
            history = [r for r in h.get("history", []) if r.get("step", 0) <= step]
            b = h.get("best") or {}
            if "eval_loss" in b and b.get("step", 0) <= step:
                best = {k: b[k] for k in ("eval_loss", "eval_perplexity", "step") if k in b}
        if step >= self.args.max_steps:
            raise ValueError(f"checkpoint is at step {step}, max_steps is {self.args.max_steps}: nothing to resume")
        return step, history, best

    def train(self) -> Dict[str, Any]:
        self.model.train()
        history: List[Dict[str, Any]] = []
        best: Dict[str, Any] = {"eval_loss": float("inf"), "step": 0}
        step = 0
        if self.resume_from is not None:
            step, history, best = self._restore(self.resume_from)
        window_loss, window_steps = 0.0, 0
        last_train_loss = float("nan")
        evals_without_improvement = 0
        t0 = time.time()

        model_desc = getattr(self.model.config, "model_name", "Hibari-Koto")
        pbar = tqdm(total=self.args.max_steps, initial=step, desc=f"Training {model_desc}", mininterval=5)
        train_iter = iter(self.train_dataloader)
        stop_reason = "max_steps"

        accum = max(1, int(self.args.gradient_accumulation_steps))

        def next_batch():
            nonlocal train_iter
            try:
                return next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                return next(train_iter)

        if step:  # resumed: skip the micro-batches the interrupted run already consumed
            for _ in range(step * accum):
                next_batch()

        nonfinite_steps = 0
        while step < self.args.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            micro = [next_batch() for _ in range(accum)]
            n_tok = [int((b["targets"] != -100).sum().item()) for b in micro]
            total_tok = max(1, sum(n_tok))
            micro_losses: List[float] = []
            used_tok = 0
            for batch, n in zip(micro, n_tok):
                if n == 0:
                    continue
                input_ids = batch["input_ids"].to(self.device)
                targets = batch["targets"].to(self.device)
                _, loss, _ = self.model(input_ids, targets=targets)
                if loss is None or not torch.isfinite(loss):
                    pbar.write(f"Skipping non-finite loss at step {step}")
                    continue
                # loss is the mean over this micro-batch's n tokens; weighting by
                # n / total gives the mean over all tokens of the step
                (loss * (n / total_tok)).backward()
                micro_losses.append(loss.item())
                used_tok += n
            if not micro_losses:
                nonfinite_steps += 1
                if nonfinite_steps >= self.args.max_nonfinite_steps:
                    raise RuntimeError(
                        f"{nonfinite_steps} consecutive steps had no finite loss (step {step}); "
                        "check the learning rate and the data."
                    )
                continue
            nonfinite_steps = 0
            if used_tok < total_tok:  # some micro-batches were skipped: re-normalise
                scale = total_tok / used_tok
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
            grad_norm = None
            if self.args.gradient_clip > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip))
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            step += 1
            window_loss += sum(micro_losses) / len(micro_losses)
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
