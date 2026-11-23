"""Verify the internal architecture of trained Hibari checkpoints (model-agnostic).

For each ``--checkpoint`` (with the tokenizer in ``--tokenizer-dir``):

  structure   config validates; parameter count = formula = unique tensors; tied embeddings share
              storage when configured; no bias tensors; every weight finite; RMSNorm gains stats
  maths       causality (changing token j never changes logits before j), KV-cache incremental
              decoding = full forward, batch rows independent, eval mode deterministic,
              loss = manual masked cross-entropy
  gguf        for each ``--gguf <checkpoint-index>=<file>`` (F32 / F16): the NumPy reference runtime
              (``hibari.export.gguf_reference``, reads only the GGUF) gives the same logits and greedy
              tokens as PyTorch -- i.e. the file really holds these weights
  heldout     optional ``--records <json>``: token-weighted loss / perplexity (``evaluate_records``)

Writes a JSON report; exit code 1 if any check fails.

    python scripts/engine/verify_architecture.py --tokenizer-dir models/Koto/tokenizer \\
        --checkpoint models/Koto/checkpoints/koto_1.1.0/best_model.pt \\
        --gguf 0=models/Koto/releases/exported/1.1.1/Hibari-Koto-1.1.1-F32.gguf \\
        --records models/Koto/data/processed_1.3.0/test_conversations.json --report out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from hibari.data.tokenizer import HibariTokenizer
from hibari.inference.generation import _stop_ids, generate_tokens
from hibari.training.checkpoint import load_checkpoint

PROMPTS = [
    [{"role": "user", "content": "Hi! How has your day been?"}],
    [{"role": "user", "content": "Rewrite this politely: give me the report now."}],
    [{"role": "user", "content": "I lost my keys today."}, {"role": "assistant", "content": "Oh no, that's stressful."},
     {"role": "user", "content": "Yeah, and I was late too."}],
]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_structure(model) -> Dict:
    cfg = model.config
    cfg.validate()
    unique = {id(p): p.numel() for p in model.parameters()}
    tied = model.lm_head.weight.data_ptr() == model.embed_tokens.embedding.weight.data_ptr()
    biases = [n for n, _ in model.named_parameters() if n.endswith(".bias")]
    nonfinite = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    gains = torch.cat([p.detach().flatten() for n, p in model.named_parameters() if "norm" in n])
    r = {
        "parameters": model.count_parameters()["total"], "formula": cfg.estimate_parameters(), "unique_tensors": sum(unique.values()),
        "tied_storage": tied, "tie_embeddings": cfg.tie_embeddings, "bias_tensors": biases, "nonfinite_tensors": nonfinite,
        "norm_gain_mean": round(gains.mean().item(), 4), "norm_gain_min": round(gains.min().item(), 4),
        "norm_gain_max": round(gains.max().item(), 4),
        "max_abs_weight": round(max(p.detach().abs().max().item() for p in model.parameters()), 4),
    }
    r["ok"] = (r["parameters"] == r["formula"] == r["unique_tensors"] and tied == cfg.tie_embeddings
               and not biases and not nonfinite)
    return r


@torch.no_grad()
def check_maths(model, ids: List[int]) -> Dict:
    x = torch.tensor([ids])
    T, V = len(ids), model.config.vocab_size
    full = model(x)[0]
    # causality
    leaks = []
    for j in sorted({1, T // 3, T // 2, T - 1}):
        y = x.clone()
        y[0, j] = (y[0, j] + 1) % V
        if not torch.equal(model(y)[0][0, :j], full[0, :j]):
            leaks.append(j)
    # KV cache: prefill 4 tokens then one token at a time
    _, _, caches = model(x[:, :4], use_cache=True)
    steps = []
    for t in range(4, T):
        lg, _, caches = model(x[:, t:t + 1], kv_caches=caches, use_cache=True)
        steps.append(lg[0, -1])
    kv_diff = (torch.stack(steps) - full[0, 4:]).abs().max().item() if steps else 0.0
    # batch independence (second row = reversed sequence)
    xb = torch.cat([x, x.flip(1)])
    batch_diff = (model(xb)[0][0] - full[0]).abs().max().item()
    # determinism
    deterministic = torch.equal(model(x)[0], full)
    # masked loss
    tgt = x.clone()
    tgt[0, : T // 2] = -100
    lg, loss, _ = model(x, targets=tgt)
    manual = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), ignore_index=-100)
    r = {"tokens": T, "causal_leaks": leaks, "kv_cache_max_abs_diff": kv_diff, "batch_max_abs_diff": batch_diff,
         "eval_deterministic": deterministic, "loss_vs_manual_abs_diff": abs(loss.item() - manual.item()),
         "logits_finite": bool(torch.isfinite(full).all())}
    r["ok"] = (not leaks and kv_diff < 1e-3 and batch_diff < 1e-3 and deterministic
               and r["loss_vs_manual_abs_diff"] < 1e-5 and r["logits_finite"])
    return r


@torch.no_grad()
def check_gguf(model, tok, gguf_path: Path, prompts_ids: List[List[int]], greedy_tokens: int) -> Dict:
    from hibari.export.gguf_reference import GGUFReferenceModel

    ref = GGUFReferenceModel(gguf_path)
    tol = 1e-3 if "F32" in gguf_path.name.upper() else 0.25
    rows, ok = [], True
    stops = _stop_ids(tok)
    for ids in prompts_ids:
        want = model(torch.tensor([ids]))[0][0].numpy()
        got = ref.forward(ids)
        diff = float(abs(got - want).max())
        argmax_agree = float((got.argmax(-1) == want.argmax(-1)).mean())
        torch_greedy = generate_tokens(model, tok, "", prompt_ids=ids, max_new_tokens=greedy_tokens,
                                       temperature=0.0, repetition_penalty=1.0)
        ref_greedy = ref.greedy(ids, greedy_tokens, stop_ids=stops)
        same = ref_greedy == torch_greedy
        rows.append({"max_abs_logit_diff": round(diff, 6), "argmax_agreement": argmax_agree, "greedy_equal": same})
        ok = ok and diff < tol and (same or "F16" in gguf_path.name.upper())
    return {"file": gguf_path.name, "sha256": _sha(gguf_path), "tolerance": tol, "prompts": rows, "ok": ok}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--gguf", action="append", default=[], help="<checkpoint index>=<F32/F16 gguf>")
    ap.add_argument("--records", default=None, help="held-out conversations JSON for loss / perplexity")
    ap.add_argument("--greedy-tokens", type=int, default=24)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    torch.manual_seed(0)
    tok = HibariTokenizer.load(a.tokenizer_dir)
    system = {"role": "system", "content": "You are a helpful assistant."}
    prompts_ids = [tok.encode(tok.apply_chat_template([system] + p, add_generation_prompt=True)) for p in PROMPTS]
    records = json.loads(Path(a.records).read_text(encoding="utf-8")) if a.records else None
    ggufs: Dict[int, List[Path]] = {}
    for g in a.gguf:
        i, p = g.split("=", 1)
        ggufs.setdefault(int(i), []).append(Path(p))

    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__, "records": a.records,
              "checkpoints": []}
    all_ok = True
    for i, ck in enumerate(a.checkpoint):
        t0 = time.time()
        loaded = load_checkpoint(ck)
        model = loaded["model"]
        assert not model.training, "load_checkpoint must return an eval-mode model"
        if tok.vocab_size != model.config.vocab_size:
            raise SystemExit(f"{ck}: tokenizer vocab {tok.vocab_size} != model vocab {model.config.vocab_size}")
        entry = {"checkpoint": Path(ck).as_posix().split("checkpoints/")[-1], "step": loaded["step"],
                 "sha256": _sha(Path(ck)), "config": model.config.to_dict(),
                 "structure": check_structure(model),
                 "maths": [check_maths(model, ids) for ids in prompts_ids],
                 "gguf": [check_gguf(model, tok, p, prompts_ids, a.greedy_tokens) for p in ggufs.get(i, [])]}
        if records is not None:
            from hibari.evaluation.evaluator import evaluate_records
            entry["heldout"] = evaluate_records(model, tok, records, by_source=False)["all"]
        entry["ok"] = entry["structure"]["ok"] and all(m["ok"] for m in entry["maths"]) and all(g["ok"] for g in entry["gguf"])
        entry["seconds"] = round(time.time() - t0, 1)
        all_ok = all_ok and entry["ok"]
        report["checkpoints"].append(entry)
        s = entry["structure"]
        kv = max(m["kv_cache_max_abs_diff"] for m in entry["maths"])
        print(f"[{'OK ' if entry['ok'] else 'FAIL'}] {entry['checkpoint']}  step {entry['step']}  params {s['parameters']:,}"
              f"  causal leaks {sum(len(m['causal_leaks']) for m in entry['maths'])}  kv-diff {kv:.2e}"
              + "".join(f"  | {g['file']}: max|Δlogit| {max(r['max_abs_logit_diff'] for r in g['prompts']):.2e}"
                        f" greedy {sum(r['greedy_equal'] for r in g['prompts'])}/{len(g['prompts'])}" for g in entry["gguf"])
              + (f"  | held-out loss {entry['heldout']['loss']} ppl {entry['heldout']['perplexity']}" if "heldout" in entry else "")
              + f"  ({entry['seconds']} s)")
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {a.report}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
