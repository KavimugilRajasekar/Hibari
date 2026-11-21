"""Verify exported GGUF files against the PyTorch checkpoint using llama.cpp.

For every GGUF file:
  1. Header: GGUF version, alignment, tensor/KV counts.
  2. Metadata: architecture, dimensions, RoPE, norm eps, tokenizer keys,
     special-token IDs, chat template.
  3. Tensors: names, shapes and dtypes vs the expected Hibari layout.
  4. Tokenizer parity: ``llama-tokenize`` IDs vs the Python tokenizer.
  5. Greedy generation parity: ``llama-completion --temp 0`` vs PyTorch
     greedy decoding (repetition penalty disabled on both sides), on single-
     and multi-turn prompts.

Writes a JSON report (``--report``). Exit code 1 if F32/F16 fail
tokenizer parity or required tensors are missing.

Usage:
    python hibari/scripts/verify_gguf.py --checkpoint <ckpt> \
        --gguf hibari/models/exported/*.gguf --llama-bin <llama.cpp/build/bin> \
        --report release/inference_parity.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import gguf
import numpy as np

from hibari.data.formatting import DEFAULT_SYSTEM_PROMPT
from hibari.data.tokenizer import HibariTokenizer
from hibari.export.gguf_export import tensor_name_map
from hibari.inference.generation import generate_tokens
from hibari.training.checkpoint import load_checkpoint

TOKENIZER_CASES = [
    "Hello, world!",
    "I'm fine, don't worry. It's 3.14 and 2025.",
    "ur just a idiot lol",
    "line one\nline two\n\n  indented",
    "Emoji test 😊💙 ok",
    "Café naïve résumé",
    "日本語テキスト",
    "What's 12 + 30? (answer: 42)",
    "   leading and trailing   ",
    "e-mail: test@example.com, url: https://example.com/a?b=c",
    "zxqvbnm qwertyuiop asdfghjkl",
    "<|system|>\nYou are Hibari.<|eos|>\n<|user|>\nhi<|eos|>\n<|assistant|>\n",
]

GEN_CASES: List[List[Dict[str, str]]] = [
    [{"role": "user", "content": "Hello!"}],
    [{"role": "user", "content": "What is your name?"}],
    [{"role": "user", "content": "I feel sad today."}],
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "ur just a idiot"}],
    [
        {"role": "user", "content": "My name is Sam."},
        {"role": "assistant", "content": "Nice to meet you, Sam! I'm Hibari."},
        {"role": "user", "content": "What's my name?"},
    ],
]


def read_header(path: Path) -> dict:
    r = gguf.GGUFReader(str(path))
    meta = {}
    for name, field in r.fields.items():
        if name.startswith("GGUF."):
            continue
        try:
            val = field.contents()
        except Exception:
            val = "<unreadable>"
        if isinstance(val, list) and len(val) > 8:
            val = f"<array len={len(val)} first={val[:3]}>"
        meta[name] = val
    tensors = {
        t.name: {"shape": [int(x) for x in t.shape], "type": t.tensor_type.name, "n_elements": int(t.n_elements)}
        for t in r.tensors
    }
    version = int(r.fields["GGUF.version"].parts[-1][0])
    return {
        "gguf_version": version,
        "alignment": int(r.alignment),
        "n_tensors": len(r.tensors),
        "n_kv": len(meta),
        "metadata": meta,
        "tensors": tensors,
        "n_vocab_tokens": len(r.fields["tokenizer.ggml.tokens"].data) if "tokenizer.ggml.tokens" in r.fields else 0,
        "n_merges": len(r.fields["tokenizer.ggml.merges"].data) if "tokenizer.ggml.merges" in r.fields else 0,
    }


def check_tensors(header: dict, cfg) -> dict:
    expected = set(tensor_name_map(cfg.n_layers, include_output=not cfg.tie_embeddings).values())
    present = set(header["tensors"])
    shapes_ok = True
    problems = []
    d, f, v = cfg.d_model, cfg.d_ff, cfg.vocab_size
    want = {"token_embd.weight": [d, v], "output_norm.weight": [d]}
    for i in range(cfg.n_layers):
        for n in ["attn_q", "attn_k", "attn_v", "attn_output"]:
            want[f"blk.{i}.{n}.weight"] = [d, d]
        want[f"blk.{i}.attn_norm.weight"] = [d]
        want[f"blk.{i}.ffn_norm.weight"] = [d]
        want[f"blk.{i}.ffn_gate.weight"] = [d, f]
        want[f"blk.{i}.ffn_up.weight"] = [d, f]
        want[f"blk.{i}.ffn_down.weight"] = [f, d]
    for name, shape in want.items():
        got = header["tensors"].get(name, {}).get("shape")
        if got != shape:
            shapes_ok = False
            problems.append(f"{name}: expected {shape} (ggml order), got {got}")
    norm_types = {header["tensors"][n]["type"] for n in present if n.endswith("norm.weight")}
    return {
        "missing": sorted(expected - present),
        "unexpected": sorted(present - expected),
        "shapes_ok": shapes_ok,
        "problems": problems,
        "norm_tensor_types": sorted(norm_types),
        "types": sorted({t["type"] for t in header["tensors"].values()}),
        "total_elements": sum(t["n_elements"] for t in header["tensors"].values()),
    }


def llama_tokenize(bin_dir: Path, model: Path, text: str) -> List[int]:
    out = subprocess.run(
        [str(bin_dir / "llama-tokenize"), "-m", str(model), "-p", text, "--ids", "--log-disable"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[-1]
    return json.loads(out)


def llama_generate(bin_dir: Path, model: Path, prompt: str, n: int) -> tuple[str, float]:
    t = time.time()
    proc = subprocess.run(
        [
            str(bin_dir / "llama-completion"), "-m", str(model), "-p", prompt, "-n", str(n),
            "--temp", "0", "--repeat-penalty", "1.0", "-c", "256", "-no-cnv", "--no-display-prompt",
            "--simple-io", "--no-warmup", "-t", "4",
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"llama-completion failed ({proc.returncode}): {proc.stderr[-800:]}")
    text = proc.stdout.replace("[end of text]", "").strip()
    return text, time.time() - t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer-dir", default="hibari/models/tokenizer")
    ap.add_argument("--gguf", nargs="+", required=True)
    ap.add_argument("--llama-bin", required=True, help="llama.cpp build/bin directory")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--report", default="release/inference_parity.json")
    a = ap.parse_args()

    bin_dir = Path(a.llama_bin)
    tok = HibariTokenizer.load(a.tokenizer_dir)
    loaded = load_checkpoint(a.checkpoint)
    model, cfg = loaded["model"].eval(), loaded["config"]

    # PyTorch reference generations (greedy, no repetition penalty)
    refs = []
    for msgs in GEN_CASES:
        prompt = tok.apply_chat_template([{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + msgs, add_generation_prompt=True)
        ids = generate_tokens(model, tok, prompt, max_new_tokens=a.max_new_tokens, temperature=0.0, repetition_penalty=1.0)
        refs.append({"prompt": prompt, "messages": msgs, "user": msgs[-1]["content"], "torch_ids": ids,
                     "torch_text": tok.decode(ids, skip_special_tokens=True).strip()})

    report = {"checkpoint": a.checkpoint, "step": loaded["step"], "files": {}}
    exit_code = 0
    for g in a.gguf:
        path = Path(g)
        print(f"\n=== {path.name}")
        header = read_header(path)
        tcheck = check_tensors(header, cfg)
        md = header["metadata"]
        print(f"GGUF v{header['gguf_version']}, {header['n_tensors']} tensors, types {tcheck['types']}, norms {tcheck['norm_tensor_types']}")

        tok_results = []
        for s in TOKENIZER_CASES:
            ll = llama_tokenize(bin_dir, path, s)
            py = tok.encode(s)
            tok_results.append({"text": s, "match": ll == py, "llama": ll if ll != py else None, "python": py if ll != py else None})
        tok_ok = all(r["match"] for r in tok_results)
        print(f"tokenizer parity: {sum(r['match'] for r in tok_results)}/{len(tok_results)}")

        gens = []
        for ref in refs:
            text, secs = llama_generate(bin_dir, path, ref["prompt"], a.max_new_tokens)
            g_ids = tok.encode(text)
            t_ids = tok.encode(ref["torch_text"])
            prefix = 0
            for x, y in zip(g_ids, t_ids):
                if x != y:
                    break
                prefix += 1
            gens.append({
                "user": ref["user"], "prompt": ref["prompt"], "messages": ref["messages"],
                "torch": ref["torch_text"], "gguf": text,
                "exact_match": text == ref["torch_text"],
                "common_prefix_tokens": prefix, "torch_tokens": len(t_ids), "gguf_tokens": len(g_ids),
                "seconds": round(secs, 3),
            })
            print(f"  [{'=' if text == ref['torch_text'] else '~'}] {ref['user']!r}\n      torch: {ref['torch_text']!r}\n      gguf : {text!r}")

        exact = sum(x["exact_match"] for x in gens)
        report["files"][path.name] = {
            "bytes": path.stat().st_size,
            "header": {k: header[k] for k in ["gguf_version", "alignment", "n_tensors", "n_kv", "n_vocab_tokens", "n_merges"]},
            "metadata": {k: v for k, v in md.items() if not k.startswith("tokenizer.ggml.") or k in (
                "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                "tokenizer.ggml.eot_token_id", "tokenizer.ggml.padding_token_id", "tokenizer.ggml.unknown_token_id",
                "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token")},
            "tensor_check": tcheck,
            "tokenizer_parity": {"passed": tok_ok, "cases": tok_results},
            "generation_parity": {"exact_matches": exact, "total": len(gens), "cases": gens},
        }
        if tcheck["missing"] or not tcheck["shapes_ok"]:
            exit_code = 1
        if not tok_ok and any(t in path.name for t in ("F32", "F16")):
            exit_code = 1

    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nReport written to {a.report}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
