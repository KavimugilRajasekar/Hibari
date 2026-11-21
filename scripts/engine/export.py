"""Export a Hibari checkpoint to GGUF (F32, F16, Q8_0, Q4_K_M).

F32 and F16 are written directly by ``hibari.export.gguf_export``. Q8_0 and
Q4_K_M are produced from the F32 file by llama.cpp's ``llama-quantize``
(the reference implementation of those formats); pass its path with
``--llama-quantize`` or put it on PATH. Variants whose tool is unavailable
are skipped and reported, never faked.

Usage:
    python hibari/scripts/export.py \
        --checkpoint models/Koto/checkpoints/koto_start/best_model.pt \
        --outdir models/Koto/releases/exported --variants f32 f16 q8_0 q4_k_m \
        --llama-quantize /path/to/llama.cpp/build/bin/llama-quantize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from hibari.data.tokenizer import HibariTokenizer
from hibari.export.gguf_export import export_to_gguf
from hibari.training.checkpoint import load_checkpoint

FILENAMES = {
    "f32": "Hibari-Koto-F32.gguf",
    "f16": "Hibari-Koto-F16.gguf",
    "q8_0": "Hibari-Koto-Q8_0.gguf",
    "q4_k_m": "Hibari-Koto-Q4_K_M.gguf",
}
QUANTIZE_TYPES = {"q8_0": "Q8_0", "q4_k_m": "Q4_K_M"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(
    checkpoint_path: str,
    tokenizer_dir: str,
    outdir: str,
    variants: list[str],
    llama_quantize: str | None,
    version: str,
) -> dict:
    loaded = load_checkpoint(checkpoint_path, device="cpu")
    model, config = loaded["model"], loaded["config"]
    config.model_name = "Hibari-Koto"
    tokenizer = HibariTokenizer.load(tokenizer_dir)
    n_params = model.count_parameters()["total"]
    print(f"Loaded {checkpoint_path}: step {loaded['step']}, {n_params:,} params")

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {"checkpoint": str(checkpoint_path), "step": loaded["step"], "parameters": n_params, "files": {}}

    f32_path = out / FILENAMES["f32"]
    need_f32 = "f32" in variants or any(v in QUANTIZE_TYPES for v in variants)
    if need_f32:
        export_to_gguf(model, tokenizer, f32_path, "f32", version=version)
    if "f16" in variants:
        export_to_gguf(model, tokenizer, out / FILENAMES["f16"], "f16", version=version)

    quant_bin = llama_quantize or shutil.which("llama-quantize")
    for v in variants:
        if v not in QUANTIZE_TYPES:
            continue
        if not quant_bin:
            results["files"][v] = {"status": "skipped", "reason": "llama-quantize not found"}
            print(f"[{v}] skipped: llama-quantize not found")
            continue
        dst = out / FILENAMES[v]
        cmd = [quant_bin, str(f32_path), str(dst), QUANTIZE_TYPES[v]]
        print("$", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            results["files"][v] = {"status": "failed", "stderr": proc.stderr[-2000:]}
            print(proc.stderr[-2000:])

    for v in variants:
        p = out / FILENAMES[v]
        if p.exists() and results["files"].get(v, {}).get("status") not in ("failed", "skipped"):
            results["files"][v] = {
                "status": "ok", "path": str(p), "bytes": p.stat().st_size,
                "mib": round(p.stat().st_size / 2**20, 2), "sha256": sha256(p),
            }
            print(f"[{v}] {p} {results['files'][v]['mib']} MiB")

    if "f32" not in variants and f32_path.exists():
        f32_path.unlink()

    with open(out / "export_manifest.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    ap = argparse.ArgumentParser(description="Export Hibari Koto to GGUF")
    ap.add_argument("--checkpoint", default="models/Koto/checkpoints/koto_start/best_model.pt")
    ap.add_argument("--tokenizer-dir", default="models/Koto/tokenizer")
    ap.add_argument("--outdir", default="models/Koto/releases/exported")
    ap.add_argument("--variants", nargs="+", default=["f32", "f16", "q8_0", "q4_k_m"], choices=list(FILENAMES))
    ap.add_argument("--llama-quantize", default=None, help="Path to llama.cpp llama-quantize binary")
    ap.add_argument("--version", default="2.0")
    a = ap.parse_args()
    export(a.checkpoint, a.tokenizer_dir, a.outdir, a.variants, a.llama_quantize, a.version)


if __name__ == "__main__":
    main()
