"""Export a Hibari checkpoint to GGUF (F32, F16, Q8_0, Q4_K_M).

Preferred: export from a release profile (``hibari.release.ModelProfile``).
The checkpoint and tokenizer hashes are checked against the profile first,
every GGUF gets ``hibari.*`` provenance metadata (model / weights version,
checkpoint and tokenizer SHA-256), the Ollama Modelfile is generated from the
profile, and an existing file is never replaced by different content unless
``--force`` is given::

    python scripts/engine/export.py --profile models/Koto/configs/release/koto_1.1.1.yaml \\
        --llama-quantize ~/llama.cpp/llama-quantize --record

Legacy (no profile, no provenance)::

    python scripts/engine/export.py --checkpoint <ckpt> --outdir <dir> --version 1.1.0

F32 and F16 are written by ``hibari.export.gguf_export``; Q8_0 and Q4_K_M are
produced from the F32 file by llama.cpp's ``llama-quantize``. Variants whose
tool is unavailable are skipped and reported, never faked.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from hibari.data.tokenizer import HibariTokenizer
from hibari.export.gguf_export import export_to_gguf
from hibari.release import ModelProfile, render_modelfile, sha256_file, tokenizer_sha256
from hibari.training.checkpoint import load_checkpoint

LEGACY_FILENAMES = {
    "f32": "Hibari-Koto-F32.gguf",
    "f16": "Hibari-Koto-F16.gguf",
    "q8_0": "Hibari-Koto-Q8_0.gguf",
    "q4_k_m": "Hibari-Koto-Q4_K_M.gguf",
}
QUANTIZE_TYPES = {"q8_0": "Q8_0", "q4_k_m": "Q4_K_M"}
sha256 = sha256_file  # backwards-compatible name


class ExportError(RuntimeError):
    pass


def _place(tmp: Path, dst: Path, force: bool) -> str:
    """Move ``tmp`` to ``dst``. Identical content: keep the existing file.
    Different content: refuse unless ``force``. Returns what happened."""
    if dst.exists():
        if sha256_file(dst) == sha256_file(tmp):
            tmp.unlink()
            return "unchanged"
        if not force:
            tmp.unlink()
            raise ExportError(f"{dst} exists with different content; refusing to replace it (use --force)")
        tmp.replace(dst)
        return "replaced"
    tmp.replace(dst)
    return "created"


def export(
    checkpoint_path: str,
    tokenizer_dir: str,
    outdir: str,
    variants: List[str],
    llama_quantize: Optional[str],
    version: str,
    filenames: Optional[Dict[str, str]] = None,
    system_prompt: Optional[str] = None,
    provenance: Optional[Dict] = None,
    force: bool = True,
    manifest_extra: Optional[Dict] = None,
) -> dict:
    filenames = filenames or LEGACY_FILENAMES
    loaded = load_checkpoint(checkpoint_path, device="cpu")
    model, config = loaded["model"], loaded["config"]
    config.model_name = "Hibari-Koto"
    tokenizer = HibariTokenizer.load(tokenizer_dir)
    n_params = model.count_parameters()["total"]
    print(f"Loaded {checkpoint_path}: step {loaded['step']}, {n_params:,} params")
    prov = dict(provenance or {})
    if prov:
        prov["hibari.checkpoint.step"] = int(loaded["step"])

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {"checkpoint": str(manifest_extra.get("checkpoint", checkpoint_path)) if manifest_extra else str(checkpoint_path), "step": loaded["step"], "parameters": n_params,
                     "created": time.strftime("%Y-%m-%d %H:%M:%S"), **(manifest_extra or {}), "files": {}}

    def write(kind: str, dst: Path) -> None:
        tmp = dst.with_name(dst.name + ".tmp")
        export_to_gguf(model, tokenizer, tmp, kind, version=version, system_prompt=system_prompt, provenance=prov)
        results["files"][kind] = {"action": _place(tmp, dst, force)}

    f32_path = out / filenames["f32"]
    need_f32 = "f32" in variants or any(v in QUANTIZE_TYPES for v in variants)
    if need_f32:
        write("f32", f32_path)
    if "f16" in variants:
        write("f16", out / filenames["f16"])

    quant_bin = llama_quantize or shutil.which("llama-quantize")
    for v in variants:
        if v not in QUANTIZE_TYPES:
            continue
        if not quant_bin:
            results["files"][v] = {"status": "skipped", "reason": "llama-quantize not found"}
            print(f"[{v}] skipped: llama-quantize not found")
            continue
        dst = out / filenames[v]
        tmp = dst.with_name(dst.name + ".tmp")
        cmd = [quant_bin, str(f32_path), str(tmp), QUANTIZE_TYPES[v]]
        print("$", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            results["files"][v] = {"status": "failed", "stderr": proc.stderr[-2000:]}
            print(proc.stderr[-2000:])
            tmp.unlink(missing_ok=True)
            continue
        results["files"][v] = {"action": _place(tmp, dst, force)}

    for v in variants:
        p = out / filenames[v]
        info = results["files"].get(v, {})
        if p.exists() and info.get("status") not in ("failed", "skipped"):
            info.update({"status": "ok", "file": p.name, "bytes": p.stat().st_size,
                         "mib": round(p.stat().st_size / 2**20, 2), "sha256": sha256_file(p)})
            results["files"][v] = info
            print(f"[{v}] {p} {info['mib']} MiB ({info.get('action')})")

    if "f32" not in variants and f32_path.exists() and results["files"].get("f32", {}).get("action") == "created":
        f32_path.unlink()
        results["files"].pop("f32", None)

    with open(out / "export_manifest.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def export_profile(profile: ModelProfile, variants: List[str], llama_quantize: Optional[str], force: bool) -> dict:
    ck, tk = profile.resolve(profile.checkpoint), profile.resolve(profile.tokenizer_dir)
    for name, recorded, actual in (("checkpoint", profile.checkpoint_sha256, sha256_file(ck)),
                                   ("tokenizer", profile.tokenizer_sha256, tokenizer_sha256(tk))):
        if not recorded:
            raise ExportError(f"profile {profile.tag} does not record the {name} sha256")
        if recorded != actual:
            raise ExportError(f"{name} sha256 {actual[:12]}… does not match the profile ({recorded[:12]}…); refusing to export")
    if not profile.gguf_dir:
        raise ExportError("profile has no gguf.dir")
    names = {g.variant: g.file for g in profile.gguf_files}
    unknown = [v for v in variants if v not in names]
    if unknown:
        raise ExportError(f"variants {unknown} are not listed in the profile")
    outdir = profile.resolve(profile.gguf_dir)
    res = export(str(ck), str(tk), str(outdir), variants, llama_quantize, profile.version,
                 filenames={**LEGACY_FILENAMES, **names}, system_prompt=profile.system_prompt,
                 provenance=profile.provenance(), force=force,
                 manifest_extra={"model": profile.model, "model_version": profile.version,
                                 "weights_version": profile.weights_version, "profile": profile.relpath(profile.source),
                                 "checkpoint": profile.relpath(ck),
                                 "checkpoint_sha256": profile.checkpoint_sha256,
                                 "tokenizer_sha256": profile.tokenizer_sha256})
    first = next((g for g in profile.gguf_files if g.variant in res["files"] and res["files"][g.variant].get("status") == "ok"), None)
    if first is None:
        detail = {v: res["files"].get(v, {}).get("reason") or res["files"].get(v, {}).get("status") for v in variants}
        raise ExportError(f"no GGUF variant was exported ({detail}); the Modelfile was not written")
    (outdir / "Modelfile").write_text(render_modelfile(profile, first.file), encoding="utf-8")
    print(f"Modelfile -> {outdir / 'Modelfile'} (FROM {first.file})")
    return res


def record_hashes(profile: ModelProfile, results: dict) -> None:
    """Write the exported files' sha256 into the profile YAML (only the gguf.files lines)."""
    text = profile.source.read_text(encoding="utf-8")
    for g in profile.gguf_files:
        info = results["files"].get(g.variant, {})
        if info.get("status") != "ok":
            continue
        pat = re.compile(rf"^(\s*{re.escape(g.variant)}:\s*)\{{[^}}\n]*\}}", re.M)
        text, n = pat.subn(lambda m: f"{m.group(1)}{{file: {g.file}, sha256: {info['sha256']}}}", text)
        if n != 1:
            raise ExportError(f"could not find a single '{g.variant}: {{...}}' line in {profile.source}")
    profile.source.write_text(text, encoding="utf-8")
    print(f"Recorded GGUF sha256 values in {profile.source}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export Hibari models to GGUF")
    ap.add_argument("--profile", default=None, help="Release profile YAML (recommended)")
    ap.add_argument("--checkpoint", default="models/Koto/checkpoints/koto_1.1.0/best_model.pt")
    ap.add_argument("--tokenizer-dir", default="models/Koto/tokenizer")
    ap.add_argument("--outdir", default=None, help="Legacy mode output directory (required without --profile)")
    ap.add_argument("--variants", nargs="+", default=["f32", "f16", "q8_0", "q4_k_m"], choices=list(LEGACY_FILENAMES))
    ap.add_argument("--llama-quantize", default=None, help="Path to llama.cpp llama-quantize binary")
    ap.add_argument("--version", default=None, help="Legacy mode: value of general.version")
    ap.add_argument("--force", action="store_true", help="Replace existing files whose content differs")
    ap.add_argument("--record", action="store_true", help="Profile mode: write the GGUF sha256 values into the profile")
    a = ap.parse_args()
    try:
        if a.profile:
            profile = ModelProfile.load(a.profile)
            res = export_profile(profile, a.variants, a.llama_quantize, a.force)
            if a.record:
                record_hashes(profile, res)
        else:
            if not a.outdir or not a.version:
                ap.error("without --profile, --outdir and --version are required")
            export(a.checkpoint, a.tokenizer_dir, a.outdir, a.variants, a.llama_quantize, a.version, force=a.force)
    except ExportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
