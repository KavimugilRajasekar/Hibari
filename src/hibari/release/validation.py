"""Validate a release profile against the files it points to.

``validate_profile`` returns a list of :class:`Finding` (``error`` /
``warning`` / ``ok``). A release is consistent only when there are no
errors. Checks:

* version strings (semver) and training status;
* model config, checkpoint, tokenizer and inference config exist;
* checkpoint architecture == model config; tokenizer vocab == model vocab;
* checkpoint / tokenizer SHA-256 == the values recorded in the profile;
* inference settings are in range and fit the context window;
* the system prompt equals the one the weights were trained with (warning);
* every GGUF exists, carries Hibari provenance metadata that matches the
  profile (version, checkpoint and tokenizer hashes), has the same vocabulary
  and dimensions, and its embedded chat template uses the profile's system
  prompt;
* the Modelfile next to the GGUFs is exactly what ``render_modelfile``
  produces.

The checks read files only; nothing is modified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from hibari.release.modelfile import render_modelfile
from hibari.release.profile import SEMVER, TRAINING_STATUSES, ModelProfile


@dataclass
class Finding:
    level: str      # "error" | "warning" | "ok"
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper():7}] {self.code}: {self.message}"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_sha256(tokenizer_dir: str | Path) -> str:
    """Hash of ``tokenizer.json`` (the file that defines vocabulary and merges)."""
    return sha256_file(Path(tokenizer_dir) / "tokenizer.json")


def errors(findings: List[Finding]) -> List[Finding]:
    return [f for f in findings if f.level == "error"]


def validate_profile(profile: ModelProfile, check_hashes: bool = True, check_gguf: bool = True,
                     modelfile_variant: Optional[str] = None) -> List[Finding]:
    F: List[Finding] = []

    def err(code, msg):
        F.append(Finding("error", code, msg))

    def warn(code, msg):
        F.append(Finding("warning", code, msg))

    def ok(code, msg):
        F.append(Finding("ok", code, msg))

    for key in ("version", "weights_version"):
        v = getattr(profile, key)
        (ok if SEMVER.match(v) else err)(f"semver:{key}", f"{key} = {v!r}")
    if profile.training_status not in TRAINING_STATUSES:
        err("training_status", f"{profile.training_status!r} not in {TRAINING_STATUSES}")
    if not profile.system_prompt.strip():
        err("system_prompt", "empty system prompt")
    if profile.training_system_prompt and profile.training_system_prompt.strip() != profile.system_prompt.strip():
        warn("system_prompt:training_mismatch",
             "the system prompt differs from the one the weights were trained with; behaviour may not follow it")

    paths = {"model_config": profile.model_config, "checkpoint": profile.checkpoint,
             "tokenizer": Path(profile.tokenizer_dir) / "tokenizer.json", "inference_config": profile.inference_config}
    present = {}
    for name, p in paths.items():
        rp = profile.resolve(p)
        present[name] = rp.exists()
        (ok if rp.exists() else err)(f"exists:{name}", str(rp))
    if not all(present.values()):
        return F

    # ── configuration consistency ──
    from hibari.data.tokenizer import HibariTokenizer
    from hibari.model.config import ModelConfig

    cfg = ModelConfig.from_yaml(profile.resolve(profile.model_config))
    tok = HibariTokenizer.load(profile.resolve(profile.tokenizer_dir))
    (ok if tok.vocab_size == cfg.vocab_size else err)("vocab", f"tokenizer {tok.vocab_size} vs model config {cfg.vocab_size}")

    import torch
    ck = torch.load(profile.resolve(profile.checkpoint), map_location="cpu", weights_only=False)
    ck_cfg = {k: v for k, v in (ck.get("config") or {}).items() if k != "model_name"}
    # A field added to ModelConfig after the checkpoint was saved (e.g. resid_dropout, engine 0.4.1)
    # was not in effect then: compare it as its default value.
    from dataclasses import fields as _fields
    for f in _fields(ModelConfig):
        ck_cfg.setdefault(f.name, f.default)
    want = {k: v for k, v in cfg.to_dict().items() if k != "model_name"}
    diff = {k: (ck_cfg.get(k), v) for k, v in want.items() if ck_cfg.get(k) != v}
    (ok if not diff else err)("checkpoint:config", "matches model config" if not diff else f"checkpoint vs config differ: {diff}")
    meta = ck.get("run_metadata") or {}
    run_ver = meta.get("model_version") or meta.get("koto_version")
    if run_ver and run_ver != profile.weights_version:
        err("checkpoint:weights_version", f"checkpoint records version {run_ver}, profile says weights {profile.weights_version}")
    step = ck.get("step")
    del ck

    try:
        inf = profile.inference
        problems = []
        if inf.temperature < 0:
            problems.append("temperature < 0")
        if not 0 < inf.top_p <= 1:
            problems.append("top_p outside (0, 1]")
        if inf.top_k < 0:
            problems.append("top_k < 0")
        if inf.repetition_penalty < 1:
            problems.append("repetition_penalty < 1")
        if inf.max_new_tokens >= cfg.max_seq_len:
            problems.append(f"max_new_tokens {inf.max_new_tokens} >= context {cfg.max_seq_len}")
        if inf.num_ctx and inf.num_ctx > cfg.max_seq_len:
            problems.append(f"num_ctx {inf.num_ctx} > trained context {cfg.max_seq_len}")
        if inf.num_candidates < 1:
            problems.append("num_candidates < 1")
        (ok if not problems else err)("inference:settings", "in range" if not problems else "; ".join(problems))
    except Exception as e:  # unknown keys, bad YAML
        err("inference:settings", str(e))

    if check_hashes:
        for name, recorded, actual_fn in (
            ("checkpoint", profile.checkpoint_sha256, lambda: sha256_file(profile.resolve(profile.checkpoint))),
            ("tokenizer", profile.tokenizer_sha256, lambda: tokenizer_sha256(profile.resolve(profile.tokenizer_dir))),
        ):
            if not recorded:
                err(f"sha256:{name}", "not recorded in the profile")
                continue
            actual = actual_fn()
            (ok if actual == recorded else err)(f"sha256:{name}", "matches" if actual == recorded else f"profile {recorded[:12]}…, file {actual[:12]}…")

    if check_gguf and profile.gguf_dir:
        F += _validate_gguf(profile, cfg, tok, step, modelfile_variant)
    return F


def _validate_gguf(profile: ModelProfile, cfg, tok, step, modelfile_variant) -> List[Finding]:
    import gguf

    from hibari.export.gguf_export import build_chat_template

    F: List[Finding] = []
    gdir = profile.resolve(profile.gguf_dir)
    if not profile.gguf_files:
        F.append(Finding("warning", "gguf:none", f"profile lists no GGUF files in {gdir}"))
    vocab = [t for t, _ in sorted(tok.get_vocab().items(), key=lambda x: x[1])]
    want_prov = profile.provenance(step)
    for g in profile.gguf_files:
        path = gdir / g.file
        code = f"gguf:{g.variant}"
        if not path.exists():
            F.append(Finding("error", code, f"missing {path}"))
            continue
        if g.sha256:
            actual = sha256_file(path)
            F.append(Finding("ok" if actual == g.sha256 else "error", f"{code}:sha256",
                             "matches" if actual == g.sha256 else f"profile {g.sha256[:12]}…, file {actual[:12]}…"))
        r = gguf.GGUFReader(str(path))

        def kv(name, default=None):
            return r.fields[name].contents() if name in r.fields else default

        missing = [k for k in ("hibari.model.version", "hibari.checkpoint.sha256", "hibari.tokenizer.sha256") if k not in r.fields]
        if missing:
            F.append(Finding("error", f"{code}:provenance", f"no Hibari provenance metadata ({missing}); cannot prove which weights it holds"))
        else:
            bad = {k: (kv(k), v) for k, v in want_prov.items()
                   if k in ("hibari.model.version", "hibari.weights.version", "hibari.checkpoint.sha256", "hibari.tokenizer.sha256")
                   and str(kv(k)) != str(v)}
            F.append(Finding("ok" if not bad else "error", f"{code}:provenance",
                             "matches profile" if not bad else f"GGUF vs profile: {bad}"))
        dims = {"llama.context_length": cfg.max_seq_len, "llama.embedding_length": cfg.d_model, "llama.block_count": cfg.n_layers,
                "llama.attention.head_count": cfg.n_heads, "llama.feed_forward_length": cfg.d_ff}
        bad = {k: (kv(k), v) for k, v in dims.items() if kv(k) is not None and int(kv(k)) != v}
        F.append(Finding("ok" if not bad else "error", f"{code}:dims", "match model config" if not bad else str(bad)))
        f_tok = r.fields.get("tokenizer.ggml.tokens")
        g_vocab = [bytes(f_tok.parts[i]).decode("utf-8") for i in f_tok.data] if f_tok else []
        F.append(Finding("ok" if g_vocab == vocab else "error", f"{code}:vocab",
                         "identical to tokenizer" if g_vocab == vocab else f"{len(g_vocab)} tokens differ from tokenizer ({len(vocab)})"))
        tmpl = kv("tokenizer.chat_template", "")
        F.append(Finding("ok" if tmpl == build_chat_template(profile.system_prompt) else "error", f"{code}:chat_template",
                         "uses the profile system prompt" if tmpl == build_chat_template(profile.system_prompt)
                         else "embedded chat template differs from the profile's"))

    mf = gdir / "Modelfile"
    if profile.gguf_files:
        variant = modelfile_variant or profile.gguf_files[0].variant
        gfile = next((g.file for g in profile.gguf_files if g.variant == variant), None)
        if gfile is None:
            F.append(Finding("error", "modelfile", f"variant {variant!r} is not one of the profile's GGUF files "
                                                   f"{[g.variant for g in profile.gguf_files]}"))
        elif not mf.exists():
            F.append(Finding("error", "modelfile", f"missing {mf}"))
        else:
            same = mf.read_text(encoding="utf-8") == render_modelfile(profile, gfile)
            F.append(Finding("ok" if same else "error", "modelfile",
                             "identical to the rendered profile Modelfile" if same
                             else "differs from render_modelfile(profile); regenerate it"))
    manifest = gdir / "export_manifest.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
            if m.get("model_version") not in (None, profile.version):
                F.append(Finding("error", "manifest", f"manifest version {m.get('model_version')} != {profile.version}"))
        except json.JSONDecodeError as e:
            F.append(Finding("error", "manifest", f"invalid JSON: {e}"))
    return F
