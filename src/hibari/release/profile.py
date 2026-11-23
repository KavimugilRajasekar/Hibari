"""Model release profiles: one file that says exactly what a model version is.

A profile (YAML, owned by the *model* repository, e.g. Koto
``configs/release/koto_1.1.1.yaml``) binds together

* the model version and the weights version it ships,
* identity (name, architecture, creator) and the system prompt,
* the model configuration, the checkpoint and the tokenizer — each with its
  SHA-256, so a GGUF can never be associated with the wrong weights,
* the inference settings (one inference YAML, referenced by path, is the
  single source of decoding defaults for Python, llama.cpp and Ollama),
* the GGUF directory and the expected files.

The engine reads profiles (``ModelProfile.load``), validates them
(``hibari.release.validation``), exports from them (``scripts/engine/export.py
--profile``) and renders the Ollama Modelfile from them
(``hibari.release.modelfile``). Nothing model-specific is hard-coded in the
engine.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$")
TRAINING_STATUSES = ("NOT_STARTED", "NOT_REQUIRED", "MANUALLY_AUTHORIZED", "COMPLETED")


@dataclass
class InferenceSettings:
    """Decoding defaults. Field names match ``HibariChatSession.reply`` / the session constructor."""

    max_new_tokens: int = 80
    temperature: float = 0.5
    top_k: int = 25
    top_p: float = 0.85
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 0
    # application layer (Python only)
    enable_safety: bool = True
    route_tasks: bool = True
    normalize_input: bool = False
    num_candidates: int = 1
    intent_candidates: Dict[str, int] = field(default_factory=dict)
    candidate_sampling: Dict[str, float] = field(default_factory=dict)
    # Ollama / llama.cpp only
    num_ctx: Optional[int] = None
    repeat_last_n: Optional[int] = None

    REPLY_KEYS = ("max_new_tokens", "temperature", "top_k", "top_p", "repetition_penalty", "no_repeat_ngram_size")
    SESSION_KEYS = ("enable_safety", "route_tasks", "normalize_input", "num_candidates", "intent_candidates", "candidate_sampling")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InferenceSettings":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(d) - known - {"model_checkpoint", "tokenizer_dir", "device"})
        if unknown:
            raise ValueError(f"Unknown inference settings: {unknown}")
        return cls(**{k: v for k, v in d.items() if k in known})

    def reply_kwargs(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.REPLY_KEYS}

    def session_kwargs(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.SESSION_KEYS}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Identity:
    name: str                       # what the model calls itself
    architecture: str = "Hibari"
    creator: str = ""
    description: str = ""
    terms: List[str] = field(default_factory=list)   # words that may appear in identity answers


@dataclass
class GGUFFile:
    variant: str
    file: str
    sha256: Optional[str] = None


@dataclass
class ModelProfile:
    model: str                       # e.g. "koto"
    version: str                     # release version (semver)
    weights_version: str             # version of the trained weights this release ships
    identity: Identity
    system_prompt: str
    model_config: str
    checkpoint: str
    tokenizer_dir: str
    inference_config: str
    checkpoint_sha256: Optional[str] = None
    tokenizer_sha256: Optional[str] = None
    training_status: str = "NOT_STARTED"
    training_system_prompt: Optional[str] = None
    gguf_dir: Optional[str] = None
    gguf_files: List[GGUFFile] = field(default_factory=list)
    experiment: Optional[str] = None
    notes: str = ""
    ollama_name: Optional[str] = None   # published Ollama name, e.g. "user/Model1.0.0:7M"; default hibari-<model>:<version>
    root: Path = field(default_factory=Path.cwd, repr=False)
    source: Optional[Path] = field(default=None, repr=False)
    _inference: Optional[InferenceSettings] = field(default=None, repr=False)

    # ───────────── loading ─────────────

    @classmethod
    def load(cls, path: str | Path, root: Optional[str | Path] = None) -> "ModelProfile":
        """Load a profile; relative paths inside it resolve against ``root``
        (default: the profile's ``root`` key, else two levels above the file,
        i.e. the model repository root for ``<repo>/configs/release/x.yaml``)."""
        path = Path(path).resolve()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        base = Path(root) if root else (path.parent / data["root"]).resolve() if "root" in data else path.parents[2]
        ident = data.get("identity") or {}
        gguf = data.get("gguf") or {}
        files = [GGUFFile(variant=v, **(spec if isinstance(spec, dict) else {"file": spec}))
                 for v, spec in (gguf.get("files") or {}).items()]
        required = ["model", "version", "weights_version", "system_prompt", "model_config", "checkpoint",
                    "tokenizer_dir", "inference_config"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"{path}: missing required profile keys {missing}")
        return cls(
            model=data["model"], version=str(data["version"]), weights_version=str(data["weights_version"]),
            identity=Identity(**ident) if ident else Identity(name=data["model"]),
            system_prompt=data["system_prompt"].strip(), model_config=data["model_config"],
            checkpoint=data["checkpoint"], tokenizer_dir=data["tokenizer_dir"], inference_config=data["inference_config"],
            checkpoint_sha256=data.get("checkpoint_sha256"), tokenizer_sha256=data.get("tokenizer_sha256"),
            training_status=data.get("training_status", "NOT_STARTED"),
            training_system_prompt=(data.get("training_system_prompt") or None),
            gguf_dir=gguf.get("dir"), gguf_files=files, experiment=data.get("experiment"), notes=data.get("notes", ""), ollama_name=data.get("ollama_name"),
            root=base, source=path,
        )

    def resolve(self, p: str | Path) -> Path:
        p = Path(p).expanduser()
        return p if p.is_absolute() else (self.root / p)

    def relpath(self, p: str | Path) -> str:
        """``p`` relative to the model repository root when it lies inside it (for reports and
        manifests that may be published: no absolute local paths)."""
        try:
            return str(Path(p).resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(p)

    @property
    def inference(self) -> InferenceSettings:
        if self._inference is None:
            data = yaml.safe_load(self.resolve(self.inference_config).read_text(encoding="utf-8")) or {}
            self._inference = InferenceSettings.from_dict(data)
        return self._inference

    @property
    def tag(self) -> str:
        """Short identifier, e.g. ``koto-1.1.1``."""
        return f"{self.model}-{self.version}"

    def gguf_path(self, variant: str) -> Path:
        for f in self.gguf_files:
            if f.variant == variant:
                return self.resolve(Path(self.gguf_dir or ".") / f.file)
        raise KeyError(f"Profile {self.tag} has no GGUF variant {variant!r}")

    def provenance(self, step: Optional[int] = None) -> Dict[str, Any]:
        """Metadata written into every GGUF exported from this profile."""
        from hibari import __version__ as engine_version

        out: Dict[str, Any] = {
            "hibari.model.name": self.model,
            "hibari.model.version": self.version,
            "hibari.weights.version": self.weights_version,
            "hibari.checkpoint.sha256": self.checkpoint_sha256 or "",
            "hibari.tokenizer.sha256": self.tokenizer_sha256 or "",
            "hibari.engine.version": engine_version,
            "hibari.identity.name": self.identity.name,
        }
        if step is not None:
            out["hibari.checkpoint.step"] = int(step)
        return out
