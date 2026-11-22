"""Packaging: every third-party module imported by src/hibari is a declared dependency.

Regression for 0.4.0: `tokenizers` and `gguf` were imported (`import hibari` needs both)
but missing from pyproject.toml, so a fresh `pip install -e .` produced a package that
could not be imported.
"""

import ast
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPORT_TO_DIST = {"yaml": "pyyaml", "torch": "torch", "numpy": "numpy", "tqdm": "tqdm",
                  "tokenizers": "tokenizers", "gguf": "gguf"}


def _third_party_imports():
    found = set()
    for f in (ROOT / "src" / "hibari").rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    std = set(sys.stdlib_module_names)
    return {m for m in found if m not in std and m != "hibari"}


def test_all_imports_are_declared():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    declared = {d.split(">")[0].split("=")[0].split("<")[0].strip().lower() for d in deps}
    imported = _third_party_imports()
    unknown = imported - set(IMPORT_TO_DIST)
    assert not unknown, f"new third-party imports, add them to IMPORT_TO_DIST and pyproject: {unknown}"
    missing = {IMPORT_TO_DIST[m] for m in imported} - declared
    assert not missing, f"imported but not declared in pyproject.toml: {missing}"


def test_package_imports_cleanly():
    import hibari
    assert hibari.__version__
    from hibari import HibariKotoLM, HibariTokenizer, InferenceEngine, ModelConfig, Trainer  # noqa: F401
