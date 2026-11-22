"""Render the Ollama Modelfile for a release profile.

The Modelfile is generated, never edited by hand: its system prompt and
decoding parameters come from the profile and its inference YAML, so the
Python engine, llama.cpp and Ollama cannot drift apart. The validator
(``hibari.release.validation``) fails if a Modelfile on disk differs from
what this function renders.

Ollama differences that cannot be configured away are listed in the header
of the generated file (no safety layer, no task router, no candidate
reranking; repeat penalty also counts prompt tokens).
"""

from __future__ import annotations

from hibari.data.tokenizer import SPECIAL_TOKENS
from hibari.release.profile import ModelProfile

# Matches HibariTokenizer.apply_chat_template exactly:
#   <|system|>\n{system}<|eos|>\n<|user|>\n{user}<|eos|>\n<|assistant|>\n
OLLAMA_TEMPLATE = '''{{- if .System }}<|system|>
{{ .System }}<|eos|>
{{ end }}
{{- range .Messages }}
{{- if eq .Role "user" }}<|user|>
{{ .Content }}<|eos|>
{{ else if eq .Role "assistant" }}<|assistant|>
{{ .Content }}<|eos|>
{{ end }}
{{- end }}<|assistant|>
'''

STOP_TOKENS = ["<|eos|>", "<|user|>", "<|system|>", "<|assistant|>"]
assert all(t in SPECIAL_TOKENS for t in STOP_TOKENS)


def _num(v: float) -> str:
    return f"{v:g}"


def render_modelfile(profile: ModelProfile, gguf_file: str) -> str:
    inf = profile.inference
    tag = f"hibari-{profile.model}:{profile.version}"
    lines = [
        f"# {profile.identity.name} {profile.version} — Ollama Modelfile (GENERATED; do not edit by hand)",
        f"# Generated from {profile.source.name if profile.source else 'profile'} by hibari.release.modelfile.",
        f"# Weights: {profile.weights_version} ({profile.checkpoint}); checkpoint sha256 {profile.checkpoint_sha256 or 'unknown'}",
        "#",
        "# Usage (from this directory):",
        f"#   ollama create {tag} -f Modelfile",
        f"#   ollama run {tag}",
        "#",
        "# Differences from the Python engine (hibari.inference.HibariChatSession):",
        "#   * no safety layer, no rewriting-task router, no informal-text normalisation,",
        "#     no candidate reranking - Ollama returns the raw model's reply;",
        "#   * repeat_penalty counts the last repeat_last_n tokens INCLUDING the prompt",
        "#     (Python counts only generated tokens).",
        f"FROM ./{gguf_file}",
        "",
        'TEMPLATE """' + OLLAMA_TEMPLATE + '"""',
        "",
        f'SYSTEM """{profile.system_prompt}"""',
        "",
    ]
    lines += [f'PARAMETER stop "{t}"' for t in STOP_TOKENS]
    lines += [
        f"PARAMETER temperature {_num(inf.temperature)}",
        f"PARAMETER top_k {inf.top_k}",
        f"PARAMETER top_p {_num(inf.top_p)}",
        f"PARAMETER repeat_penalty {_num(inf.repetition_penalty)}",
    ]
    if inf.repeat_last_n is not None:
        lines.append(f"PARAMETER repeat_last_n {inf.repeat_last_n}")
    lines.append(f"PARAMETER num_ctx {inf.num_ctx or 256}")
    lines.append(f"PARAMETER num_predict {inf.max_new_tokens}")
    return "\n".join(lines) + "\n"
