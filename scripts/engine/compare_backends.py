"""Compare a release across PyTorch, llama.cpp and Ollama (raw model, no app layer).

For each GGUF variant of the profile:

1. **Tokenizer parity** — ``llama-tokenize`` IDs vs the Python tokenizer on
   edge-case strings and on every rendered chat prompt.
2. **Chat-template parity (Ollama)** — the number of prompt tokens Ollama
   evaluates for ``/api/chat`` (its Modelfile TEMPLATE) vs the Python
   ``apply_chat_template`` token count.
3. **Greedy parity** — PyTorch greedy (no repetition penalty) vs
   ``llama-completion --temp 0`` on the Python-rendered prompt, and vs
   Ollama ``/api/chat`` (template rendered by Ollama) and ``/api/generate``
   (``raw``: the Python-rendered prompt) with temperature 0.
4. **Default sampling (Ollama)** — the reply with the Modelfile's own
   parameters, for human review only (sampling is not expected to match).

    LD_LIBRARY_PATH=~/llama.cpp python scripts/engine/compare_backends.py \\
        --profile models/Koto/configs/release/koto_1.1.1.yaml --llama-bin ~/llama.cpp \\
        --ollama-host http://127.0.0.1:11434 --report models/Koto/outputs/evaluation/backend_parity_1.1.1.json

Ollama models are created as ``hibari-<model>-parity:<variant>`` from the
generated Modelfile; pass ``--no-ollama`` to skip Ollama.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_gguf import TOKENIZER_CASES  # noqa: E402

from hibari.data.tokenizer import HibariTokenizer  # noqa: E402
from hibari.inference.generation import generate_tokens  # noqa: E402
from hibari.release import ModelProfile, render_modelfile  # noqa: E402
from hibari.training.checkpoint import load_checkpoint  # noqa: E402

CASES: List[List[Dict[str, str]]] = [
    [{"role": "user", "content": "Hi"}],
    [{"role": "user", "content": "ur name?"}],
    [{"role": "user", "content": "wat r u doing?"}],
    [{"role": "user", "content": "My grandmother passed away last week."}],
    [{"role": "user", "content": "I'm so frustrated with my project."}],
    [{"role": "user", "content": "You're so pathetic!"}],
    [{"role": "user", "content": "make this more professional: send me the documents"}],
    [{"role": "user", "content": "Task: Correct grammar\nInput: i has a interview tomorrow and im nervous"}],
    [{"role": "user", "content": "Write an email to my manager asking for leave on Friday."}],
    [{"role": "user", "content": "I have an interview tomorrow."},
     {"role": "assistant", "content": "That's exciting! Are you feeling prepared?"},
     {"role": "user", "content": "Not really. I'm nervous."}],
    [{"role": "user", "content": "Hey, my name is Tanvir."},
     {"role": "assistant", "content": "Nice to meet you, Tanvir!"},
     {"role": "user", "content": "What is my name?"}],
]


def http_json(url: str, payload: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def llama_env(bin_dir: Path) -> dict:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def llama_tokenize(bin_dir: Path, model: Path, text: str) -> List[int]:
    out = subprocess.run([str(bin_dir / "llama-tokenize"), "-m", str(model), "-p", text, "--ids", "--log-disable"],
                         capture_output=True, text=True, check=True, env=llama_env(bin_dir)).stdout.strip().splitlines()[-1]
    return json.loads(out)


def trim_to_tokens(text: str, tok, n: int) -> str:
    """First ``n`` tokens of ``text`` (re-tokenised), so both backends are compared at the same length."""
    ids = tok.encode(text)
    return text.strip() if len(ids) <= n else tok.decode(ids[:n], skip_special_tokens=True).strip()


def llama_greedy(bin_dir: Path, model: Path, prompt: str, n: int) -> str:
    """Greedy continuation of at most ``n`` tokens.

    ``llama-completion -n k`` prints k-1 tokens (measured 2026-11-23 for k = 5, 10, 20, 64 with the
    1.1.1 and 1.3.0 GGUFs), so k = n + 1 is requested; callers trim with ``trim_to_tokens``.
    Before this fix, replies that reached the cap were reported as mismatches.
    """
    n = n + 1
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(prompt)
        pfile = f.name
    try:
        proc = subprocess.run(
            [str(bin_dir / "llama-completion"), "-m", str(model), "-f", pfile, "-n", str(n), "--temp", "0",
             "--repeat-penalty", "1.0", "-c", "256", "-no-cnv", "--no-display-prompt", "--simple-io", "--no-warmup", "-t", "4"],
            capture_output=True, text=True, env=llama_env(bin_dir))
    finally:
        os.unlink(pfile)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-completion failed ({proc.returncode}): {proc.stderr[-600:]}")
    return proc.stdout.replace("[end of text]", "").strip()


def ollama_create(host: str, name: str, gguf_dir: Path, modelfile: str) -> None:
    mf = gguf_dir / f".Modelfile.{name.replace(':', '_').replace('/', '_')}"
    mf.write_text(modelfile, encoding="utf-8")
    try:
        env = dict(os.environ, OLLAMA_HOST=host.replace("http://", ""))
        proc = subprocess.run(["ollama", "create", name, "-f", str(mf)], cwd=str(gguf_dir), capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"ollama create failed: {proc.stderr[-600:]}")
    finally:
        mf.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--variants", nargs="*", default=None, help="default: every GGUF in the profile")
    ap.add_argument("--llama-bin", default=None, help="directory with llama-tokenize and llama-completion")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--no-ollama", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    prof = ModelProfile.load(a.profile)
    tok = HibariTokenizer.load(prof.resolve(prof.tokenizer_dir))
    model = load_checkpoint(prof.resolve(prof.checkpoint))["model"].eval()
    system = {"role": "system", "content": prof.system_prompt}

    refs = []
    for msgs in CASES:
        prompt = tok.apply_chat_template([system] + msgs, add_generation_prompt=True)
        ids = generate_tokens(model, tok, prompt, max_new_tokens=a.max_new_tokens, temperature=0.0, repetition_penalty=1.0)
        refs.append({"messages": msgs, "prompt": prompt, "prompt_tokens": len(tok.encode(prompt)),
                     "torch": tok.decode(ids, skip_special_tokens=True).strip()})

    report = {"profile": prof.relpath(prof.source), "model_version": prof.version, "weights_version": prof.weights_version,
              "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "max_new_tokens": a.max_new_tokens, "variants": {}}
    variants = a.variants or [g.variant for g in prof.gguf_files]
    for v in variants:
        path = prof.gguf_path(v)
        gfile = path.name
        print(f"\n=== {v}: {gfile}")
        vr: Dict = {"file": gfile, "cases": []}
        if a.llama_bin:
            bin_dir = Path(a.llama_bin).expanduser()
            texts = TOKENIZER_CASES + [r["prompt"] for r in refs]
            mism = [t for t in texts if llama_tokenize(bin_dir, path, t) != tok.encode(t)]
            vr["tokenizer_parity"] = {"cases": len(texts), "mismatches": mism}
            print(f"tokenizer parity (llama.cpp): {len(texts) - len(mism)}/{len(texts)}")
        oname = None
        if not a.no_ollama:
            oname = f"hibari-{prof.model}-parity:{prof.version}-{v}"
            ollama_create(a.ollama_host, oname, path.parent, render_modelfile(prof, gfile))
        for ref in refs:
            row = {"user": ref["messages"][-1]["content"], "turns": len(ref["messages"]), "torch": ref["torch"]}
            if a.llama_bin:
                row["llama_cpp"] = trim_to_tokens(
                    llama_greedy(Path(a.llama_bin).expanduser(), path, ref["prompt"], a.max_new_tokens), tok, a.max_new_tokens)
            if oname:
                greedy = {"temperature": 0, "repeat_penalty": 1.0, "num_predict": a.max_new_tokens, "seed": 0}
                chat = http_json(f"{a.ollama_host}/api/chat", {"model": oname, "stream": False, "messages": ref["messages"], "options": greedy})
                raw = http_json(f"{a.ollama_host}/api/generate", {"model": oname, "stream": False, "raw": True, "prompt": ref["prompt"], "options": greedy})
                dflt = http_json(f"{a.ollama_host}/api/chat", {"model": oname, "stream": False, "messages": ref["messages"], "options": {"seed": 0}})
                row["ollama_chat"] = chat["message"]["content"].strip()
                row["ollama_raw"] = raw["response"].strip()
                row["ollama_prompt_tokens"] = chat.get("prompt_eval_count")
                row["python_prompt_tokens"] = ref["prompt_tokens"]
                row["ollama_default_sampling"] = dflt["message"]["content"].strip()
            for k in ("llama_cpp", "ollama_chat", "ollama_raw"):
                if k in row:
                    row[f"{k}_matches_torch"] = row[k] == ref["torch"]
            if "ollama_prompt_tokens" in row:
                row["template_parity"] = row["ollama_prompt_tokens"] == row["python_prompt_tokens"]
            vr["cases"].append(row)
            flags = " ".join(f"{k.replace('_matches_torch', '')}={'=' if row[k] else 'DIFF'}" for k in row if k.endswith("_matches_torch"))
            print(f"  {row['user'][:50]!r:54} {flags} {'tmpl=' + ('ok' if row.get('template_parity') else 'DIFF') if 'template_parity' in row else ''}")
        n = len(vr["cases"])
        vr["summary"] = {k: f"{sum(c.get(k, False) for c in vr['cases'])}/{n}" for k in
                         ("llama_cpp_matches_torch", "ollama_chat_matches_torch", "ollama_raw_matches_torch", "template_parity")
                         if any(k in c for c in vr["cases"])}
        print("  summary:", vr["summary"])
        report["variants"][v] = vr

    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport written to {a.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
