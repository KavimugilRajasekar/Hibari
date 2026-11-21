"""Verify Hibari GGUF files with llama-cpp-python (run in an env that has llama_cpp).

Reads the parity report written by verify_gguf.py (PyTorch reference outputs)
and checks, for each GGUF:
  * raw completion of the exact chat-formatted prompt (greedy) vs PyTorch,
  * create_chat_completion() using the chat template embedded in the GGUF,
  * a 3-turn conversation through create_chat_completion (no errors, non-empty).

Usage:
    python hibari/scripts/verify_llama_cpp_python.py --parity release/inference_parity.json \
        --gguf release/*.gguf --report release/llama_cpp_python_check.json
"""

import argparse
import json
from pathlib import Path

import llama_cpp
from llama_cpp import Llama


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", required=True)
    ap.add_argument("--gguf", nargs="+", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--max-tokens", type=int, default=48)
    a = ap.parse_args()
    parity = json.loads(Path(a.parity).read_text())
    out = {"llama_cpp_python_version": llama_cpp.__version__, "files": {}}

    for g in a.gguf:
        name = Path(g).name
        cases = next(iter(parity["files"].values()))["generation_parity"]["cases"]
        llm = Llama(model_path=g, n_ctx=256, n_threads=4, verbose=False, seed=0)
        rows = []
        for c in cases:
            raw = llm(c["prompt"], max_tokens=a.max_tokens, temperature=0.0, top_k=1, repeat_penalty=1.0)
            raw_text = raw["choices"][0]["text"].strip()
            chat = llm.create_chat_completion(messages=c["messages"], max_tokens=a.max_tokens,
                                              temperature=0.0, top_k=1, repeat_penalty=1.0)
            chat_text = chat["choices"][0]["message"]["content"].strip()
            rows.append({"user": c["user"], "torch": c["torch"], "raw": raw_text, "chat_template": chat_text,
                         "raw_matches_torch": raw_text == c["torch"], "chat_matches_torch": chat_text == c["torch"],
                         "finish_reason": raw["choices"][0]["finish_reason"]})
            print(f"[{name}] {'=' if raw_text == c['torch'] else '~'}{'=' if chat_text == c['torch'] else '~'} {c['user']!r} -> {chat_text!r}")
        conv = [{"role": "user", "content": "Hi there!"}]
        for follow in ["I'm Maya.", "What's my name?"]:
            r = llm.create_chat_completion(messages=conv, max_tokens=40, temperature=0.0, top_k=1)
            conv += [{"role": "assistant", "content": r["choices"][0]["message"]["content"].strip()},
                     {"role": "user", "content": follow}]
        r = llm.create_chat_completion(messages=conv, max_tokens=40, temperature=0.0, top_k=1)
        conv.append({"role": "assistant", "content": r["choices"][0]["message"]["content"].strip()})
        out["files"][name] = {
            "chat_format": getattr(llm, "chat_format", None),
            "raw_exact": sum(r["raw_matches_torch"] for r in rows), "chat_exact": sum(r["chat_matches_torch"] for r in rows),
            "total": len(rows), "cases": rows, "multi_turn": conv,
        }
        del llm
    Path(a.report).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: {kk: v[kk] for kk in ("chat_format", "raw_exact", "chat_exact", "total")} for k, v in out["files"].items()}, indent=1))


if __name__ == "__main__":
    main()
