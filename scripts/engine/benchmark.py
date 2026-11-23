"""CPU benchmark for Hibari Koto (PyTorch checkpoint and GGUF variants).

Measures, per GGUF file (llama.cpp):
  * llama-bench prompt-processing (pp64) and generation (tg128) tokens/s
  * end-to-end llama-completion run: wall time, load time, first-token
    latency (prompt eval of a chat prompt + 1 token), peak RSS (/usr/bin/time)
  * stability: N repeated greedy runs must give identical text
And for the PyTorch checkpoint: load time, first-token latency, tokens/s,
peak RSS of a fresh process.

Usage:
    python hibari/scripts/benchmark.py --checkpoint <ckpt> --gguf release/*.gguf \
        --llama-bin <llama.cpp/build/bin> --out release/benchmarks.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

PROMPT = "<|system|>\nYou are Hibari Koto, a friendly and calm AI companion. Keep replies short, kind and honest.<|eos|>\n<|user|>\nHi! How was your day?<|eos|>\n<|assistant|>\n"


def run_timed(cmd):
    proc = subprocess.run(["/usr/bin/time", "-v"] + cmd, capture_output=True, text=True)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", proc.stderr)
    wall = re.search(r"Elapsed \(wall clock\) time .*: ([\d:.]+)", proc.stderr)
    secs = None
    if wall:
        parts = [float(p) for p in wall.group(1).split(":")]
        secs = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    return proc, (int(rss.group(1)) / 1024 if rss else None), secs


def bench_gguf(bin_dir: Path, path: Path, threads: int, repeats: int) -> dict:
    out = {"file": path.name, "mib": round(path.stat().st_size / 2**20, 2)}
    b = subprocess.run([str(bin_dir / "llama-bench"), "-m", str(path), "-p", "64", "-n", "128",
                        "-t", str(threads), "-r", "5", "-o", "json"], capture_output=True, text=True)
    try:
        for row in json.loads(b.stdout):
            key = "pp64_tok_s" if row["n_prompt"] > 0 else "tg128_tok_s"
            out[key] = round(row["avg_ts"], 1)
            out[key.replace("tok_s", "stddev")] = round(row["stddev_ts"], 1)
    except Exception:
        out["llama_bench_error"] = b.stderr[-500:]

    cmd = [str(bin_dir / "llama-completion"), "-m", str(path), "-p", PROMPT, "-n", "64", "--temp", "0",
           "--repeat-penalty", "1.0", "-c", "256", "-no-cnv", "--no-display-prompt", "--simple-io",
           "--no-warmup", "-t", str(threads)]
    texts = []
    for i in range(repeats):
        proc, rss, wall = run_timed(cmd)
        texts.append(proc.stdout.replace("[end of text]", "").strip())
        if i == 0:
            err = proc.stderr
            load = re.search(r"load time =\s+([\d.]+) ms", err)
            pe = re.search(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens", err)
            ev = re.search(r"\s eval time =\s+([\d.]+) ms /\s+(\d+) runs", err)
            out.update({
                "process_wall_s": wall, "peak_rss_mib": round(rss, 1) if rss else None,
                "load_ms": float(load.group(1)) if load else None,
                "prompt_tokens": int(pe.group(2)) if pe else None,
                "prompt_eval_ms": float(pe.group(1)) if pe else None,
                "first_token_ms": round(float(pe.group(1)) + float(ev.group(1)) / max(1, int(ev.group(2))), 2) if pe and ev else None,
                "generated_tokens": int(ev.group(2)) + 1 if ev else None,
                "gen_tok_s": round(int(ev.group(2)) / (float(ev.group(1)) / 1000), 1) if ev and float(ev.group(1)) > 0 else None,
            })
    out["repeat_runs"] = repeats
    out["deterministic_across_runs"] = len(set(texts)) == 1
    out["sample_output"] = texts[0]
    return out


TORCH_SNIPPET = textwrap.dedent('''
    import json, sys, time, resource
    t0 = time.perf_counter()
    import torch
    from hibari.data.tokenizer import HibariTokenizer
    from hibari.inference.generation import generate_tokens
    from hibari.training.checkpoint import load_checkpoint
    torch.set_num_threads(int(sys.argv[3]))
    m = load_checkpoint(sys.argv[1])["model"].eval(); tok = HibariTokenizer.load(sys.argv[2])
    load_s = time.perf_counter() - t0
    ids = tok.encode(sys.argv[4])
    generate_tokens(m, tok, "", prompt_ids=ids, max_new_tokens=4, temperature=0)  # warm-up
    ts = []
    for _ in range(10):
        t = time.perf_counter(); generate_tokens(m, tok, "", prompt_ids=ids, max_new_tokens=1, temperature=0); ts.append(time.perf_counter() - t)
    ts.sort()
    # force exactly 64 tokens for a throughput figure: disable stop tokens
    from hibari.inference import generation as g
    t = time.perf_counter()
    stop = g._stop_ids; g._stop_ids = lambda tk: set()
    out = generate_tokens(m, tok, "", prompt_ids=ids, max_new_tokens=64, temperature=0)
    g._stop_ids = stop
    dt = time.perf_counter() - t
    print(json.dumps({"load_s": round(load_s, 3), "first_token_ms": round(ts[5] * 1000, 2),
                      "gen_tok_s": round(len(out) / dt, 1), "generated_tokens": len(out),
                      "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}))
''')


def bench_torch(ckpt: str, tok_dir: str, threads: int) -> dict:
    p = subprocess.run([sys.executable, "-c", TORCH_SNIPPET, ckpt, tok_dir, str(threads), PROMPT],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": p.stderr[-800:]}
    return {"file": Path(ckpt).name, "mib": round(Path(ckpt).stat().st_size / 2**20, 2), **json.loads(p.stdout.strip().splitlines()[-1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer-dir", default="models/Koto/tokenizer")
    ap.add_argument("--gguf", nargs="+", required=True)
    ap.add_argument("--llama-bin", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="release/benchmarks.json")
    a = ap.parse_args()

    cpu = subprocess.run(["bash", "-c", "lscpu | grep 'Model name' | sed 's/.*: *//'"], capture_output=True, text=True).stdout.strip()
    res = {"cpu": cpu, "threads": a.threads, "pytorch": bench_torch(a.checkpoint, a.tokenizer_dir, a.threads), "gguf": []}
    print("pytorch", res["pytorch"])
    for g in a.gguf:
        r = bench_gguf(Path(a.llama_bin), Path(g), a.threads, a.repeats)
        print(json.dumps({k: v for k, v in r.items() if k != "sample_output"}))
        res["gguf"].append(r)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
