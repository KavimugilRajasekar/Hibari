# 12 — Model Export and Quantization (GGUF)

Hibari Koto is exported to GGUF using the llama.cpp **`llama`**
architecture. The file is self-contained: weights, hyper-parameters, the full
BPE vocabulary, every merge rule, token types, special-token IDs and a chat
template are embedded. No PyTorch checkpoint or Python tokenizer is needed at
inference time.

## Why the `llama` architecture is a correct fit

| Hibari component | llama.cpp `llama` |
|---|---|
| Pre-RMSNorm (eps 1e-5), no biases | `attn_norm` / `ffn_norm`, `layer_norm_rms_epsilon` |
| Multi-head attention, n_head_kv = n_head = 8 | `head_count`, `head_count_kv` |
| RoPE, θ = 10000, rotation of **adjacent pairs** `(x0,x1),(x2,x3)…` | `LLAMA_ROPE_TYPE_NORM` — the same rotation, so Q/K are written **unpermuted** |
| SwiGLU `down(silu(gate(x)) * up(x))` | `ffn_gate`, `ffn_up`, `ffn_down` |
| Final RMSNorm, tied output head | `output_norm`; `output.weight` omitted → runtime reuses `token_embd` |

HF LLaMA checkpoints need a Q/K permutation during conversion because HF uses
the rotate-half (NeoX) layout. Hibari does not, and the numerical checks below
confirm that no permutation is needed.

## Export

```bash
python scripts/engine/export.py \
  --checkpoint models/Koto/checkpoints/koto_v4/best_model.pt \
  --outdir models/Koto/releases/exported \
  --variants f32 f16 q8_0 q4_k_m \
  --llama-quantize ~/llama.cpp/llama-quantize
```

`llama-quantize` from any llama.cpp build works, including the copy bundled
with Ollama (`/usr/local/lib/ollama/llama-quantize`). Exporting overwrites
files with the same name in `--outdir`; models already created in Ollama are
not affected, because Ollama stores its own copy.

Output names are `Hibari-Koto-<TYPE>.gguf`:

| Suffix | Storage | v4 size |
|---|---|---|
| `F32` | 32-bit float, the exact training weights | 23.7 MiB |
| `F16` | 16-bit float (norms stay F32) | 11.9 MiB |
| `Q8_0` | 8-bit integers, one scale per 32-weight block | 6.4 MiB |
| `Q4_K_M` | 4-bit k-quant, "medium" mix (some tensors get more bits) | 3.9 MiB |

> **llama.cpp shared libraries.** The binaries in `~/llama.cpp` were built in
> `~/Desktop/llama.cpp/build/bin` and copied; their RUNPATH still points at
> the build folder, so outside `~/llama.cpp` they fail with
> `error while loading shared libraries: libllama-cli-impl.so`. Fix:
> `export LD_LIBRARY_PATH="$LLAMA_CPP_HOME:$LD_LIBRARY_PATH"` (add it to
> `~/.bashrc` after the `LLAMA_CPP_HOME` line).

* **F32 / F16** are written by `src/hibari/export/gguf_export.py`.
  2-D weights use the requested type; 1-D RMSNorm weights are always F32.
* **Q8_0 / Q4_K_M** are produced from the F32 file by llama.cpp's
  `llama-quantize`, the reference implementation of those formats. If it is
  not available the variant is reported as skipped, never faked. All tensor
  rows (256 / 768 elements) are multiples of the 256-element K-quant block.
* `export_manifest.json` records size and SHA-256 for every file.

## Metadata written

| Key | Value |
|---|---|
| `general.architecture` | `llama` |
| `general.name` / `basename` / `version` / `author` / `license` / `size_label` | `Hibari-Koto`, `2.0`, `Kavimugil Rajasekar`, `mit`, `6.2M` |
| `llama.context_length` | 256 |
| `llama.embedding_length` / `block_count` / `feed_forward_length` | 256 / 6 / 768 |
| `llama.attention.head_count` / `head_count_kv` | 8 / 8 |
| `llama.rope.dimension_count` / `rope.freq_base` | 32 / 10000 |
| `llama.attention.layer_norm_rms_epsilon` | 1e-5 |
| `llama.vocab_size` | 4096 |
| `tokenizer.ggml.model` / `pre` | `gpt2` / `gpt-2` |
| `tokenizer.ggml.tokens` / `merges` / `token_type` / `scores` | 4096 tokens / 3833 merges / CONTROL for the 7 special tokens |
| `bos` / `eos` / `eot` / `padding` / `unknown` token ids | 2 / 3 / 3 / 0 / 1 |
| `tokenizer.ggml.add_bos_token` / `add_eos_token` | false / false (training never used BOS) |
| `tokenizer.chat_template` | Jinja template identical to `HibariTokenizer.apply_chat_template` |

Chat format (training, Python, llama.cpp, Ollama):

```
<|system|>\n{system}<|eos|>\n<|user|>\n{user}<|eos|>\n<|assistant|>\n{reply}<|eos|>\n
```

## Defects fixed in the original exporter

| Defect | Effect | Fix |
|---|---|---|
| RMSNorm weights cast to F16 in the F16 file | llama.cpp aborted: `binary_op: unsupported types: dst: f32, src0: f32, src1: f16` | 1-D tensors always F32 |
| `tokenizer.ggml.pre = "default"` | llama.cpp applied an extra punctuation split, e.g. `I'm` → `I`,`'`,`m`; 14 vs 18 tokens on a test sentence. The GGUF model saw token sequences it was never trained on | `gpt-2` (identical regex to the HF ByteLevel pre-tokenizer) |
| No `add_bos_token` | runtime-dependent BOS insertion | explicitly `false` |
| No chat template, no version/author/license | runtimes could not format chats | added |
| `output.weight` duplicated although tied | +4 MiB per F32 file | omitted; runtimes fall back to `token_embd` |
| Only F32/F16 | — | Q8_0 and Q4_K_M via `llama-quantize` |
| Tokenizer/model vocab mismatch not detected | invalid files possible | validated on export |

## Verification

```bash
# header, metadata, tensors, tokenizer parity and greedy parity vs PyTorch in llama.cpp
python scripts/engine/verify_gguf.py --checkpoint <ckpt> --gguf models/Koto/releases/exported/*.gguf \
  --llama-bin ~/llama.cpp --report models/Koto/outputs/metrics/parity_<run>.json

# the same prompts through llama-cpp-python (needs `pip install llama-cpp-python`)
python scripts/engine/verify_llama_cpp_python.py --parity models/Koto/outputs/metrics/parity_<run>.json \
  --gguf models/Koto/releases/exported/*.gguf --report models/Koto/outputs/metrics/llama_cpp_python_<run>.json

# speed
llama-bench -m models/Koto/releases/exported/Hibari-Koto-Q8_0.gguf -p 128 -n 64 -t 6
```

The test suite additionally runs a pure-NumPy reference implementation
(`src/hibari/export/gguf_reference.py`) that reads **only** the GGUF
and must reproduce the PyTorch logits (F32 max abs error ~5e-6).

### Results for v4 (`koto_v4/best_model.pt`, step 11000)

Measured 2026-11-21 with llama.cpp build 12580 (`36cd3aff4`) on a 12-core
CPU. Report: `models/Koto/outputs/metrics/parity_v4.json`.

| File | Tokenizer parity | Greedy replies identical to PyTorch | Prompt t/s (pp128) | Generation t/s (tg64, 6 threads) |
|---|---|---|---|---|
| F32 | 12 / 12 | 6 / 6 | ~26,200 | ~2,080 |
| F16 | 12 / 12 | 6 / 6 | — | — |
| Q8_0 | 12 / 12 | 6 / 6 | ~27,100 | ~4,980 |
| Q4_K_M | 12 / 12 | 6 / 6 | — | — |

The `release/` files referenced by earlier versions of this page
(`inference_parity.json`, `llama_cpp_python_check.json`, `RELEASE_NOTES.md`)
are not in the repository at present; `release/` currently holds only the
Modelfile.

## Running a GGUF in llama.cpp directly

```bash
# interactive chat (uses the embedded chat template and default system prompt)
llama-cli -m models/Koto/releases/exported/Hibari-Koto-Q8_0.gguf \
  -c 256 -n 96 --temp 0.5 --top-k 25 --top-p 0.85 --repeat-penalty 1.15
#   /regen  new answer · /clear  wipe history (do it every 4–5 turns) · /exit

# one question, then exit
llama-cli -m models/Koto/releases/exported/Hibari-Koto-Q8_0.gguf -c 256 --temp 0 -p "who developed you?" -st

# OpenAI-compatible server (web chat at http://127.0.0.1:8080)
llama-server -m models/Koto/releases/exported/Hibari-Koto-Q8_0.gguf -c 256 --port 8080

# raw prompt: -no-cnv is required, otherwise llama-completion aborts in the chat-template code
llama-completion -m models/Koto/releases/exported/Hibari-Koto-Q8_0.gguf -c 256 -n 60 --temp 0 \
  -no-cnv --no-display-prompt \
  -p $'<|system|>\nYou are Hibari Koto, a friendly and calm AI companion. Keep replies short, kind and honest.<|eos|>\n<|user|>\nwho developed you?<|eos|>\n<|assistant|>\n'
```

`llama-server` rejects any request longer than the context instead of
trimming it (`request (257 tokens) exceeds the available context size (256
tokens)`), which happens after about six turns. Clients must send only the
last few messages; the Python `HibariChatSession` and Ollama trim
automatically.

## Known differences between Python and GGUF tokenization

The Python tokenizer applies Unicode NFKC normalisation before BPE; llama.cpp
does not. Output differs only for compatibility characters (full-width
letters, ligatures such as `ﬁ`). All 12 tokenizer parity cases, including
emoji, accents, CJK, newlines, numbers, URLs and special tokens, match.
