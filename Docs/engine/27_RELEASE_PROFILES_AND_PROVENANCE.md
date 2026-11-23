# 27 — Release Profiles and GGUF Provenance

## Why
Until Koto 1.1.1 the GGUFs in `models/Koto/releases/exported/` (loaded by the
Modelfile and therefore by Ollama) contained the **1.0.0** weights, labelled
"koto_v4", `general.name = Hibari-Koto-Plus`, `general.version = 2.0`, while
the documentation described 1.1.0. Nothing in a GGUF said which checkpoint
it came from, and decoding defaults were defined in six places.

## Release profile (`hibari.release.ModelProfile`)
One YAML per model version in the model repository
(`configs/release/<model>_X.Y.Z.yaml`, template `hibari/configs/release/template.yaml`):
version, weights version, training status, identity, system prompt (and the
one used in training), model / inference config paths, checkpoint and
tokenizer with SHA-256, GGUF directory and files with SHA-256, and optionally
`ollama_name` — the published Ollama name used in the generated Modelfile's usage lines
(default `hibari-<model>:<version>`; Koto 1.3.1: `kavimugil-r/Hibari-Koto1.3.1:29M`).

`InferenceEngine.from_profile(path)` loads the checkpoint and tokenizer and
creates sessions with the profile's system prompt, identity terms and
application settings; `engine.reply_settings()` gives the decoding kwargs.

## Provenance metadata in every GGUF
`export_to_gguf(..., system_prompt=..., provenance=profile.provenance())` writes

| key | example |
|---|---|
| `hibari.model.name` / `hibari.model.version` | `koto` / `1.1.1` |
| `hibari.weights.version` | `1.1.0` |
| `hibari.checkpoint.sha256`, `hibari.checkpoint.step` | `df8acd08…`, `12000` |
| `hibari.tokenizer.sha256` | `d5a2924e…` |
| `hibari.engine.version`, `hibari.identity.name` | `0.3.0`, `Koto` |

plus `general.version` = the release version and a chat template using the
profile's system prompt. llama.cpp and Ollama ignore unknown keys;
`llama-quantize` copies them into Q8_0 / Q4_K_M (verified).

## Validation (`validate_release.py`)
Semver; files exist; checkpoint architecture = model config; tokenizer vocab
= model vocab; checkpoint `run_metadata.model_version` = `weights_version`;
hashes = profile; inference settings in range (`max_new_tokens` < context,
`num_ctx` ≤ trained context); system prompt = training prompt (warning); per
GGUF: SHA-256, provenance = profile, dimensions, vocabulary identical to the
tokenizer, embedded chat template; the Modelfile is byte-identical to
`render_modelfile(profile)`. Read-only.

## Export (`export.py --profile`)
Refuses to export if the checkpoint or tokenizer hash differs from the
profile; writes `X.tmp`, then keeps an identical existing file, refuses to
replace a different one without `--force`; quantizes with `llama-quantize`;
generates the Modelfile and `export_manifest.json`; `--record` writes the
GGUF hashes into the profile.

## Backend parity (`compare_backends.py`)
See doc 23 §3. Koto 1.1.1 result: tokenizer 23/23 on every variant; greedy
PyTorch = llama.cpp = Ollama chat = Ollama raw on 11/11 prompts for F32,
F16 and Q8_0; Q4_K_M 6/11 (llama.cpp) and 8/11 (Ollama), the expected
effect of 4-bit weights on a 6M model; chat-template token counts 11/11.

`llama-completion -n k` prints k-1 tokens (measured 2026-11-23), so the script requests n+1 and trims
both outputs to n tokens (engine 0.4.1). Before that, replies that reached the cap counted as mismatches:
Koto 1.3.0 F32 showed 9/11; with the fix F32 and F16 are 11/11, Q8_0 10/11, Q4_K_M 7/11
(`models/Koto/outputs/evaluation/backend_parity_1.3.0_r2.json`).
