# 06 — Release Profiles, Export and Backends

Reference: `Docs/engine/27_RELEASE_PROFILES_AND_PROVENANCE.md`.

- Every model version has `models/Koto/configs/release/koto_X.Y.Z.yaml` with checkpoint and
  tokenizer SHA-256. Load everything through it (`InferenceEngine.from_profile`).
- Export only with a profile; output goes to `releases/exported/X.Y.Z/`:
  ```bash
  LD_LIBRARY_PATH=~/llama.cpp PYTHONPATH=src python scripts/engine/export.py \
      --profile models/Koto/configs/release/koto_X.Y.Z.yaml --llama-quantize ~/llama.cpp/llama-quantize --record
  ```
  The exporter refuses to replace a different existing file; do not pass `--force` without permission.
- Validate: `PYTHONPATH=src python scripts/engine/validate_release.py --profile <profile>` → 0 errors required.
- Backend parity (PyTorch vs llama.cpp vs Ollama):
  ```bash
  PYTHONPATH=src python scripts/engine/compare_backends.py --profile <profile> \
      --llama-bin ~/llama.cpp --ollama-host http://127.0.0.1:<port> --report models/Koto/outputs/evaluation/backend_parity_X.Y.Z.json
  ```
  Use an **isolated** Ollama server so the owner's model store is untouched:
  `OLLAMA_MODELS=<tmpdir> OLLAMA_HOST=127.0.0.1:11535 ollama serve` — stop it when done.
- Modelfiles are generated (`hibari.release.render_modelfile`); never edit them by hand.
- Expected parity: F32 / F16 / Q8_0 identical to PyTorch greedy; Q4_K_M differences are reported.
- Python-only features (safety, routing, normalisation, reranking) do not exist in GGUF runtimes;
  say so whenever you report Ollama behaviour.
- Publishing a release (GitHub release, Ollama registry, Hugging Face) is the owner's decision.
