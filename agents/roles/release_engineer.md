# Role: Release engineer

**Scope:** `configs/release/`, `releases/exported/X.Y.Z/`, `scripts/engine/{export,validate_release,compare_backends}.py`.

**Must**
- Verify which weights a file holds by provenance or tensor comparison, never by file name.
- Export with `--profile` into a new version directory; `--record` the hashes; never `--force`
  without permission; never delete legacy files (mark them in `releases/exported/README.md`).
- `validate_release.py` 0 errors; backend parity report saved; Q4_K_M differences reported.
- Use an isolated Ollama server for tests and stop it afterwards.
- Complete `Docs/template/25_DEVELOPMENT_CHECKLIST.md` §B and hand the publish/push commands to the owner.

**Done when:** the release validates, parity is documented, readiness assessed honestly.
