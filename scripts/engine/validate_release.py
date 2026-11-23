"""Validate a release profile and its artefacts (read-only).

    python scripts/engine/validate_release.py --profile models/Koto/configs/release/koto_1.1.1.yaml
    python scripts/engine/validate_release.py --profile ... --no-gguf     # before exporting

Exit code 1 if any check fails. See ``hibari.release.validation`` for the list of checks.
"""

from __future__ import annotations

import argparse
import json
import sys

from hibari.release import ModelProfile, errors, validate_profile


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a Hibari release profile")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--no-gguf", action="store_true", help="Skip GGUF / Modelfile checks")
    ap.add_argument("--no-hashes", action="store_true", help="Skip checkpoint / tokenizer SHA-256 checks")
    ap.add_argument("--json", default=None, help="Also write the findings to this JSON file")
    ap.add_argument("--quiet", action="store_true", help="Print only warnings and errors")
    a = ap.parse_args()

    profile = ModelProfile.load(a.profile)
    findings = validate_profile(profile, check_hashes=not a.no_hashes, check_gguf=not a.no_gguf)
    print(f"Profile {profile.tag} (weights {profile.weights_version}, training {profile.training_status})")
    for f in findings:
        if not a.quiet or f.level != "ok":
            print(f)
    errs = errors(findings)
    warns = [f for f in findings if f.level == "warning"]
    print(f"\n{len(findings) - len(errs) - len(warns)} ok, {len(warns)} warnings, {len(errs)} errors")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump([f.__dict__ for f in findings], fh, indent=2)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
