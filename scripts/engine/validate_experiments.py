"""Validate experiment records (<model repo>/experiments/EXP-*.yaml) against the template.

    python scripts/engine/validate_experiments.py models/Koto/experiments

Checks: required fields, id matches the file name, semver model_version,
training.status is one of NOT_STARTED / NOT_REQUIRED / MANUALLY_AUTHORIZED /
COMPLETED, a COMPLETED or MANUALLY_AUTHORIZED run names who authorised it and
its training config, and every referenced report / config path exists
(relative to the model repository). Exit code 1 on any problem.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import yaml

from hibari.release.profile import SEMVER, TRAINING_STATUSES

REQUIRED = ["id", "model", "model_version", "date", "objective", "problem", "source_changes", "datasets", "configs",
            "checkpoint", "training", "evaluation", "results", "failures", "improvements", "limitations", "conclusion"]


def validate_experiment(path: Path, repo_root: Path) -> List[str]:
    p: List[str] = []
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for k in REQUIRED:
        if k not in d:
            p.append(f"missing field {k!r}")
    if d.get("id") != path.stem:
        p.append(f"id {d.get('id')!r} != file name {path.stem!r}")
    if not SEMVER.match(str(d.get("model_version", ""))):
        p.append(f"model_version {d.get('model_version')!r} is not semver")
    tr = d.get("training") or {}
    st = tr.get("status")
    if st not in TRAINING_STATUSES:
        p.append(f"training.status {st!r} not in {TRAINING_STATUSES}")
    if st in ("COMPLETED", "MANUALLY_AUTHORIZED"):
        if not tr.get("authorized_by"):
            p.append("training run without training.authorized_by")
        if not tr.get("config"):
            p.append("training run without training.config")
    refs = list(d.get("configs") or []) + list(d.get("datasets") or []) + list((d.get("evaluation") or {}).get("reports") or [])
    if tr.get("config"):
        refs.append(tr["config"])
    for r in refs:
        if isinstance(r, str) and not r.startswith(("<", "data/processed")) and not (repo_root / r).exists():
            p.append(f"referenced path does not exist: {r}")
    return p


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    exp_dir = Path(sys.argv[1]).resolve()
    files = sorted(exp_dir.glob("EXP-*.yaml"))
    bad = 0
    for f in files:
        probs = validate_experiment(f, exp_dir.parent)
        print(f"{'OK  ' if not probs else 'FAIL'} {f.name}" + "".join(f"\n     - {x}" for x in probs))
        bad += bool(probs)
    print(f"{len(files) - bad}/{len(files)} experiment records valid")
    return 1 if bad or not files else 0


if __name__ == "__main__":
    sys.exit(main())
