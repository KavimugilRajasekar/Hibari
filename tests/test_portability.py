"""Windows portability: every text read / write names its encoding.

On Windows, Python's default text encoding is the ANSI code page (cp1252), not
UTF-8. Koto's data and reports contain emoji and non-English text, so a
``read_text()`` / ``write_text()`` / ``open(..., "r")`` without ``encoding=``
can crash or corrupt data there (found while preparing training on a Windows
laptop, 2026-11-22).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = [ROOT / "src", ROOT / "scripts", *[ROOT / "models" / "Koto" / d for d in ("koto", "training", "evaluation", "scripts")]]
CALL = re.compile(r"\.(read_text|write_text)\(|(?<!\.)\bopen\(")   # builtin open only (not zf.open)


def _calls(text):
    for m in CALL.finditer(text):
        depth, j = 1, m.end()
        while depth and j < len(text):
            depth += {"(": 1, ")": -1}.get(text[j], 0)
            j += 1
        yield m.group(0), text[m.end():j - 1], text.count("\n", 0, m.start()) + 1


def test_text_io_names_its_encoding():
    bad = []
    for d in DIRS:
        for f in d.rglob("*.py") if d.exists() else []:
            for name, args, line in _calls(f.read_text(encoding="utf-8")):
                if "encoding=" in args:
                    continue
                if name.startswith("open(") and (re.search(r"['\"][rwa]b['\"]", args) or "zf.open" in args or not args.strip()):
                    continue  # binary mode
                if name.startswith("open(") and "ZipFile" in args:
                    continue
                bad.append(f"{f.relative_to(ROOT)}:{line}: {name}{args[:60]})")
    assert not bad, "text I/O without encoding=:\n" + "\n".join(bad)
