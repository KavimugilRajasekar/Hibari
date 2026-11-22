"""Independent byte-level BPE reference encoder (GPT-2 algorithm, pure Python).

An independent re-implementation, informed by OpenAI's GPT-2 ``src/encoder.py``
(MIT licence), of the algorithm that Hibari's production tokenizer
(Hugging Face ``tokenizers`` BPE + ByteLevel) is supposed to implement:

1. split the text into pre-tokens with GPT-2's pattern
   ``'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+``
   (implemented here as a scanner over Unicode categories, so no ``regex``
   dependency is needed);
2. map every UTF-8 byte to a printable character (``bytes_to_unicode``), so
   the vocabulary never contains raw control bytes or whitespace;
3. repeatedly merge the adjacent pair with the lowest merge rank;
4. decode by the inverse byte map, with ``errors="replace"`` for token
   sequences that end inside a multi-byte character.

It does not replace the production tokenizer (vocabulary and IDs are read from
``tokenizer.json`` and never changed). It exists to *verify* it:
``tests/test_bpe_reference.py`` checks token-for-token agreement on edge cases
and on real training text, and ``nfkc=False`` reproduces what llama.cpp /
Ollama do (they do not apply Hibari's NFKC normaliser).
"""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


@lru_cache()
def bytes_to_unicode() -> Dict[int, str]:
    """Reversible map byte -> printable unicode character (the GPT-2 table).

    Printable Latin-1 bytes map to themselves; the remaining 68 bytes (controls,
    space, ...) map to code points 256+ so every byte has a visible symbol.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def _cls(ch: str) -> str:
    if ch.isspace():
        return "S"
    cat = unicodedata.category(ch)
    return "L" if cat[0] == "L" else "N" if cat[0] == "N" else "O"


def pretokenize(text: str) -> List[str]:
    """Split ``text`` exactly like GPT-2's pre-tokenisation pattern."""
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        hit = next((c for c in CONTRACTIONS if text.startswith(c, i)), None)
        if hit:
            out.append(hit)
            i += len(hit)
            continue
        c = _cls(text[i])
        if text[i] == " " and i + 1 < n and _cls(text[i + 1]) != "S":   # " ?\p{L}+" etc.
            k = _cls(text[i + 1])
            j = i + 2
            while j < n and _cls(text[j]) == k:
                j += 1
            out.append(text[i:j])
            i = j
            continue
        if c != "S":
            j = i + 1
            while j < n and _cls(text[j]) == c:
                j += 1
            out.append(text[i:j])
            i = j
            continue
        j = i
        while j < n and _cls(text[j]) == "S":
            j += 1
        if j == n or j - i == 1:      # "\s+(?!\S)" to the end, or a single whitespace (falls to "\s+")
            out.append(text[i:j])
            i = j
        else:                          # "\s+(?!\S)" backtracks one char before the next non-space
            out.append(text[i:j - 1])
            i = j - 1
    return out


def _pairs(word: Sequence[str]) -> set:
    return {(a, b) for a, b in zip(word, word[1:])}


class ReferenceBPE:
    def __init__(self, vocab: Dict[str, int], merges: Iterable[Tuple[str, str]], special_tokens: Iterable[str] = (),
                 nfkc: bool = True):
        self.encoder = dict(vocab)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.ranks = {tuple(m): i for i, m in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.special = sorted(special_tokens, key=len, reverse=True)
        self.nfkc = nfkc
        self.cache: Dict[str, Tuple[str, ...]] = {}

    @classmethod
    def from_tokenizer_json(cls, path: str | Path, nfkc: bool = True) -> "ReferenceBPE":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in data["model"]["merges"]]
        special = [t["content"] for t in data.get("added_tokens", []) if t.get("special")]
        return cls(data["model"]["vocab"], merges, special, nfkc=nfkc)

    def bpe(self, token: str) -> Tuple[str, ...]:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = _pairs(word)
        while pairs:
            first, second = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if (first, second) not in self.ranks:
                break
            merged, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = tuple(merged)
            pairs = _pairs(word) if len(word) > 1 else set()
        self.cache[token] = word
        return word

    def _encode_plain(self, text: str) -> List[int]:
        if self.nfkc:
            text = unicodedata.normalize("NFKC", text)
        ids: List[int] = []
        for piece in pretokenize(text):
            mapped = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self.bpe(mapped))
        return ids

    def encode(self, text: str) -> List[int]:
        """Special-token strings are matched first (as added tokens are), then BPE."""
        ids: List[int] = []
        i = start = 0
        while i < len(text):
            sp = next((s for s in self.special if text.startswith(s, i)), None)
            if sp:
                if start < i:
                    ids += self._encode_plain(text[start:i])
                ids.append(self.encoder[sp])
                i += len(sp)
                start = i
            else:
                i += 1
        if start < len(text):
            ids += self._encode_plain(text[start:])
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        out = bytearray()
        for t in ids:
            tok = self.decoder[t]
            if tok in self.special:
                if not skip_special:
                    out += tok.encode("utf-8")
                continue
            out += bytes(self.byte_decoder[c] for c in tok)
        return out.decode("utf-8", errors="replace")
