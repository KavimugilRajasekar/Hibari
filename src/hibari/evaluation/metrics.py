"""Model-independent text metrics for evaluating generated replies.

These are automatic, surface-level measurements. They are meant to be
combined with task-specific rule checks and read alongside the transcripts;
none of them (including the character n-gram F-score, a chrF-style overlap
with a reference) decides on its own whether a reply is natural or correct.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

_WORD = re.compile(r"[a-z0-9']+")
_SENT_END = re.compile(r"[.!?)\"']$")


def words(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def ngram_repetition(text: str, n: int = 3) -> float:
    """Fraction of word n-grams that are repeats (0 = no repetition)."""
    w = words(text)
    grams = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def repeated_sentence(text: str) -> bool:
    """True if the same sentence occurs twice ("I'm fine. I'm fine.")."""
    sents = [re.sub(r"\W+", " ", s.lower()).strip() for s in re.split(r"(?<=[.!?])\s+", text)]
    sents = [s for s in sents if len(s.split()) >= 2]
    return len(sents) != len(set(sents))


def distinct_n(texts: Iterable[str], n: int = 2) -> float:
    """Distinct-n across a set of replies (higher = more varied)."""
    grams, total = set(), 0
    for t in texts:
        w = words(t)
        g = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
        grams.update(g)
        total += len(g)
    return len(grams) / max(1, total)


def grammar_heuristic(text: str) -> Dict[str, bool]:
    """Surface checks only: capitalised start, terminal punctuation, no doubled
    words, no space before punctuation, sensible length."""
    t = text.strip()
    w = re.findall(r"[A-Za-z']+", t)
    return {
        "capitalised": bool(t) and (t[0].isupper() or not t[0].isalpha()),
        "terminal_punctuation": bool(_SENT_END.search(t)),
        "no_doubled_word": not re.search(r"\b(\w+) \1\b", t.lower()),
        "no_space_before_punct": not re.search(r"\s[,.!?]", t),
        "reasonable_length": 1 <= len(w) <= 120,
    }


def echo_ratio(reply: str, prompt: str) -> float:
    """Share of the reply's word 3-grams copied from the prompt (repeating the question)."""
    r, p = words(reply), words(prompt)
    rg = [tuple(r[i : i + 3]) for i in range(len(r) - 2)]
    if not rg:
        return 0.0
    pg = {tuple(p[i : i + 3]) for i in range(len(p) - 2)}
    return sum(g in pg for g in rg) / len(rg)


def char_ngram_f(hypothesis: str, reference: str, n_max: int = 6, beta: float = 2.0) -> float:
    """chrF-style character n-gram F-score (0..1) between hypothesis and reference."""
    def grams(s: str, n: int) -> Counter:
        s = re.sub(r"\s+", " ", s.strip().lower())
        return Counter(s[i : i + n] for i in range(len(s) - n + 1))

    precs, recs = [], []
    for n in range(1, n_max + 1):
        h, r = grams(hypothesis, n), grams(reference, n)
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precs.append(overlap / sum(h.values()))
        recs.append(overlap / sum(r.values()))
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p == 0 and r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def length_ratio(output: str, source: str) -> float:
    return len(words(output)) / max(1, len(words(source)))


_LEAD = re.compile(r"^\s*(here'?s|here is|sure[,!]|certainly[,!]|of course[,!]|the (corrected|rewritten|rephrased) (sentence|version))", re.I)
_SELF_INTRO = re.compile(r"\b(i'?m|i am|my name is|this is)\s+hibari\b", re.I)
_AI_DISCLAIMER = re.compile(r"\bas an ai( language)? model\b|\bi am (just )?an ai language model\b", re.I)


def has_lead(text: str) -> bool:
    """Reply starts with a preamble ("Here's ...", "Sure, ...") instead of the result."""
    return bool(_LEAD.search(text))


def self_introduction(text: str) -> bool:
    return bool(_SELF_INTRO.search(text))


def ai_disclaimer(text: str) -> bool:
    return bool(_AI_DISCLAIMER.search(text))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
