"""Reusable conversation-data pipeline steps (model-agnostic).

    raw source --adapter--> records --validate--> --normalise--> --deduplicate-->
    --quality filter--> --window--> --split--> tokenise / mask (ConversationDataset) --> collate

Dataset *adapters* (how to read DailyDialog, EmpatheticDialogues, ...) belong
to the model repository (Koto ``koto/data``); this module provides the steps
every model needs, each deterministic and side-effect free:

* :func:`validate_conversation` — schema problems of one conversation;
* :func:`normalize_text` — the key used for duplicate and leak detection;
* :func:`deduplicate` — drop exact duplicates (by normalised turn text);
* :class:`QualityFilter` — named, configurable rejection rules;
* :func:`stable_split` — hash split: a text is always in the same split;
* :func:`conversation_stats` — reproducible per-source statistics.

Windowing and SFT masking already live in ``hibari.data.formatting``
(``window_conversation``) and ``hibari.data.dataset`` (``ConversationDataset``).
"""

from __future__ import annotations

import collections
import hashlib
import re
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hibari.data.tokenizer import SPECIAL_TOKENS

Message = Mapping[str, str]
ROLES = ("system", "user", "assistant")
_SPECIAL_RE = re.compile("|".join(re.escape(t) for t in SPECIAL_TOKENS))


def messages_of(record) -> List[Message]:
    """Accept ``[...]`` or ``{"messages": [...], "source": ...}``."""
    return list(record["messages"]) if isinstance(record, Mapping) else list(record)


def source_of(record, default: str = "unknown") -> str:
    return record.get("source", default) if isinstance(record, Mapping) else default


def validate_conversation(messages: Sequence[Message]) -> List[str]:
    """Problems that make a conversation unusable for SFT (empty list = valid).

    Rules: known roles only; at most one system turn and only first; user and
    assistant strictly alternate starting with user; the last turn is an
    assistant turn; no empty turn; no special-token strings inside text.
    """
    problems: List[str] = []
    if not messages:
        return ["empty conversation"]
    roles = [m.get("role") for m in messages]
    for i, m in enumerate(messages):
        if m.get("role") not in ROLES:
            problems.append(f"turn {i}: unknown role {m.get('role')!r}")
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            problems.append(f"turn {i}: empty content")
        elif _SPECIAL_RE.search(content):
            problems.append(f"turn {i}: contains a special-token string")
    if "system" in roles[1:]:
        problems.append("system turn after the first position")
    turns = roles[1:] if roles[0] == "system" else roles
    if not turns:
        problems.append("no user / assistant turns")
    else:
        if turns[0] != "user":
            problems.append("first non-system turn is not a user turn")
        if turns[-1] != "assistant":
            problems.append("last turn is not an assistant turn")
        if any(a == b for a, b in zip(turns, turns[1:])):
            problems.append("roles do not alternate")
    return problems


def normalize_text(text: str) -> str:
    """Lowercase, punctuation-free form used for duplicate / leak keys."""
    return re.sub(r"\W+", " ", text.lower()).strip()


def conversation_key(messages: Sequence[Message], roles: Tuple[str, ...] = ("user", "assistant")) -> str:
    return " | ".join(normalize_text(m["content"]) for m in messages if m.get("role") in roles)


def deduplicate(records: Iterable, key: Callable[[Sequence[Message]], str] = conversation_key) -> Tuple[List, int]:
    """Keep the first occurrence of each key; returns (kept, number dropped)."""
    seen, kept, dropped = set(), [], 0
    for r in records:
        k = key(messages_of(r))
        if k in seen:
            dropped += 1
            continue
        seen.add(k)
        kept.append(r)
    return kept, dropped


@dataclass
class QualityFilter:
    """Named rejection rules; ``check`` returns the rules a conversation breaks.

    Defaults are deliberately permissive (they remove clear junk, not style):
    very short or very long assistant turns, assistant turns that repeat the
    same sentence, turns that are mostly non-letters, URLs in replies.
    """

    min_assistant_chars: int = 2
    max_assistant_chars: int = 1200
    max_repeated_sentences: int = 1
    min_letter_ratio: float = 0.5
    reject_urls_in_replies: bool = True
    extra_rules: Dict[str, Callable[[Sequence[Message]], bool]] = field(default_factory=dict)

    def check(self, messages: Sequence[Message]) -> List[str]:
        broken: List[str] = []
        for m in messages:
            if m.get("role") != "assistant":
                continue
            t = m["content"].strip()
            if len(t) < self.min_assistant_chars:
                broken.append("assistant_too_short")
            if len(t) > self.max_assistant_chars:
                broken.append("assistant_too_long")
            sents = [normalize_text(s) for s in re.split(r"(?<=[.!?])\s+", t)]
            sents = [s for s in sents if len(s.split()) >= 3]
            if len(sents) - len(set(sents)) > self.max_repeated_sentences - 1:
                broken.append("assistant_repeats_sentence")
            non_space = [c for c in t if not c.isspace()]
            if non_space and sum(c.isalpha() for c in non_space) / len(non_space) < self.min_letter_ratio:
                broken.append("assistant_mostly_non_letters")
            if self.reject_urls_in_replies and re.search(r"https?://|www\.", t):
                broken.append("assistant_contains_url")
        for name, rule in self.extra_rules.items():
            if rule(messages):
                broken.append(name)
        return sorted(set(broken))

    def apply(self, records: Iterable) -> Tuple[List, Dict[str, int]]:
        kept, reasons = [], collections.Counter()
        for r in records:
            broken = self.check(messages_of(r))
            if broken:
                reasons.update(broken)
            else:
                kept.append(r)
        return kept, dict(reasons)


def stable_split(text: str, val_pct: int = 4, test_pct: int = 4) -> str:
    """``"train"`` / ``"val"`` / ``"test"`` from a hash of the normalised text.

    The same text always lands in the same split, whatever the order or the
    other data (identical to Koto's ``koto.data.tasks.split_of``).
    """
    h = int(hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < test_pct:
        return "test"
    if h < test_pct + val_pct:
        return "val"
    return "train"


def conversation_stats(records: Iterable, count_tokens: Optional[Callable[[str], int]] = None) -> Dict[str, Dict]:
    """Per-source statistics: conversations, turn pairs, reply length, duplicate rate."""
    by: Dict[str, List[Sequence[Message]]] = collections.defaultdict(list)
    for r in records:
        by[source_of(r)].append(messages_of(r))
    out: Dict[str, Dict] = {}
    for src, convs in sorted(by.items()):
        pairs = [sum(1 for m in c if m.get("role") == "assistant") for c in convs]
        replies = [m["content"] for c in convs for m in c if m.get("role") == "assistant"]
        words = [len(t.split()) for t in replies]
        keys = [conversation_key(c) for c in convs]
        s = {
            "conversations": len(convs),
            "mean_turn_pairs": round(statistics.mean(pairs), 2) if pairs else 0,
            "multi_turn_share": round(sum(p > 1 for p in pairs) / len(pairs), 3) if pairs else 0,
            "mean_reply_words": round(statistics.mean(words), 1) if words else 0,
            "median_reply_words": statistics.median(words) if words else 0,
            "duplicate_share": round(1 - len(set(keys)) / len(keys), 3) if keys else 0,
        }
        if count_tokens:
            s["mean_tokens"] = round(statistics.mean(count_tokens(" ".join(m["content"] for m in c)) for c in convs), 1)
        out[src] = s
    return out
