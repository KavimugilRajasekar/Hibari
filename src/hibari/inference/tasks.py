"""Text-rewriting tasks: one shared representation for training and inference.

A rewriting request is sent to the model as an ordinary user turn in a fixed,
two-line form::

    Task: Rephrase
    Input: i am learning python from yesterday.

and the model replies with the rewritten text only. Training data builders
must use :func:`format_task_request` so that the prompt seen in training and
the prompt built at inference are byte-identical.

The module also provides

* :func:`parse_task_request` — recognises explicit requests typed in chat
  ("Make this professional: ...", "Rephrase this sentence: \"...\"") so the
  chat session can route them to the canonical form. Anything it does not
  recognise is treated as normal conversation.
* :data:`TASK_DECODING` — decoding settings for rewrites (greedy, no
  repetition penalty, length bounded by the input).
* :func:`protected_terms` / :func:`missing_protected_terms` — numbers, times,
  names and technical tokens that a rewrite must keep unchanged.

Nothing here contains expected answers; it only shapes the request and checks
the output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label: str        # the text after "Task: "
    verb: str         # used in user-facing messages ("rephrase", "correct", ...)
    description: str


TASKS: Dict[str, TaskSpec] = {
    t.name: t
    for t in [
        TaskSpec("rephrase", "Rephrase", "rephrase",
                 "Rewrite more clearly and naturally; fix mistakes; keep the meaning."),
        TaskSpec("correct_grammar", "Correct grammar", "correct",
                 "Fix grammar, spelling and punctuation; change as little as possible."),
        TaskSpec("formal", "Make formal", "rewrite",
                 "Rewrite in a professional, formal tone; keep the meaning."),
        TaskSpec("casual", "Make casual", "rewrite",
                 "Rewrite in a relaxed, casual tone; keep the meaning."),
        TaskSpec("shorten", "Make shorter", "shorten",
                 "Remove unnecessary words; keep every fact."),
        TaskSpec("simplify", "Simplify", "simplify",
                 "Use simpler words and structure; keep the meaning."),
        TaskSpec("polite", "Make polite", "rewrite",
                 "Remove rudeness and make the request courteous; keep the meaning."),
    ]
}

_LABEL_TO_TASK = {t.label.lower(): t.name for t in TASKS.values()}


def format_task_request(task: str, text: str) -> str:
    """Canonical user-turn content for a rewriting task."""
    if task not in TASKS:
        raise KeyError(f"Unknown task {task!r}; expected one of {sorted(TASKS)}")
    return f"Task: {TASKS[task].label}\nInput: {text.strip()}"


@dataclass(frozen=True)
class TaskRequest:
    task: str
    text: str


_CANONICAL = re.compile(r"^\s*task:\s*(?P<label>[^\n]+?)\s*\n\s*input:\s*(?P<text>.*)$", re.I | re.S)

# Instruction phrasings → task. Order matters: more specific first.
_INSTRUCTIONS: List[tuple] = [
    ("correct_grammar", r"(?:please\s+)?(?:correct|fix|check)\s+(?:the\s+|my\s+)?(?:grammar|spelling|mistakes|errors)(?:\s+(?:in|of)\s+this(?:\s+sentence)?)?"),
    ("correct_grammar", r"(?:please\s+)?(?:correct|fix)\s+this(?:\s+sentence)?"),
    ("formal", r"(?:please\s+)?(?:make|rewrite|write)\s+(?:this|it)(?:\s+sentence|\s+message|\s+email)?\s+(?:more\s+)?(?:professional|formal)(?:ly)?"),
    ("formal", r"(?:please\s+)?(?:make|rewrite)\s+(?:this|it)\s+(?:sound\s+)?(?:more\s+)?(?:professional|formal)"),
    ("casual", r"(?:please\s+)?(?:make|rewrite|write)\s+(?:this|it)(?:\s+sentence|\s+message)?\s+(?:more\s+)?(?:casual|informal|relaxed)(?:ly)?"),
    ("casual", r"(?:please\s+)?(?:make|rewrite)\s+(?:this|it)\s+sound\s+(?:more\s+)?(?:casual|informal|relaxed)"),
    ("polite", r"(?:please\s+)?(?:make|rewrite)\s+(?:this|it)(?:\s+sentence|\s+message)?\s+(?:more\s+)?polite(?:ly)?"),
    ("shorten", r"(?:please\s+)?(?:make\s+(?:this|it)(?:\s+sentence)?\s+(?:shorter|more\s+concise|concise|brief)|shorten\s+(?:this|it)(?:\s+sentence)?)"),
    ("simplify", r"(?:please\s+)?(?:simplify\s+(?:this|it)(?:\s+sentence)?|make\s+(?:this|it)(?:\s+sentence)?\s+simpler)"),
    ("rephrase", r"(?:please\s+)?(?:rephrase|reword|paraphrase|rewrite)(?:\s+(?:this|it|the\s+following))?(?:\s+sentence)?"),
]
# The instruction must be followed by a separator (":", "-", a newline, an
# opening quote) or by nothing, so "rewrite the story about dragons" stays chat.
_SEP = r"(?:\s*[:\-]\s*|\s*\n\s*|\s+(?=[\"'“‘`])|\s*$)"
_LEAD = r"(?:(?:can|could|would|will)\s+(?:you|u)\s+(?:please\s+|pls\s+)?)?"
_TAIL = r"(?:\s+(?:pls|plz|please))?"
_PREFIX_RE = [(task, re.compile(rf"^\s*{_LEAD}{pat}{_TAIL}{_SEP}(?P<text>.*)$", re.I | re.S)) for task, pat in _INSTRUCTIONS]
# Quoted text first, instruction after: "'cant come today' make it polite".
_SUFFIX_RE = [(task, re.compile(rf"^\s*(?P<text>([\"'“‘`]).+[\"'”’`])\s*,?\s*{pat}\s*[.!?]*\s*$", re.I | re.S))
              for task, pat in _INSTRUCTIONS]

_QUOTES = {'"': '"', "'": "'", "“": "”", "‘": "’", "`": "`"}


def _unquote(text: str) -> str:
    t = text.strip()
    if len(t) >= 2 and t[0] in _QUOTES and t[-1] == _QUOTES[t[0]]:
        t = t[1:-1].strip()
    return t


def parse_task_request(message: str) -> Optional[TaskRequest]:
    """Return the rewriting task in ``message``, or ``None`` for normal chat.

    Recognised forms: the canonical ``Task: ...\\nInput: ...`` block, and an
    explicit instruction at the start of the message followed by the text
    (``Rephrase: "..."``, ``Make this professional: ...``). The text may be
    empty (``Rephrase: ""``); callers decide how to answer that.
    """
    if not message or not message.strip():
        return None
    m = _CANONICAL.match(message)
    if m:
        task = _LABEL_TO_TASK.get(m.group("label").strip().rstrip(".").lower())
        if task:
            return TaskRequest(task, _unquote(m.group("text")))
    for rx_list in (_PREFIX_RE, _SUFFIX_RE):
        for task, rx in rx_list:
            m = rx.match(message)
            if m:
                return TaskRequest(task, _unquote(m.group("text")))
    return None


# ───────────────────────── decoding ─────────────────────────

# Rewrites should be deterministic and must be free to repeat input words, so
# no repetition penalty; the n-gram block stops copy loops.
TASK_DECODING = dict(temperature=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0, no_repeat_ngram_size=4)


def task_max_new_tokens(input_tokens: int, task: str, cap: int = 160) -> int:
    """Length budget proportional to the input (rewrites rarely grow much)."""
    factor = 1.0 if task in ("shorten", "simplify") else 1.6
    return max(16, min(cap, int(input_tokens * factor) + 16))


# ─────────────────── meaning preservation checks ───────────────────

_NUM = re.compile(r"\d+(?:[.,:/]\d+)*")
_TECH = re.compile(r"\b(?:[A-Za-z]+[0-9+#.][A-Za-z0-9+#.]*|[a-z]+[A-Z][A-Za-z]*|[A-Z]{2,}[a-z]*)\b")
_TIME_WORD = re.compile(r"\b(?:\d{1,2}\s*(?:am|pm)|noon|midnight)\b", re.I)
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS.update({"thirty": "30", "forty": "40", "fifty": "50", "hundred": "100", "thousand": "1000"})
DIGIT_WORDS = {v: k for k, v in NUMBER_WORDS.items()}
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTHS = ("january", "february", "march", "april", "june", "july", "august", "september",
           "october", "november", "december")


def protected_terms(text: str) -> List[str]:
    """Terms a rewrite must keep: numbers, times, weekdays/months, mid-sentence
    capitalised words (names) and technical tokens (iOS, C++, HTTP, v2.1)."""
    out: List[str] = []
    out += _NUM.findall(text)
    out += [w for w in re.findall(r"[A-Za-z]+", text) if w.lower() in NUMBER_WORDS]
    out += [m.group(0).replace(" ", "").lower() for m in _TIME_WORD.finditer(text)]
    out += [w for w in re.findall(r"[A-Za-z]+", text) if w.lower() in _WEEKDAYS + _MONTHS]
    out += [t.rstrip(".") for t in _TECH.findall(text) if t.lower() not in ("am", "pm")]
    words = re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+\b", text.strip())
    out += [w for w in words if w.lower() not in ("i",)]
    seen, uniq = set(), []
    for t in out:
        k = t.lower()
        if k and k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq


def missing_protected_terms(source: str, output: str) -> List[str]:
    """Protected terms of ``source`` that do not appear in ``output``
    (case-insensitive; spaces ignored for times such as "10 AM")."""
    low = output.lower()
    missing = []
    out_words = set(re.findall(r"[a-z0-9]+", low))
    for t in protected_terms(source):
        k = t.lower()
        if k in NUMBER_WORDS:  # "three" is kept by "three" or "3"
            if k in out_words or NUMBER_WORDS[k] in out_words:
                continue
        elif k.isdigit() and k in DIGIT_WORDS and DIGIT_WORDS[k] in out_words:
            continue
        elif re.search(rf"(?<![0-9.]){re.escape(k)}(?![0-9])" if k[0].isdigit()
                       else rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", low):
            continue
        elif re.fullmatch(r"\d{1,2}(am|pm)", k) and re.search(rf"(?<!\d){k[:-2]}\s*{k[-2:]}\b", low):
            continue
        missing.append(t)
    return missing
