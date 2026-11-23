"""Conservative normalisation of informal English for the chat path.

Users write ``"wat r u doing"``, ``"ur name?"``, ``"thx!!!"``. A small model
trained mostly on edited dialogue text understands the expanded form better,
so the chat session may feed the *normalised* message to the model (see
``HibariChatSession(normalize_input=True)``). Rewriting requests are routed
before normalisation and always keep the user's original text: correcting
the user's mistakes is the job of the model, not of this table.

Rules
* Only unambiguous chat shorthand is expanded (``u``, ``r``, ``wat``,
  ``pls``, ``thx``, ``idk`` ...). Words that are also ordinary English words
  (``ill``, ``wed``, ``id``, ``its``) are never touched.
* ``ur`` becomes ``you're`` before an adjective / participle / article
  (``ur so kind``) and ``your`` otherwise (``ur name``).
* Runs of ``!`` / ``?`` / ``.`` longer than the conventional ``...`` are
  shortened, stretched letters (``soooo``) are squeezed, a lone lowercase
  ``i`` becomes ``I``. Nothing else (case, word order, spelling) is changed.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# word -> expansion (lowercase keys; matched as whole words, case-insensitive)
SHORTHAND: Dict[str, str] = {
    "u": "you", "r": "are", "wat": "what", "wut": "what", "wht": "what", "wats": "what's",
    "whats": "what's", "pls": "please", "plz": "please", "plss": "please", "thx": "thanks", "thnx": "thanks",
    "thanx": "thanks", "tnx": "thanks", "ty": "thank you", "tysm": "thank you so much", "idk": "I don't know",
    "idc": "I don't care", "imo": "in my opinion", "btw": "by the way", "rn": "right now", "nvm": "never mind",
    "hru": "how are you", "wbu": "what about you", "hbu": "how about you", "wyd": "what are you doing",
    "im": "I'm", "ive": "I've", "dont": "don't", "cant": "can't", "didnt": "didn't", "doesnt": "doesn't",
    "isnt": "isn't", "wasnt": "wasn't", "havent": "haven't", "youre": "you're", "thats": "that's",
    "hows": "how's", "lemme": "let me", "gimme": "give me", "wanna": "want to", "gonna": "going to",
    "gotta": "got to", "cuz": "because", "coz": "because", "bcoz": "because", "bcz": "because",
    "tmrw": "tomorrow", "tmr": "tomorrow", "2moro": "tomorrow", "tomoro": "tomorrow", "b4": "before",
    "gr8": "great", "msg": "message", "abt": "about", "ppl": "people", "sry": "sorry", "srry": "sorry",
    "k": "okay", "kk": "okay", "ok": "okay", "y": "why", "n": "and", "nd": "and",
}

# after "ur", these mean "you're" (ur so kind / ur right / ur doing great)
_UR_YOU_ARE = re.compile(
    r"(?:so|very|too|really|not|the|a|an|welcome|right|wrong|amazing|awesome|funny|smart|kind|nice|cute|sweet|"
    r"stupid|pathetic|useless|dumb|kidding|joking|crazy|annoying|boring|helpful|great|good|bad|best|worst|"
    r"\w+ing|\w+ed)\b",
    re.I,
)
# words that are commonly stretched and whose single-letter form is the intended word
_SQUEEZE_TO_ONE = {"so", "no", "yes", "hi", "hey", "please", "hello", "oh", "ok", "okay", "wow", "what", "why", "now",
                   "bye", "yay", "ugh", "omg", "lol", "thanks", "sad", "bad", "mad", "love", "tired", "sorry"}

_WORD = re.compile(r"[A-Za-z0-9']+|[^A-Za-z0-9'\s]+|\s+")


def _match_case(src: str, rep: str) -> str:
    if rep[:1] == "I" and rep[:2] in ("I ", "I'"):  # "I'm", "I don't know" keep their capital
        return rep
    if src[:1].isupper() and src[1:].islower():
        return rep[:1].upper() + rep[1:]
    if len(src) > 1 and src.isupper():
        return rep.upper() if len(rep) <= 4 else rep[:1].upper() + rep[1:]
    return rep


def _squeeze(word: str) -> str:
    if not re.search(r"([A-Za-z])\1{2,}", word):
        return word
    two = re.sub(r"([A-Za-z])\1{2,}", r"\1\1", word)
    one = re.sub(r"([A-Za-z])\1{2,}", r"\1", word)
    if one.lower() in _SQUEEZE_TO_ONE:
        return one
    return two


def normalize_informal(text: str) -> str:
    """Return ``text`` with chat shorthand expanded; see the module docstring."""
    if not text or not text.strip():
        return text
    t = re.sub(r"([!?])\1+", r"\1", text)          # "!!!" -> "!", "???" -> "?"
    t = re.sub(r"\?!+|!\?+", "?", t)
    t = re.sub(r"\.{4,}", "...", t)
    tokens: List[str] = _WORD.findall(t)
    out: List[str] = []
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if not re.match(r"[A-Za-z0-9']+$", tok):
            out.append(tok)
            continue
        tok = _squeeze(tok)
        low = tok.lower()
        nxt = next((x for x in tokens[i + 1:] if not x.isspace()), "")
        if low == "ur":
            rep = "you're" if _UR_YOU_ARE.match(nxt or "") else "your"
            out.append(_match_case(tok, rep))
        elif low == "i":
            out.append("I")
        elif low in ("k", "kk", "ok", "y", "n", "r") and not _standalone_ok(low, tokens, i):
            out.append(tok)
        elif low in SHORTHAND:
            out.append(_match_case(tok, SHORTHAND[low]))
        else:
            out.append(tok)
    return "".join(out)


def _standalone_ok(low: str, tokens: List[str], i: int) -> bool:
    """Short ambiguous tokens are expanded only in clear contexts."""
    words = [t.lower() for t in tokens if re.match(r"[A-Za-z0-9']+$", t)]
    if low in ("k", "kk", "ok"):
        return len(words) == 1          # "k" / "ok" as a whole message
    prev = next((t.lower() for t in reversed(tokens[:i]) if re.match(r"[A-Za-z0-9']+$", t)), "")
    nxt = next((t.lower() for t in tokens[i + 1:] if re.match(r"[A-Za-z0-9']+$", t)), "")
    if low == "r":                     # "how r u", "wat r ur plans", "r u ok"
        return nxt in ("u", "you", "ur", "your", "we", "they", "these", "those", "there") or prev in (
            "how", "what", "wat", "who", "where", "why", "when", "u", "you", "we", "they")
    if low == "y":                     # "y not", "y r u sad"
        return nxt in ("not", "r", "are", "is", "do", "did", "would", "u", "you") and i == _first_word_index(tokens)
    if low == "n":                     # "me n my friend"
        return bool(prev) and bool(nxt)
    return False


def _first_word_index(tokens: List[str]) -> int:
    for j, t in enumerate(tokens):
        if re.match(r"[A-Za-z0-9']+$", t):
            return j
    return -1


def changed_words(original: str, normalized: str) -> List[Tuple[str, str]]:
    """Pairs (original word, replacement) for logging and tests."""
    a = re.findall(r"[A-Za-z0-9']+", original)
    b = re.findall(r"[A-Za-z0-9']+", normalized)
    if len(a) == len(b):
        return [(x, y) for x, y in zip(a, b) if x != y]
    return [(original, normalized)] if original != normalized else []
