"""Deterministic application-level safety layer for Hibari Koto.

Scope (documented in Docs/09_EVALUATION_AND_TESTING.md):

* This module runs **outside** the neural model, in the Python chat session.
  It is NOT present when the GGUF is run directly in llama.cpp / Ollama; there
  only the behaviour the model learned from its training data applies.
* It intercepts exactly three high-risk categories and returns a fixed,
  reviewed response without calling the model:

    - ``SELF_HARM``  — suicidal ideation or self-harm intent
    - ``VIOLENCE``   — stated intent to kill / seriously hurt someone
    - ``HARMFUL_REQUEST`` — requests for instructions to cause serious harm

* Everything else (sadness, loneliness, anger, insults, profanity, identity
  questions, small talk) is answered by the model itself. There are no
  hardcoded identity answers and no automatic emoji injection.

Detection is pattern based. It normalises informal spelling ("wanna", "u",
"kms", stretched letters) and excludes common idioms ("killing it",
"kill time", "dying to see") so that ordinary profanity or slang is not
mistaken for an emergency. It is a safety net, not a classifier with
guaranteed recall — see the known-limitations document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List

# ──────────────────────────────────────────────────────────────
# Categories and fixed responses
# ──────────────────────────────────────────────────────────────


class SafetyCategory(str, Enum):
    NONE = "none"
    SELF_HARM = "self_harm"
    VIOLENCE = "violence"
    HARMFUL_REQUEST = "harmful_request"


# Helplines: India (the project's home region) plus the generic fallback.
# Tele-MANAS is the Government of India 24/7 mental-health line; 112 is the
# national emergency number.
CRISIS_RESOURCES = (
    "If you are in India, you can call Tele-MANAS at 14416 (free, 24/7) or "
    "iCall at 9152987821. If you are somewhere else, please contact your local "
    "crisis line. If you might act on these thoughts right now, call 112 or "
    "your local emergency number, or go to the nearest hospital."
)

SAFETY_RESPONSES = {
    SafetyCategory.SELF_HARM: (
        "I'm really sorry you're feeling this much pain, and I'm glad you told me. "
        "You deserve support from a real person right now. "
        + CRISIS_RESOURCES
        + " If you can, reach out to someone you trust and let them know how "
        "you're feeling. I'm still here to talk."
    ),
    SafetyCategory.VIOLENCE: (
        "It sounds like you're feeling a lot of anger right now. I can't help "
        "with hurting anyone, but I'm willing to listen. If you feel you might "
        "act on it, please step away from the situation and from anything you "
        "could use to hurt someone, and contact someone you trust. If anyone is "
        "in immediate danger, call 112 or your local emergency number. "
        "Do you want to tell me what happened?"
    ),
    SafetyCategory.HARMFUL_REQUEST: (
        "I can't help with that because it could seriously hurt someone. "
        "If something is going on that's making you think about this, I'm happy "
        "to talk about it."
    ),
}


@dataclass
class SafetyAssessment:
    category: SafetyCategory = SafetyCategory.NONE
    matched: List[str] = field(default_factory=list)

    @property
    def is_flagged(self) -> bool:
        return self.category is not SafetyCategory.NONE


# ──────────────────────────────────────────────────────────────
# Normalisation of informal text
# ──────────────────────────────────────────────────────────────

_SHORTHAND = [
    (r"\bwanna\b", "want to"),
    (r"\bgonna\b", "going to"),
    (r"\bgotta\b", "got to"),
    (r"\bim\b", "i am"),
    (r"\bi'm\b", "i am"),
    # without these, "i'll hurt him" / "i'd kill him" never reached the "i will" / "i would" patterns
    (r"\bi'll\b", "i will"),
    (r"\bi'd\b", "i would"),
    (r"\bu\b", "you"),
    (r"\bur\b", "your"),
    (r"\bmyslef\b", "myself"),
    (r"\bmy self\b", "myself"),
    (r"\bdont\b", "don't"),
    (r"\bcant\b", "can't"),
    (r"\bwont\b", "won't"),
    (r"\bkms\b", "kill myself"),
    (r"\bkys\b", "kill yourself"),
    (r"\bunalive\b", "kill"),
    (r"\bsuicidal\b", "suicide"),
]


def normalize_text(text: str) -> str:
    """Lowercase, replace curly quotes, squeeze stretched letters and expand shorthand."""
    t = text.lower()
    t = t.replace("’", "'").replace("‘", "'")
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    # "diieeee" -> "die", "killl" -> "kill" (keep doubled letters such as 'kill')
    t = re.sub(r"([a-z])\1{2,}", r"\1\1", t)
    t = re.sub(r"\bdiie+\b|\bdie+\b", "die", t)
    for pat, rep in _SHORTHAND:
        t = re.sub(pat, rep, t)
    return re.sub(r"\s+", " ", t).strip()


# ──────────────────────────────────────────────────────────────
# Patterns
# ──────────────────────────────────────────────────────────────

# First-person intent: "i want to", "i am going to", "i will", "i feel like" ...
_INTENT = (
    r"\bi (?:really |just |seriously |actually |literally |swear i )?(?:am )?"
    r"(?:want(?:ed)? to|going to|plan(?:ning)? to|will|'ll|about to|try(?:ing)? to|"
    r"wish i could|feel like|should|might|have to|need to|thinking (?:about|of)|could)"
)
_VIOLENT_VERB = (
    r"(?:kill|killing|murder|murdering|stab|stabbing|shoot|shooting|strangle|choke|poison|poisoning|"
    r"beat up|beating up|attack|attacking|hurt|hurting|harm|harming|punch|punching|slap|slapping|"
    r"burn down|set fire to)"
)

_SELF_HARM_PATTERNS = [
    r"\bkill(?:ing)? myself\b",
    r"\bend(?:ing)? my (?:own )?life\b",
    r"\btake my (?:own )?life\b",
    r"\bend it all\b",
    r"\bsuicide\b",
    r"\bself harm\b|\bselfharm\b",
    r"\b(?:hurt|harm|cut|cutting|starve) myself\b",
    r"\bhang myself\b",
    r"\boverdos(?:e|ing)\b",
    r"\b(?:want|wish|deserve|going|ready) to die\b",
    r"\bi (?:just )?(?:want|wish) i (?:was|were) dead\b",
    r"\bi wish i (?:was|were) never born\b",
    r"\b(?:don't|do not|no longer) want to (?:live|be alive|exist|wake up)\b",
    r"\bnot worth living\b",
    r"\bno reason to live\b",
    r"\bbetter off (?:dead|without me)\b",
    r"\b(?:better|happier|easier) (?:off )?(?:if i (?:was|were|am) (?:gone|dead|not here|not around)|if i (?:wasn't|weren't|disappeared|died)|without me)\b",
    r"\b(?:no one|nobody) would (?:care|notice|miss me) if i (?:died|was gone|disappeared|killed myself)\b",
    r"\bcan't go on (?:living|anymore|like this)\b",
    r"\bjump off (?:a|the|this) (?:bridge|building|roof|cliff)\b",
    r"\b(?:sleep|go to sleep) and never wake up\b",
]

_VIOLENCE_PATTERNS = [
    # explicit intent + violent verb (+ optional object)
    _INTENT + r" " + _VIOLENT_VERB + r"\b",
    r"\bi(?: am| will|'ll| would| want to)? (?:kill|murder|stab|shoot) (?:you|him|her|them|everyone|somebody|someone|people|my \w+)\b",
    r"\bi am going to make (?:him|her|them) pay\b",
    r"\b(?:bring|take) a (?:gun|knife) to (?:school|work|class)\b",
]

_HARMFUL_REQUEST_PATTERNS = [
    r"\bhow (?:do i|to|can i|would i|could i) (?:make|build|create) (?:a )?(?:bomb|explosive|pipe bomb|poison|nerve gas|meth)\b",
    r"\bhow (?:do i|to|can i|would i|could i) (?:kill|murder|poison|hurt|stab|shoot) (?:someone|somebody|a person|people|my \w+|him|her|them)\b",
    r"\bhow (?:do i|to|can i) (?:get away with|hide a body|dispose of a body)\b",
    r"\b(?:mix|use|make|give|buy|put)\b.{0,40}\bto (?:poison|kill|hurt|drug) (?:someone|somebody|a person|people|my \w+|him|her|them)\b",
    r"\b(?:poison|drug) (?:my|his|her|their) (?:neighbou?r|wife|husband|boss|friend|teacher|family|roommate|parents?|brother|sister)\b",
    r"\bbest way to (?:kill|poison|hurt) (?:someone|somebody|a person|people|myself)\b",
    r"\bhow (?:do i|to|can i|much .* to) (?:overdose|kill myself|hang myself)\b",
]

# Idioms that contain violent words but are not a threat.
_BENIGN_IDIOMS = [
    r"\bkill(?:ing|ed)? (?:it|time|the game|the lights|the vibe|the engine|the process)\b",
    r"\b(?:this|that|it|you|homework|work|traffic|my back|my feet) (?:is|are) killing me\b",
    r"\bkilling me softly\b",
    r"\b(?:dying|die) (?:to|for) (?:see|know|try|hear|go|meet|eat|have)\b",
    r"\bdie (?:of|from) (?:laughter|laughing|embarrassment|boredom)\b",
    r"\bkill (?:for|a) (?:a )?(?:pizza|coffee|burger|nap|beer)\b",
    r"\b(?:video )?game\b.*\bkill\b|\bkill\b.*\b(?:in the game|boss|zombies?|monsters?|enemies|npc)\b",
    # negated intent: "i don't want to hurt you", "i would never kill anyone"
    r"\b(?:don't|do not|never|wouldn't|would never|not) (?:want to |going to |try to |mean to )?" + _VIOLENT_VERB + r"\b\s*\w*",
    r"\bhurt (?:your|his|her|their|my) feelings\b",
    r"\bshoot (?:you|me|him|her|them|us) (?:a |an )?(?:quick )?(?:message|text|email|e mail|mail|line|dm|note|reply)\b",
]


def _matches(patterns: List[str], text: str) -> List[str]:
    return [p for p in patterns if re.search(p, text)]


def assess_message(text: str) -> SafetyAssessment:
    """Classify a user message into a safety category (deterministic)."""
    if not text or not text.strip():
        return SafetyAssessment()
    norm = normalize_text(text)

    # Self-harm is checked first: "kill myself" must never be treated as violence.
    hits = _matches(_SELF_HARM_PATTERNS, norm)
    if hits:
        return SafetyAssessment(SafetyCategory.SELF_HARM, hits)

    hits = _matches(_HARMFUL_REQUEST_PATTERNS, norm)
    if hits:
        return SafetyAssessment(SafetyCategory.HARMFUL_REQUEST, hits)

    stripped = norm
    for idiom in _BENIGN_IDIOMS:
        stripped = re.sub(idiom, " ", stripped)
    hits = _matches(_VIOLENCE_PATTERNS, stripped)
    if hits:
        return SafetyAssessment(SafetyCategory.VIOLENCE, hits)

    return SafetyAssessment()


def get_safety_response(category: SafetyCategory) -> str:
    """Return the fixed, reviewed response for a flagged category."""
    if category is SafetyCategory.NONE:
        raise ValueError("No safety response for SafetyCategory.NONE")
    return SAFETY_RESPONSES[category]


# ──────────────────────────────────────────────────────────────
# Backwards-compatible helpers
# ──────────────────────────────────────────────────────────────


def detect_crisis(text: str) -> bool:
    """True when the message indicates self-harm or suicidal ideation."""
    return assess_message(text).category is SafetyCategory.SELF_HARM


def detect_violence(text: str) -> bool:
    """True when the message states an intent to seriously hurt someone."""
    return assess_message(text).category is SafetyCategory.VIOLENCE


def get_crisis_response() -> str:
    return SAFETY_RESPONSES[SafetyCategory.SELF_HARM]


# ──────────────────────────────────────────────────────────────
# Output post-processing (light, deterministic, no content injection)
# ──────────────────────────────────────────────────────────────

_CHAT_MARKERS = ["<|eos|>", "<|user|>", "<|assistant|>", "<|system|>", "<|pad|>", "<|bos|>", "<|unk|>"]
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF]️?")


def clean_response(text: str, max_emojis: int = 1) -> str:
    """Strip chat markers, collapse whitespace, cap emoji count, drop a dangling fragment.

    This never adds content: it only removes artefacts. If the model's reply
    ends in an unfinished sentence after at least one complete sentence, the
    unfinished tail is removed (common when ``max_new_tokens`` is reached).
    """
    for marker in _CHAT_MARKERS:
        text = text.replace(marker, " ")
    text = re.sub(r"[ \t]+", " ", text).strip()

    seen = 0

    def _cap(m: re.Match) -> str:
        nonlocal seen
        seen += 1
        return m.group(0) if seen <= max_emojis else ""

    text = _EMOJI_RE.sub(_cap, text)
    text = re.sub(r"\s+([.,!?])", r"\1", text).strip()

    if text and text[-1] not in ".!?\"')":
        last = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
        if last > 0:
            text = text[: last + 1]
    return text


def process_response(model_response: str, user_message: str) -> str:
    """Final pipeline: safety override for flagged inputs, otherwise clean the model text."""
    assessment = assess_message(user_message)
    if assessment.is_flagged:
        return get_safety_response(assessment.category)
    return clean_response(model_response)
