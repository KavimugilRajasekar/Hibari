"""Rule-based analysis of a user message: intent, emotion, hostility.

This is a *routing and checking* component. It never writes a reply. It is
used to

* route rewriting / e-mail requests (``hibari.inference.tasks``),
* tell the reply critic (``hibari.inference.reranking``) what kind of answer
  would be appropriate, so it can choose between model-generated candidates
  (e.g. reject "That's great!" after "My grandmother passed away"),
* label evaluation cases.

The analysis works on the informal-normalised text
(``hibari.inference.normalization``), handles simple negation ("I'm not
sad", "not happy at all") and distinguishes the user's own feelings from a
third person's ("my friend is going through a hard time" -> sympathy).

It is deliberately transparent: every decision lists the rule that fired in
``MessageAnalysis.evidence``. It is an English-only heuristic, not a trained
classifier; see Koto ``docs/18_KNOWN_LIMITATIONS.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from hibari.inference.normalization import normalize_informal
from hibari.inference.tasks import parse_task_request


class Intent(str, Enum):
    EMPTY = "empty"
    NOISE = "noise"                       # punctuation / symbols only
    GREETING = "greeting"
    FAREWELL = "farewell"
    THANKS = "thanks"
    APOLOGY = "apology"
    COMPLIMENT = "compliment"
    HOSTILE = "hostile"                   # insult / abuse aimed at the assistant
    IDENTITY = "identity"                 # who / what are you, who made you
    REWRITE_REQUEST = "rewrite_request"
    EMAIL_REQUEST = "email_request"
    EMOTIONAL = "emotional_disclosure"    # the user shares a feeling or an emotional event
    QUESTION = "question"
    REQUEST = "request"                   # asks for help / an action
    STATEMENT = "statement"


class Emotion(str, Enum):
    NONE = "none"
    GRIEF = "grief"
    SADNESS = "sadness"
    LONELINESS = "loneliness"
    FRUSTRATION = "frustration"
    ANGER = "anger"
    ANXIETY = "anxiety"                   # nervous, worried, scared, stressed
    DISAPPOINTMENT = "disappointment"
    CONFUSION = "confusion"
    EXCITEMENT = "excitement"
    HAPPINESS = "happiness"


NEGATIVE = {Emotion.GRIEF, Emotion.SADNESS, Emotion.LONELINESS, Emotion.FRUSTRATION, Emotion.ANGER,
            Emotion.ANXIETY, Emotion.DISAPPOINTMENT}
POSITIVE = {Emotion.EXCITEMENT, Emotion.HAPPINESS}


@dataclass
class MessageAnalysis:
    text: str
    normalized: str
    intent: Intent
    emotion: Emotion = Emotion.NONE
    subject: str = "self"                 # whose feeling / situation: "self" or "other"
    hostile: bool = False
    informal: bool = False
    is_question: bool = False
    rewrite_task: Optional[str] = None
    needs_live_data: bool = False         # weather, live scores, news, prices, "right now"
    evidence: List[str] = field(default_factory=list)

    @property
    def valence(self) -> int:
        if self.emotion in NEGATIVE or self.hostile:
            return -1
        if self.emotion in POSITIVE:
            return 1
        return 0

    def to_dict(self) -> Dict[str, object]:
        return {"intent": self.intent.value, "emotion": self.emotion.value, "valence": self.valence,
                "subject": self.subject, "hostile": self.hostile, "informal": self.informal,
                "is_question": self.is_question, "rewrite_task": self.rewrite_task, "needs_live_data": self.needs_live_data, "evidence": self.evidence}


# ───────────────────────── lexicons ─────────────────────────

_FAMILY = (r"(?:mom|mum|mother|dad|father|grand(?:ma|pa|mother|father|mom|dad)|nana|granny|brother|sister|son|"
           r"daughter|wife|husband|uncle|aunt|aunty|auntie|cousin|friend|best friend|dog|cat|pet|puppy|kitten|"
           r"grandparents?|parents?|baby|partner|boyfriend|girlfriend|teacher|neighbou?r)")
_PEOPLE = _FAMILY + r"|(?:colleague|classmate|roommate|coworker|co-worker|team ?mate|team)"

EMOTION_PATTERNS: List[Tuple[Emotion, str]] = [
    (Emotion.GRIEF, rf"\b(?:passed away|passed on|died|has died|is dead|death of|funeral|lost my {_FAMILY}|rest in peace|\brip\b|miss (?:him|her|them) so much)\b"),
    (Emotion.GRIEF, rf"\bmy {_FAMILY} (?:passed|died|is gone)\b"),
    (Emotion.LONELINESS, r"\b(?:lonely|loneliness|so alone|all alone|feel alone|no friends|nobody (?:wants|cares|talks|likes|understands)|no one (?:wants|cares|talks|likes|understands)|left out|isolated)\b"),
    (Emotion.SADNESS, r"\b(?:sad|sadness|unhappy|depressed|down lately|feel(?:ing)? down|heartbroken|broke up|break ?up|miserable|crying|cried|cry|upset|devastated|feel(?:ing)? (?:low|empty|awful|terrible|horrible|bad|hopeless|worthless)|hopeless|having a (?:hard|rough|tough|bad) (?:time|day|week))\b"),
    (Emotion.FRUSTRATION, r"\b(?:frustrat\w*|annoy\w*|irritat\w*|fed up|sick of|tired of|keeps? (?:failing|crashing|breaking)|nothing (?:works|is working)|stuck on|so done with|this (?:day|week) sucks|sucks|ugh+)\b"),
    (Emotion.ANGER, r"\b(?:angry|anger|mad at|so mad|furious|pissed|rage|i hate (?:my|this|it|when|that|everything|everyone))\b"),
    (Emotion.ANXIETY, r"\b(?:nervous|anxious|anxiety|worried|worry|scared|afraid|stress(?:ed|ful)?|panic\w*|terrified|overwhelm\w*|freaking out|can't sleep|pressure)\b"),
    (Emotion.DISAPPOINTMENT, r"\b(?:disappoint\w*|let down|failed (?:my|the|an?|again|in)|fail(?:ed)? again|didn't (?:get|pass|make it|get selected|get the)|got rejected|rejected|not selected|lost (?:the|my|our) (?:match|game|job|final|competition)|got fired|laid off)\b"),
    (Emotion.DISAPPOINTMENT, r"\b(?:said|told me) my \w+ (?:was|is) (?:bad|terrible|awful|not good|wrong|poor)\b"),
    (Emotion.FRUSTRATION, r"\b(?:not|never) (?:happy|satisfied|pleased|okay) with\b"),
    (Emotion.CONFUSION, r"\b(?:confus\w*|don't understand|do not understand|makes no sense|not making sense|i'm lost|so lost|no idea what)\b"),
    (Emotion.EXCITEMENT, r"\b(?:excited|exciting|can't wait|thrilled|pumped|got (?:the job|accepted|selected|promoted|admission|in(?:to)?)|accepted (?:into|to|at)|passed (?:my|the|an?) \w+|i won|we won|got engaged|got married|yay+|woo+|great news|good news)\b"),
    (Emotion.HAPPINESS, r"\b(?:happy|glad|so good|amazing day|great day|wonderful day|proud of myself|feeling (?:great|good|awesome|amazing)|love (?:it|this|my))\b"),
]

# "not sad", "not happy at all", "never felt lonely" -> the emotion does not apply
_NEGATION = r"\b(?:not|never|no longer|isn't|wasn't|aren't|don't|didn't|won't|n't)\s+(?:\w+\s+){0,2}$"

_HOSTILE = [
    r"\b(?:fuck|screw|f\*+k|fk) (?:you|u|off|this bot)\b",
    r"\bstfu\b|\bshut up\b|\bgo to hell\b|\bgo die\b",
    r"\byou(?:'re| are|r)? (?:so |such an? |a |an |really |fucking |the most )*(?:stupid|pathetic|useless|dumb|idiot|moron|worthless|trash|garbage|terrible|awful|annoying|lame|bad bot|loser|joke|disgrace)\b",
    r"\b(?:stupid|pathetic|useless|dumb|idiot|worthless) (?:bot|ai|machine|thing)\b",
    r"\bi hate you\b|\byou suck\b|\bnobody likes you\b",
]
_COMPLIMENT = r"\byou(?:'re| are)? (?:so |really |very )?(?:smart|clever|helpful|kind|nice|sweet|funny|awesome|amazing|great|the best|cool|cute)\b|\b(?:good|great|nice) (?:job|work|answer)\b|\bi (?:like|love) (?:you|talking to you)\b"
_GREETING = r"^\W*(?:hi+|hello+|hey+|heya|hiya|yo|sup|wass?up|what's up|howdy|good (?:morning|afternoon|evening)|greetings|namaste|vanakkam)\b"
_FAREWELL = r"\b(?:bye+|goodbye|good night|see (?:you|ya)|talk (?:to you )?later|gotta go|got to go|i have to go|i'm leaving|take care|later)\W*$"
_THANKS = r"\b(?:thanks|thank you|thank u|cheers|much appreciated|appreciate it)\b"
_APOLOGY = r"^\W*(?:sorry|my bad|apologies|i apologi[sz]e)\b"
_IDENTITY = (r"\b(?:your name|who are you|what are you\b(?!\s+(?:\w+ing|up to|thinking|planning|into))|who (?:made|built|created|trained|designed) you|are you (?:a |an )?(?:human|real|person|bot|robot|ai|machine|computer|alive|chatgpt)|"
             r"what (?:model|architecture) are you|what is hibari|what is koto|tell me about yourself|introduce yourself)\b")
_EMAIL = r"\b(?:write|draft|compose|create|prepare|help me (?:write|with))\b.{0,40}\b(?:e-?mail|mail|letter)\b"
_QUESTION_START = r"^\W*(?:what|who|whom|whose|where|when|why|how|which|is|are|am|was|were|do|does|did|can|could|would|will|should|shall|may|have|has)\b"
_REQUEST = r"\b(?:can you|could you|would you|will you|please|help me|tell me|give me|show me|explain|suggest|recommend|i need (?:help|advice|you to)|let's)\b"
_OTHER_SUBJECT = rf"\b(?:my|our|his|her|their) (?:{_PEOPLE})\b(?:'s)?"
_LIVE_DATA = (r"\b(?:weather|forecast|temperature (?:in|at|today|tomorrow)|will it rain|score (?:of|in)|live score|who (?:is )?winning|"
              r"latest news|news today|today'?s news|stock price|share price|exchange rate|bitcoin price|price of \w+ (?:today|now)|"
              r"what time is it|today'?s date|what day is (?:it|today)|current (?:president|prime minister|price|score|time|news|ceo)|"
              r"right now|at the moment)\b")
_INFORMAL_MARK = r"\b(?:u|ur|r|wat|pls|plz|thx|idk|lol|lmao|omg|im|dont|cant|gonna|wanna|btw|rn|nvm|tmrw|cuz|bro|bruh)\b"


def _emotion(norm: str, evidence: List[str]) -> Emotion:
    for emo, pat in EMOTION_PATTERNS:
        for m in re.finditer(pat, norm):
            if re.search(_NEGATION, norm[: m.start()]):
                evidence.append(f"negated:{emo.value}:{m.group(0)}")
                continue
            evidence.append(f"emotion:{emo.value}:{m.group(0)}")
            return emo
    return Emotion.NONE


def analyze_message(text: str) -> MessageAnalysis:
    """Classify ``text``; see the module docstring."""
    raw = text or ""
    if not raw.strip():
        return MessageAnalysis(raw, "", Intent.EMPTY)
    if not re.search(r"[A-Za-z0-9]", raw):
        return MessageAnalysis(raw, raw.strip(), Intent.NOISE, is_question="?" in raw)

    norm_text = normalize_informal(raw)
    norm = norm_text.lower().replace("’", "'")
    evidence: List[str] = []
    informal = bool(re.search(_INFORMAL_MARK, raw.lower())) or norm_text != raw
    is_question = "?" in raw or bool(re.match(_QUESTION_START, norm))
    a = MessageAnalysis(raw, norm_text, Intent.STATEMENT, informal=informal, is_question=is_question, evidence=evidence)

    if is_question or re.search(r"\b(?:tell me|check)\b", norm):
        m = re.search(_LIVE_DATA, norm)
        if m:
            a.needs_live_data = True
            evidence.append(f"live_data:{m.group(0)}")

    req = parse_task_request(raw)
    if req is not None:
        a.intent, a.rewrite_task = Intent.REWRITE_REQUEST, req.task
        evidence.append(f"task:{req.task}")
        return a
    if re.search(_EMAIL, norm):
        a.intent = Intent.EMAIL_REQUEST
        evidence.append("email")
        return a

    a.emotion = _emotion(norm, evidence)
    other = re.search(_OTHER_SUBJECT, norm)
    if a.emotion not in (Emotion.NONE, Emotion.GRIEF) and other and not re.search(r"\bi(?:'m| am| feel| was|'ve)\b", norm):
        a.subject = "other"
        evidence.append(f"subject:other:{other.group(0)}")
    elif a.emotion is Emotion.NONE and other and re.search(
            r"\b(?:going through|struggling|sick|ill|in (?:the )?hospital|hard time|difficult time|tough time|not well|unwell|depressed|surgery|cancer)\b", norm):
        a.emotion, a.subject = Emotion.SADNESS, "other"
        evidence.append(f"subject:other-difficulty:{other.group(0)}")

    for pat in _HOSTILE:
        m = re.search(pat, norm)
        if m:
            a.hostile = True
            evidence.append(f"hostile:{m.group(0)}")
            break

    n_words = len(re.findall(r"[a-z0-9']+", norm))
    if a.hostile:
        a.intent = Intent.HOSTILE
    elif re.search(_GREETING, norm) and n_words <= 6 and a.emotion is Emotion.NONE:
        a.intent = Intent.GREETING
    elif re.search(_FAREWELL, norm) and n_words <= 8:
        a.intent = Intent.FAREWELL
    elif re.search(_THANKS, norm) and n_words <= 10 and a.emotion is Emotion.NONE:
        a.intent = Intent.THANKS
    elif re.search(_APOLOGY, norm) and n_words <= 10:
        a.intent = Intent.APOLOGY
    elif re.search(_IDENTITY, norm):
        a.intent = Intent.IDENTITY
    elif a.emotion is not Emotion.NONE:
        a.intent = Intent.EMOTIONAL
    elif re.search(_COMPLIMENT, norm):
        a.intent = Intent.COMPLIMENT
    elif is_question:
        a.intent = Intent.QUESTION
    elif re.search(_REQUEST, norm):
        a.intent = Intent.REQUEST
    evidence.append(f"intent:{a.intent.value}")
    return a
