"""Choose the best reply among several model-generated candidates.

Why: a small model's single most likely reply is often generic or
mismatched ("I'm glad I could help." after an insult, "That is so nice of
her." after a bereavement), while a handful of sampled replies usually
contains an appropriate one. Selecting among the model's *own* candidates
improves relevance without hand-written answers and without retraining.

Every candidate is produced by the model. Its score combines

1. **fluency** — mean log-probability per token of the reply given the
   conversation (the end-of-turn token included, so complete replies win);
2. **context relevance** — pointwise mutual information: how much more
   likely the reply is given *this* conversation than given an empty user
   turn. Generic replies that fit any message ("I'm glad I could help.")
   score low; replies that respond to what was said score high
   (Li et al., 2016, "A Diversity-Promoting Objective Function");
3. **appropriateness checks** from :class:`ReplyCritic` — transparent,
   named rules driven by the message analysis
   (``hibari.inference.intent``): affect mismatch (cheerful reply to grief,
   condolences for good news), thanking an insult, claiming human
   experiences, entities and topics that appear from nowhere, echoing,
   repetition, truncation, length that does not fit the message, and e-mail
   structure for e-mail requests.

The critic never edits or writes text; it only ranks. If every candidate is
poor, the best poor candidate is still returned — the limitation stays
visible instead of being hidden behind a canned reply.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from hibari.evaluation.metrics import char_ngram_f, echo_ratio, ngram_repetition, repeated_sentence, words
from hibari.inference.intent import NEGATIVE, POSITIVE, Emotion, Intent, MessageAnalysis

# ───────────────────────── reply patterns ─────────────────────────

POSITIVE_REPLY = (r"\b(?:that'?s|that is|how|sounds?|it'?s) (?:so |really |very )?(?:great|nice|wonderful|awesome|amazing|cool|fun|exciting|lovely|good)\b"
                  r"|\bcongrat\w*|\bglad (?:to hear|you|that|i)\b|\bhappy (?:for you|to hear)\b|\bi'?m glad\b|\bgood for you\b"
                  r"|\bnice of (?:her|him|them|you)\b|\blucky\b|\byou did (?:well|great)\b|\bi'?m sure you did\b|\bblessed\b|\byay\b|\ba lot of fun\b")
NEGATIVE_REPLY = r"\bsorry (?:to hear|for)\b|\bthat'?s (?:terrible|awful|sad|too bad|a shame|rough|tough)\b|\boh no\b|\bthat sucks\b|\bcondolences\b"
EMPATHY_REPLY = (r"\bsorry\b|\bthat sounds\b|\bsounds (?:hard|tough|difficult|stressful|frustrating|painful|exhausting|lonely|scary)\b|\bmust (?:be|feel|have been)\b"
                 r"|\bunderstand\b|\bunderstandable\b|\b(?:terrible|awful|horrible|heartbreaking)\b|\b(?:hard|tough|difficult|painful|stressful|frustrating)\b|\bhere for you\b|\bhere to listen\b"
                 r"|\bit'?s (?:okay|ok|normal|natural) to\b|\bi hear you\b|\bthat can be\b|\bloss\b|\bwhat'?s (?:going on|wrong|on your mind)\b|\bwhat happened\b|\bwant to talk\b")
CELEBRATE_REPLY = r"\bcongrat\w*|\b(?:great|wonderful|awesome|amazing|fantastic|exciting|brilliant) (?:news)?\b|\bhappy for you\b|\bwell done\b|\bproud\b|\bthat'?s (?:great|wonderful|awesome|amazing)\b"
DEESCALATE_REPLY = (r"\bsorry\b|\bfrustrat\w*|\bwhat went wrong\b|\bwhat (?:happened|is going on|'?s going on|'?s wrong)\b|\b(?:keep|keeping) (?:things|it|this) respectful\b"
                    r"|\b(?:try|trying) to (?:help|do better|fix)\b|\bdo better\b|\btell me (?:what|how)\b|\bupset\b|\bhow can i help\b|\bi'?m here\b")
UNSURE_REPLY = (r"\b(?:don'?t know|not sure|can'?t (?:check|look|access|see|get|browse)|cannot (?:check|access|browse)|no way (?:of|to) (?:know|check)|"
                r"don'?t have (?:access|live|real-time|current|up-to-date)|no (?:internet|live|real-time) (?:access|data|information)|weather (?:app|website|service)|check (?:a|an|the) )\b")
# where a user states a fact about themselves (used to tell "never told" from "forgotten")
USER_FACT_GIVEN = {
    "name": r"\bmy name(?:'s| is)\b|\bi'?m called\b|\bcall me\b|\bi am [a-z]+\b(?=[.!,]|$)|\bthis is [a-z]+\b(?=[.!,]|$)",
    "age": r"\bi(?:'m| am) \d+\b|\byears old\b",
    "city": r"\bi live in\b|\bi'?m from\b|\bi am from\b",
    "job": r"\bi work (?:as|at|in)\b|\bi(?:'m| am) an? (?:teacher|student|developer|engineer|nurse|doctor|designer)\b",
}
# replies written for "someone you care about is struggling", wrong when nobody was mentioned
THIRD_PARTY_REPLY = r"\bsomeone (?:close to you|you care about|you love)\b|\bfor them\b|\bwatching someone\b|\bbeing there for them\b|\beasier for them\b"
THIRD_PARTY_USER = (r"\b(?:my|our|his|her|their) (?:friend|family|mom|mum|dad|mother|father|brother|sister|son|daughter|wife|husband|partner|"
                    r"boyfriend|girlfriend|cousin|uncle|aunt|grand\w+|colleague|roommate|neighbou?r|boss|manager|teacher|team|kid|child)\b|\b(?:he|she|they)\b")
NOT_TOLD_REPLY = (r"\b(?:haven'?t (?:told|said|mentioned|shared)|didn'?t (?:tell|say|mention|share)|don'?t know (?:your|it|that)|not sure what your|"
                  r"you (?:haven'?t|didn'?t)|what(?:'s| is) your name\?)")
ACKNOWLEDGE_VIEW_REPLY = (r"\b(?:fair (?:enough|point)|good point|i see|that'?s (?:okay|fine|fair)|you (?:might|may|could) be right|"
                          r"why (?:do|would) you|what do you think|how do you see|i understand|interesting|tell me (?:why|more)|"
                          r"different (?:view|opinion|ways?)|that'?s a (?:good|fair) point)\b")
REPAIR_REPLY = (r"\b(?:sorry|my mistake|my bad|i misunderstood|what did you mean|what do you mean|could you (?:explain|clarify|say)|"
                r"can you (?:explain|clarify|tell me)|let me try again|i see,? so)\b")
TOPIC_CHANGE_REPLY = (r"\b(?:sure|of course|okay|ok|alright|happy to|let'?s|what would you like|what do you want|what'?s on your mind|"
                      r"anything in mind|what else)\b")
CONDOLENCE_REPLY = r"\bsorry for your loss\b|\bcondolences\b|\bmay (?:she|he|they) rest in peace\b"
GRATEFUL_REPLY = r"\bglad (?:i|to) (?:could |was able to |can )?help\b|\byou'?re welcome\b|\bthank you\b|\bthanks\b|\bi'?m glad\b|\bglad to hear\b|\bmy pleasure\b"
AGREE_INSULT_REPLY = r"^\W*(?:(?:yes|yeah|yep|true)\W*)?(?:i am|i'?m|yes|true|you'?re right|you are right)?\W*$|\byou(?:'re| are) (?:so )?right\b|\bi(?:'m| am) (?:so )?(?:pathetic|stupid|useless|dumb|an idiot|worthless)\b"
INSULT_WORDS = r"\b(?:stupid|idiot|pathetic|useless|dumb|moron|loser|shut up|fuck\w*|shit|bitch|hate you)\b"
WELCOME_REPLY = r"\b(?:welcome|anytime|glad|pleasure|happy to help|no problem|no worries)\b"
GREETING_REPLY = r"\b(?:hi|hello|hey|good (?:morning|afternoon|evening))\b|\?"
FAREWELL_REPLY = r"\b(?:bye|goodbye|see you|take care|talk (?:to you )?(?:soon|later)|good night|nice (?:talking|chatting))\b"
HUMAN_CLAIM = (r"\bmy (?:wife|husband|kids?|son|daughter|mom|mum|dad|mother|father|girlfriend|boyfriend|parents|family|brother|sister|grand\w+)\b"
               r"|\bwhen i was (?:a kid|young|little|in school|a child)\b|\bi (?:also )?(?:have|had) (?:a|two|three|my) (?:dog|cat|kids?|children|brother|sister|job)\b"
               r"|\bi lost my\b|\b(?<!not )i(?:'m| am) (?:a )?(?:human|person|real person)\b|\bi (?:went|go) to (?:school|college|work)\b|\bmy (?:job|boss|house|car)\b")
EMAIL_SUBJECT = re.compile(r"^\s*subject\s*:.*$", re.I | re.M)
EMAIL_GREETING = re.compile(r"^\s*(?:dear|hi|hello|good (?:morning|afternoon))\b.*$", re.I | re.M)
EMAIL_CLOSING = re.compile(r"^\s*(?:best regards|kind regards|regards|sincerely|thank you|thanks|best|warm regards|yours (?:sincerely|faithfully))\b.*$", re.I | re.M)
# a yes/no question about what the user has just said ("Did she pass away?")
_YESNO_Q = re.compile(r"(?:^|(?<=[.!?]\s))((?:did|is|was|are|were|do|does|has|have)\b[^.!?]*\?)", re.I)
_STOP = set("i me my you your he she it we they them her him his its our their a an the is are was were be been do does did "
            "have has had to of in on at for with and or but so that this what why how when where who which can could would "
            "will just really very about".split())
# topics that should never appear unless the user brought them up
OUT_OF_NOWHERE_TOPICS = (r"\b(?:holocaust|hitler|nazi\w*|genocide|terroris\w*|bomb\w*|murder\w*|suicid\w*|rape\w*|porn\w*|sex\w*|drugs?|cocaine|heroin|"
                         r"war|weapons?|guns?|kill\w*|dead|death|cancer|racis\w*|slave\w*|religio\w*|politic\w*|elections?)\b")
_COMMON_CAPITALISED = {"i", "i'm", "i've", "i'll", "i'd", "ok", "okay", "ai", "mr", "mrs", "ms", "dr", "english", "monday", "tuesday",
                       "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march", "april", "may",
                       "june", "july", "august", "september", "october", "november", "december", "subject", "dear", "regards",
                       "best", "thanks", "thank", "sincerely", "hi", "hello", "hey", "your", "name", "manager", "team"}

def _has_word(word: str, text: str) -> bool:
    """``word`` occurs in ``text`` as a whole word, singular or plural (both lowercase):
    "war" is not in "award", "sam" is not in "same", "weapon" is in "weapons"."""
    stem = word[:-1] if word.endswith("s") and len(word) > 3 else word
    return re.search(rf"(?<![a-z0-9]){re.escape(stem)}s?(?![a-z0-9])", text) is not None


INTENT_MAX_WORDS = {Intent.GREETING: 30, Intent.THANKS: 25, Intent.FAREWELL: 25, Intent.APOLOGY: 30, Intent.COMPLIMENT: 30,
                    Intent.HOSTILE: 40, Intent.NOISE: 30, Intent.IDENTITY: 45, Intent.EMOTIONAL: 70, Intent.EMAIL_REQUEST: 220}
DEFAULT_MAX_WORDS = 90


@dataclass
class CriticWeights:
    """Weights of the score terms. Penalties are positive numbers that are subtracted."""

    fluency: float = 1.0
    relevance: float = 0.4
    affect_mismatch: float = 3.0
    empathy_bonus: float = 1.0
    hostile_thanks: float = 3.0
    hostile_agree: float = 3.0
    hostile_retaliate: float = 3.0
    deescalate_bonus: float = 1.0
    intent_bonus: float = 0.6
    human_claim: float = 1.5
    out_of_nowhere_topic: float = 3.0
    unknown_entity: float = 0.75
    echo: float = 1.5
    repetition: float = 1.5
    incomplete: float = 1.0
    too_long: float = 1.0
    too_short_emotional: float = 0.5
    question_dodged: float = 0.75
    no_acknowledgement: float = 0.75
    honest_limit: float = 0.75
    redundant_question: float = 0.75
    relevance_cap: float = 1.5
    email_structure: float = 1.5
    empty: float = 10.0


@dataclass
class Candidate:
    text: str
    token_ids: List[int]
    stopped_on_eos: bool
    source: str
    logprob: float = 0.0          # sum log p(reply [+ eos] | conversation)
    generic_logprob: float = 0.0  # sum log p(reply [+ eos] | empty user turn)
    n_scored_tokens: int = 1
    adjustments: List[Tuple[str, float]] = field(default_factory=list)
    score: float = 0.0

    @property
    def fluency(self) -> float:
        return self.logprob / max(1, self.n_scored_tokens)

    @property
    def relevance(self) -> float:
        return (self.logprob - self.generic_logprob) / max(1, self.n_scored_tokens)

    def to_dict(self) -> Dict[str, object]:
        return {"text": self.text, "source": self.source, "score": round(self.score, 3), "fluency": round(self.fluency, 3),
                "relevance": round(self.relevance, 3), "stopped_on_eos": self.stopped_on_eos,
                "adjustments": [(k, round(v, 3)) for k, v in self.adjustments]}


class ReplyCritic:
    """Named, explainable appropriateness rules for a candidate reply."""

    def __init__(self, identity_terms: Iterable[str] = (), weights: Optional[CriticWeights] = None):
        self.identity_terms = {t.lower() for t in identity_terms}
        self.w = weights or CriticWeights()

    def judge(self, reply: str, analysis: MessageAnalysis, context: str, stopped_on_eos: bool = True,
              previous_replies: Sequence[str] = ()) -> List[Tuple[str, float]]:
        """Signed adjustments (bonus > 0, penalty < 0) with the rule name.

        ``previous_replies`` are Koto's earlier replies in this conversation
        (most recent last); repeating one of them is penalised."""
        w, out = self.w, []
        low = reply.lower().strip()
        n = len(words(reply))
        if not low:
            return [("empty", -w.empty)]
        ctx_low = context.lower()
        user_low = analysis.text.lower()

        emo = analysis.emotion
        if emo in NEGATIVE and re.search(POSITIVE_REPLY, low):
            out.append(("affect_mismatch:cheerful_reply_to_negative", -w.affect_mismatch))
        if emo in POSITIVE and re.search(NEGATIVE_REPLY, low) and not analysis.hostile:
            out.append(("affect_mismatch:condolence_for_good_news", -w.affect_mismatch))
        if emo in NEGATIVE:
            if re.search(EMPATHY_REPLY, low):
                out.append(("empathy", w.empathy_bonus))
            else:
                out.append(("no_acknowledgement_of_feeling", -w.no_acknowledgement))
        if self.asks_what_user_said(reply, analysis.normalized or analysis.text):
            out.append(("asks_what_user_already_said", -w.redundant_question))
        if emo in POSITIVE and re.search(CELEBRATE_REPLY, low):
            out.append(("shares_good_news", w.empathy_bonus))

        if re.search(CONDOLENCE_REPLY, low) and emo is not Emotion.GRIEF and not re.search(r"\b(?:died|passed away|death|funeral|lost my)\b", user_low):
            out.append(("condolence_without_loss", -w.affect_mismatch))
        if analysis.hostile:
            if re.search(POSITIVE_REPLY, low):
                out.append(("hostile:cheerful_reply", -w.hostile_thanks))
            if re.search(GRATEFUL_REPLY, low):
                out.append(("hostile:thanks_or_glad", -w.hostile_thanks))
            if re.search(AGREE_INSULT_REPLY, low):
                out.append(("hostile:agrees_with_insult", -w.hostile_agree))
            if re.search(INSULT_WORDS, low):
                out.append(("hostile:retaliates", -w.hostile_retaliate))
            if re.search(DEESCALATE_REPLY, low):
                out.append(("hostile:de_escalates", w.deescalate_bonus))
            else:
                out.append(("hostile:ignores_the_user", -w.no_acknowledgement))

        if re.search(THIRD_PARTY_REPLY, low) and not re.search(THIRD_PARTY_USER, user_low) and not re.search(THIRD_PARTY_USER, ctx_low[-400:]):
            out.append(("talks_about_third_party_not_mentioned", -w.affect_mismatch / 2))

        if analysis.asks_user_fact:
            told = USER_FACT_GIVEN.get(analysis.asks_user_fact)
            if told and not re.search(told, ctx_low):
                # the user never said it: admitting that is right, answering with Koto's own identity is not
                if re.search(NOT_TOLD_REPLY, low):
                    out.append(("user_fact:admits_not_told", w.honest_limit))
                if self.identity_terms and any(re.search(rf"\b{re.escape(t)}\b", low) for t in self.identity_terms):
                    out.append(("user_fact:answers_with_own_identity", -w.honest_limit * 2))
        for prev in previous_replies[-3:]:
            if prev and (prev.strip().lower() == low or (len(words(prev)) >= 4 and char_ngram_f(reply, prev) >= 0.85)):
                out.append(("repeats_previous_reply", -w.repetition * 1.5))
                break

        if analysis.needs_live_data:
            if re.search(UNSURE_REPLY, low):
                out.append(("live_data:admits_cannot_check", w.honest_limit))
            else:
                out.append(("live_data:answers_without_access", -w.honest_limit * 2))

        it = analysis.intent
        if it is Intent.GREETING and re.search(GREETING_REPLY, low):
            out.append(("greeting_returned", w.intent_bonus))
        if it is Intent.THANKS:
            if re.search(WELCOME_REPLY, low):
                out.append(("thanks_acknowledged", w.intent_bonus))
            if re.search(r"\bsorry\b", low):
                out.append(("thanks:apologises", -w.affect_mismatch / 2))
        if it is Intent.DISAGREEMENT:
            if re.search(ACKNOWLEDGE_VIEW_REPLY, low):
                out.append(("disagreement:acknowledged", w.intent_bonus))
            if re.search(GRATEFUL_REPLY, low) or re.search(r"\bi agree\b|\byou'?re right\b", low) and len(words(reply)) < 4:
                out.append(("disagreement:ignored", -w.no_acknowledgement))
        if it is Intent.REPAIR:
            if re.search(REPAIR_REPLY, low):
                out.append(("repair:asks_or_apologises", w.intent_bonus))
            else:
                out.append(("repair:ignored", -w.no_acknowledgement))
        if it is Intent.TOPIC_CHANGE and re.search(TOPIC_CHANGE_REPLY, low):
            out.append(("topic_change:accepted", w.intent_bonus))
        if it is Intent.FAREWELL and re.search(FAREWELL_REPLY, low):
            out.append(("farewell_returned", w.intent_bonus))
        if it is Intent.IDENTITY and self.identity_terms and any(re.search(rf"\b{re.escape(t)}\b", low) for t in self.identity_terms):
            out.append(("identity_stated", w.intent_bonus))
        if it is Intent.QUESTION and n < 6 and low.endswith("?") and not re.search(r"\bhow are you\b|\bwhat about you\b|\band you\b", user_low):
            out.append(("question_dodged", -w.question_dodged))
        if it is Intent.EMAIL_REQUEST:
            parts = [bool(EMAIL_SUBJECT.search(reply)), bool(EMAIL_GREETING.search(reply)), bool(EMAIL_CLOSING.search(reply))]
            body = EMAIL_CLOSING.sub("", EMAIL_GREETING.sub("", EMAIL_SUBJECT.sub("", reply)))
            parts.append(len(words(body)) >= 10)
            missing = 4 - sum(parts)
            if missing:
                out.append((f"email:missing_{missing}_of_subject_greeting_closing_body", -w.email_structure * missing / 2))

        if re.search(HUMAN_CLAIM, low) and not re.search(HUMAN_CLAIM, user_low):
            out.append(("claims_human_experience", -w.human_claim))
        # Whole words ("war" is not mentioned by "award"). If the conversation is already on one of the
        # reply's topics, related ones are fine ("What was the Holocaust?" -> "... a genocide"); this no
        # longer depends on which topic word comes first in the reply.
        topics = [m.group(0) for m in re.finditer(OUT_OF_NOWHERE_TOPICS, low)]
        if topics and not any(_has_word(t, ctx_low) for t in topics):
            out.append((f"topic_from_nowhere:{topics[0]}", -w.out_of_nowhere_topic))
        unknown = self.unknown_entities(reply, context)
        if unknown:
            out.append((f"unknown_entities:{','.join(unknown[:3])}", -w.unknown_entity * min(2, len(unknown))))

        if max(echo_ratio(reply, analysis.text), echo_ratio(reply, analysis.normalized)) >= 0.5 and n >= 4 and it is not Intent.EMAIL_REQUEST:
            out.append(("echoes_user", -w.echo))
        if ngram_repetition(reply) > 0.2 or repeated_sentence(reply):
            out.append(("repetition", -w.repetition))
        if not stopped_on_eos:
            out.append(("incomplete", -w.incomplete))
        limit = INTENT_MAX_WORDS.get(it, DEFAULT_MAX_WORDS)
        if n > limit:
            out.append((f"too_long:{n}>{limit}", -w.too_long))
        if it is Intent.EMOTIONAL and n < 4:
            out.append(("too_short_for_disclosure", -w.too_short_emotional))
        return out

    @staticmethod
    def asks_what_user_said(reply: str, user: str) -> bool:
        """A yes/no question whose content words were all in the user's message."""
        def stems(t):
            return {w[:4] for w in words(t) if w not in _STOP and len(w) > 2}
        u = stems(user)
        for m in _YESNO_Q.finditer(reply):
            q = stems(m.group(1))
            if len(q) >= 1 and q <= u:
                return True
        return False

    def unknown_entities(self, reply: str, context: str) -> List[str]:
        """Capitalised mid-sentence words that are neither in the conversation nor identity terms."""
        ctx = context.lower()
        found = []
        for m in re.finditer(r"(?<![.!?:\n]\s)(?<!^)(?<![.!?:\n])\b([A-Z][a-z]{2,})\b", reply.strip()):
            w = m.group(1).lower()
            if w in _COMMON_CAPITALISED or w in self.identity_terms or _has_word(w, ctx):
                continue
            found.append(m.group(1))
        return found


# ───────────────────────── model scoring ─────────────────────────


@torch.no_grad()
def reply_logprobs(model, prompt_ids: Sequence[int], replies: Sequence[Sequence[int]], device="cpu") -> List[float]:
    """Sum of log p(reply tokens | prompt) for each reply (one forward pass each).

    Sequences longer than the model window are left-truncated from the prompt side.
    """
    max_len = model.config.max_seq_len
    out = []
    for rep in replies:
        rep = list(rep)
        if not rep:
            out.append(-math.inf)
            continue
        ids = list(prompt_ids) + rep
        if len(ids) > max_len:
            ids = ids[-max_len:]
        n_prompt = len(ids) - len(rep)
        if n_prompt < 1:  # the reply alone fills the window: score what fits
            ids, n_prompt = ids[-max_len:], 1
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _, _ = model(x)
        lp = F.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = x[0, 1:]
        tok_lp = lp.gather(1, tgt.unsqueeze(1)).squeeze(1)[n_prompt - 1:]
        out.append(float(tok_lp.sum()))
    return out


def score_candidates(candidates: List[Candidate], weights: CriticWeights) -> List[Candidate]:
    for c in candidates:
        rel = max(-weights.relevance_cap, min(weights.relevance_cap, c.relevance))
        c.score = weights.fluency * c.fluency + weights.relevance * rel + sum(v for _, v in c.adjustments)
    return sorted(candidates, key=lambda c: c.score, reverse=True)
