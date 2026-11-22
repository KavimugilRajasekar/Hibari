"""Regression tests for the 0.3.0 conversational layer (Koto 1.1.1 repair).

N1-N3 informal normalisation · A1-A8 message analysis · C1-C9 reply critic ·
S1-S5 candidate reranking in the chat session · R1 rewrite routing phrasings.
Each test pins a failure observed on the real 1.1.0 weights (see Koto
docs/20_KOTO_1.1.1_REPAIR_REPORT.md).
"""

import pytest
import torch

from hibari.data.tokenizer import HibariTokenizer
from hibari.inference.conversation import HibariChatSession
from hibari.inference.intent import Emotion, Intent, analyze_message
from hibari.inference.normalization import normalize_informal
from hibari.inference.reranking import Candidate, CriticWeights, ReplyCritic, reply_logprobs, score_candidates
from hibari.inference.tasks import parse_task_request
from hibari.model.config import ModelConfig
from hibari.model.llm import HibariKotoLM


# ───────────── normalisation ─────────────

@pytest.mark.parametrize("src,want", [
    ("wat r u doing?", "what are you doing?"),
    ("ur name?", "your name?"),
    ("ur so kind", "you're so kind"),
    ("thx!!!", "thanks!"),
    ("how r u", "how are you"),
    ("soooo tired", "so tired"),
    ("me n my friend", "me and my friend"),
])
def test_n1_shorthand_expanded(src, want):
    assert normalize_informal(src) == want


@pytest.mark.parametrize("text", ["I will be ill tomorrow", "its fine", "ok I will", "The R language", "Plan B"])
def test_n2_ordinary_words_untouched(text):
    assert normalize_informal(text) == text


def test_n3_empty_and_lone_i():
    assert normalize_informal("") == ""
    assert normalize_informal("i think i can") == "I think I can"


# ───────────── message analysis ─────────────

@pytest.mark.parametrize("text,intent,emotion", [
    ("Hi", Intent.GREETING, Emotion.NONE),
    ("My grandmother passed away last week.", Intent.EMOTIONAL, Emotion.GRIEF),
    ("I'm so frustrated with my project.", Intent.EMOTIONAL, Emotion.FRUSTRATION),
    ("I got the job!!", Intent.EMOTIONAL, Emotion.EXCITEMENT),
    ("Fuck you!", Intent.HOSTILE, Emotion.NONE),
    ("You're so pathetic!", Intent.HOSTILE, Emotion.NONE),
    ("ur name?", Intent.IDENTITY, Emotion.NONE),
    ("wat r u doing?", Intent.QUESTION, Emotion.NONE),          # not an identity question
    ("Write an email to my manager asking for leave.", Intent.EMAIL_REQUEST, Emotion.NONE),
    ("make this more professional: send me the documents", Intent.REWRITE_REQUEST, Emotion.NONE),
    ("???", Intent.NOISE, Emotion.NONE),
    ("", Intent.EMPTY, Emotion.NONE),
])
def test_a1_intent_and_emotion(text, intent, emotion):
    a = analyze_message(text)
    assert (a.intent, a.emotion) == (intent, emotion), a.evidence


def test_a2_negation():
    assert analyze_message("I'm not sad at all").emotion is Emotion.NONE
    assert analyze_message("I am not happy with this").emotion is Emotion.FRUSTRATION


def test_a3_third_party_vs_own_grief():
    other = analyze_message("My friend is going through a difficult time.")
    assert (other.emotion, other.subject) == (Emotion.SADNESS, "other")
    assert analyze_message("my grandmother passed away").subject == "self"


def test_a4_profanity_is_not_hostility_towards_assistant():
    a = analyze_message("fuck this day sucks")
    assert not a.hostile and a.emotion is Emotion.FRUSTRATION


def test_a5_idioms_are_not_emotions_or_threats():
    assert analyze_message("killing it at work today").emotion is Emotion.NONE


def test_a6_live_data_questions():
    assert analyze_message("What will the weather be like in Madurai next Tuesday?").needs_live_data
    assert not analyze_message("What is the capital of Japan?").needs_live_data


def test_a7_valence():
    assert analyze_message("I feel so lonely").valence == -1
    assert analyze_message("I passed my exam!").valence == 1
    assert analyze_message("What is Python?").valence == 0


def test_a8_informal_flag():
    assert analyze_message("idk lol").informal
    assert not analyze_message("I do not know.").informal


# ───────────── reply critic ─────────────

CRITIC = ReplyCritic(identity_terms=["hibari", "koto"])


def adj(reply, user, context=None, eos=True):
    a = analyze_message(user)
    return dict(CRITIC.judge(reply, a, context or user, eos))


def total(reply, user, **kw):
    return sum(adj(reply, user, **kw).values())


def test_c1_cheerful_reply_to_grief_penalised():
    user = "My grandmother passed away last week."
    assert "affect_mismatch:cheerful_reply_to_negative" in adj("That is so nice of her.", user)
    assert total("I'm so sorry for your loss.", user) > total("That is so nice of her.", user)


def test_c2_thanking_an_insult_penalised():
    user = "Fuck you!"
    assert "hostile:thanks_or_glad" in adj("I'm glad I could help.", user)
    assert total("Sounds like you're frustrated. What's going on?", user) > total("I'm glad I could help.", user)


def test_c3_agreeing_with_insult_penalised():
    assert "hostile:agrees_with_insult" in adj("Yes, I am.", "You're so pathetic!")


def test_c4_condolence_for_good_news_penalised():
    assert "affect_mismatch:condolence_for_good_news" in adj("Oh no, I'm sorry to hear that.", "I got the job!!")


def test_c5_topic_from_nowhere():
    user = "Hi Brother, I hope you are doing well"
    assert any(k.startswith("topic_from_nowhere") for k in adj("I'm glad you're in the way of the Holocaust.", user))
    # the same word is fine when the user brought it up
    assert not any(k.startswith("topic_from_nowhere") for k in adj("The Holocaust was a genocide.", "What was the Holocaust?"))


def test_c6_human_claim():
    assert "claims_human_experience" in adj("I lost my dog last year too.", "My dog died.")


def test_c7_live_data():
    q = "What will the weather be like in Madurai next Tuesday?"
    assert total("I can't check live weather. A weather app will have the forecast.", q) > total("It will be sunny and 30 degrees.", q)


def test_c8_email_structure():
    q = "Write an email to my manager asking for leave on Friday."
    full = "Subject: Leave on Friday\n\nDear Manager,\n\nI would like to request leave on Friday for a personal matter. I will finish my tasks first.\n\nBest regards,\n[Your name]"
    partial = "Subject: Leave Request for Friday.\n\nThank you,\n[Your name]"
    assert not any(k.startswith("email:") for k in adj(full, q))
    assert any(k.startswith("email:") for k in adj(partial, q))


def test_c9_redundant_question_and_truncation():
    assert "asks_what_user_already_said" in adj("That is terrible. Did she pass away?", "My grandmother passed away last week.")
    assert "incomplete" in adj("Well, the thing is that", "hello", eos=False)


def test_c10_score_ordering_uses_relevance_cap():
    w = CriticWeights()
    a = Candidate("a", [1], True, "x", logprob=-1.0, generic_logprob=-100.0, n_scored_tokens=1)
    b = Candidate("b", [1], True, "y", logprob=-1.0, generic_logprob=-1.5, n_scored_tokens=1)
    ranked = score_candidates([a, b], w)
    assert ranked[0].text == "a"
    assert ranked[0].score == pytest.approx(-1.0 + w.relevance * w.relevance_cap)


# ───────────── session integration ─────────────

CORPUS = ["hello world", "how are you", "I am fine thanks", "what is your name", "I am sorry to hear that",
          "that sounds hard", "tell me more about it"] * 4


@pytest.fixture(scope="module")
def tok():
    t = HibariTokenizer()
    t.train_from_iterator(CORPUS, vocab_size=300, min_frequency=1)
    return t


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    return HibariKotoLM(ModelConfig(vocab_size=tok.vocab_size, max_seq_len=96, d_model=32, n_layers=1, n_heads=2, d_ff=64)).eval()


DEC = dict(max_new_tokens=12, temperature=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0)


def test_s1_single_candidate_path_unchanged(model, tok):
    a = HibariChatSession(model, tok)
    b = HibariChatSession(model, tok, num_candidates=1, normalize_input=False)
    assert a.reply("hello world", **DEC) == b.reply("hello world", **DEC)
    assert b.last_reply_info["candidates"] == []


def test_s2_candidates_recorded_and_winner_returned(model, tok):
    s = HibariChatSession(model, tok, num_candidates=4)
    out = s.reply("how are you", seed=1, **DEC)
    cands = s.last_reply_info["candidates"]
    assert 1 <= len(cands) <= 4
    assert out == cands[0]["text"] or (not cands[0]["text"] and out)
    assert cands == sorted(cands, key=lambda c: c["score"], reverse=True)
    assert s.messages[-1] == {"role": "assistant", "content": out}


def test_s3_reranking_is_deterministic(model, tok):
    outs = {HibariChatSession(model, tok, num_candidates=4).reply("tell me more", seed=3, **DEC) for _ in range(2)}
    assert len(outs) == 1


def test_s4_intent_pool_size(model, tok):
    s = HibariChatSession(model, tok, num_candidates=2, intent_candidates={"emotional_disclosure": 5})
    s.reply("I feel so lonely", **DEC)
    assert s.last_reply_info["pool_size"] == 5
    s.reply("hello", **DEC)
    assert s.last_reply_info["pool_size"] == 2


def test_s5_normalised_text_reaches_model_but_safety_sees_original(model, tok):
    s = HibariChatSession(model, tok, normalize_input=True)
    s.reply("wat r u doing", **DEC)
    assert s.messages[1]["content"] == "what are you doing"
    s.reply("i wanna kms", **DEC)          # safety normalises shorthand itself
    assert s.last_safety_category == "self_harm"


def test_s6_reply_logprobs_matches_manual(model, tok):
    prompt = tok.encode("hello")
    rep = tok.encode(" world")
    lp = reply_logprobs(model, prompt, [rep])[0]
    with torch.no_grad():
        logits, _, _ = model(torch.tensor([prompt + rep]))
    ls = torch.log_softmax(logits[0], -1)
    manual = sum(float(ls[len(prompt) - 1 + i, t]) for i, t in enumerate(rep))
    assert lp == pytest.approx(manual, abs=1e-4)


# ───────────── rewrite routing ─────────────

@pytest.mark.parametrize("text,task", [
    ("Rewrite this properly: Hi Brother!, I hope ur Doing Well", "rephrase"),
    ("improve this: i has a interview", "rephrase"),
    ("make this better: hello", "rephrase"),
    ("correct this: i has a interview tomorrow", "correct_grammar"),
])
def test_r1_rewrite_phrasings(text, task):
    req = parse_task_request(text)
    assert req is not None and req.task == task


@pytest.mark.parametrize("text", ["rewrite the story about dragons", "I want to improve my writing", "make this weekend fun"])
def test_r2_chat_not_routed(text):
    assert parse_task_request(text) is None
