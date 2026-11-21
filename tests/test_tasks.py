"""Tests for rewriting tasks: request format, parsing, preservation checks,
n-gram blocking, chat-session routing and the engine rewrite API."""

import pytest
import torch

from hibari.data.tokenizer import HibariTokenizer
from hibari.inference.conversation import TOO_LONG_TASK_RESPONSE, HibariChatSession, TaskResult, empty_task_response
from hibari.inference.engine import InferenceEngine
from hibari.inference.generation import generate_tokens
from hibari.inference.safety import get_crisis_response
from hibari.inference.sampling import banned_ngram_tokens, sample_next_token
from hibari.inference.tasks import (
    TASK_DECODING,
    TASKS,
    format_task_request,
    missing_protected_terms,
    parse_task_request,
    protected_terms,
    task_max_new_tokens,
)
from hibari.model.config import ModelConfig
from hibari.model.llm import HibariKotoLM

CORPUS = ["hello world", "how are you", "Task: Rephrase", "Input: she go home", "She goes home."] * 4


@pytest.fixture(scope="module")
def tok():
    t = HibariTokenizer()
    t.train_from_iterator(CORPUS, vocab_size=300, min_frequency=1)
    return t


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tok.vocab_size, max_seq_len=256, d_model=64, n_layers=2, n_heads=2, d_ff=128)
    return HibariKotoLM(cfg).eval()


# ───────────── request format and parsing ─────────────

@pytest.mark.parametrize("task", sorted(TASKS))
def test_canonical_request_round_trips(task):
    msg = format_task_request(task, "  she go to school  ")
    assert msg == f"Task: {TASKS[task].label}\nInput: she go to school"
    req = parse_task_request(msg)
    assert req is not None and req.task == task and req.text == "she go to school"


def test_unknown_task_rejected():
    with pytest.raises(KeyError):
        format_task_request("translate", "hola")


@pytest.mark.parametrize("message,task,text", [
    ('Rephrase this sentence: "i am go to college yesterday and meet my friends"', "rephrase",
     "i am go to college yesterday and meet my friends"),
    ('Rephrase: "i am learning programming from two years."', "rephrase", "i am learning programming from two years."),
    ('Make this professional: "I need leave tomorrow."', "formal", "I need leave tomorrow."),
    ('Make this sentence more professional: "I can\'t come today."', "formal", "I can't come today."),
    ('Make this casual: "I am currently occupied."', "casual", "I am currently occupied."),
    ('Make this sentence shorter: "Due to the fact that I was ill, I stayed home."', "shorten",
     "Due to the fact that I was ill, I stayed home."),
    ("Correct grammar: She go to school every day.", "correct_grammar", "She go to school every day."),
    ("fix my grammar pls: he dont know", "correct_grammar", "he dont know"),
    ("Can you rephrase this: i go home", "rephrase", "i go home"),
    ("Simplify this\nThe utilisation of resources was suboptimal.", "simplify", "The utilisation of resources was suboptimal."),
    ("'hey cant come tonite' make it more polite", "polite", "hey cant come tonite"),
    ('Rephrase: ""', "rephrase", ""),
])
def test_parse_natural_requests(message, task, text):
    req = parse_task_request(message)
    assert req is not None and (req.task, req.text) == (task, text)


@pytest.mark.parametrize("message", [
    "Hello!", "What is Python?", "rewrite the story about dragons", "Can you rephrase this?",
    "I want to make this better", "Make this weekend fun", "What should I learn after that?", "", "   ",
])
def test_ordinary_chat_is_not_a_task(message):
    assert parse_task_request(message) is None


# ───────────── preservation checks ─────────────

def test_protected_terms_cover_numbers_times_names_and_tech():
    terms = {t.lower() for t in protected_terms("Send it to Priya on Monday at 10 AM using Python 3.11 on iOS, three copies.")}
    assert {"priya", "monday", "10am", "3.11", "ios", "three"} <= terms


@pytest.mark.parametrize("src,out,missing", [
    ("The meeting is on Monday at 10 AM.", "The meeting is set for Monday at 10am.", []),
    ("The meeting is on Monday at 10 AM.", "The meeting is on Tuesday at 10 AM.", ["Monday"]),
    ("Version 10 is out", "Version 100 is out", ["10"]),
    ("The project has three modules.", "The project has 3 modules.", []),
    ("It has 5 tests.", "It has five tests.", []),
    ("It supports Android but not iOS.", "It supports Android only.", ["iOS"]),
])
def test_missing_protected_terms(src, out, missing):
    assert [m.lower() for m in missing_protected_terms(src, out)] == [m.lower() for m in missing]


def test_task_length_budget_scales_and_is_bounded():
    assert task_max_new_tokens(10, "rephrase") > task_max_new_tokens(10, "shorten")
    assert task_max_new_tokens(1, "rephrase") >= 16
    assert task_max_new_tokens(10_000, "rephrase") == 160


# ───────────── n-gram blocking ─────────────

def test_banned_ngram_tokens():
    assert banned_ngram_tokens([1, 2, 3, 1, 2], 3) == {3}
    assert banned_ngram_tokens([1, 2, 3, 4], 3) == set()
    assert banned_ngram_tokens([1, 2, 1, 2], 0) == set()


def test_sampler_never_picks_banned_token():
    logits = torch.tensor([0.0, 10.0, 1.0])
    assert sample_next_token(logits, temperature=0.0, banned_token_ids={1}) == 2


def test_generation_with_ngram_block_has_no_repeated_ngram(model, tok, monkeypatch):
    import hibari.inference.generation as gen
    real = gen.sample_next_token
    # a sampler that always prefers token 10 then 11 would loop "10 11 10 11 ..."
    def looping(logits, **kw):
        logits = logits.clone()
        logits[10] += 100.0
        logits[11] += 50.0
        return real(logits, **kw)
    monkeypatch.setattr(gen, "sample_next_token", looping)
    out = generate_tokens(model, tok, "hello", max_new_tokens=12, temperature=0.0, no_repeat_ngram_size=2)
    bigrams = list(zip(out, out[1:]))
    assert len(bigrams) == len(set(bigrams))


def test_generation_default_unchanged_without_ngram_block(model, tok):
    a = generate_tokens(model, tok, "hello", max_new_tokens=10, temperature=0.0)
    b = generate_tokens(model, tok, "hello", max_new_tokens=10, temperature=0.0, no_repeat_ngram_size=0)
    assert a == b


# ───────────── chat-session routing ─────────────

def test_reply_routes_task_requests_in_canonical_form(model, tok):
    s = HibariChatSession(model, tok)
    out = s.reply('Rephrase: "she go home"')
    assert s.last_task is not None and s.last_task.task == "rephrase"
    assert s.messages[-2] == {"role": "user", "content": "Task: Rephrase\nInput: she go home"}
    assert s.messages[-1]["content"] == out
    assert s.last_stopped_on_eos is not None


def test_reply_without_routing_sends_text_as_typed(model, tok):
    s = HibariChatSession(model, tok, route_tasks=False)
    s.reply('Rephrase: "she go home"', max_new_tokens=4, temperature=0.0)
    assert s.last_task is None
    assert s.messages[1]["content"] == 'Rephrase: "she go home"'


def test_task_prompt_contains_only_system_and_request(model, tok, monkeypatch):
    import hibari.inference.conversation as conv
    seen = {}

    def fake_generate(**kw):
        seen["prompt"] = tok.decode(kw["prompt_ids"])
        seen["decoding"] = {k: kw[k] for k in TASK_DECODING}
        return []
    monkeypatch.setattr(conv, "generate_tokens", fake_generate)
    s = HibariChatSession(model, tok)
    s.messages.append({"role": "user", "content": "earlier chat turn"})
    s.messages.append({"role": "assistant", "content": "earlier reply"})
    s.run_task("correct_grammar", "she go home")
    assert "earlier chat turn" not in seen["prompt"]
    assert seen["prompt"].endswith("<|user|>\nTask: Correct grammar\nInput: she go home<|eos|>\n<|assistant|>\n")
    assert seen["decoding"] == TASK_DECODING


def test_empty_task_input_answered_without_model(model, tok, monkeypatch):
    import hibari.inference.conversation as conv
    monkeypatch.setattr(conv, "generate_tokens", lambda **kw: pytest.fail("model must not be called"))
    s = HibariChatSession(model, tok)
    assert s.reply('Rephrase: ""') == empty_task_response("rephrase")
    assert s.last_stopped_on_eos is True


def test_task_input_still_goes_through_safety(model, tok):
    s = HibariChatSession(model, tok)
    assert s.reply('Rephrase: "i want to die"') == get_crisis_response()
    assert s.last_task.safety_category == "self_harm"


def test_run_task_retries_until_protected_terms_kept(model, tok, monkeypatch):
    outputs = iter([("Meet on Tuesday.", True), ("Meet on Monday at 9.", True), ("Meet on Monday at 10 AM.", True)])
    s = HibariChatSession(model, tok)
    monkeypatch.setattr(s, "_task_candidate", lambda *a, **k: next(outputs))
    res = s.run_task("rephrase", "Meet on Monday at 10 AM.")
    assert res.output == "Meet on Monday at 10 AM." and res.missing_terms == [] and res.candidates_tried == 3


def test_run_task_reports_terms_it_could_not_keep(model, tok, monkeypatch):
    s = HibariChatSession(model, tok)
    monkeypatch.setattr(s, "_task_candidate", lambda *a, **k: ("Meet soon.", True))
    res = s.run_task("rephrase", "Meet on Monday.", retries=2)
    assert res.missing_terms == ["Monday"] and res.candidates_tried == 3


def test_engine_rewrite_returns_result_and_keeps_no_history(model, tok):
    eng = InferenceEngine(model, tok)
    res = eng.rewrite("formal", "send me the file")
    assert isinstance(res, TaskResult) and res.task == "formal" and res.input == "send me the file"


def test_long_task_input_is_split_at_sentences_and_header_never_truncated(model, tok, monkeypatch):
    import hibari.inference.conversation as conv
    prompts = []

    def fake_generate(**kw):
        prompts.append(tok.decode(kw["prompt_ids"]))
        return []
    monkeypatch.setattr(conv, "generate_tokens", fake_generate)
    s = HibariChatSession(model, tok)
    text = " ".join(f"she go home number {i}." for i in range(40))
    res = s.run_task("rephrase", text, retries=0)
    assert res.chunks > 1 and len(prompts) == res.chunks
    for p in prompts:
        assert "<|user|>\nTask: Rephrase\nInput: " in p          # header intact in every chunk
        assert len(tok.encode(p)) <= model.config.max_seq_len
    assert " ".join(p.split("Input: ")[1].split("<|eos|>")[0] for p in prompts) == text  # nothing dropped


def test_single_sentence_too_long_gets_fixed_message(model, tok, monkeypatch):
    import hibari.inference.conversation as conv
    monkeypatch.setattr(conv, "generate_tokens", lambda **kw: pytest.fail("model must not be called"))
    s = HibariChatSession(model, tok)
    res = s.run_task("rephrase", "she go home " * 200)
    assert res.output == TOO_LONG_TASK_RESPONSE and res.chunks == 0
