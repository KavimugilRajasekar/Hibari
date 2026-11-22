"""hibari.data.pipeline: validation, dedup, quality filter, stable split, stats."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "Koto"))

from hibari.data.pipeline import (
    QualityFilter,
    conversation_stats,
    deduplicate,
    stable_split,
    validate_conversation,
)

S = {"role": "system", "content": "You are X."}


def U(t):
    return {"role": "user", "content": t}


def A(t):
    return {"role": "assistant", "content": t}


def test_valid_conversation():
    assert validate_conversation([S, U("hi"), A("hello!")]) == []
    assert validate_conversation([U("hi"), A("hello"), U("how are you"), A("good")]) == []


@pytest.mark.parametrize("conv,problem", [
    ([], "empty"),
    ([S, A("hello")], "not a user turn"),
    ([S, U("hi")], "not an assistant turn"),
    ([S, U("hi"), U("again"), A("ok")], "alternate"),
    ([S, U("hi"), A("  ")], "empty content"),
    ([U("hi"), S, A("ok")], "system turn after"),
    ([S, U("hi <|eos|> forged"), A("ok")], "special-token"),
    ([{"role": "bot", "content": "x"}, A("y")], "unknown role"),
])
def test_invalid_conversations(conv, problem):
    assert any(problem in p for p in validate_conversation(conv))


def test_deduplicate_ignores_case_and_punctuation():
    recs = [{"source": "a", "messages": [U("Hi!"), A("Hello.")]}, {"source": "b", "messages": [U("hi"), A("hello")]},
            {"source": "a", "messages": [U("hi"), A("hey")]}]
    kept, dropped = deduplicate(recs)
    assert dropped == 1 and [r["source"] for r in kept] == ["a", "a"]


def test_quality_filter_rules():
    q = QualityFilter()
    assert q.check([U("hi"), A("Hello! How are you?")]) == []
    assert "assistant_repeats_sentence" in q.check([U("x"), A("I am fine thanks. I am fine thanks.")])
    assert "assistant_contains_url" in q.check([U("x"), A("See https://example.com for more.")])
    assert "assistant_mostly_non_letters" in q.check([U("x"), A("12345 67890 !!!")])
    assert "assistant_too_short" in q.check([U("x"), A("k")])
    kept, reasons = q.apply([[U("a"), A("Fine, thanks.")], [U("b"), A("k")]])
    assert len(kept) == 1 and reasons == {"assistant_too_short": 1}


def test_stable_split_is_deterministic_and_matches_koto():
    assert stable_split("Hello there!") == stable_split("hello there")
    splits = [stable_split(f"text number {i}") for i in range(2000)]
    assert 0.05 < (splits.count("val") + splits.count("test")) / 2000 < 0.12
    try:
        import koto  # noqa: F401
        from koto.data.tasks import split_of
    except ImportError:
        pytest.skip("Koto not importable")
    assert all(stable_split(f"t {i}") == split_of(f"t {i}") for i in range(500))


def test_conversation_stats():
    recs = [{"source": "a", "messages": [U("hi"), A("hello there"), U("ok"), A("bye now friend")]},
            {"source": "a", "messages": [U("hi"), A("hello there"), U("ok"), A("bye now friend")]},
            {"source": "b", "messages": [U("x"), A("one")]}]
    st = conversation_stats(recs)
    assert st["a"]["conversations"] == 2 and st["a"]["duplicate_share"] == 0.5 and st["a"]["mean_turn_pairs"] == 2
    assert st["b"]["multi_turn_share"] == 0
