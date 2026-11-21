"""Tests for conversation formatting and dataset preparation."""

import pytest
import torch
from hibari.data.collator import DataCollatorForCausalLM
from hibari.data.dataset import ConversationDataset
from hibari.data.formatting import (
    format_blended_skill_talk_example,
    format_dailydialog_example,
    format_oasst1_example,
    format_prosocial_dialog_example,
)
from hibari.data.tokenizer import HibariTokenizer


def test_format_blended_skill_talk_example():
    ex = {
        "personas": ["i like music.", "i enjoy coding."],
        "free_messages": ["Do you like rock music?", "Who is your favorite artist?"],
        "guided_messages": ["I love classic rock!", "I really like Queen."],
        "previous_utterance": ["Hi!", "Hello!"],
    }
    messages = format_blended_skill_talk_example(ex)
    assert len(messages) >= 5
    assert messages[0]["role"] == "system"
    # Personas are excluded by default (they taught invented personal facts)
    assert "i like music." not in messages[0]["content"]
    with_persona = format_blended_skill_talk_example(ex, include_personas=True)
    assert "i like music." in with_persona[0]["content"]
    assert any(m["role"] == "user" for m in messages)
    assert any(m["role"] == "assistant" for m in messages)


def test_format_dailydialog_example():
    dialogue = ["Hello there!", "Hi, how are you doing?", "Doing great, thanks!"]
    messages = format_dailydialog_example(dialogue)
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello there!"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Hi, how are you doing?"


def test_format_prosocial_dialog_example():
    ex = {
        "context": "I feel like giving up on everything today.",
        "response": "I'm really sorry to hear that. Please remember that you're not alone.",
    }
    messages = format_prosocial_dialog_example(ex)
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "I feel like giving up on everything today."
    assert messages[2]["role"] == "assistant"
    assert "not alone" in messages[2]["content"]


def test_format_oasst1_example():
    messages = format_oasst1_example("Can you explain how empathy helps in friendship?", "Empathy allows friends to connect deeply.")
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_collator():
    collator = DataCollatorForCausalLM(pad_id=0, ignore_index=-100)
    batch = [
        {"input_ids": torch.tensor([1, 2, 3]), "targets": torch.tensor([2, 3, 4])},
        {"input_ids": torch.tensor([5, 6]), "targets": torch.tensor([6, 7])},
    ]
    out = collator(batch)
    assert out["input_ids"].shape == (2, 3)
    assert out["targets"].shape == (2, 3)
    assert out["targets"][1, 2].item() == -100


# ───────────── cleaning, normalisation, windowing ─────────────

from hibari.data.formatting import (
    DEFAULT_SYSTEM_PROMPT,
    clean_text,
    normalize_conversation,
    window_conversation,
)


def test_clean_text_fixes_corpus_spacing():
    assert clean_text("Hello , how are you ?") == "Hello, how are you?"
    assert clean_text("it ' s fine . I do n't know !") == "it's fine. I don't know!"
    assert clean_text("I’m ok") == "I'm ok"
    assert clean_text("nice 😊💙", strip_emojis=True) == "nice"


def test_normalize_conversation_alternation_and_ends_on_assistant():
    raw = [
        {"role": "system", "content": "You are a knight."},
        {"role": "assistant", "content": "leading assistant turn"},
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "reply 😊"},
        {"role": "user", "content": "trailing user"},
    ]
    out = normalize_conversation(raw)
    assert out[0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    assert [m["role"] for m in out] == ["system", "user", "assistant"]
    assert out[1]["content"] == "a\nb" and out[2]["content"] == "reply"
    assert normalize_conversation([{"role": "user", "content": "only"}]) == []


def test_window_conversation_never_cuts_replies():
    conv = [{"role": "system", "content": "S"}]
    for i in range(10):
        conv += [{"role": "user", "content": f"u{i} " * 5}, {"role": "assistant", "content": f"a{i} " * 5}]
    count = lambda msgs: sum(len(m["content"].split()) + 2 for m in msgs)
    wins = window_conversation(conv, count, max_tokens=30)
    assert len(wins) > 1
    for w in wins:
        assert count(w) <= 30
        assert w[0]["role"] == "system" and w[1]["role"] == "user" and w[-1]["role"] == "assistant"
    # every assistant turn appears in exactly one window
    replies = [m["content"] for w in wins for m in w if m["role"] == "assistant"]
    assert len(replies) == 10 and len(set(replies)) == 10


def test_window_drops_oversized_pair():
    conv = [{"role": "system", "content": "S"},
            {"role": "user", "content": "x " * 100}, {"role": "assistant", "content": "y"},
            {"role": "user", "content": "ok"}, {"role": "assistant", "content": "fine"}]
    count = lambda msgs: sum(len(m["content"].split()) + 2 for m in msgs)
    wins = window_conversation(conv, count, max_tokens=20)
    assert wins == [[conv[0], conv[3], conv[4]]]
