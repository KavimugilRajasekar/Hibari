"""Tests for HibariTokenizer."""

import pytest
from hibari.data.tokenizer import HibariTokenizer


def test_tokenizer_train_and_encode():
    corpus = [
        "Hello! I am Hibari Koto, your AI companion.",
        "How are you doing today? I like music and reading.",
        "That is awesome, tell me more about your hobbies!",
    ]
    tok = HibariTokenizer()
    tok.train_from_iterator(corpus, vocab_size=300, min_frequency=1)

    assert tok.vocab_size <= 300
    assert tok.pad_id is not None
    assert tok.eos_id is not None

    text = "Hello! I am Hibari Koto"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert "Hibari Koto" in decoded


def test_chat_template():
    tok = HibariTokenizer()
    messages = [
        {"role": "system", "content": "You are Hibari Koto."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    formatted = tok.apply_chat_template(messages, add_generation_prompt=True)
    assert "<|system|>" in formatted
    assert "<|user|>" in formatted
    assert "<|assistant|>" in formatted
    assert formatted.endswith("<|assistant|>\n")
