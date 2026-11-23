"""Tests for hibari.evaluation.metrics and gradient accumulation in the trainer."""

import torch
from torch.utils.data import DataLoader

from hibari.data.collator import DataCollatorForCausalLM
from hibari.data.dataset import ConversationDataset
from hibari.data.formatting import DEFAULT_SYSTEM_PROMPT
from hibari.data.tokenizer import HibariTokenizer
from hibari.evaluation.metrics import (
    ai_disclaimer,
    char_ngram_f,
    distinct_n,
    echo_ratio,
    grammar_heuristic,
    has_lead,
    length_ratio,
    ngram_repetition,
    repeated_sentence,
    self_introduction,
)
from hibari.model.config import ModelConfig
from hibari.model.llm import HibariKotoLM
from hibari.training.optimizer import configure_optimizers
from hibari.training.trainer import Trainer, TrainingArgs


def test_repetition_metrics():
    assert ngram_repetition("a b c a b c a b c") > 0.3
    assert ngram_repetition("I like reading books.") == 0.0
    assert repeated_sentence("I'm doing well today. I'm doing well today.")
    assert not repeated_sentence("I'm doing well. How are you?")


def test_echo_ratio():
    assert echo_ratio("What should I learn next?", "What should I learn next?") == 1.0
    assert echo_ratio("Try Git and data structures.", "What should I learn next?") == 0.0


def test_char_ngram_f():
    assert abs(char_ngram_f("She goes home.", "She goes home.") - 1.0) < 1e-9
    assert char_ngram_f("She goes home.", "She goes home!") > char_ngram_f("Totally different", "She goes home.")
    assert char_ngram_f("", "x") == 0.0


def test_style_detectors():
    assert has_lead("Here's a more polite version: Could you help?")
    assert has_lead("Sure, here it is.")
    assert not has_lead("She goes to school every day.")
    assert self_introduction("Hi! I'm Hibari, nice to meet you.")
    assert not self_introduction("Hi! Nice to meet you.")
    assert ai_disclaimer("As an AI language model, I cannot do that.")
    assert not ai_disclaimer("I can help with that.")


def test_other_metrics():
    assert all(grammar_heuristic("I'm fine, thanks.").values())
    assert not grammar_heuristic("i am the the best")["no_doubled_word"]
    assert length_ratio("one two", "one two three four") == 0.5
    assert distinct_n(["a b", "a b"]) == 0.5


def _setup():
    convs = [[{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": "hi"},
              {"role": "assistant", "content": "Hello there."}]] * 4
    tok = HibariTokenizer()
    tok.train_from_iterator([m["content"] for c in convs for m in c] * 3, vocab_size=200, min_frequency=1)
    ds = ConversationDataset(convs, tok, max_seq_len=128)
    assert len(ds) == 4
    return tok, ds


def _one_step(tok, ds, batch_size, accum, tmp_path):
    torch.manual_seed(0)
    model = HibariKotoLM(ModelConfig(vocab_size=tok.vocab_size, max_seq_len=128, d_model=32, n_layers=1, n_heads=2, d_ff=64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=DataCollatorForCausalLM(tok.pad_id))
    args = TrainingArgs(learning_rate=1e-2, max_steps=1, batch_size=batch_size, gradient_accumulation_steps=accum,
                        eval_every=100, checkpoint_every=100, log_every=1, save_best=False, output_dir=str(tmp_path))
    opt = configure_optimizers(model, learning_rate=args.learning_rate)
    Trainer(model, opt, None, loader, None, args).train()
    return [p.detach().clone() for p in model.parameters()]


def test_gradient_accumulation_matches_full_batch(tmp_path):
    tok, ds = _setup()
    full = _one_step(tok, ds, batch_size=4, accum=1, tmp_path=tmp_path / "a")
    acc = _one_step(tok, ds, batch_size=2, accum=2, tmp_path=tmp_path / "b")
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(full, acc))
