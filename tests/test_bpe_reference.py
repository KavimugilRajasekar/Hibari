"""The production tokenizer agrees token-for-token with an independent GPT-2-algorithm BPE.

`hibari.data.bpe_reference` re-implements byte-level BPE (GPT-2 `encoder.py`
algorithm: byte->unicode table, rank-ordered merges, pre-tokenisation pattern).
Agreement shows the HF-tokenizers pipeline does what the GGUF exporter and
llama.cpp assume (tokenizer.ggml.model = "gpt2", pre = "gpt-2").
"""

import json
from pathlib import Path

import pytest

from hibari.data.bpe_reference import ReferenceBPE, bytes_to_unicode, pretokenize
from hibari.data.tokenizer import HibariTokenizer

KOTO_TOK = Path(__file__).resolve().parents[1] / "models" / "Koto" / "tokenizer"
EDGE = ["", " ", "  ", "hello", " hello", "hello  world", "a\n\nb", "x \n y", "tabs\there", "  leading", "trailing  ",
        "I'm OK, don't", "IT'S CAPS", "3.14 and 2025", "Café naïve résumé", "日本語テキスト", "Emoji 😊💙 ok",
        "ﬁ ligature ＡＢＣ full-width", "e-mail: a@b.com", "<|user|>\nhi<|eos|>\n<|assistant|>\n", "x²³ ½ Ⅷ",
        "​ zero width", "a\r\nb", "!!!???", "ur so dumb lol", "she dont like apples"]


def test_bytes_to_unicode_is_a_bijection_over_256_bytes():
    m = bytes_to_unicode()
    assert sorted(m) == list(range(256)) and len(set(m.values())) == 256
    assert all(not c.isspace() and c.isprintable() for c in m.values())


def test_pretokenize_matches_gpt2_pattern():
    # "\s+(?!\S)" backtracks one char before "ok", the last "\n" is matched by "\s+" alone
    assert pretokenize("I'm here  now\n\nok") == ["I", "'m", " here", " ", " now", "\n", "\n", "ok"]
    assert pretokenize("a  ") == ["a", "  "]
    assert "".join(pretokenize("x \n y\t\tz  ")) == "x \n y\t\tz  "


@pytest.fixture(scope="module")
def tiny():
    t = HibariTokenizer()
    t.train_from_iterator(["hello world", "how are you", "I'm fine, thanks!", "café 😊"] * 5, vocab_size=320, min_frequency=1)
    return t


def _ref(tok: HibariTokenizer, nfkc=True) -> ReferenceBPE:
    data = json.loads(tok._tokenizer.to_str())
    merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in data["model"]["merges"]]
    special = [t["content"] for t in data["added_tokens"] if t.get("special")]
    return ReferenceBPE(data["model"]["vocab"], merges, special, nfkc=nfkc)


def test_tiny_tokenizer_parity(tiny):
    ref = _ref(tiny)
    for t in EDGE:
        assert tiny.encode(t) == ref.encode(t), t
        assert ref.decode(ref.encode(t), skip_special=False) == tiny.decode(tiny.encode(t)), t


@pytest.mark.skipif(not (KOTO_TOK / "tokenizer.json").exists(), reason="Koto tokenizer not checked out")
def test_koto_tokenizer_parity_on_edge_cases_and_training_text():
    tok = HibariTokenizer.load(KOTO_TOK)
    ref = ReferenceBPE.from_tokenizer_json(KOTO_TOK / "tokenizer.json")
    for t in EDGE:
        assert tok.encode(t) == ref.encode(t), t
    val = KOTO_TOK.parent / "data" / "processed_1.1.0" / "val_conversations.json"
    if val.exists():
        texts = [m["content"] for r in json.loads(val.read_text())[:600] for m in r["messages"]]
        assert all(tok.encode(t) == ref.encode(t) for t in texts)


def test_nfkc_is_the_only_normalisation(tiny):
    """Hibari applies NFKC (llama.cpp does not): compatibility characters differ, others do not."""
    with_n, without = _ref(tiny, True), _ref(tiny, False)
    assert with_n.encode("ﬁ") != without.encode("ﬁ")
    assert with_n.encode("Café 😊") == without.encode("Café 😊")


def test_decode_replaces_incomplete_utf8(tiny):
    ref = _ref(tiny)
    ids = ref.encode("😊")
    if len(ids) > 1:
        assert "�" in ref.decode(ids[:1])
