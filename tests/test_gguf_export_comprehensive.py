"""Comprehensive tests for Hibari GGUF export and tokenizer compatibility.

Covers:
- Tokenizer vocabulary preservation and BPE merge ordering
- Token ID consistency and special token IDs
- GGUF metadata integrity (all required fields present)
- Tensor mapping for FP32 and FP16 exports
- Model loading via GGUFReader
- Basic model inference (generate call)
- Chat template and special token handling
- Context length enforcement (256-token limit)
- Edge cases: missing tokenizer files, invalid merges, missing metadata,
  incorrect token IDs, and unsupported quantization types
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import gguf

from hibari.data.tokenizer import HibariTokenizer, SPECIAL_TOKENS
from hibari.export.gguf_export import _get_bpe_merges, _build_token_types, export_to_gguf
from hibari.model.config import ModelConfig
from hibari.model.llm import HibariKotoLM


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CORPUS = [
    "Hello! I am Hibari Koto, your friendly AI companion.",
    "I love music, books, and interesting conversations.",
    "How are you doing today? Tell me about your hobbies.",
    "That sounds great! Do you enjoy cooking or travelling?",
    "What is your favorite movie or TV show? I love anime.",
]


@pytest.fixture(scope="module")
def trained_tokenizer() -> HibariTokenizer:
    """A fully-trained HibariTokenizer with real BPE merges."""
    tok = HibariTokenizer()
    tok.train_from_iterator(CORPUS, vocab_size=512, min_frequency=1)
    return tok


@pytest.fixture(scope="module")
def small_model(trained_tokenizer) -> HibariKotoLM:
    """Tiny model for fast export tests (vocab matches the tokenizer, as the
    exporter now enforces: a GGUF whose token list is shorter than its
    embedding matrix is rejected by llama.cpp)."""
    config = ModelConfig(
        vocab_size=trained_tokenizer.vocab_size,
        max_seq_len=64,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        model_name="Hibari-Test",
    )
    return HibariKotoLM(config)


@pytest.fixture(scope="module")
def hibari_model(trained_tokenizer) -> HibariKotoLM:
    """Model with the default ModelConfig context length (256)."""
    config = ModelConfig(vocab_size=trained_tokenizer.vocab_size)
    return HibariKotoLM(config)


# ---------------------------------------------------------------------------
# 1. Tokenizer vocabulary preservation
# ---------------------------------------------------------------------------

class TestTokenizerVocabularyPreservation:

    def test_vocab_contains_all_special_tokens(self, trained_tokenizer):
        vocab = trained_tokenizer.get_vocab()
        for tok in SPECIAL_TOKENS:
            assert tok in vocab, f"Special token {tok!r} missing from vocab"

    def test_special_token_ids_are_stable(self, trained_tokenizer):
        """Special token IDs must match the order in SPECIAL_TOKENS list."""
        assert trained_tokenizer.pad_id == 0
        assert trained_tokenizer.unk_id == 1
        assert trained_tokenizer.bos_id == 2
        assert trained_tokenizer.eos_id == 3
        assert trained_tokenizer.system_id == 4
        assert trained_tokenizer.user_id == 5
        assert trained_tokenizer.assistant_id == 6

    def test_vocab_size_bounded(self, trained_tokenizer):
        assert trained_tokenizer.vocab_size <= 512

    def test_vocab_sorted_by_id_is_contiguous(self, trained_tokenizer):
        vocab = trained_tokenizer.get_vocab()
        ids = sorted(vocab.values())
        # IDs must be a contiguous range starting at 0
        assert ids == list(range(len(ids))), "Vocabulary IDs are not contiguous"

    def test_encode_decode_roundtrip(self, trained_tokenizer):
        text = "Hello Hibari"
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        assert "Hibari" in decoded

    def test_encode_with_special_tokens(self, trained_tokenizer):
        text = "hi"
        ids = trained_tokenizer.encode(text, add_special_tokens=True)
        assert ids[0] == trained_tokenizer.bos_id
        assert ids[-1] == trained_tokenizer.eos_id

    def test_save_and_load_preserves_vocab(self, trained_tokenizer, tmp_path):
        trained_tokenizer.save(tmp_path)
        loaded = HibariTokenizer.load(tmp_path)
        assert loaded.get_vocab() == trained_tokenizer.get_vocab()

    def test_save_and_load_preserves_special_token_ids(self, trained_tokenizer, tmp_path):
        trained_tokenizer.save(tmp_path)
        loaded = HibariTokenizer.load(tmp_path)
        assert loaded.pad_id == trained_tokenizer.pad_id
        assert loaded.bos_id == trained_tokenizer.bos_id
        assert loaded.eos_id == trained_tokenizer.eos_id


# ---------------------------------------------------------------------------
# 2. BPE merge preservation and ordering
# ---------------------------------------------------------------------------

class TestBPEMergePreservation:

    def test_merges_exist(self, trained_tokenizer):
        merges = _get_bpe_merges(trained_tokenizer)
        assert len(merges) > 0, "Tokenizer must have at least one BPE merge rule"

    def test_merges_format_is_space_separated(self, trained_tokenizer):
        merges = _get_bpe_merges(trained_tokenizer)
        for merge in merges:
            parts = merge.split(" ")
            assert len(parts) == 2, (
                f"Each merge must be 'left right', got {merge!r}"
            )

    def test_merges_order_matches_tokenizer_json(self, trained_tokenizer, tmp_path):
        """Order exported by _get_bpe_merges must match the tokenizer's internal order."""
        trained_tokenizer.save(tmp_path)
        with open(tmp_path / "tokenizer.json") as f:
            raw = json.load(f)
        json_merges = [f"{a} {b}" for a, b in raw["model"]["merges"]]
        exported_merges = _get_bpe_merges(trained_tokenizer)
        assert exported_merges == json_merges

    def test_merges_count_consistent_with_tokenizer_json(self, trained_tokenizer, tmp_path):
        trained_tokenizer.save(tmp_path)
        with open(tmp_path / "tokenizer.json") as f:
            raw = json.load(f)
        n_json = len(raw["model"]["merges"])
        n_exported = len(_get_bpe_merges(trained_tokenizer))
        assert n_exported == n_json

    def test_save_load_preserves_merges(self, trained_tokenizer, tmp_path):
        """Merge rules must survive save → load round-trip."""
        orig_merges = _get_bpe_merges(trained_tokenizer)
        trained_tokenizer.save(tmp_path)
        loaded = HibariTokenizer.load(tmp_path)
        assert _get_bpe_merges(loaded) == orig_merges


# ---------------------------------------------------------------------------
# 3. Token type classification
# ---------------------------------------------------------------------------

class TestTokenTypeClassification:

    def test_special_tokens_are_control_type(self, trained_tokenizer):
        vocab = trained_tokenizer.get_vocab()
        sorted_tokens = [tok for tok, _ in sorted(vocab.items(), key=lambda x: x[1])]
        types = _build_token_types(sorted_tokens, SPECIAL_TOKENS)
        for special in SPECIAL_TOKENS:
            if special in vocab:
                idx = vocab[special]
                assert types[idx] == int(gguf.TokenType.CONTROL), (
                    f"Token {special!r} should be CONTROL type"
                )

    def test_normal_tokens_are_normal_type(self, trained_tokenizer):
        vocab = trained_tokenizer.get_vocab()
        sorted_tokens = [tok for tok, _ in sorted(vocab.items(), key=lambda x: x[1])]
        types = _build_token_types(sorted_tokens, SPECIAL_TOKENS)
        special_set = set(SPECIAL_TOKENS)
        for tok, idx in vocab.items():
            if tok not in special_set:
                assert types[idx] == int(gguf.TokenType.NORMAL), (
                    f"Token {tok!r} should be NORMAL type"
                )

    def test_token_types_length_matches_vocab(self, trained_tokenizer):
        vocab = trained_tokenizer.get_vocab()
        sorted_tokens = [tok for tok, _ in sorted(vocab.items(), key=lambda x: x[1])]
        types = _build_token_types(sorted_tokens, SPECIAL_TOKENS)
        assert len(types) == len(sorted_tokens)


# ---------------------------------------------------------------------------
# 4. GGUF metadata integrity
# ---------------------------------------------------------------------------

REQUIRED_TOKENIZER_FIELDS = [
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.unknown_token_id",
]

REQUIRED_ARCH_FIELDS = [
    "general.name",
    "llama.context_length",
    "llama.embedding_length",
    "llama.block_count",
    "llama.feed_forward_length",
    "llama.attention.head_count",
    "llama.rope.freq_base",
]


def _make_exported_gguf(
    tmp_path: Path,
    model: HibariKotoLM,
    tokenizer: HibariTokenizer | None,
    quantization: str = "f32",
) -> Path:
    out = tmp_path / f"test_{quantization}.gguf"
    export_to_gguf(model, tokenizer=tokenizer, output_path=out, quantization_type=quantization)
    return out


class TestGGUFMetadataIntegrity:

    def test_required_tokenizer_fields_present_f32(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f32")
        reader = gguf.GGUFReader(str(out))
        for field in REQUIRED_TOKENIZER_FIELDS:
            assert field in reader.fields, f"Missing required field: {field}"

    def test_required_tokenizer_fields_present_f16(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f16")
        reader = gguf.GGUFReader(str(out))
        for field in REQUIRED_TOKENIZER_FIELDS:
            assert field in reader.fields, f"Missing field in f16 export: {field}"

    def test_required_arch_fields_present(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f32")
        reader = gguf.GGUFReader(str(out))
        for field in REQUIRED_ARCH_FIELDS:
            assert field in reader.fields, f"Missing arch field: {field}"

    def test_tokenizer_model_is_gpt2(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f32")
        reader = gguf.GGUFReader(str(out))
        field = reader.fields["tokenizer.ggml.model"]
        value = bytes(field.parts[-1]).decode("utf-8")
        assert value == "gpt2"

    def test_merges_count_correct(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f32")
        reader = gguf.GGUFReader(str(out))
        merges_field = reader.fields["tokenizer.ggml.merges"]
        expected_count = len(_get_bpe_merges(trained_tokenizer))
        assert len(merges_field.data) == expected_count, (
            f"Expected {expected_count} merges in GGUF, found {len(merges_field.data)}"
        )

    def test_token_list_count_correct(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f32")
        reader = gguf.GGUFReader(str(out))
        tokens_field = reader.fields["tokenizer.ggml.tokens"]
        assert len(tokens_field.data) == trained_tokenizer.vocab_size

    def test_bos_eos_ids_match_tokenizer(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f32")
        reader = gguf.GGUFReader(str(out))
        bos = int(reader.fields["tokenizer.ggml.bos_token_id"].parts[-1][0])
        eos = int(reader.fields["tokenizer.ggml.eos_token_id"].parts[-1][0])
        assert bos == trained_tokenizer.bos_id
        assert eos == trained_tokenizer.eos_id

    def test_pad_unk_ids_match_tokenizer(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f32")
        reader = gguf.GGUFReader(str(out))
        pad = int(reader.fields["tokenizer.ggml.padding_token_id"].parts[-1][0])
        unk = int(reader.fields["tokenizer.ggml.unknown_token_id"].parts[-1][0])
        assert pad == trained_tokenizer.pad_id
        assert unk == trained_tokenizer.unk_id

    def test_scores_array_length_matches_vocab(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f32")
        reader = gguf.GGUFReader(str(out))
        scores_field = reader.fields["tokenizer.ggml.scores"]
        assert len(scores_field.data) == trained_tokenizer.vocab_size

    def test_token_type_array_length_matches_vocab(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f32")
        reader = gguf.GGUFReader(str(out))
        types_field = reader.fields["tokenizer.ggml.token_type"]
        assert len(types_field.data) == trained_tokenizer.vocab_size

    def test_context_length_matches_config(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f32")
        reader = gguf.GGUFReader(str(out))
        ctx = int(reader.fields["llama.context_length"].parts[-1][0])
        assert ctx == small_model.config.max_seq_len

    def test_hibari_context_length_is_256(self, hibari_model, trained_tokenizer, tmp_path):
        """Canonical Hibari-Koto-1M must have a 256-token context limit."""
        out = _make_exported_gguf(tmp_path, hibari_model, trained_tokenizer, "f32")
        reader = gguf.GGUFReader(str(out))
        ctx = int(reader.fields["llama.context_length"].parts[-1][0])
        assert ctx == 256


# ---------------------------------------------------------------------------
# 5. Tensor mapping
# ---------------------------------------------------------------------------

class TestTensorMapping:

    EXPECTED_TENSOR_NAMES_2_LAYER = [
        "token_embd.weight",
        "output_norm.weight",
        # no "output.weight": tied embeddings -> runtimes reuse token_embd
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.1.attn_norm.weight",
        "blk.1.attn_q.weight",
        "blk.1.attn_k.weight",
        "blk.1.attn_v.weight",
        "blk.1.attn_output.weight",
        "blk.1.ffn_norm.weight",
        "blk.1.ffn_gate.weight",
        "blk.1.ffn_up.weight",
        "blk.1.ffn_down.weight",
    ]

    def test_all_expected_tensors_present_f32(self, small_model, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, None, "f32")
        reader = gguf.GGUFReader(str(out))
        tensor_names = {t.name for t in reader.tensors}
        for name in self.EXPECTED_TENSOR_NAMES_2_LAYER:
            assert name in tensor_names, f"Missing tensor: {name}"

    def test_all_expected_tensors_present_f16(self, small_model, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, None, "f16")
        reader = gguf.GGUFReader(str(out))
        tensor_names = {t.name for t in reader.tensors}
        for name in self.EXPECTED_TENSOR_NAMES_2_LAYER:
            assert name in tensor_names, f"Missing tensor in f16 export: {name}"

    def test_f32_tensors_are_float32(self, small_model, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, None, "f32")
        reader = gguf.GGUFReader(str(out))
        for tensor in reader.tensors:
            assert tensor.tensor_type == gguf.GGMLQuantizationType.F32, (
                f"Tensor {tensor.name} has wrong dtype in f32 export"
            )

    def test_f16_matrices_are_float16_norms_float32(self, small_model, tmp_path):
        """2-D weights are F16; 1-D RMSNorm weights must stay F32 (ggml's CPU
        kernels abort on F16 norm weights — the original exporter's bug)."""
        out = _make_exported_gguf(tmp_path, small_model, None, "f16")
        reader = gguf.GGUFReader(str(out))
        for tensor in reader.tensors:
            want = gguf.GGMLQuantizationType.F32 if len(tensor.shape) == 1 else gguf.GGMLQuantizationType.F16
            assert tensor.tensor_type == want, f"Tensor {tensor.name} has wrong dtype in f16 export"

    def test_no_extra_unknown_tensors(self, small_model, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, None, "f32")
        reader = gguf.GGUFReader(str(out))
        tensor_names = {t.name for t in reader.tensors}
        expected = set(self.EXPECTED_TENSOR_NAMES_2_LAYER)
        unexpected = tensor_names - expected
        assert not unexpected, f"Unexpected tensors in GGUF: {unexpected}"


# ---------------------------------------------------------------------------
# 6. FP32 / FP16 exports
# ---------------------------------------------------------------------------

class TestQuantizationFormats:

    def test_f32_export_creates_file(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f32")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_f16_export_creates_file(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f16")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_f16_smaller_than_f32(self, small_model, trained_tokenizer, tmp_path):
        out32 = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f32")
        out16 = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f16")
        assert out16.stat().st_size < out32.stat().st_size

    def test_invalid_quantization_raises(self, small_model, trained_tokenizer, tmp_path):
        with pytest.raises(ValueError, match="quantization_type"):
            export_to_gguf(
                small_model,
                tokenizer=trained_tokenizer,
                output_path=tmp_path / "bad.gguf",
                quantization_type="q4_k_m",
            )

    def test_export_without_tokenizer_succeeds(self, small_model, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, None, "f32")
        assert out.exists()
        reader = gguf.GGUFReader(str(out))
        # Tokenizer fields must NOT be present when tokenizer is None
        assert "tokenizer.ggml.merges" not in reader.fields
        assert "tokenizer.ggml.tokens" not in reader.fields

    def test_export_returns_path_object(self, small_model, trained_tokenizer, tmp_path):
        result = export_to_gguf(
            small_model,
            tokenizer=trained_tokenizer,
            output_path=tmp_path / "ret.gguf",
        )
        assert isinstance(result, Path)
        assert result.exists()


# ---------------------------------------------------------------------------
# 7. Model loading validation
# ---------------------------------------------------------------------------

class TestModelLoading:

    def test_gguf_reader_can_open_f32_export(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f32")
        reader = gguf.GGUFReader(str(out))
        assert len(reader.tensors) > 0

    def test_gguf_reader_can_open_f16_export(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f16")
        reader = gguf.GGUFReader(str(out))
        assert len(reader.tensors) > 0

    def test_embedding_shape_preserved_f32(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f32")
        reader = gguf.GGUFReader(str(out))
        emb = next(t for t in reader.tensors if t.name == "token_embd.weight")
        # Shape is [vocab_size, d_model] but GGUF stores row-major
        # gguf stores shape as (cols, rows) in some versions; at minimum both
        # dimensions must equal the config values.
        assert small_model.config.vocab_size in emb.shape
        assert small_model.config.d_model in emb.shape

    def test_all_layer_tensors_present(self, small_model, trained_tokenizer, tmp_path):
        out = _make_exported_gguf(tmp_path, small_model, trained_tokenizer, "f32")
        reader = gguf.GGUFReader(str(out))
        tensor_names = {t.name for t in reader.tensors}
        for i in range(small_model.config.n_layers):
            for suffix in ["attn_q", "attn_k", "attn_v", "attn_output",
                           "ffn_gate", "ffn_up", "ffn_down"]:
                name = f"blk.{i}.{suffix}.weight"
                assert name in tensor_names, f"Missing tensor: {name}"


# ---------------------------------------------------------------------------
# 8. Basic inference
# ---------------------------------------------------------------------------

class TestModelInference:

    def test_model_generates_tokens(self, small_model):
        input_ids = torch.tensor([[2, 71, 85]])  # bos, 'a', 'e'
        output = small_model.generate(input_ids, max_new_tokens=5, temperature=0.0)
        assert output.shape[1] == input_ids.shape[1] + 5

    def test_model_logits_shape(self, small_model):
        input_ids = torch.randint(0, small_model.config.vocab_size, (1, 10))
        logits, loss, _ = small_model(input_ids)
        assert logits.shape == (1, 10, small_model.config.vocab_size)
        assert loss is None

    def test_model_loss_with_targets(self, small_model):
        input_ids = torch.randint(0, small_model.config.vocab_size, (2, 8))
        targets = torch.randint(0, small_model.config.vocab_size, (2, 8))
        _, loss, _ = small_model(input_ids, targets=targets)
        assert loss is not None
        assert loss.item() > 0

    def test_inference_respects_context_limit(self, small_model):
        """The model clips its attention window to max_seq_len during generation.

        ``HibariKotoLM.generate`` does not stop after max_seq_len tokens; it
        continues generating but trims the *input* to the last ``max_seq_len``
        tokens on each step so the model never attends to more tokens than its
        context length.  The total output length is therefore
        ``initial_len + max_new_tokens`` (minus any early EOS stop), not
        capped at ``max_seq_len``.  We verify the sliding-window clipping
        mechanism here rather than testing an early-stop property the model
        does not have.
        """
        max_len = small_model.config.max_seq_len  # 64 for small_model
        # Start with a prompt longer than max_seq_len to exercise clipping
        long_prompt = torch.randint(0, small_model.config.vocab_size, (1, max_len + 10))
        max_new = 5
        output = small_model.generate(long_prompt, max_new_tokens=max_new, temperature=0.0)
        # Output must be exactly prompt_len + max_new_tokens (no EOS triggered)
        assert output.shape[1] == long_prompt.shape[1] + max_new

    def test_hibari_1m_context_limit_256(self, hibari_model):
        """Canonical model must enforce 256-token context."""
        assert hibari_model.config.max_seq_len == 256


# ---------------------------------------------------------------------------
# 9. Chat template and special token handling
# ---------------------------------------------------------------------------

class TestChatTemplateAndSpecialTokens:

    def test_system_role_formatting(self, trained_tokenizer):
        msgs = [{"role": "system", "content": "You are Hibari."}]
        out = trained_tokenizer.apply_chat_template(msgs)
        assert out.startswith("<|system|>\n")
        assert "<|eos|>" in out

    def test_user_role_formatting(self, trained_tokenizer):
        msgs = [{"role": "user", "content": "Hello!"}]
        out = trained_tokenizer.apply_chat_template(msgs)
        assert "<|user|>\n" in out
        assert "<|eos|>" in out

    def test_assistant_role_formatting(self, trained_tokenizer):
        msgs = [{"role": "assistant", "content": "Hi there!"}]
        out = trained_tokenizer.apply_chat_template(msgs)
        assert "<|assistant|>\n" in out
        assert "<|eos|>" in out

    def test_generation_prompt_appended(self, trained_tokenizer):
        msgs = [{"role": "user", "content": "Hello"}]
        out = trained_tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        assert out.endswith("<|assistant|>\n")

    def test_full_conversation_format(self, trained_tokenizer):
        msgs = [
            {"role": "system", "content": "You are Hibari Koto."},
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": "My name is Hibari."},
        ]
        out = trained_tokenizer.apply_chat_template(msgs)
        # All three role markers present
        assert "<|system|>" in out
        assert "<|user|>" in out
        assert "<|assistant|>" in out
        # Three EOS separators (one per turn)
        assert out.count("<|eos|>") == 3

    def test_special_tokens_are_encodable(self, trained_tokenizer):
        """Each special token must encode to a single token ID."""
        for tok in SPECIAL_TOKENS:
            tok_id = trained_tokenizer._tokenizer.token_to_id(tok)
            assert tok_id is not None, f"Special token {tok!r} not in vocabulary"

    def test_eos_token_in_chat_output_is_encodable(self, trained_tokenizer):
        msgs = [{"role": "user", "content": "hi"}]
        text = trained_tokenizer.apply_chat_template(msgs)
        ids = trained_tokenizer.encode(text)
        # Encoding must succeed without raising
        assert len(ids) > 0


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_missing_tokenizer_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            HibariTokenizer.load(tmp_path / "nonexistent_dir")

    def test_empty_tokenizer_dir_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            HibariTokenizer.load(tmp_path / "empty")

    def test_untrained_tokenizer_has_no_merges(self):
        """An untrained tokenizer has no BPE merges."""
        tok = HibariTokenizer()
        merges = _get_bpe_merges(tok)
        assert merges == []

    def test_export_with_untrained_tokenizer_raises(self, small_model, tmp_path):
        """Exporting with an untrained tokenizer (no merges) must raise RuntimeError."""
        tok = HibariTokenizer()
        with pytest.raises(RuntimeError, match="no BPE merge rules"):
            export_to_gguf(
                small_model,
                tokenizer=tok,
                output_path=tmp_path / "bad_merges.gguf",
            )

    def test_missing_metadata_fields_detected(self, small_model, trained_tokenizer, tmp_path):
        """Verify that a GGUF file missing merges is distinguishable from one that has them."""
        # Build a GGUF without merges (simulate the old broken exporter)
        broken_path = tmp_path / "broken.gguf"
        writer = gguf.GGUFWriter(str(broken_path), "llama")
        writer.add_name("broken")
        writer.add_context_length(64)
        writer.add_embedding_length(64)
        writer.add_block_count(0)
        writer.add_feed_forward_length(128)
        writer.add_head_count(2)
        writer.add_layer_norm_rms_eps(1e-5)
        writer.add_rope_freq_base(10000.0)
        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(["a", "b"])
        # NOTE: intentionally NOT calling add_token_merges()
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        reader = gguf.GGUFReader(str(broken_path))
        assert "tokenizer.ggml.merges" not in reader.fields, (
            "The broken GGUF should NOT contain merges"
        )

        # The fixed export MUST have merges
        fixed_path = tmp_path / "fixed.gguf"
        export_to_gguf(small_model, trained_tokenizer, fixed_path)
        fixed_reader = gguf.GGUFReader(str(fixed_path))
        assert "tokenizer.ggml.merges" in fixed_reader.fields

    def test_unsupported_model_config_zero_layers(self):
        """A model with zero layers is degenerate but must not crash export."""
        config = ModelConfig(
            vocab_size=64,
            max_seq_len=16,
            d_model=32,
            n_layers=0,
            n_heads=2,
            d_ff=64,
        )
        model = HibariKotoLM(config)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "zero_layer.gguf"
            export_to_gguf(model, tokenizer=None, output_path=out)
            assert out.exists()

    def test_incorrect_bos_id_would_fail_chat(self, trained_tokenizer):
        """Sanity check: wrong BOS ID breaks the encoding contract."""
        # We test that the actual BOS ID encodes to exactly that ID
        bos_text = "<|bos|>"
        actual_id = trained_tokenizer._tokenizer.token_to_id("<|bos|>")
        assert actual_id == trained_tokenizer.bos_id

    def test_merge_string_does_not_contain_extra_spaces(self, trained_tokenizer):
        """Each merge must be exactly 'left right' — no leading/trailing spaces."""
        for merge in _get_bpe_merges(trained_tokenizer):
            assert merge == merge.strip(), f"Merge has extra whitespace: {merge!r}"
            parts = merge.split(" ")
            assert len(parts) == 2, f"Merge split into ≠2 parts: {parts}"
