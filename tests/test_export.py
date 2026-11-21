"""Tests for model export to GGUF."""

from pathlib import Path
import pytest
import gguf
import torch

from hibari.export.gguf_export import export_to_gguf
from hibari.model.config import ModelConfig
from hibari.model.llm import HibariKotoLM


def test_gguf_export(tmp_path: Path):
    config = ModelConfig(
        vocab_size=128,
        max_seq_len=32,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        model_name="Hibari-Koto-Test",
    )
    model = HibariKotoLM(config)
    out_file = tmp_path / "test.gguf"

    export_to_gguf(model, tokenizer=None, output_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0

    reader = gguf.GGUFReader(str(out_file))
    assert len(reader.tensors) > 0
