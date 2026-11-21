"""Tests for Hibari model architecture."""

import pytest
import torch

from hibari.model.config import ModelConfig
from hibari.model.llm import HibariKotoLM


def test_model_config():
    config = ModelConfig(
        vocab_size=1024,
        max_seq_len=256,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=384,
        tie_embeddings=True,
    )
    assert config.head_dim == 32
    est_params = config.estimate_parameters()
    # Should be ~984K to 1M
    assert 900_000 <= est_params <= 1_100_000


def test_model_parameter_count():
    config = ModelConfig(
        vocab_size=1024,
        max_seq_len=256,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=384,
        tie_embeddings=True,
    )
    model = HibariKotoLM(config)
    counts = model.count_parameters()
    assert counts["total"] == 984_192
    assert counts["trainable"] == 984_192


def test_model_forward():
    config = ModelConfig(
        vocab_size=1024,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
    )
    model = HibariKotoLM(config)
    input_ids = torch.randint(0, 1024, (2, 16))
    targets = torch.randint(0, 1024, (2, 16))

    logits, loss, _ = model(input_ids, targets=targets)
    assert logits.shape == (2, 16, 1024)
    assert loss is not None
    assert loss.item() > 0.0


def test_model_generate():
    config = ModelConfig(
        vocab_size=1024,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
    )
    model = HibariKotoLM(config)
    input_ids = torch.randint(0, 1024, (1, 5))
    output = model.generate(input_ids, max_new_tokens=5, temperature=0.0)
    assert output.shape == (1, 10)


# ───────────── Hibari Koto (actual released configuration) ─────────────

from pathlib import Path

from hibari.model.embedding import apply_rotary_emb, precompute_rope_freqs
from hibari.model.normalization import RMSNorm
from hibari.model.transformer import SwiGLU

# Koto's configs live in the Koto submodule (models/Koto) since the engine/model split.
KOTO_YAML = Path(__file__).resolve().parents[1] / "models" / "Koto" / "configs" / "model" / "koto.yaml"
needs_koto = pytest.mark.skipif(not KOTO_YAML.exists(), reason="Koto submodule not checked out")


@needs_koto
def test_koto_config_values():
    c = ModelConfig.from_yaml(KOTO_YAML)
    assert (c.vocab_size, c.max_seq_len, c.d_model, c.n_layers, c.n_heads, c.d_ff) == (4096, 256, 256, 6, 8, 768)
    assert c.tie_embeddings and c.head_dim == 32 and c.model_name == "Hibari-Koto"


@needs_koto
def test_koto_parameter_count_is_6163712():
    c = ModelConfig.from_yaml(KOTO_YAML)
    m = HibariKotoLM(c)
    assert m.count_parameters()["total"] == 6_163_712
    assert c.estimate_parameters() == 6_163_712


def test_backward_pass_all_grads():
    m = HibariKotoLM(ModelConfig(vocab_size=128, max_seq_len=32, d_model=32, n_layers=2, n_heads=2, d_ff=64))
    ids = torch.randint(0, 128, (2, 16))
    _, loss, _ = m(ids, targets=ids)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_causality_future_tokens_do_not_affect_past():
    torch.manual_seed(0)
    m = HibariKotoLM(ModelConfig(vocab_size=128, max_seq_len=32, d_model=32, n_layers=2, n_heads=2, d_ff=64)).eval()
    a = torch.randint(0, 128, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % 128
    with torch.no_grad():
        la, lb = m(a)[0], m(b)[0]
    assert torch.allclose(la[:, :8], lb[:, :8], atol=1e-5)
    assert not torch.allclose(la[:, 8:], lb[:, 8:])


def test_rope_rotates_adjacent_pairs_and_preserves_norm():
    cos, sin = precompute_rope_freqs(head_dim=8, max_seq_len=16)
    x = torch.randn(1, 1, 16, 8)
    y = apply_rotary_emb(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)
    assert torch.allclose(y[0, 0, 0], x[0, 0, 0])  # position 0: identity
    # pair (0,1) at position p rotates by angle p * theta^0 = p
    p = 3
    c, s = torch.cos(torch.tensor(float(p))), torch.sin(torch.tensor(float(p)))
    x0, x1 = x[0, 0, p, 0], x[0, 0, p, 1]
    assert torch.allclose(y[0, 0, p, 0], x0 * c - x1 * s, atol=1e-5)
    assert torch.allclose(y[0, 0, p, 1], x0 * s + x1 * c, atol=1e-5)


def test_rope_relative_position_property():
    """q·k after RoPE depends only on the position difference."""
    cos, sin = precompute_rope_freqs(head_dim=8, max_seq_len=64)
    q, k = torch.randn(8), torch.randn(8)
    def rot(v, pos):
        x = v.view(1, 1, 1, 8).expand(1, 1, pos + 1, 8).clone()
        return apply_rotary_emb(x, cos, sin)[0, 0, pos]
    d1 = torch.dot(rot(q, 10), rot(k, 7))
    d2 = torch.dot(rot(q, 33), rot(k, 30))
    assert torch.allclose(d1, d2, atol=1e-4)


def test_rmsnorm():
    n = RMSNorm(16, eps=1e-6)
    x = torch.randn(4, 16) * 5
    y = n(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4), atol=1e-3)


def test_swiglu_matches_formula():
    f = SwiGLU(8, 16)
    x = torch.randn(3, 8)
    ref = f.down_proj(torch.nn.functional.silu(f.gate_proj(x)) * f.up_proj(x))
    assert torch.allclose(f(x), ref)
    assert f.gate_proj.bias is None and f.up_proj.bias is None
