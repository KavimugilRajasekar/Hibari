"""Pytest configuration and shared fixtures."""
import pytest

@pytest.fixture
def dummy_config():
    return {
        "vocab_size": 100,
        "context_length": 16,
        "embedding_dim": 32,
        "num_layers": 2,
        "num_heads": 2,
        "ffn_hidden_dim": 64,
        "dropout": 0.0,
    }
