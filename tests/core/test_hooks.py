"""Tests for activation hooks."""

import pytest
import torch
from torch import nn

from steering_vectors_agents.core.hooks import (
    ActivationCache,
    ActivationHook,
    extract_activations,
)


class SimpleTransformerLayer(nn.Module):
    """Minimal transformer layer for testing (not a real transformer)."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.mlp = nn.Linear(hidden_size, hidden_size)
        self.self_attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return x + self.mlp(x)


class SimpleModel(nn.Module):
    """Minimal model with layers for testing hooks."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 64):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [SimpleTransformerLayer(hidden_size) for _ in range(num_layers)]
        )
        self.config = type("Config", (), {"hidden_size": hidden_size})()

    def forward(self, input_ids=None, **kwargs):
        # simulate transformer forward
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        seq_len = input_ids.shape[1] if input_ids is not None else 10
        x = torch.randn(batch_size, seq_len, self.config.hidden_size)

        for layer in self.model.layers:
            x = layer(x)

        return x


class TestActivationCache:
    def test_store_and_get(self):
        cache = ActivationCache()
        tensor = torch.randn(2, 10, 64)

        cache.store("test", tensor)
        retrieved = cache.get("test")

        assert retrieved is not None
        assert torch.allclose(retrieved, tensor)

    def test_get_missing_returns_none(self):
        cache = ActivationCache()
        assert cache.get("missing") is None

    def test_clear(self):
        cache = ActivationCache()
        cache.store("test", torch.randn(2, 10, 64))

        cache.clear()

        assert cache.get("test") is None
        assert len(cache) == 0

    def test_contains(self):
        cache = ActivationCache()
        cache.store("exists", torch.randn(2, 10, 64))

        assert "exists" in cache
        assert "missing" not in cache

    def test_keys(self):
        cache = ActivationCache()
        cache.store("a", torch.randn(2, 10, 64))
        cache.store("b", torch.randn(2, 10, 64))

        keys = cache.keys()
        assert set(keys) == {"a", "b"}


class TestActivationHook:
    @pytest.fixture
    def model(self):
        return SimpleModel(num_layers=4, hidden_size=64)

    def test_attach_and_detach(self, model):
        hook = ActivationHook(model, layer_indices=[0, 1])

        hook.attach()
        assert hook._attached
        assert len(hook._handles) == 2

        hook.detach()
        assert not hook._attached
        assert len(hook._handles) == 0

    def test_context_manager(self, model):
        hook = ActivationHook(model, layer_indices=[0])

        with hook:
            assert hook._attached

        assert not hook._attached

    def test_captures_activations(self, model):
        hook = ActivationHook(model, layer_indices=[1, 2])

        with hook:
            model(input_ids=torch.zeros(1, 5, dtype=torch.long))

        activations = hook.get_activations()
        assert 1 in activations
        assert 2 in activations
        assert activations[1].shape[-1] == 64

    def test_token_position_last(self, model):
        hook = ActivationHook(model, layer_indices=[0], token_position="last")

        with hook:
            model(input_ids=torch.zeros(1, 10, dtype=torch.long))

        act = hook.cache.get("layer_0")
        assert act.shape[1] == 1  # only last token

    def test_token_position_first(self, model):
        hook = ActivationHook(model, layer_indices=[0], token_position="first")

        with hook:
            model(input_ids=torch.zeros(1, 10, dtype=torch.long))

        act = hook.cache.get("layer_0")
        assert act.shape[1] == 1

    def test_token_position_all(self, model):
        hook = ActivationHook(model, layer_indices=[0], token_position="all")

        with hook:
            model(input_ids=torch.zeros(1, 10, dtype=torch.long))

        act = hook.cache.get("layer_0")
        assert act.shape[1] == 10


class TestExtractActivations:
    def test_extract_activations(self):
        model = SimpleModel(num_layers=4, hidden_size=64)
        inputs = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}

        activations = extract_activations(
            model=model,
            inputs=inputs,
            layer_indices=[0, 2],
            token_position="last",
        )

        assert 0 in activations
        assert 2 in activations
        assert activations[0].shape == (1, 1, 64)
