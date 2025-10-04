"""Tests for activation injection."""

import pytest
import torch
from torch import nn

from steering_vectors_agents.core.injection import (
    ActivationInjector,
    MultiVectorInjector,
)
from steering_vectors_agents.core.vectors import SteeringVector, SteeringVectorSet


class SimpleTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.mlp = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return x + self.mlp(x) * 0.1  # small change to track


class SimpleModel(nn.Module):
    def __init__(self, num_layers: int = 4, hidden_size: int = 64):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [SimpleTransformerLayer(hidden_size) for _ in range(num_layers)]
        )
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.hidden_size = hidden_size

    def forward(self, x=None, input_ids=None, **kwargs):
        if x is None:
            batch = input_ids.shape[0] if input_ids is not None else 1
            seq = input_ids.shape[1] if input_ids is not None else 10
            x = torch.randn(batch, seq, self.hidden_size)

        for layer in self.model.layers:
            x = layer(x)
        return x


class TestActivationInjector:
    @pytest.fixture
    def model(self):
        return SimpleModel(num_layers=4, hidden_size=64)

    @pytest.fixture
    def steering_vector(self):
        return SteeringVector(
            behavior="refusal",
            layer_index=1,
            vector=torch.randn(64),
            model_name="test",
        )

    def test_attach_and_detach(self, model, steering_vector):
        injector = ActivationInjector(model, [steering_vector])

        injector.attach()
        assert injector._attached
        assert len(injector._handles) == 1

        injector.detach()
        assert not injector._attached
        assert len(injector._handles) == 0

    def test_context_manager(self, model, steering_vector):
        injector = ActivationInjector(model, [steering_vector])

        with injector:
            assert injector._attached

        assert not injector._attached

    def test_injection_changes_output(self, model, steering_vector):
        x = torch.randn(1, 5, 64)

        # baseline
        with torch.no_grad():
            baseline = model(x=x.clone()).clone()

        # with injection
        injector = ActivationInjector(model, [steering_vector], strength=5.0)
        with injector:
            with torch.no_grad():
                injected = model(x=x.clone())

        assert not torch.allclose(baseline, injected)

    def test_strength_zero_no_change(self, model, steering_vector):
        x = torch.randn(1, 5, 64)

        with torch.no_grad():
            baseline = model(x=x.clone()).clone()

        injector = ActivationInjector(model, [steering_vector], strength=0.0)
        with injector:
            with torch.no_grad():
                injected = model(x=x.clone())

        assert torch.allclose(baseline, injected)

    def test_strength_property(self, model, steering_vector):
        injector = ActivationInjector(model, [steering_vector], strength=1.0)

        assert injector.strength == 1.0

        injector.strength = 2.0
        assert injector.strength == 2.0

    def test_injection_mode_all(self, model, steering_vector):
        # just verify it runs without error
        injector = ActivationInjector(
            model, [steering_vector], strength=1.0, injection_mode="all"
        )
        with injector:
            model(input_ids=torch.zeros(1, 5, dtype=torch.long))

    def test_injection_mode_last(self, model, steering_vector):
        injector = ActivationInjector(
            model, [steering_vector], strength=1.0, injection_mode="last"
        )
        with injector:
            model(input_ids=torch.zeros(1, 5, dtype=torch.long))

    def test_injection_mode_first(self, model, steering_vector):
        injector = ActivationInjector(
            model, [steering_vector], strength=1.0, injection_mode="first"
        )
        with injector:
            model(input_ids=torch.zeros(1, 5, dtype=torch.long))


class TestMultiVectorInjector:
    @pytest.fixture
    def model(self):
        return SimpleModel(num_layers=4, hidden_size=64)

    @pytest.fixture
    def vector_sets(self):
        return {
            "refusal": SteeringVectorSet(
                behavior="refusal",
                vectors=[SteeringVector("refusal", 1, torch.randn(64), "test")],
            ),
            "uncertainty": SteeringVectorSet(
                behavior="uncertainty",
                vectors=[SteeringVector("uncertainty", 2, torch.randn(64), "test")],
            ),
        }

    def test_creation(self, model, vector_sets):
        injector = MultiVectorInjector(
            model,
            vector_sets,
            strengths={"refusal": 1.0, "uncertainty": 0.5},
        )

        assert injector.get_strength("refusal") == 1.0
        assert injector.get_strength("uncertainty") == 0.5

    def test_set_strength(self, model, vector_sets):
        injector = MultiVectorInjector(model, vector_sets)

        injector.set_strength("refusal", 2.0)
        assert injector.get_strength("refusal") == 2.0

    def test_set_strength_unknown_raises(self, model, vector_sets):
        injector = MultiVectorInjector(model, vector_sets)

        with pytest.raises(KeyError):
            injector.set_strength("unknown_behavior", 1.0)

    def test_context_manager(self, model, vector_sets):
        injector = MultiVectorInjector(model, vector_sets)

        with injector:
            assert injector._attached

        assert not injector._attached

    def test_multiple_behaviors_inject(self, model, vector_sets):
        x = torch.randn(1, 5, 64)

        with torch.no_grad():
            baseline = model(x=x.clone()).clone()

        injector = MultiVectorInjector(
            model,
            vector_sets,
            strengths={"refusal": 5.0, "uncertainty": 5.0},
        )

        with injector:
            with torch.no_grad():
                injected = model(x=x.clone())

        assert not torch.allclose(baseline, injected)

    def test_zero_strength_no_effect(self, model, vector_sets):
        x = torch.randn(1, 5, 64)

        with torch.no_grad():
            baseline = model(x=x.clone()).clone()

        injector = MultiVectorInjector(
            model,
            vector_sets,
            strengths={"refusal": 0.0, "uncertainty": 0.0},
        )

        with injector:
            with torch.no_grad():
                injected = model(x=x.clone())

        assert torch.allclose(baseline, injected)
