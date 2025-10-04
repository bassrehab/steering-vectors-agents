"""Tests for steering vectors."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from steering_vectors_agents.core.vectors import SteeringVector, SteeringVectorSet


class TestSteeringVector:
    def test_creation(self):
        vec = SteeringVector(
            behavior="refusal",
            layer_index=15,
            vector=torch.randn(4096),
            model_name="test-model",
        )

        assert vec.behavior == "refusal"
        assert vec.layer_index == 15
        assert vec.dim == 4096
        assert "created_at" in vec.metadata

    def test_norm(self):
        # create vector with known norm
        v = torch.zeros(100)
        v[0] = 3.0
        v[1] = 4.0  # norm should be 5

        vec = SteeringVector(
            behavior="test",
            layer_index=0,
            vector=v,
            model_name="test",
        )

        assert abs(vec.norm - 5.0) < 1e-5

    def test_normalize(self):
        vec = SteeringVector(
            behavior="test",
            layer_index=0,
            vector=torch.randn(64) * 10,
            model_name="test",
        )

        normalized = vec.normalize()

        assert abs(normalized.norm - 1.0) < 1e-5
        assert normalized.metadata.get("normalized") is True

    def test_scale(self):
        original = torch.randn(64)
        vec = SteeringVector(
            behavior="test",
            layer_index=0,
            vector=original.clone(),
            model_name="test",
        )

        scaled = vec.scale(2.0)

        assert torch.allclose(scaled.vector, original * 2.0)
        assert scaled.metadata.get("scale_factor") == 2.0

    def test_to_device(self):
        vec = SteeringVector(
            behavior="test",
            layer_index=0,
            vector=torch.randn(64),
            model_name="test",
        )

        moved = vec.to("cpu")
        assert moved.vector.device.type == "cpu"

    def test_save_and_load(self):
        original = SteeringVector(
            behavior="refusal",
            layer_index=15,
            vector=torch.randn(128),
            model_name="test-model",
            extraction_method="caa",
            metadata={"test_key": "test_value"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_vector"
            original.save(path)

            # check files exist
            assert (path.with_suffix(".pt")).exists()
            assert (path.with_suffix(".json")).exists()

            # load and verify
            loaded = SteeringVector.load(path)

            assert loaded.behavior == original.behavior
            assert loaded.layer_index == original.layer_index
            assert loaded.model_name == original.model_name
            assert loaded.extraction_method == original.extraction_method
            assert torch.allclose(loaded.vector, original.vector)


class TestSteeringVectorSet:
    def test_creation_empty(self):
        vec_set = SteeringVectorSet(behavior="refusal")
        assert vec_set.behavior == "refusal"
        assert len(vec_set) == 0

    def test_creation_with_vectors(self):
        vectors = [
            SteeringVector("refusal", 14, torch.randn(64), "test"),
            SteeringVector("refusal", 15, torch.randn(64), "test"),
            SteeringVector("refusal", 16, torch.randn(64), "test"),
        ]

        vec_set = SteeringVectorSet(behavior="refusal", vectors=vectors)

        assert len(vec_set) == 3
        assert vec_set.layers == [14, 15, 16]

    def test_behavior_mismatch_raises(self):
        wrong_vec = SteeringVector("uncertainty", 14, torch.randn(64), "test")

        with pytest.raises(ValueError, match="behavior"):
            SteeringVectorSet(behavior="refusal", vectors=[wrong_vec])

    def test_add_and_get(self):
        vec_set = SteeringVectorSet(behavior="refusal")
        vec = SteeringVector("refusal", 15, torch.randn(64), "test")

        vec_set.add(vec)

        assert vec_set.get(15) is vec
        assert vec_set.get(99) is None

    def test_get_best(self):
        # create vectors with different norms
        v1 = torch.randn(64)
        v2 = torch.randn(64) * 2  # larger norm

        vec_set = SteeringVectorSet(
            behavior="refusal",
            vectors=[
                SteeringVector("refusal", 14, v1, "test"),
                SteeringVector("refusal", 15, v2, "test"),
            ],
        )

        best = vec_set.get_best(metric="norm")
        assert best.layer_index == 15  # v2 has larger norm

    def test_get_best_empty_raises(self):
        vec_set = SteeringVectorSet(behavior="refusal")

        with pytest.raises(ValueError, match="No vectors"):
            vec_set.get_best()

    def test_iteration(self):
        vectors = [
            SteeringVector("refusal", i, torch.randn(64), "test")
            for i in range(3)
        ]
        vec_set = SteeringVectorSet(behavior="refusal", vectors=vectors)

        iterated = list(vec_set)
        assert len(iterated) == 3

    def test_getitem(self):
        vec = SteeringVector("refusal", 15, torch.randn(64), "test")
        vec_set = SteeringVectorSet(behavior="refusal", vectors=[vec])

        assert vec_set[15] is vec

        with pytest.raises(KeyError):
            _ = vec_set[99]

    def test_save_and_load(self):
        vectors = [
            SteeringVector("refusal", 14, torch.randn(64), "test-model"),
            SteeringVector("refusal", 15, torch.randn(64), "test-model"),
        ]
        original = SteeringVectorSet(behavior="refusal", vectors=vectors)

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir) / "refusal_vectors"
            original.save(dir_path)

            # check structure
            assert (dir_path / "metadata.json").exists()
            assert (dir_path / "layer_14.pt").exists()
            assert (dir_path / "layer_15.pt").exists()

            # load and verify
            loaded = SteeringVectorSet.load(dir_path)

            assert loaded.behavior == original.behavior
            assert loaded.layers == original.layers
            assert len(loaded) == len(original)

            for layer_idx in original.layers:
                orig_vec = original.get(layer_idx)
                load_vec = loaded.get(layer_idx)
                assert torch.allclose(orig_vec.vector, load_vec.vector)

    def test_to_device(self):
        vectors = [
            SteeringVector("refusal", i, torch.randn(64), "test")
            for i in range(2)
        ]
        vec_set = SteeringVectorSet(behavior="refusal", vectors=vectors)

        moved = vec_set.to("cpu")

        for vec in moved:
            assert vec.vector.device.type == "cpu"
