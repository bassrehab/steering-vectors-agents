"""Steering vector representation and manipulation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class SteeringVector:
    """A steering vector for a specific behavior and layer."""

    behavior: str
    layer_index: int
    vector: torch.Tensor
    model_name: str
    extraction_method: str = "caa"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()

    @property
    def norm(self) -> float:
        """L2 norm of the vector."""
        return self.vector.norm().item()

    @property
    def dim(self) -> int:
        """Dimension of the vector."""
        return self.vector.shape[-1]

    def normalize(self) -> "SteeringVector":
        """Return L2-normalized copy."""
        normalized = self.vector / self.vector.norm()
        return SteeringVector(
            behavior=self.behavior,
            layer_index=self.layer_index,
            vector=normalized,
            model_name=self.model_name,
            extraction_method=self.extraction_method,
            metadata={**self.metadata, "normalized": True},
        )

    def scale(self, factor: float) -> "SteeringVector":
        """Return scaled copy."""
        return SteeringVector(
            behavior=self.behavior,
            layer_index=self.layer_index,
            vector=self.vector * factor,
            model_name=self.model_name,
            extraction_method=self.extraction_method,
            metadata={**self.metadata, "scale_factor": factor},
        )

    def to(self, device: str) -> "SteeringVector":
        """Move vector to device."""
        return SteeringVector(
            behavior=self.behavior,
            layer_index=self.layer_index,
            vector=self.vector.to(device),
            model_name=self.model_name,
            extraction_method=self.extraction_method,
            metadata=self.metadata,
        )

    def save(self, path: Path) -> None:
        """Save vector to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "behavior": self.behavior,
            "layer_index": self.layer_index,
            "model_name": self.model_name,
            "extraction_method": self.extraction_method,
            "metadata": self.metadata,
            "vector_shape": list(self.vector.shape),
        }

        # save metadata + tensor separately (easier to inspect metadata without loading torch)
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(data, f, indent=2)

        torch.save(self.vector, path.with_suffix(".pt"))

    @classmethod
    def load(cls, path: Path) -> "SteeringVector":
        """Load vector from file."""
        path = Path(path)

        # handle both .pt and .json paths (users pass either, we figure it out)
        if path.suffix == ".pt":
            meta_path = path.with_suffix(".json")
            tensor_path = path
        elif path.suffix == ".json":
            meta_path = path
            tensor_path = path.with_suffix(".pt")
        else:
            meta_path = path.with_suffix(".json")
            tensor_path = path.with_suffix(".pt")

        with open(meta_path) as f:
            data = json.load(f)

        vector = torch.load(tensor_path, weights_only=True)

        return cls(
            behavior=data["behavior"],
            layer_index=data["layer_index"],
            vector=vector,
            model_name=data["model_name"],
            extraction_method=data.get("extraction_method", "caa"),
            metadata=data.get("metadata", {}),
        )


class SteeringVectorSet:
    """Collection of steering vectors for a behavior across multiple layers."""

    def __init__(self, behavior: str, vectors: Optional[List[SteeringVector]] = None):
        self.behavior = behavior
        self._vectors: Dict[int, SteeringVector] = {}

        if vectors:
            for v in vectors:
                if v.behavior != behavior:
                    raise ValueError(f"Vector behavior '{v.behavior}' != set behavior '{behavior}'")
                self._vectors[v.layer_index] = v

    def add(self, vector: SteeringVector) -> None:
        """Add a vector to the set."""
        if vector.behavior != self.behavior:
            raise ValueError(f"Vector behavior mismatch: {vector.behavior} != {self.behavior}")
        self._vectors[vector.layer_index] = vector

    def get(self, layer_index: int) -> Optional[SteeringVector]:
        """Get vector for specific layer."""
        return self._vectors.get(layer_index)

    def get_best(self, metric: str = "norm") -> SteeringVector:
        """Get the 'best' vector based on a metric (just norm for now)."""
        if not self._vectors:
            raise ValueError("empty vector set")

        if metric == "norm":
            return max(self._vectors.values(), key=lambda v: v.norm)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    @property
    def layers(self) -> List[int]:
        """List of layer indices with vectors."""
        return sorted(self._vectors.keys())

    @property
    def model_name(self) -> Optional[str]:
        """Model name (from first vector)."""
        if self._vectors:
            return next(iter(self._vectors.values())).model_name
        return None

    def to(self, device: str) -> "SteeringVectorSet":
        """Move all vectors to device."""
        new_set = SteeringVectorSet(self.behavior)
        for layer_idx, vec in self._vectors.items():
            new_set.add(vec.to(device))
        return new_set

    def save(self, dir_path: Path) -> None:
        """Save all vectors to directory."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # save each vector
        for layer_idx, vector in self._vectors.items():
            vector.save(dir_path / f"layer_{layer_idx}")

        # save set metadata
        meta = {
            "behavior": self.behavior,
            "layers": self.layers,
            "model_name": self.model_name,
        }
        with open(dir_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, dir_path: Path) -> "SteeringVectorSet":
        """Load all vectors from directory."""
        dir_path = Path(dir_path)

        # load set metadata
        with open(dir_path / "metadata.json") as f:
            meta = json.load(f)

        vec_set = cls(behavior=meta["behavior"])

        # load each vector
        for layer_idx in meta["layers"]:
            vector = SteeringVector.load(dir_path / f"layer_{layer_idx}")
            vec_set.add(vector)

        return vec_set

    def __len__(self) -> int:
        return len(self._vectors)

    def __iter__(self):
        return iter(self._vectors.values())

    def __getitem__(self, layer_index: int) -> SteeringVector:
        if layer_index not in self._vectors:
            raise KeyError(f"No vector for layer {layer_index}")
        return self._vectors[layer_index]
