"""Activation injection for applying steering vectors at runtime."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, Union

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from .vectors import SteeringVector, SteeringVectorSet


class ActivationInjector:
    """
    Injects steering vectors into model activations during inference.

    Use as context manager for automatic cleanup.
    """

    def __init__(
        self,
        model: nn.Module,
        vectors: List[SteeringVector],
        strength: float = 1.0,
        injection_mode: Literal["all", "last", "first"] = "all",
    ):
        self.model = model
        self._vectors = {v.layer_index: v for v in vectors}
        self._strength = strength
        self.injection_mode = injection_mode
        self._handles: List[RemovableHandle] = []
        self._attached = False

    @property
    def strength(self) -> float:
        return self._strength

    @strength.setter
    def strength(self, value: float) -> None:
        self._strength = value

    def _get_layer_module(self, layer_idx: int) -> nn.Module:
        """Get the module for a specific layer."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx]
        else:
            raise ValueError(f"can't find layers in model: {type(self.model).__name__}")

    def _create_injection_fn(self, layer_idx: int) -> Callable:
        """Create injection hook function for a layer."""
        vector = self._vectors[layer_idx].vector

        def hook(module: nn.Module, input: Any, output: Any) -> Any:
            if self._strength == 0:
                return output

            # handle tuple outputs (attention layers)
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            # ensure vector is on same device and dtype
            steering = vector.to(hidden_states.device).to(hidden_states.dtype)

            # apply injection based on mode
            if self.injection_mode == "all":
                # add to all positions: (batch, seq, hidden) + (hidden,)
                modified = hidden_states + (self._strength * steering)
            elif self.injection_mode == "last":
                modified = hidden_states.clone()
                modified[:, -1, :] = modified[:, -1, :] + (self._strength * steering)
            elif self.injection_mode == "first":
                modified = hidden_states.clone()
                modified[:, 0, :] = modified[:, 0, :] + (self._strength * steering)
            else:
                modified = hidden_states

            if rest is not None:
                return (modified,) + rest
            return modified

        return hook

    def attach(self) -> "ActivationInjector":
        """Attach injection hooks to model."""
        if self._attached:
            return self

        for layer_idx in self._vectors:
            module = self._get_layer_module(layer_idx)
            handle = module.register_forward_hook(self._create_injection_fn(layer_idx))
            self._handles.append(handle)

        self._attached = True
        return self

    def detach(self) -> None:
        """Remove all injection hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._attached = False

    def __enter__(self) -> "ActivationInjector":
        return self.attach()

    def __exit__(self, *args) -> None:
        self.detach()


class MultiVectorInjector:
    """Apply multiple steering vectors with independent strength control."""

    def __init__(
        self,
        model: nn.Module,
        vector_sets: Dict[str, SteeringVectorSet],
        strengths: Optional[Dict[str, float]] = None,
        injection_mode: Literal["all", "last", "first"] = "all",
        default_layer: Optional[int] = None,
    ):
        self.model = model
        self.vector_sets = vector_sets
        self._strengths = strengths or {k: 1.0 for k in vector_sets}
        self.injection_mode = injection_mode
        self.default_layer = default_layer
        self._handles: List[RemovableHandle] = []
        self._attached = False

        # build layer -> [(behavior, vector)] mapping
        self._layer_vectors: Dict[int, List[tuple]] = {}
        for behavior, vec_set in vector_sets.items():
            if default_layer is not None:
                vec = vec_set.get(default_layer)
                if vec:
                    self._layer_vectors.setdefault(default_layer, []).append((behavior, vec))
            else:
                # use best layer from each set
                try:
                    vec = vec_set.get_best()
                    self._layer_vectors.setdefault(vec.layer_index, []).append((behavior, vec))
                except ValueError:
                    pass  # empty set

    def set_strength(self, behavior: str, strength: float) -> None:
        """Set strength for a specific behavior."""
        if behavior not in self._strengths:
            raise KeyError(f"no such behavior: '{behavior}'")
        self._strengths[behavior] = strength

    def get_strength(self, behavior: str) -> float:
        """Get current strength for a behavior."""
        return self._strengths.get(behavior, 0.0)

    def _get_layer_module(self, layer_idx: int) -> nn.Module:
        # FIXME: duplicated from ActivationInjector, should extract to common helper
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx]
        raise ValueError(f"model type not supported: {type(self.model)}")

    def _create_multi_injection_fn(self, layer_idx: int) -> Callable:
        """Create hook that applies multiple vectors at one layer."""
        behavior_vectors = self._layer_vectors[layer_idx]

        def hook(module: nn.Module, input: Any, output: Any) -> Any:
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            modified = hidden_states

            for behavior, vec in behavior_vectors:
                strength = self._strengths.get(behavior, 0.0)
                if strength == 0:
                    continue

                steering = vec.vector.to(modified.device).to(modified.dtype)

                if self.injection_mode == "all":
                    modified = modified + (strength * steering)
                elif self.injection_mode == "last":
                    modified = modified.clone()
                    modified[:, -1, :] = modified[:, -1, :] + (strength * steering)
                elif self.injection_mode == "first":
                    modified = modified.clone()
                    modified[:, 0, :] = modified[:, 0, :] + (strength * steering)

            if rest is not None:
                return (modified,) + rest
            return modified

        return hook

    def attach(self) -> "MultiVectorInjector":
        """Attach injection hooks."""
        if self._attached:
            return self

        for layer_idx in self._layer_vectors:
            module = self._get_layer_module(layer_idx)
            handle = module.register_forward_hook(self._create_multi_injection_fn(layer_idx))
            self._handles.append(handle)

        self._attached = True
        return self

    def detach(self) -> None:
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._attached = False

    def __enter__(self) -> "MultiVectorInjector":
        return self.attach()

    def __exit__(self, *args) -> None:
        self.detach()
