"""Activation extraction hooks for transformer models."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, Union

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class ActivationCache:
    """Cache for storing activations captured during forward pass."""

    def __init__(self):
        self._activations: Dict[str, torch.Tensor] = {}

    def store(self, name: str, activation: torch.Tensor) -> None:
        """Store activation tensor under given name."""
        self._activations[name] = activation.detach().clone()

    def get(self, name: str) -> Optional[torch.Tensor]:
        """Retrieve activation by name, or None if not found."""
        return self._activations.get(name)

    def clear(self) -> None:
        """Clear all stored activations."""
        self._activations.clear()

    def keys(self) -> List[str]:
        """Return list of stored activation names."""
        return list(self._activations.keys())

    def __len__(self) -> int:
        return len(self._activations)

    def __contains__(self, name: str) -> bool:
        return name in self._activations


class ActivationHook:
    """
    Hook for capturing activations from transformer layers.

    Supports extracting from residual stream, MLP output, or attention output.
    Use as context manager for automatic cleanup.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: List[int],
        component: Literal["residual", "mlp", "attn"] = "residual",
        token_position: Literal["last", "first", "all"] = "all",
    ):
        self.model = model
        self.layer_indices = layer_indices
        self.component = component
        self.token_position = token_position
        self.cache = ActivationCache()
        self._handles: List[RemovableHandle] = []
        self._attached = False

    def _get_layer_module(self, layer_idx: int) -> nn.Module:
        # XXX: this is fragile - each model family has different structure
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Llama, Qwen, Mistral style
            base = self.model.model.layers[layer_idx]
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT-2 style
            base = self.model.transformer.h[layer_idx]
        else:
            # TODO: add support for more architectures (falcon, mpt, etc)
            raise ValueError(f"unsupported model type: {type(self.model).__name__}")

        if self.component == "residual":
            return base
        elif self.component == "mlp":
            return base.mlp
        elif self.component == "attn":
            if hasattr(base, "self_attn"):
                return base.self_attn
            return base.attn
        else:
            raise ValueError(f"Unknown component: {self.component}")

    def _create_hook_fn(self, layer_idx: int) -> Callable:
        """Create hook function for a layer."""
        def hook(module: nn.Module, input: Any, output: Any) -> None:
            # output can be tensor or tuple (for attn with past_key_values)
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output

            # select token position
            if self.token_position == "last":
                act = act[:, -1:, :]
            elif self.token_position == "first":
                act = act[:, :1, :]
            # "all" keeps full sequence

            self.cache.store(f"layer_{layer_idx}", act)

        return hook

    def attach(self) -> "ActivationHook":
        """Attach hooks to model layers."""
        if self._attached:
            return self

        for idx in self.layer_indices:
            module = self._get_layer_module(idx)
            handle = module.register_forward_hook(self._create_hook_fn(idx))
            self._handles.append(handle)

        self._attached = True
        return self

    def detach(self) -> None:
        """Remove all hooks from model."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._attached = False

    def __enter__(self) -> "ActivationHook":
        return self.attach()

    def __exit__(self, *args) -> None:
        self.detach()

    def get_activations(self) -> Dict[int, torch.Tensor]:
        """Return dict mapping layer index to activation tensor."""
        result = {}
        for idx in self.layer_indices:
            act = self.cache.get(f"layer_{idx}")
            if act is not None:
                result[idx] = act
        return result


def extract_activations(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    layer_indices: List[int],
    component: Literal["residual", "mlp", "attn"] = "residual",
    token_position: Literal["last", "first", "all"] = "last",
) -> Dict[int, torch.Tensor]:
    """
    Extract activations from model for given inputs.

    Convenience function that handles hook setup/teardown.
    """
    hook = ActivationHook(
        model=model,
        layer_indices=layer_indices,
        component=component,
        token_position=token_position,
    )

    with hook:
        with torch.no_grad():
            model(**inputs)

    return hook.get_activations()
