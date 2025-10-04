"""
steering-vectors-agents: Control agent behaviors through activation steering.

This package provides tools for extracting and applying steering vectors
to control LLM agent behaviors at inference time, without retraining.
"""

from steering_vectors_agents.version import __version__

from steering_vectors_agents.core import (
    ActivationCache,
    ActivationHook,
    ActivationInjector,
    MODEL_CONFIGS,
    ModelConfig,
    MultiVectorInjector,
    SteeringVector,
    SteeringVectorSet,
    extract_activations,
    get_model_config,
    infer_model_config,
)

__all__ = [
    "__version__",
    "ActivationCache",
    "ActivationHook",
    "ActivationInjector",
    "MODEL_CONFIGS",
    "ModelConfig",
    "MultiVectorInjector",
    "SteeringVector",
    "SteeringVectorSet",
    "extract_activations",
    "get_model_config",
    "infer_model_config",
]
