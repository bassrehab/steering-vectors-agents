"""Model configurations for different architectures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for a specific model architecture."""

    name: str
    num_layers: int
    hidden_size: int
    layer_template: str = "model.layers.{i}"
    residual_template: str = "model.layers.{i}"
    mlp_template: str = "model.layers.{i}.mlp"
    attn_template: str = "model.layers.{i}.self_attn"

    # recommended layers for different behaviors (empirically determined)
    recommended_layers: Dict[str, List[int]] = field(default_factory=dict)

    def get_recommended_layers(self, behavior: str) -> List[int]:
        """Get recommended layers for a behavior, or middle layers as default."""
        if behavior in self.recommended_layers:
            return self.recommended_layers[behavior]
        # default: middle third of layers
        start = self.num_layers // 3
        end = 2 * self.num_layers // 3
        return list(range(start, end))


# Pre-configured models
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    # Qwen3 family
    "Qwen/Qwen3-8B": ModelConfig(
        name="Qwen/Qwen3-8B",
        num_layers=36,
        hidden_size=4096,
        recommended_layers={
            "refusal": [14, 15, 16, 17, 18],
            "uncertainty": [12, 13, 14, 15, 16],
            "tool_restraint": [16, 17, 18, 19, 20],
            "instruction_hierarchy": [14, 15, 16, 17, 18],
        },
    ),
    "Qwen/Qwen3-4B": ModelConfig(
        name="Qwen/Qwen3-4B",
        num_layers=36,
        hidden_size=2560,
        recommended_layers={
            "refusal": [14, 15, 16, 17, 18],
            "uncertainty": [12, 13, 14, 15, 16],
            "tool_restraint": [16, 17, 18, 19, 20],
            "instruction_hierarchy": [14, 15, 16, 17, 18],
        },
    ),
    "Qwen/Qwen3-14B": ModelConfig(
        name="Qwen/Qwen3-14B",
        num_layers=48,
        hidden_size=5120,
        recommended_layers={
            "refusal": [20, 21, 22, 23, 24],
            "uncertainty": [16, 17, 18, 19, 20],
            "tool_restraint": [22, 23, 24, 25, 26],
            "instruction_hierarchy": [20, 21, 22, 23, 24],
        },
    ),
    # DeepSeek
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": ModelConfig(
        name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        num_layers=48,
        hidden_size=5120,
        recommended_layers={
            "refusal": [20, 21, 22, 23, 24],
            "uncertainty": [16, 17, 18, 19, 20],
            "tool_restraint": [22, 23, 24, 25, 26],
            "instruction_hierarchy": [20, 21, 22, 23, 24],
        },
    ),
    # Llama family (for reference)
    "meta-llama/Llama-3.1-8B-Instruct": ModelConfig(
        name="meta-llama/Llama-3.1-8B-Instruct",
        num_layers=32,
        hidden_size=4096,
        recommended_layers={
            "refusal": [14, 15, 16],
            "uncertainty": [12, 13, 14],
            "tool_restraint": [16, 17, 18],
            "instruction_hierarchy": [14, 15, 16],
        },
    ),
    "meta-llama/Llama-3.1-70B-Instruct": ModelConfig(
        name="meta-llama/Llama-3.1-70B-Instruct",
        num_layers=80,
        hidden_size=8192,
        recommended_layers={
            "refusal": [35, 36, 37, 38, 39, 40],
            "uncertainty": [30, 31, 32, 33, 34, 35],
            "tool_restraint": [40, 41, 42, 43, 44, 45],
            "instruction_hierarchy": [35, 36, 37, 38, 39, 40],
        },
    ),
    # Mistral
    "mistralai/Mistral-7B-Instruct-v0.2": ModelConfig(
        name="mistralai/Mistral-7B-Instruct-v0.2",
        num_layers=32,
        hidden_size=4096,
        recommended_layers={
            "refusal": [12, 14, 16, 18, 20],
            "uncertainty": [10, 12, 14, 16],
            "tool_restraint": [14, 16, 18, 20],
            "instruction_hierarchy": [12, 14, 16, 18],
        },
    ),
    "mistralai/Mistral-7B-Instruct-v0.3": ModelConfig(
        name="mistralai/Mistral-7B-Instruct-v0.3",
        num_layers=32,
        hidden_size=4096,
        recommended_layers={
            "refusal": [12, 14, 16, 18, 20],
            "uncertainty": [10, 12, 14, 16],
            "tool_restraint": [14, 16, 18, 20],
            "instruction_hierarchy": [12, 14, 16, 18],
        },
    ),
    # OpenAI GPT-OSS (August 2025)
    "openai/gpt-oss-20b": ModelConfig(
        name="openai/gpt-oss-20b",
        num_layers=24,
        hidden_size=2880,
        recommended_layers={
            "refusal": [8, 10, 12, 14, 16],  # middle layers for 24-layer model
            "uncertainty": [6, 8, 10, 12],
            "tool_restraint": [10, 12, 14, 16],
            "instruction_hierarchy": [8, 10, 12, 14],
        },
    ),
    # Google Gemma 2 (June 2024)
    "google/gemma-2-9b-it": ModelConfig(
        name="google/gemma-2-9b-it",
        num_layers=42,  # Gemma 2 9B has 42 layers
        hidden_size=3584,  # Gemma 2 9B hidden size
        recommended_layers={
            "refusal": [14, 16, 18, 20, 22],  # middle layers for 42-layer model
            "uncertainty": [12, 14, 16, 18],
            "tool_restraint": [16, 18, 20, 22],
            "instruction_hierarchy": [14, 16, 18, 20],
        },
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get config for a model by name.

    Tries exact match first, then partial match.
    """
    # exact match
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]

    # partial match (useful for versioned model names)
    model_lower = model_name.lower()
    for key, config in MODEL_CONFIGS.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return config

    raise ValueError(
        f"No config found for model: {model_name}. "
        f"Available: {list(MODEL_CONFIGS.keys())}"
    )


def infer_model_config(model) -> ModelConfig:
    """Infer config from a loaded model."""
    model_name = getattr(model.config, "_name_or_path", None)

    if model_name:
        try:
            return get_model_config(model_name)
        except ValueError:
            pass

    # fallback: infer from model structure
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
        hidden_size = model.config.hidden_size
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        num_layers = len(model.transformer.h)
        hidden_size = model.config.n_embd
    else:
        raise ValueError("Cannot infer model config from unknown architecture")

    return ModelConfig(
        name=model_name or "unknown",
        num_layers=num_layers,
        hidden_size=hidden_size,
    )
