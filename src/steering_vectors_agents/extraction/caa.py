"""Contrastive Activation Addition (CAA) extraction algorithm."""

from __future__ import annotations

from typing import List, Literal, Optional, Union

import torch
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.hooks import ActivationHook
from ..core.vectors import SteeringVector, SteeringVectorSet
from ..datasets.base import ContrastPairDataset


def extract_caa_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    contrast_pairs: Union[ContrastPairDataset, List[dict]],
    layer_idx: int,
    token_position: Literal["last", "first", "mean"] = "last",
    batch_size: int = 1,  # not actually used yet, kept for API compat
    show_progress: bool = True,
) -> SteeringVector:
    """
    Extract steering vector using CAA: mean(positive) - mean(negative) activations.
    """
    device = next(model.parameters()).device
    model.eval()

    # get pairs as list
    if isinstance(contrast_pairs, ContrastPairDataset):
        pairs = list(contrast_pairs)
        behavior = contrast_pairs.behavior
    else:
        pairs = contrast_pairs
        behavior = "unknown"

    positive_activations = []
    negative_activations = []

    iterator = tqdm(pairs, desc=f"Layer {layer_idx}", disable=not show_progress)

    for pair in iterator:
        # get texts
        if hasattr(pair, "positive"):
            pos_text = pair.positive
            neg_text = pair.negative
        else:
            pos_text = pair["positive"]
            neg_text = pair["negative"]

        # process positive
        pos_act = _get_activation(
            model, tokenizer, pos_text, layer_idx, token_position, device
        )
        positive_activations.append(pos_act)

        # process negative
        neg_act = _get_activation(
            model, tokenizer, neg_text, layer_idx, token_position, device
        )
        negative_activations.append(neg_act)

    # compute mean activations
    pos_mean = torch.stack(positive_activations).mean(dim=0)
    neg_mean = torch.stack(negative_activations).mean(dim=0)

    # steering vector = difference of means
    # this is the core of CAA - surprisingly simple but it works
    steering_vector = pos_mean - neg_mean

    model_name = getattr(model.config, "_name_or_path", "unknown")

    return SteeringVector(
        behavior=behavior,
        layer_index=layer_idx,
        vector=steering_vector.cpu(),
        model_name=model_name,
        extraction_method="caa",
        metadata={
            "num_pairs": len(pairs),
            "token_position": token_position,
            "pos_mean_norm": pos_mean.norm().item(),
            "neg_mean_norm": neg_mean.norm().item(),
            "vector_norm": steering_vector.norm().item(),
        },
    )


def _get_activation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    layer_idx: int,
    token_position: str,
    device: torch.device,
) -> torch.Tensor:
    """Get activation for a single text."""
    # tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # set up hook
    hook = ActivationHook(model, [layer_idx], component="residual", token_position="all")

    with hook:
        with torch.no_grad():
            model(**inputs)

    # get activation
    activation = hook.cache.get(f"layer_{layer_idx}")  # (1, seq_len, hidden)

    if activation is None:
        raise RuntimeError(f"Failed to capture activation at layer {layer_idx}")

    # select token position
    if token_position == "last":
        result = activation[0, -1, :]  # (hidden,)
    elif token_position == "first":
        result = activation[0, 0, :]
    elif token_position == "mean":
        result = activation[0].mean(dim=0)
    else:
        raise ValueError(f"Unknown token_position: {token_position}")

    return result


def extract_caa_vectors(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    contrast_pairs: Union[ContrastPairDataset, List[dict]],
    layer_indices: List[int],
    token_position: Literal["last", "first", "mean"] = "last",
    show_progress: bool = True,
) -> SteeringVectorSet:
    """
    Extract steering vectors for multiple layers.

    Args:
        model: HuggingFace model
        tokenizer: Corresponding tokenizer
        contrast_pairs: Dataset of positive/negative pairs
        layer_indices: Layers to extract from
        token_position: Which token's activation to use
        show_progress: Show progress bar

    Returns:
        SteeringVectorSet containing vectors for all specified layers
    """
    if isinstance(contrast_pairs, ContrastPairDataset):
        behavior = contrast_pairs.behavior
    else:
        behavior = "unknown"

    vectors = []

    for layer_idx in layer_indices:
        vector = extract_caa_vector(
            model=model,
            tokenizer=tokenizer,
            contrast_pairs=contrast_pairs,
            layer_idx=layer_idx,
            token_position=token_position,
            show_progress=show_progress,
        )
        vectors.append(vector)

    return SteeringVectorSet(behavior=behavior, vectors=vectors)
