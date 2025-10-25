"""Base classes for extraction methods."""

from abc import ABC, abstractmethod
from typing import List

from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.vectors import SteeringVectorSet
from ..datasets.base import ContrastPairDataset


class ExtractionMethod(ABC):
    """Abstract base class for steering vector extraction methods."""

    name: str = "base"

    @abstractmethod
    def extract(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        contrast_pairs: ContrastPairDataset,
        layer_indices: List[int],
    ) -> SteeringVectorSet:
        """Extract steering vectors from contrast pairs."""
        pass
