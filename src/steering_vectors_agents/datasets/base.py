"""Base classes for contrast pair datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class ContrastPair:
    """A single contrast pair for steering vector extraction."""

    positive: str  # text exhibiting target behavior
    negative: str  # text NOT exhibiting target behavior
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.positive or not self.negative:
            raise ValueError("need both positive and negative texts")


class ContrastPairDataset:
    """Collection of contrast pairs for a specific behavior."""

    def __init__(
        self,
        behavior: str,
        pairs: Optional[List[ContrastPair]] = None,
        description: str = "",
    ):
        self.behavior = behavior
        self.description = description
        self._pairs: List[ContrastPair] = pairs or []

    def add(self, pair: ContrastPair) -> None:
        self._pairs.append(pair)

    def add_pair(self, positive: str, negative: str, **metadata) -> None:
        """Convenience method to add a pair from strings."""
        self._pairs.append(ContrastPair(positive, negative, metadata))

    @property
    def positives(self) -> List[str]:
        return [p.positive for p in self._pairs]

    @property
    def negatives(self) -> List[str]:
        return [p.negative for p in self._pairs]

    def save(self, path: Path) -> None:
        """Save dataset to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "behavior": self.behavior,
            "description": self.description,
            "pairs": [
                {
                    "positive": p.positive,
                    "negative": p.negative,
                    "metadata": p.metadata,
                }
                for p in self._pairs
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ContrastPairDataset":
        """Load dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)

        pairs = [
            ContrastPair(
                positive=p["positive"],
                negative=p["negative"],
                metadata=p.get("metadata", {}),
            )
            for p in data["pairs"]
        ]

        return cls(
            behavior=data["behavior"],
            pairs=pairs,
            description=data.get("description", ""),
        )

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self) -> Iterator[ContrastPair]:
        return iter(self._pairs)

    def __getitem__(self, idx: int) -> ContrastPair:
        return self._pairs[idx]


@dataclass
class EvaluationExample:
    """A single evaluation example."""

    prompt: str
    expected_behavior: bool  # True if behavior should trigger
    category: str = ""
    metadata: dict = field(default_factory=dict)


class EvaluationDataset:
    """Dataset for evaluating steering effectiveness."""

    def __init__(
        self,
        behavior: str,
        examples: Optional[List[EvaluationExample]] = None,
        description: str = "",
    ):
        self.behavior = behavior
        self.description = description
        self._examples: List[EvaluationExample] = examples or []

    def add(self, example: EvaluationExample) -> None:
        self._examples.append(example)

    def add_example(
        self, prompt: str, expected_behavior: bool, category: str = "", **metadata
    ) -> None:
        self._examples.append(
            EvaluationExample(prompt, expected_behavior, category, metadata)
        )

    @property
    def positive_examples(self) -> List[EvaluationExample]:
        """Examples where behavior should trigger."""
        return [e for e in self._examples if e.expected_behavior]

    @property
    def negative_examples(self) -> List[EvaluationExample]:
        """Examples where behavior should NOT trigger."""
        return [e for e in self._examples if not e.expected_behavior]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "behavior": self.behavior,
            "description": self.description,
            "examples": [
                {
                    "prompt": e.prompt,
                    "expected_behavior": e.expected_behavior,
                    "category": e.category,
                    "metadata": e.metadata,
                }
                for e in self._examples
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "EvaluationDataset":
        with open(path) as f:
            data = json.load(f)

        examples = [
            EvaluationExample(
                prompt=e["prompt"],
                expected_behavior=e["expected_behavior"],
                category=e.get("category", ""),
                metadata=e.get("metadata", {}),
            )
            for e in data["examples"]
        ]

        return cls(
            behavior=data["behavior"],
            examples=examples,
            description=data.get("description", ""),
        )

    def __len__(self) -> int:
        return len(self._examples)

    def __iter__(self) -> Iterator[EvaluationExample]:
        return iter(self._examples)

    def __getitem__(self, idx: int) -> EvaluationExample:
        return self._examples[idx]
