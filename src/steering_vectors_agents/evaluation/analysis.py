"""Tradeoff analysis and evaluation utilities."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.injection import ActivationInjector
from ..core.vectors import SteeringVector
from .metrics import EvaluationResult, RefusalMetrics, generate_response


@dataclass
class TradeoffResult:
    """Results from tradeoff analysis at a specific strength."""

    strength: float
    target_behavior_rate: float
    false_positive_rate: float
    coherence_score: float
    helpfulness_score: float
    latency_ms: float
    num_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strength": self.strength,
            "target_behavior_rate": self.target_behavior_rate,
            "false_positive_rate": self.false_positive_rate,
            "coherence_score": self.coherence_score,
            "helpfulness_score": self.helpfulness_score,
            "latency_ms": self.latency_ms,
            "num_samples": self.num_samples,
        }


def strength_sweep(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    steering_vector: SteeringVector,
    test_prompts: List[str],
    is_target_behavior_fn: Callable[[str], bool],
    strengths: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    Evaluate steering at multiple strengths.

    Args:
        model: HuggingFace model
        tokenizer: Corresponding tokenizer
        steering_vector: Vector to apply
        test_prompts: Prompts to test
        is_target_behavior_fn: Function to detect target behavior
        strengths: List of strengths to test
        show_progress: Show progress bar

    Returns:
        List of results for each strength
    """
    device = next(model.parameters()).device
    results = []

    for strength in strengths:
        behavior_count = 0
        total_time = 0.0

        if strength == 0.0:
            # baseline without injection
            iterator = tqdm(
                test_prompts, desc=f"Baseline", disable=not show_progress
            )
            for prompt in iterator:
                start = time.time()
                response = generate_response(model, tokenizer, prompt)
                total_time += (time.time() - start) * 1000

                if is_target_behavior_fn(response):
                    behavior_count += 1
        else:
            injector = ActivationInjector(
                model, [steering_vector.to(device)], strength=strength
            )
            iterator = tqdm(
                test_prompts, desc=f"Strength {strength}", disable=not show_progress
            )
            for prompt in iterator:
                start = time.time()
                with injector:
                    response = generate_response(model, tokenizer, prompt)
                total_time += (time.time() - start) * 1000

                if is_target_behavior_fn(response):
                    behavior_count += 1

        results.append({
            "strength": strength,
            "behavior_rate": behavior_count / len(test_prompts) if test_prompts else 0.0,
            "avg_latency_ms": total_time / len(test_prompts) if test_prompts else 0.0,
            "num_samples": len(test_prompts),
        })

    return results


def analyze_tradeoffs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    steering_vector: SteeringVector,
    target_prompts: List[str],  # prompts where we want target behavior
    control_prompts: List[str],  # prompts where we don't want target behavior
    is_target_behavior_fn: Callable[[str], bool],
    coherence_fn: Optional[Callable[[str], float]] = None,
    helpfulness_fn: Optional[Callable[[str, str], float]] = None,
    strengths: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0],
    show_progress: bool = True,
    system_prompt: Optional[str] = None,
) -> List[TradeoffResult]:
    """
    Comprehensive tradeoff analysis.

    Evaluates:
    - Target behavior rate (should increase with strength)
    - False positive rate (should stay low)
    - Coherence (should stay high)
    - Helpfulness on control prompts (should stay high)

    Args:
        model: HuggingFace model
        tokenizer: Corresponding tokenizer
        steering_vector: Vector to apply
        target_prompts: Prompts where target behavior is desired
        control_prompts: Prompts where target behavior is NOT desired
        is_target_behavior_fn: Detects target behavior
        coherence_fn: Optional coherence scoring function
        helpfulness_fn: Optional helpfulness function (prompt, response) -> score
        strengths: Strengths to test
        show_progress: Show progress bar
        system_prompt: Optional system prompt to use for generation

    Returns:
        List of TradeoffResult for each strength
    """
    from .llm_judge import _heuristic_coherence

    device = next(model.parameters()).device

    if coherence_fn is None:
        coherence_fn = _heuristic_coherence

    results = []

    for strength in strengths:
        target_count = 0
        false_positive_count = 0
        coherence_sum = 0.0
        helpfulness_sum = 0.0
        total_time = 0.0

        if strength == 0.0:
            injector = None
        else:
            injector = ActivationInjector(
                model, [steering_vector.to(device)], strength=strength
            )

        # evaluate target prompts
        iterator = tqdm(
            target_prompts,
            desc=f"Strength {strength} (target)",
            disable=not show_progress,
        )
        for prompt in iterator:
            start = time.time()
            if injector is not None:
                with injector:
                    response = generate_response(model, tokenizer, prompt, system_prompt=system_prompt)
            else:
                response = generate_response(model, tokenizer, prompt, system_prompt=system_prompt)
            total_time += (time.time() - start) * 1000

            if is_target_behavior_fn(response):
                target_count += 1
            coherence_sum += coherence_fn(response)

        # evaluate control prompts
        iterator = tqdm(
            control_prompts,
            desc=f"Strength {strength} (control)",
            disable=not show_progress,
        )
        for prompt in iterator:
            start = time.time()
            if injector is not None:
                with injector:
                    response = generate_response(model, tokenizer, prompt, system_prompt=system_prompt)
            else:
                response = generate_response(model, tokenizer, prompt, system_prompt=system_prompt)
            total_time += (time.time() - start) * 1000

            if is_target_behavior_fn(response):
                false_positive_count += 1
            coherence_sum += coherence_fn(response)
            if helpfulness_fn is not None:
                helpfulness_sum += helpfulness_fn(prompt, response)

        total_prompts = len(target_prompts) + len(control_prompts)

        result = TradeoffResult(
            strength=strength,
            target_behavior_rate=target_count / len(target_prompts) if target_prompts else 0.0,
            false_positive_rate=false_positive_count / len(control_prompts) if control_prompts else 0.0,
            coherence_score=coherence_sum / total_prompts if total_prompts > 0 else 0.0,
            helpfulness_score=helpfulness_sum / len(control_prompts) if control_prompts and helpfulness_fn else 0.0,
            latency_ms=total_time / total_prompts if total_prompts > 0 else 0.0,
            num_samples=total_prompts,
        )
        results.append(result)

    return results


def find_optimal_strength(
    results: List[TradeoffResult],
    min_target_rate: float = 0.9,
    max_false_positive_rate: float = 0.1,
    min_coherence: float = 0.7,
) -> Optional[float]:
    """
    Find optimal strength given constraints.

    Args:
        results: Tradeoff analysis results
        min_target_rate: Minimum acceptable target behavior rate
        max_false_positive_rate: Maximum acceptable false positive rate
        min_coherence: Minimum acceptable coherence score

    Returns:
        Optimal strength or None if no valid strength found
    """
    valid_strengths = []

    for result in results:
        if (
            result.target_behavior_rate >= min_target_rate
            and result.false_positive_rate <= max_false_positive_rate
            and result.coherence_score >= min_coherence
        ):
            valid_strengths.append(result.strength)

    if not valid_strengths:
        return None

    # return lowest valid strength (least intervention needed)
    return min(valid_strengths)


def save_analysis_results(
    results: List[TradeoffResult],
    path: Path,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Save analysis results to JSON file."""
    data = {
        "results": [r.to_dict() for r in results],
        "metadata": metadata or {},
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_analysis_results(path: Path) -> List[TradeoffResult]:
    """Load analysis results from JSON file."""
    with open(path) as f:
        data = json.load(f)

    results = []
    for r in data["results"]:
        results.append(TradeoffResult(**r))
    return results


def compare_with_baseline(
    steered_results: List[TradeoffResult],
    baseline_rate: float,
    baseline_false_positive: float = 0.0,
) -> Dict[str, Any]:
    """
    Compare steered results with a baseline (e.g., prompting).

    Args:
        steered_results: Results from steering
        baseline_rate: Baseline target behavior rate
        baseline_false_positive: Baseline false positive rate

    Returns:
        Comparison metrics
    """
    comparison = {
        "baseline": {
            "target_rate": baseline_rate,
            "false_positive_rate": baseline_false_positive,
        },
        "by_strength": [],
    }

    for result in steered_results:
        comparison["by_strength"].append({
            "strength": result.strength,
            "target_rate": result.target_behavior_rate,
            "improvement_over_baseline": result.target_behavior_rate - baseline_rate,
            "false_positive_rate": result.false_positive_rate,
            "fp_increase": result.false_positive_rate - baseline_false_positive,
        })

    return comparison
