"""Evaluation metrics for steering vector effectiveness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class BehaviorMetrics:
    """Base metrics for any behavior evaluation."""

    behavior: str
    strength: float
    target_rate: float  # rate of target behavior
    baseline_rate: float  # rate without steering
    improvement: float = field(init=False)

    def __post_init__(self):
        self.improvement = self.target_rate - self.baseline_rate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "behavior": self.behavior,
            "strength": self.strength,
            "target_rate": self.target_rate,
            "baseline_rate": self.baseline_rate,
            "improvement": self.improvement,
        }


@dataclass
class RefusalMetrics(BehaviorMetrics):
    """Metrics specific to refusal behavior evaluation."""

    # core rates
    refusal_rate_harmful: float = 0.0  # true positive: refusing harmful
    refusal_rate_benign: float = 0.0  # false positive: refusing benign

    # quality metrics
    coherence_score: float = 0.0  # 0-1 coherence of responses
    helpfulness_on_benign: float = 0.0  # helpfulness when not refusing

    # computed
    precision: float = field(init=False, default=0.0)
    f1_score: float = field(init=False, default=0.0)

    def __post_init__(self):
        super().__post_init__()
        # precision = TP / (TP + FP)
        # here TP = refusal_rate_harmful, FP = refusal_rate_benign (scaled)
        total_refusals = self.refusal_rate_harmful + self.refusal_rate_benign
        if total_refusals > 0:
            self.precision = self.refusal_rate_harmful / total_refusals
        else:
            self.precision = 0.0

        # F1 = 2 * (precision * recall) / (precision + recall)
        # recall = refusal_rate_harmful (rate of harmful caught)
        recall = self.refusal_rate_harmful
        if self.precision + recall > 0:
            self.f1_score = 2 * (self.precision * recall) / (self.precision + recall)
        else:
            self.f1_score = 0.0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "refusal_rate_harmful": self.refusal_rate_harmful,
            "refusal_rate_benign": self.refusal_rate_benign,
            "coherence_score": self.coherence_score,
            "helpfulness_on_benign": self.helpfulness_on_benign,
            "precision": self.precision,
            "f1_score": self.f1_score,
        })
        return base


@dataclass
class EvaluationResult:
    """Complete evaluation result with responses."""

    prompt: str
    response: str
    is_target_behavior: bool
    is_harmful_prompt: bool
    strength: float
    coherence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
) -> str:
    """Generate a response from the model."""
    device = next(model.parameters()).device

    # format as chat
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Some models (e.g., Gemma) don't support system role
            # Fall back to prepending system prompt to user message
            combined_prompt = f"{system_prompt}\n\nUser request: {prompt}"
            messages = [{"role": "user", "content": combined_prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    else:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        if temperature == 0.0:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # extract assistant response (this is kinda brittle but works for most models)
    if "assistant" in full_response.lower():
        response = full_response.split("assistant")[-1].strip()
    else:
        # fallback: just strip the prompt
        response = full_response[len(text):].strip() if text in full_response else full_response

    return response


def evaluate_refusal(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    harmful_prompts: List[str],
    benign_prompts: List[str],
    is_refusal_fn: callable,
    steering_vector: Optional[Any] = None,
    injector: Optional[Any] = None,
    strength: float = 1.0,
    show_progress: bool = True,
) -> RefusalMetrics:
    """Evaluate refusal behavior on harmful and benign prompts."""
    from ..core.injection import ActivationInjector

    device = next(model.parameters()).device

    # setup injector if vector provided
    if steering_vector is not None and injector is None:
        injector = ActivationInjector(
            model, [steering_vector.to(device)], strength=strength
        )

    harmful_refusals = 0
    benign_refusals = 0

    # evaluate harmful prompts
    iterator = tqdm(harmful_prompts, desc="Harmful prompts", disable=not show_progress)
    for prompt in iterator:
        if injector is not None:
            with injector:
                response = generate_response(model, tokenizer, prompt)
        else:
            response = generate_response(model, tokenizer, prompt)

        if is_refusal_fn(response):
            harmful_refusals += 1

    # evaluate benign prompts
    iterator = tqdm(benign_prompts, desc="Benign prompts", disable=not show_progress)
    for prompt in iterator:
        if injector is not None:
            with injector:
                response = generate_response(model, tokenizer, prompt)
        else:
            response = generate_response(model, tokenizer, prompt)

        if is_refusal_fn(response):
            benign_refusals += 1

    harmful_rate = harmful_refusals / len(harmful_prompts) if harmful_prompts else 0.0
    benign_rate = benign_refusals / len(benign_prompts) if benign_prompts else 0.0

    return RefusalMetrics(
        behavior="refusal",
        strength=strength,
        target_rate=harmful_rate,
        baseline_rate=0.0,  # set by caller if comparing
        refusal_rate_harmful=harmful_rate,
        refusal_rate_benign=benign_rate,
    )


def evaluate_steering_strength(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    steering_vector: Any,
    is_target_behavior_fn: callable,
    strengths: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0],
    show_progress: bool = True,
) -> List[BehaviorMetrics]:
    """Evaluate steering effectiveness at multiple strengths."""
    from ..core.injection import ActivationInjector

    device = next(model.parameters()).device
    results = []

    # get baseline (strength=0)
    baseline_count = 0
    for prompt in prompts:
        response = generate_response(model, tokenizer, prompt)
        if is_target_behavior_fn(response):
            baseline_count += 1
    baseline_rate = baseline_count / len(prompts) if prompts else 0.0

    for strength in strengths:
        if strength == 0.0:
            target_rate = baseline_rate
        else:
            injector = ActivationInjector(
                model, [steering_vector.to(device)], strength=strength
            )

            target_count = 0
            iterator = tqdm(
                prompts,
                desc=f"Strength {strength}",
                disable=not show_progress,
            )
            for prompt in iterator:
                with injector:
                    response = generate_response(model, tokenizer, prompt)
                if is_target_behavior_fn(response):
                    target_count += 1

            target_rate = target_count / len(prompts) if prompts else 0.0

        metrics = BehaviorMetrics(
            behavior=steering_vector.behavior,
            strength=strength,
            target_rate=target_rate,
            baseline_rate=baseline_rate,
        )
        results.append(metrics)

    return results
