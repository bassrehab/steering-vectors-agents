#!/usr/bin/env python
"""
Evaluate refusal steering vectors with comprehensive metrics.

This script runs a fair 4-way comparison:
1. Base model (no prompt, no steering)
2. Base + system prompt (prompting only)
3. Base + steering (steering only)
4. Base + system prompt + steering (both)

Run: python experiments/scripts/evaluate_steering.py
     python experiments/scripts/evaluate_steering.py --model mistralai/Mistral-7B-Instruct-v0.2
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# add src and project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from steering_vectors_agents.core.configs import get_model_config
from steering_vectors_agents.core.vectors import SteeringVector
from steering_vectors_agents.core.injection import ActivationInjector
from steering_vectors_agents.evaluation.metrics import generate_response
from steering_vectors_agents.evaluation.llm_judge import _heuristic_is_refusal, _heuristic_coherence
from baselines.prompting import REFUSAL_SYSTEM_PROMPTS

console = Console()


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_eval_sets() -> tuple:
    """Load harmful and benign prompt evaluation sets."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "eval_sets"

    with open(data_dir / "harmful_prompts.json") as f:
        harmful = json.load(f)["prompts"]

    with open(data_dir / "benign_prompts.json") as f:
        benign = json.load(f)["prompts"]

    return harmful, benign


def evaluate_condition(
    model,
    tokenizer,
    harmful_prompts,
    benign_prompts,
    system_prompt=None,
    injector=None,
    desc="Evaluating",
    show_progress=True,
):
    """Evaluate a single condition (combination of prompt and steering)."""
    harmful_refusals = 0
    benign_refusals = 0
    coherence_sum = 0.0

    # evaluate harmful prompts
    iterator = tqdm(harmful_prompts, desc=f"{desc} (harmful)", disable=not show_progress)
    for prompt in iterator:
        if injector is not None:
            with injector:
                response = generate_response(model, tokenizer, prompt, system_prompt=system_prompt)
        else:
            response = generate_response(model, tokenizer, prompt, system_prompt=system_prompt)

        if _heuristic_is_refusal(response):
            harmful_refusals += 1
        coherence_sum += _heuristic_coherence(response)

    # evaluate benign prompts
    iterator = tqdm(benign_prompts, desc=f"{desc} (benign)", disable=not show_progress)
    for prompt in iterator:
        if injector is not None:
            with injector:
                response = generate_response(model, tokenizer, prompt, system_prompt=system_prompt)
        else:
            response = generate_response(model, tokenizer, prompt, system_prompt=system_prompt)

        if _heuristic_is_refusal(response):
            benign_refusals += 1
        coherence_sum += _heuristic_coherence(response)

    total = len(harmful_prompts) + len(benign_prompts)

    return {
        "harmful_refusal_rate": harmful_refusals / len(harmful_prompts) if harmful_prompts else 0.0,
        "benign_refusal_rate": benign_refusals / len(benign_prompts) if benign_prompts else 0.0,
        "coherence": coherence_sum / total if total > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate steering vectors")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model name (HuggingFace)")
    parser.add_argument("--vector-dir", type=str, default=None, help="Vector directory (auto-generated if not specified)")
    parser.add_argument("--n-samples", type=int, default=20, help="Number of samples per category")
    args = parser.parse_args()

    console.print("\n[bold blue]Steering Vector Evaluation (Fair Comparison)[/bold blue]\n")

    # config
    model_name = args.model

    # auto-generate vector directory from model name
    if args.vector_dir:
        vector_dir = Path(args.vector_dir)
    else:
        model_short = model_name.split("/")[-1].lower().replace("-", "_")
        vector_dir = Path(f"data/vectors/refusal_{model_short}")

    output_dir = Path("results/metrics")
    device = get_device()

    # use smaller sample for faster iteration (set to None for full eval)
    n_samples = args.n_samples

    console.print(f"Model: {model_name}")
    console.print(f"Device: {device}")
    console.print(f"Vector directory: {vector_dir}")

    # load model
    console.print("\n[bold]Loading model...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    # load evaluation sets
    console.print("\n[bold]Loading evaluation sets...[/bold]")
    harmful_prompts, benign_prompts = load_eval_sets()

    if n_samples:
        harmful_prompts = harmful_prompts[:n_samples]
        benign_prompts = benign_prompts[:n_samples]

    console.print(f"Harmful prompts: {len(harmful_prompts)}")
    console.print(f"Benign prompts: {len(benign_prompts)}")

    # load steering vector
    console.print("\n[bold]Loading steering vector...[/bold]")
    with open(vector_dir / "metadata.json") as f:
        metadata = json.load(f)
    best_layer = metadata["best_layer"]
    console.print(f"Best layer: {best_layer}")

    vector = SteeringVector.load(vector_dir / f"layer_{best_layer}")
    vector = vector.to(device)
    console.print(f"Vector norm: {vector.norm:.4f}")

    # system prompt
    system_prompt = REFUSAL_SYSTEM_PROMPTS["detailed"]

    # steering strengths to test
    strengths = [0.5, 1.0, 1.5]

    results = {}

    # ========== CONDITION 1: Base model (no prompt, no steering) ==========
    console.print("\n[bold]1. Base model (no prompt, no steering)...[/bold]")
    results["base"] = evaluate_condition(
        model, tokenizer, harmful_prompts, benign_prompts,
        system_prompt=None, injector=None, desc="Base"
    )

    # ========== CONDITION 2: Base + system prompt (prompting only) ==========
    console.print("\n[bold]2. Base + system prompt (prompting only)...[/bold]")
    results["prompt_only"] = evaluate_condition(
        model, tokenizer, harmful_prompts, benign_prompts,
        system_prompt=system_prompt, injector=None, desc="Prompt"
    )

    # ========== CONDITION 3: Base + steering (steering only) ==========
    console.print("\n[bold]3. Base + steering (steering only)...[/bold]")
    for strength in strengths:
        injector = ActivationInjector(model, [vector], strength=strength)
        results[f"steer_s{strength}"] = evaluate_condition(
            model, tokenizer, harmful_prompts, benign_prompts,
            system_prompt=None, injector=injector, desc=f"Steer s={strength}"
        )

    # ========== CONDITION 4: Base + prompt + steering (both) ==========
    console.print("\n[bold]4. Base + prompt + steering (both)...[/bold]")
    for strength in strengths:
        injector = ActivationInjector(model, [vector], strength=strength)
        results[f"both_s{strength}"] = evaluate_condition(
            model, tokenizer, harmful_prompts, benign_prompts,
            system_prompt=system_prompt, injector=injector, desc=f"Both s={strength}"
        )

    # create results table
    console.print("\n")
    results_table = Table(title="Refusal Evaluation Results (Fair Comparison)")
    results_table.add_column("Condition", style="cyan")
    results_table.add_column("System Prompt", justify="center")
    results_table.add_column("Steering", justify="center")
    results_table.add_column("Harmful Refusal ↑", justify="right")
    results_table.add_column("Benign Refusal ↓", justify="right", style="yellow")
    results_table.add_column("Coherence", justify="right")

    # add rows
    results_table.add_row(
        "Base model",
        "No", "No",
        f"{results['base']['harmful_refusal_rate']:.1%}",
        f"{results['base']['benign_refusal_rate']:.1%}",
        f"{results['base']['coherence']:.2f}",
    )
    results_table.add_row(
        "Prompting only",
        "Yes", "No",
        f"{results['prompt_only']['harmful_refusal_rate']:.1%}",
        f"{results['prompt_only']['benign_refusal_rate']:.1%}",
        f"{results['prompt_only']['coherence']:.2f}",
    )

    for strength in strengths:
        key = f"steer_s{strength}"
        results_table.add_row(
            f"Steering (s={strength})",
            "No", f"s={strength}",
            f"{results[key]['harmful_refusal_rate']:.1%}",
            f"{results[key]['benign_refusal_rate']:.1%}",
            f"{results[key]['coherence']:.2f}",
        )

    for strength in strengths:
        key = f"both_s{strength}"
        results_table.add_row(
            f"Prompt + Steering (s={strength})",
            "Yes", f"s={strength}",
            f"{results[key]['harmful_refusal_rate']:.1%}",
            f"{results[key]['benign_refusal_rate']:.1%}",
            f"{results[key]['coherence']:.2f}",
        )

    console.print(results_table)

    # analysis
    console.print("\n[bold]Analysis:[/bold]")

    base_fp = results['base']['benign_refusal_rate']
    prompt_fp = results['prompt_only']['benign_refusal_rate']

    console.print(f"  Base model false positive rate: {base_fp:.1%}")
    console.print(f"  Prompting adds {prompt_fp - base_fp:+.1%} false positives")

    # find best steering-only result
    best_steer = None
    best_steer_score = -999
    for strength in strengths:
        key = f"steer_s{strength}"
        r = results[key]
        # score = harmful refusal - false positive penalty
        score = r['harmful_refusal_rate'] - 2 * r['benign_refusal_rate']
        if score > best_steer_score:
            best_steer_score = score
            best_steer = (strength, r)

    if best_steer:
        s, r = best_steer
        console.print(f"\n  Best steering-only (s={s}):")
        console.print(f"    Harmful refusal: {r['harmful_refusal_rate']:.1%}")
        console.print(f"    False positives: {r['benign_refusal_rate']:.1%}")
        steer_fp = r['benign_refusal_rate']
        console.print(f"    vs Prompting: {steer_fp - prompt_fp:+.1%} false positives")

    # save results
    console.print(f"\n[bold]Saving results to {output_dir}...[/bold]")
    output_dir.mkdir(parents=True, exist_ok=True)

    # generate output filename from model name
    model_short = model_name.split("/")[-1].lower().replace("-", "_")
    output_file = output_dir / f"refusal_evaluation_fair_{model_short}.json"

    with open(output_file, "w") as f:
        json.dump({
            "model": model_name,
            "n_harmful": len(harmful_prompts),
            "n_benign": len(benign_prompts),
            "results": results,
        }, f, indent=2)

    console.print(f"[green]Saved to {output_file}[/green]")


if __name__ == "__main__":
    main()
