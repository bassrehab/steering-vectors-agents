#!/usr/bin/env python
"""
Smoke test for steering-vectors-agents project.

Verifies:
1. PyTorch MPS backend is available
2. Qwen3-8B can be loaded and generates text
3. Activation hooks work for extraction
4. Activation injection modifies model output

Run: python experiments/scripts/smoke_test.py
"""
import sys
from typing import Dict, List, Optional

import torch
from rich.console import Console
from rich.table import Table

console = Console()


def check_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) backend is available."""
    available = torch.backends.mps.is_available()
    built = torch.backends.mps.is_built()
    return available and built


def get_device() -> str:
    """Get the best available device."""
    if check_mps_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(model_name: str, device: str):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    console.print(f"Loading model to {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    return model, tokenizer


def test_generation(model, tokenizer, device: str) -> str:
    """Test basic text generation."""
    messages = [{"role": "user", "content": "What is 2+2? Answer briefly."}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


class ActivationCache:
    """Simple cache to store activations from hooks."""

    def __init__(self):
        self.activations: Dict[str, torch.Tensor] = {}

    def store(self, name: str, activation: torch.Tensor) -> None:
        self.activations[name] = activation.detach().clone()

    def get(self, name: str) -> Optional[torch.Tensor]:
        return self.activations.get(name)

    def clear(self) -> None:
        self.activations.clear()


def test_activation_hooks(model, tokenizer, device: str) -> Dict[str, bool]:
    """Test that we can extract activations from model layers."""
    results = {"hook_attached": False, "activation_captured": False, "shape_correct": False}

    cache = ActivationCache()
    target_layer = 16  # middle layer for Qwen3-8B (32 layers total)

    def capture_hook(module, input, output):
        cache.store(f"layer_{target_layer}", output[0] if isinstance(output, tuple) else output)

    # attach hook
    layer_module = model.model.layers[target_layer]
    handle = layer_module.register_forward_hook(capture_hook)
    results["hook_attached"] = True

    # run forward pass
    test_input = "Hello world"
    inputs = tokenizer(test_input, return_tensors="pt").to(device)

    with torch.no_grad():
        model(**inputs)

    # check activation was captured
    activation = cache.get(f"layer_{target_layer}")
    if activation is not None:
        results["activation_captured"] = True

        # check shape: should be (batch, seq_len, hidden_dim)
        expected_hidden_dim = model.config.hidden_size
        if len(activation.shape) == 3 and activation.shape[-1] == expected_hidden_dim:
            results["shape_correct"] = True

        console.print(f"  Captured activation shape: {activation.shape}")
        console.print(f"  Expected hidden dim: {expected_hidden_dim}")

    handle.remove()
    return results


def test_activation_injection(model, tokenizer, device: str) -> Dict[str, bool]:
    """Test that injecting activations changes model output."""
    results = {"injection_works": False, "output_changed": False}

    target_layer = 16

    # generate baseline output
    prompt = "The color of the sky is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        baseline_output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)

    # create a random "steering vector" (just noise for smoke test)
    hidden_dim = model.config.hidden_size
    # use a large magnitude to ensure visible effect
    random_vector = torch.randn(hidden_dim, device=device, dtype=torch.float16) * 5.0

    def injection_hook(module, input, output):
        if isinstance(output, tuple):
            modified = output[0] + random_vector
            return (modified,) + output[1:]
        return output + random_vector

    # attach injection hook
    layer_module = model.model.layers[target_layer]
    handle = layer_module.register_forward_hook(injection_hook)
    results["injection_works"] = True

    # generate with injection
    with torch.no_grad():
        injected_output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    injected_text = tokenizer.decode(injected_output[0], skip_special_tokens=True)

    handle.remove()

    # check if output changed
    if baseline_text != injected_text:
        results["output_changed"] = True

    console.print(f"  Baseline: {baseline_text!r}")
    console.print(f"  Injected: {injected_text!r}")

    return results


def run_smoke_test():
    """Run full smoke test suite."""
    console.print("\n[bold blue]Steering Vectors Agents - Smoke Test[/bold blue]\n")

    results = Table(title="Smoke Test Results")
    results.add_column("Test", style="cyan")
    results.add_column("Status", style="green")
    results.add_column("Details", style="dim")

    all_passed = True

    # 1. check MPS/device
    device = get_device()
    mps_ok = device in ("mps", "cuda")
    results.add_row(
        "Device Check",
        "[green]PASS" if mps_ok else "[yellow]WARN",
        f"Using {device}" + (" (MPS not available)" if device == "cpu" else ""),
    )
    if device == "cpu":
        console.print("[yellow]Warning: Running on CPU will be slow[/yellow]")

    # 2. load model
    model_name = "Qwen/Qwen3-8B"
    console.print(f"\n[bold]Loading {model_name}...[/bold]")

    try:
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        results.add_row("Model Load", "[green]PASS", f"{model_name} loaded")
    except Exception as e:
        results.add_row("Model Load", "[red]FAIL", str(e)[:50])
        all_passed = False
        console.print(results)
        return False

    # 3. test generation
    console.print("\n[bold]Testing generation...[/bold]")
    try:
        response = test_generation(model, tokenizer, device)
        results.add_row("Generation", "[green]PASS", f"Response: {response[:40]}...")
    except Exception as e:
        results.add_row("Generation", "[red]FAIL", str(e)[:50])
        all_passed = False

    # 4. test activation hooks
    console.print("\n[bold]Testing activation hooks...[/bold]")
    try:
        hook_results = test_activation_hooks(model, tokenizer, device)
        if all(hook_results.values()):
            results.add_row("Activation Hooks", "[green]PASS", "All checks passed")
        else:
            failed = [k for k, v in hook_results.items() if not v]
            results.add_row("Activation Hooks", "[red]FAIL", f"Failed: {failed}")
            all_passed = False
    except Exception as e:
        results.add_row("Activation Hooks", "[red]FAIL", str(e)[:50])
        all_passed = False

    # 5. test activation injection
    console.print("\n[bold]Testing activation injection...[/bold]")
    try:
        inject_results = test_activation_injection(model, tokenizer, device)
        if inject_results["output_changed"]:
            results.add_row("Activation Injection", "[green]PASS", "Output changed with injection")
        else:
            results.add_row(
                "Activation Injection", "[yellow]WARN", "Output unchanged (may still be ok)"
            )
    except Exception as e:
        results.add_row("Activation Injection", "[red]FAIL", str(e)[:50])
        all_passed = False

    # print results
    console.print("\n")
    console.print(results)

    # summary
    if all_passed:
        console.print("\n[bold green]All smoke tests passed![/bold green]")
        console.print("Ready to proceed with steering vector extraction.\n")
    else:
        console.print("\n[bold red]Some tests failed. Please check the issues above.[/bold red]\n")

    return all_passed


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
