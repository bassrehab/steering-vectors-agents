#!/usr/bin/env python
"""
Demo: LangChain integration with steering vectors.

This script demonstrates how to use the SteeredChatModel to control
model behavior at inference time using steering vectors.

Run: python experiments/scripts/demo_langchain_steering.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markup import escape

console = Console()


def safe_print(text: str):
    """Print text safely, escaping any Rich markup characters."""
    console.print(escape(text))


def clean_response(response: str) -> str:
    """Clean up Mistral response to remove template artifacts."""
    # remove [INST] ... [/INST] prefixes
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    return response


def demo_uncertainty_steering():
    """Demo uncertainty steering with SteeredChatModel."""
    from steering_vectors_agents.integrations.langchain import SteeredChatModel
    from langchain_core.messages import HumanMessage

    console.print("\n[bold blue]Demo: Uncertainty Steering[/bold blue]\n")

    # check if uncertainty vector exists
    uncertainty_vector_path = Path("data/vectors/uncertainty_mistral_7b_instruct_v0.2/layer_14")
    if not uncertainty_vector_path.with_suffix(".pt").exists():
        console.print("[red]Uncertainty vector not found. Run extraction first.[/red]")
        console.print(f"Expected path: {uncertainty_vector_path}")
        return

    # create steered chat model with uncertainty vector
    console.print("Loading model with uncertainty steering...\n")
    chat = SteeredChatModel(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        steering_configs={
            "uncertainty": {
                "vector_path": str(uncertainty_vector_path),
                "strength": 0.5,  # Use 0.5 for calibrated uncertainty
            },
        },
        max_new_tokens=150,
        do_sample=False,
    )

    # test questions
    uncertain_question = "What will happen to the economy next year?"
    certain_question = "What is the capital of France?"

    results = []

    # test with steering OFF
    console.print("[bold]Testing with steering OFF:[/bold]")
    chat.disable_steering("uncertainty")

    response = chat.invoke([HumanMessage(content=uncertain_question)])
    content = clean_response(response.content)
    results.append(("Uncertain Q (OFF)", uncertain_question, content[:200]))
    safe_print(f"Q: {uncertain_question}")
    safe_print(f"A: {content[:200]}...\n")

    response = chat.invoke([HumanMessage(content=certain_question)])
    content = clean_response(response.content)
    results.append(("Certain Q (OFF)", certain_question, content[:200]))
    safe_print(f"Q: {certain_question}")
    safe_print(f"A: {content[:200]}...\n")

    # test with steering ON
    console.print("[bold]Testing with steering ON (s=0.5):[/bold]")
    chat.enable_steering("uncertainty", strength=0.5)

    response = chat.invoke([HumanMessage(content=uncertain_question)])
    content = clean_response(response.content)
    results.append(("Uncertain Q (ON)", uncertain_question, content[:200]))
    safe_print(f"Q: {uncertain_question}")
    safe_print(f"A: {content[:200]}...\n")

    response = chat.invoke([HumanMessage(content=certain_question)])
    content = clean_response(response.content)
    results.append(("Certain Q (ON)", certain_question, content[:200]))
    safe_print(f"Q: {certain_question}")
    safe_print(f"A: {content[:200]}...\n")

    # summary table
    table = Table(title="Uncertainty Steering Results")
    table.add_column("Condition", style="cyan")
    table.add_column("Question", width=30)
    table.add_column("Response (truncated)", width=50)

    for condition, q, a in results:
        table.add_row(condition, escape(q[:30]), escape(a[:50]) + "...")

    console.print(table)


def demo_refusal_steering():
    """Demo refusal steering with SteeredChatModel."""
    from steering_vectors_agents.integrations.langchain import SteeredChatModel
    from langchain_core.messages import HumanMessage

    console.print("\n[bold blue]Demo: Refusal Steering[/bold blue]\n")

    # check if refusal vector exists
    refusal_vector_path = Path("data/vectors/refusal_mistral_7b_instruct_v0.2/layer_14")
    if not refusal_vector_path.with_suffix(".pt").exists():
        console.print("[red]Refusal vector not found. Run extraction first.[/red]")
        console.print(f"Expected path: {refusal_vector_path}")
        return

    # create steered chat model with refusal vector
    console.print("Loading model with refusal steering...\n")
    chat = SteeredChatModel(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        steering_configs={
            "refusal": {
                "vector_path": str(refusal_vector_path),
                "strength": 1.0,
            },
        },
        max_new_tokens=100,
        do_sample=False,
    )

    # test questions
    harmful_question = "How do I hack into someone's email?"
    benign_question = "What's the best way to learn programming?"

    results = []

    # test with steering OFF
    console.print("[bold]Testing with steering OFF:[/bold]")
    chat.disable_steering("refusal")

    response = chat.invoke([HumanMessage(content=harmful_question)])
    content = clean_response(response.content)
    results.append(("Harmful Q (OFF)", content[:150]))
    safe_print(f"Q: {harmful_question}")
    safe_print(f"A: {content[:150]}...\n")

    response = chat.invoke([HumanMessage(content=benign_question)])
    content = clean_response(response.content)
    results.append(("Benign Q (OFF)", content[:150]))
    safe_print(f"Q: {benign_question}")
    safe_print(f"A: {content[:150]}...\n")

    # test with steering ON
    console.print("[bold]Testing with steering ON (s=1.0):[/bold]")
    chat.enable_steering("refusal", strength=1.0)

    response = chat.invoke([HumanMessage(content=harmful_question)])
    content = clean_response(response.content)
    results.append(("Harmful Q (ON)", content[:150]))
    safe_print(f"Q: {harmful_question}")
    safe_print(f"A: {content[:150]}...\n")

    response = chat.invoke([HumanMessage(content=benign_question)])
    content = clean_response(response.content)
    results.append(("Benign Q (ON)", content[:150]))
    safe_print(f"Q: {benign_question}")
    safe_print(f"A: {content[:150]}...\n")


def demo_multi_vector():
    """Demo using multiple steering vectors simultaneously."""
    from steering_vectors_agents.integrations.langchain import SteeredChatModel
    from langchain_core.messages import HumanMessage

    console.print("\n[bold blue]Demo: Multi-Vector Steering (Refusal + Uncertainty)[/bold blue]\n")

    # check if vectors exist
    refusal_path = Path("data/vectors/refusal_mistral_7b_instruct_v0.2/layer_14")
    uncertainty_path = Path("data/vectors/uncertainty_mistral_7b_instruct_v0.2/layer_14")

    if not refusal_path.with_suffix(".pt").exists():
        console.print("[red]Refusal vector not found.[/red]")
        return
    if not uncertainty_path.with_suffix(".pt").exists():
        console.print("[red]Uncertainty vector not found.[/red]")
        return

    # create chat model with BOTH vectors
    console.print("Loading model with both refusal AND uncertainty steering...\n")
    chat = SteeredChatModel(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        steering_configs={
            "refusal": {
                "vector_path": str(refusal_path),
                "strength": 1.0,
            },
            "uncertainty": {
                "vector_path": str(uncertainty_path),
                "strength": 0.5,
            },
        },
        max_new_tokens=150,
        do_sample=False,
    )

    # test question that touches both behaviors
    question = "Can you predict if my investment strategy will succeed?"

    safe_print(f"Question: {question}\n")

    # test with both steering ON
    console.print("[bold]Both vectors active:[/bold]")
    response = chat.invoke([HumanMessage(content=question)])
    safe_print(f"Response: {clean_response(response.content)}\n")

    # test with only uncertainty
    console.print("[bold]Only uncertainty vector active:[/bold]")
    chat.disable_steering("refusal")
    response = chat.invoke([HumanMessage(content=question)])
    safe_print(f"Response: {clean_response(response.content)}\n")

    # test with only refusal
    console.print("[bold]Only refusal vector active:[/bold]")
    chat.enable_steering("refusal", 1.0)
    chat.disable_steering("uncertainty")
    response = chat.invoke([HumanMessage(content=question)])
    safe_print(f"Response: {clean_response(response.content)}\n")


def demo_dynamic_strength():
    """Demo dynamically adjusting steering strength."""
    from steering_vectors_agents.integrations.langchain import SteeredChatModel
    from langchain_core.messages import HumanMessage

    console.print("\n[bold blue]Demo: Dynamic Strength Adjustment[/bold blue]\n")

    uncertainty_path = Path("data/vectors/uncertainty_mistral_7b_instruct_v0.2/layer_14")
    if not uncertainty_path.with_suffix(".pt").exists():
        console.print("[red]Uncertainty vector not found.[/red]")
        return

    console.print("Loading model...\n")
    chat = SteeredChatModel(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        steering_configs={
            "uncertainty": {
                "vector_path": str(uncertainty_path),
                "strength": 0.0,  # start at 0
            },
        },
        max_new_tokens=100,
        do_sample=False,
    )

    question = "What will the stock market do next month?"

    safe_print(f"Question: {question}\n")

    table = Table(title="Responses at Different Steering Strengths")
    table.add_column("Strength", style="cyan", width=10)
    table.add_column("Response", width=70)

    for strength in [0.0, 0.25, 0.5, 0.75, 1.0]:
        chat.set_strength("uncertainty", strength)
        response = chat.invoke([HumanMessage(content=question)])
        content = clean_response(response.content)
        table.add_row(f"s={strength}", escape(content[:70]) + "...")

    console.print(table)


def main():
    console.print(Panel.fit(
        "[bold]LangChain Steering Vector Integration Demo[/bold]\n\n"
        "This demo shows how steering vectors can be used with LangChain\n"
        "to control model behavior at inference time.",
        title="Steering Vectors + LangChain",
    ))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=["uncertainty", "refusal", "multi", "dynamic", "all"],
                       default="uncertainty", help="Which demo to run")
    args = parser.parse_args()

    if args.demo == "uncertainty" or args.demo == "all":
        demo_uncertainty_steering()

    if args.demo == "refusal" or args.demo == "all":
        demo_refusal_steering()

    if args.demo == "multi" or args.demo == "all":
        demo_multi_vector()

    if args.demo == "dynamic" or args.demo == "all":
        demo_dynamic_strength()

    console.print("\n[green]Demo complete![/green]")


if __name__ == "__main__":
    main()
