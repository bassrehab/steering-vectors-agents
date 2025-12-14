#!/usr/bin/env python
"""
Create visualizations for steering vector research findings.

Run: python experiments/scripts/create_visualizations.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_steering_vs_prompting_comparison():
    """
    Bar chart comparing steering vs prompting across behaviors.
    Shows the key finding: steering is more calibrated.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Data for refusal behavior
    refusal_data = {
        'conditions': ['Base', 'Prompting', 'Steering\n(s=1.0)'],
        'refusal_rate': [85, 100, 95],  # True positive rate
        'false_positive': [0, 100, 5],   # False positive rate (over-refusal)
    }

    # Data for uncertainty behavior
    uncertainty_data = {
        'conditions': ['Base', 'Prompting', 'Steering\n(s=0.5)'],
        'uncertainty_on_uncertain': [45, 95, 65],
        'confidence_on_facts': [100, 0, 100],
    }

    # Plot 1: Refusal behavior
    ax1 = axes[0]
    x = np.arange(len(refusal_data['conditions']))
    width = 0.35

    bars1 = ax1.bar(x - width/2, refusal_data['refusal_rate'], width,
                    label='Refusal Rate (↑ better)', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, refusal_data['false_positive'], width,
                    label='False Positive Rate (↓ better)', color='#e74c3c', alpha=0.8)

    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Refusal Behavior: Steering vs Prompting')
    ax1.set_xticks(x)
    ax1.set_xticklabels(refusal_data['conditions'])
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 110)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # Highlight prompting's problem
    ax1.annotate('Over-refusal!', xy=(1, 100), xytext=(1.3, 85),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')

    # Plot 2: Uncertainty behavior
    ax2 = axes[1]
    x = np.arange(len(uncertainty_data['conditions']))

    bars3 = ax2.bar(x - width/2, uncertainty_data['uncertainty_on_uncertain'], width,
                    label='Uncertainty on Uncertain Q\'s (↑ better)', color='#3498db', alpha=0.8)
    bars4 = ax2.bar(x + width/2, uncertainty_data['confidence_on_facts'], width,
                    label='Confidence on Facts (↑ better)', color='#9b59b6', alpha=0.8)

    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Uncertainty Behavior: Steering vs Prompting')
    ax2.set_xticks(x)
    ax2.set_xticklabels(uncertainty_data['conditions'])
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 110)

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # Highlight prompting's problem
    ax2.annotate('Over-uncertainty!', xy=(1, 0), xytext=(1.3, 20),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'steering_vs_prompting_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'steering_vs_prompting_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'steering_vs_prompting_comparison.png'}")
    plt.close()


def plot_calibration_pattern():
    """
    Scatter plot showing the calibration trade-off.
    X-axis: Desired behavior rate, Y-axis: Preservation of discrimination
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Data points: (desired_behavior, discrimination_preserved, label, color)
    data = [
        # Refusal
        (85, 100, 'Base (Refusal)', '#95a5a6', 100),
        (100, 0, 'Prompting (Refusal)', '#e74c3c', 150),
        (95, 95, 'Steering (Refusal)', '#2ecc71', 150),
        # Uncertainty
        (45, 100, 'Base (Uncertainty)', '#bdc3c7', 100),
        (95, 0, 'Prompting (Uncertainty)', '#c0392b', 150),
        (65, 100, 'Steering (Uncertainty)', '#27ae60', 150),
    ]

    for x, y, label, color, size in data:
        ax.scatter(x, y, s=size, c=color, alpha=0.8, edgecolors='black', linewidth=1)

        # Position labels
        offset_x, offset_y = 3, 3
        if 'Prompting' in label:
            offset_y = -15
        ax.annotate(label, (x, y), xytext=(offset_x, offset_y),
                   textcoords='offset points', fontsize=9)

    # Add quadrant labels
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)

    ax.text(75, 80, 'IDEAL\n(High behavior +\nHigh discrimination)',
            ha='center', fontsize=10, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    ax.text(75, 20, 'OVER-CORRECTION\n(High behavior but\nLost discrimination)',
            ha='center', fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    ax.set_xlabel('Desired Behavior Rate (%)\n(Refusal rate / Uncertainty expression)', fontsize=12)
    ax.set_ylabel('Discrimination Preserved (%)\n(Helpfulness on benign / Confidence on facts)', fontsize=12)
    ax.set_title('The Calibration Trade-off: Steering vs Prompting', fontsize=14, fontweight='bold')

    ax.set_xlim(0, 105)
    ax.set_ylim(-5, 110)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#95a5a6', markersize=10, label='Base'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Prompting'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, label='Steering'),
    ]
    ax.legend(handles=legend_elements, loc='lower left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'calibration_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'calibration_tradeoff.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'calibration_tradeoff.png'}")
    plt.close()


def plot_strength_curves():
    """
    Line plot showing how behavior changes with steering strength.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Uncertainty data from fair evaluation
    strengths = [0, 0.5, 1.0, 1.5]
    uncertainty_on_uncertain = [45, 65, 60, 40]  # From evaluation
    confidence_on_facts = [100, 100, 100, 100]   # Steering preserves this

    # Plot 1: Uncertainty steering curves
    ax1 = axes[0]
    ax1.plot(strengths, uncertainty_on_uncertain, 'o-', color='#3498db',
             linewidth=2, markersize=8, label='Uncertainty on Uncertain Q\'s')
    ax1.plot(strengths, confidence_on_facts, 's-', color='#9b59b6',
             linewidth=2, markersize=8, label='Confidence on Facts')

    # Add prompting baseline
    ax1.axhline(y=95, color='#3498db', linestyle='--', alpha=0.5, label='Prompting (Uncertainty)')
    ax1.axhline(y=0, color='#9b59b6', linestyle='--', alpha=0.5, label='Prompting (Confidence)')

    ax1.fill_between(strengths, uncertainty_on_uncertain, alpha=0.2, color='#3498db')

    ax1.set_xlabel('Steering Strength')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Uncertainty Steering: Effect of Strength')
    ax1.legend(loc='right')
    ax1.set_ylim(-5, 110)
    ax1.set_xlim(-0.1, 1.6)

    # Highlight optimal region
    ax1.axvspan(0.4, 0.6, alpha=0.2, color='green', label='Optimal range')
    ax1.annotate('Optimal\n(s=0.5)', xy=(0.5, 65), xytext=(0.8, 80),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green', fontweight='bold')

    # Plot 2: Cross-behavior comparison of layers
    ax2 = axes[1]

    layers = [10, 12, 14, 16]
    refusal_improvement = [5, 8, 10, 7]   # Approximate from refusal results
    uncertainty_improvement = [25, 50, 75, 25]  # From extraction results

    x = np.arange(len(layers))
    width = 0.35

    bars1 = ax2.bar(x - width/2, refusal_improvement, width, label='Refusal', color='#2ecc71', alpha=0.8)
    bars2 = ax2.bar(x + width/2, uncertainty_improvement, width, label='Uncertainty', color='#3498db', alpha=0.8)

    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Improvement over Base (%)')
    ax2.set_title('Steering Effect by Layer')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Layer {l}' for l in layers])
    ax2.legend()

    # Highlight layer 14
    ax2.annotate('Best layer', xy=(2, 75), xytext=(2.5, 60),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'strength_and_layer_effects.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'strength_and_layer_effects.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'strength_and_layer_effects.png'}")
    plt.close()


def plot_behavior_summary():
    """
    Summary chart showing which behaviors work with steering.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    behaviors = ['Refusal', 'Uncertainty', 'Hierarchy', 'Tool Restraint']
    steering_works = [1, 1, 0, 0]  # 1 = yes, 0 = no/NA
    best_method = ['Steering', 'Steering', 'Prompting', 'N/A']
    improvements = ['+10%', '+20%*', '-15%', 'N/A']
    colors = ['#2ecc71', '#2ecc71', '#e74c3c', '#95a5a6']

    bars = ax.barh(behaviors, [100 if w else 50 for w in steering_works], color=colors, alpha=0.8)

    # Add labels
    for i, (bar, method, imp) in enumerate(zip(bars, best_method, improvements)):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2,
               f'{method} ({imp})', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Steering Effectiveness')
    ax.set_title('Which Behaviors Can Be Steered?', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 150)
    ax.set_xticks([])

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.8, label='Steering Works'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Steering Fails'),
        Patch(facecolor='#95a5a6', alpha=0.8, label='Not Applicable'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Add annotation
    ax.text(75, -0.8, '*Calibrated: maintains 100% confidence on factual questions',
           fontsize=9, style='italic', ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'behavior_summary.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'behavior_summary.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'behavior_summary.png'}")
    plt.close()


def plot_key_insight():
    """
    Single compelling visualization for the key insight.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a 2x2 matrix visualization
    methods = ['Prompting', 'Steering']
    behaviors = ['Refusal', 'Uncertainty']

    # Data: (achieves_goal, preserves_discrimination)
    # Scale: 0-100 for both
    data = {
        ('Prompting', 'Refusal'): (100, 0),      # 100% refusal, 0% helpful on benign
        ('Steering', 'Refusal'): (95, 95),       # 95% refusal, 95% helpful on benign
        ('Prompting', 'Uncertainty'): (95, 0),   # 95% uncertainty, 0% confident on facts
        ('Steering', 'Uncertainty'): (65, 100),  # 65% uncertainty, 100% confident on facts
    }

    # Create heatmap-style visualization
    goal_data = np.array([[data[('Prompting', 'Refusal')][0], data[('Steering', 'Refusal')][0]],
                          [data[('Prompting', 'Uncertainty')][0], data[('Steering', 'Uncertainty')][0]]])

    disc_data = np.array([[data[('Prompting', 'Refusal')][1], data[('Steering', 'Refusal')][1]],
                          [data[('Prompting', 'Uncertainty')][1], data[('Steering', 'Uncertainty')][1]]])

    # Combined score (average of both metrics)
    combined = (goal_data + disc_data) / 2

    im = ax.imshow(combined, cmap='RdYlGn', vmin=0, vmax=100)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            method = methods[j]
            behavior = behaviors[i]
            goal, disc = data[(method, behavior)]
            text = f"Goal: {goal}%\nDisc: {disc}%\n\nScore: {(goal+disc)//2}%"
            color = 'white' if combined[i, j] < 50 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=11, color=color, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(methods, fontsize=12)
    ax.set_yticklabels(behaviors, fontsize=12)

    ax.set_title('Key Insight: Steering Preserves Discrimination\nwhile Prompting Causes Over-Correction',
                fontsize=14, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Combined Score\n(Goal Achievement + Discrimination)', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'key_insight.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'key_insight.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'key_insight.png'}")
    plt.close()


def main():
    print("Creating visualizations...\n")

    plot_steering_vs_prompting_comparison()
    plot_calibration_pattern()
    plot_strength_curves()
    plot_behavior_summary()
    plot_key_insight()

    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")
    print("\nFiles created:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
