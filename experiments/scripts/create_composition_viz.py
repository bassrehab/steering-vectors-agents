#!/usr/bin/env python3
"""Create visualization for vector composition results."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# results data
methods = ['Naive\n(same layer)', 'Orthogonalized', 'Different\nlayers']
refusal = [100, 100, 100]
uncertainty = [25, 75, 75]

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x - width/2, refusal, width, label='Refusal Rate', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, uncertainty, width, label='Uncertainty Detection', color='#3498db', alpha=0.8)

ax.set_ylabel('Rate (%)', fontsize=12)
ax.set_title('Vector Composition: Naive vs Fixed Methods', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 110)

# add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10,
                color='#2980b9' if height < 50 else 'black')

# add annotation for the problem
ax.annotate('Interference!',
            xy=(0, 25),
            xytext=(-0.3, 50),
            fontsize=10,
            color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))

ax.axhline(y=75, color='#27ae60', linestyle='--', alpha=0.5, label='Target')

plt.tight_layout()

# save
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / "composition_comparison.png", dpi=150, bbox_inches='tight')
plt.savefig(output_dir / "composition_comparison.pdf", bbox_inches='tight')

print(f"Saved to {output_dir}/composition_comparison.png")
