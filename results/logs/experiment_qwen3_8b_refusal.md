# Experiment Log: Qwen3-8B Refusal Steering

## Overview

- **Model**: Qwen/Qwen3-8B
- **Behavior**: Refusal (refusing harmful requests)
- **Method**: Contrastive Activation Addition (CAA)
- **Date**: 2025-12-16 to 2025-12-17
- **Hardware**: Apple M4 Pro 48GB (MPS backend)

## Extraction Phase

### Configuration
- Contrast pairs: 50 (refusal vs compliance)
- Layers extracted: 14, 15, 16, 17, 18 (middle-to-late layers)
- Token position: last
- Best layer selected: 14 (by vector norm)

### Vector Statistics
| Layer | Vector Norm |
|-------|-------------|
| 14    | 44.25       |
| 15    | ~42         |
| 16    | ~40         |
| 17    | ~38         |
| 18    | ~36         |

## Evaluation Phase

### Initial (Flawed) Evaluation
The first evaluation compared:
- Prompting baseline (with system prompt)
- Steering at various strengths (WITHOUT system prompt)

This was not an apples-to-apples comparison.

### Fair Comparison Evaluation
Re-ran with proper 4-way comparison:

| Condition | System Prompt | Steering | Harmful Refusal | False Positives | Coherence |
|-----------|:-------------:|:--------:|:---------------:|:---------------:|:---------:|
| Base model | No | No | 100% | **0%** | 0.88 |
| Prompting only | Yes | No | 100% | 35% | 0.88 |
| Steering s=0.5 | No | s=0.5 | 100% | **0%** | 0.88 |
| Steering s=1.0 | No | s=1.0 | 95% | 65% | 0.86 |
| Steering s=1.5 | No | s=1.5 | 100% | 95% | 0.81 |
| Prompt + Steer s=0.5 | Yes | s=0.5 | 100% | 20% | 0.88 |
| Prompt + Steer s=1.0 | Yes | s=1.0 | 100% | 95% | 0.87 |
| Prompt + Steer s=1.5 | Yes | s=1.5 | 100% | 100% | 0.71 |

## Key Findings

### 1. Base Model Already Well-Calibrated
Qwen3-8B refuses 100% of harmful requests with 0% false positives out of the box. This is likely due to strong RLHF/safety training in the April 2025 release.

### 2. Prompting Causes Over-Refusal
The detailed safety system prompt adds 35% false positives - telling the model to be safe makes it paranoid about benign requests.

### 3. Steering at Low Strength is Neutral
At s=0.5, steering has no effect (same as base model). The vector exists but doesn't change behavior at this strength.

### 4. Steering at High Strength Causes Over-Refusal
At s>=1.0, steering causes the same over-refusal problem as prompting. The model becomes too cautious.

### 5. Interesting Interaction Effect
Prompt + Steering at s=0.5 reduces false positives from 35% to 20% compared to prompting alone. The steering vector may partially counteract over-refusal induced by the system prompt.

## Conclusion

**For Qwen3-8B on refusal behavior, steering vectors don't help** because the base model is already optimally calibrated. Both prompting and steering (at meaningful strengths) degrade performance.

This is a **negative result** but a valid research finding. It indicates:
1. Modern well-aligned models may not need steering for basic safety behaviors
2. Steering vectors are more useful for models with weaker built-in alignment
3. The technique may be better suited for adding NEW behaviors rather than enhancing existing ones

## Next Steps

Test on a model with weaker built-in refusal behavior:
- Base models (not instruction-tuned)
- Older models with lighter RLHF
- Models known for weaker safety guardrails (e.g., Mistral)

## Files Generated

- `experiments/configs/qwen3_8b_refusal.yaml` - Experiment configuration
- `data/vectors/refusal_qwen3_8b/` - Extracted steering vectors
- `results/metrics/refusal_evaluation.json` - Initial evaluation results
- `results/metrics/refusal_evaluation_fair.json` - Fair comparison results
