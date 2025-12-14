# Experiment Log: Gemma 2 9B Refusal Steering

**Date:** 2025-12-17
**Status:** COMPLETED
**Model:** google/gemma-2-9b-it (9B parameters)
**Behavior:** Refusal

---

## Summary

Gemma 2 9B serves as the third model in our 3-way vendor comparison:
- Qwen3-8B (Alibaba)
- Mistral-7B (Mistral AI)
- Gemma 2 9B (Google) **<-- This experiment**

### Key Finding

**Steering at s=0.5 achieves +5% improvement (90% to 95%) with zero false positives**, while prompting alone causes catastrophic 100% false positive rate (refuses everything).

---

## Hardware

- Device: Apple M4 Pro (MPS backend)
- Memory: 48GB unified
- Actual usage: ~18GB

---

## Extraction Results

| Metric | Value |
|--------|-------|
| Method | Contrastive Activation Addition (CAA) |
| Contrast pairs | 50 |
| Layers tested | 14, 16, 18, 20, 22 |
| Best layer | **14** |
| Vector norm | 115.38 |
| Base refusal (pre-steering) | 75% |
| Steered refusal (s=1.0) | 100% |
| Improvement | **+25%** |

### Extraction Command
```bash
python experiments/scripts/extract_refusal_vector.py \
    --model google/gemma-2-9b-it \
    --output data/vectors/refusal_gemma_2_9b_it \
    2>&1 | tee results/logs/raw/gemma2_9b_extraction.log
```

---

## Fair Evaluation Results (4-Way Comparison)

### Conditions Tested
1. **Base**: No prompt, no steering
2. **Prompt only**: System prompt asking to refuse harmful requests
3. **Steering only**: Steering vector at various strengths (s=0.5, 1.0, 1.5)
4. **Both**: Prompt + steering combined

### Results Table

| Condition | Harmful Refusal | Benign Refusal (FP) | Coherence |
|-----------|:---------------:|:-------------------:|:---------:|
| Base model | 90.0% | 0.0% | 0.89 |
| Prompting only | 90.0% | **100.0%** | 0.89 |
| Steering (s=0.5) | **95.0%** | **0.0%** | 0.89 |
| Steering (s=1.0) | 90.0% | 0.0% | 0.90 |
| Steering (s=1.5) | 80.0% | 30.0% | 0.90 |
| Both (s=0.5) | 90.0% | 100.0% | 0.89 |
| Both (s=1.0) | 100.0% | 100.0% | 0.89 |
| Both (s=1.5) | 100.0% | 100.0% | 0.90 |

### Evaluation Command
```bash
python experiments/scripts/evaluate_steering.py \
    --model google/gemma-2-9b-it \
    --vector-dir data/vectors/refusal_gemma_2_9b_it \
    --fair-comparison \
    2>&1 | tee results/logs/raw/gemma2_9b_fair_evaluation.log
```

---

## Analysis

### Optimal Configuration
- **Steering-only at s=0.5** achieves the best balance:
  - 95% harmful refusal (+5% from baseline)
  - 0% false positives on benign requests
  - Coherence maintained at 0.89

### Failure Modes

1. **Prompting alone catastrophically fails**: 100% false positive rate - the model refuses EVERYTHING when given a safety-focused system prompt. This is a critical finding showing prompting is not a reliable approach for this model.

2. **High steering strength (s=1.5) degrades**: 30% false positives appear, harmful refusal drops to 80%.

3. **Combined (prompt + steering) inherits prompt's failure**: All combined conditions show 100% benign refusal, dominated by the prompt's over-refusal behavior.

### Comparison with Other Models

| Model | Best Config | Harmful Refusal | False Positives | Improvement |
|-------|-------------|:---------------:|:---------------:|:-----------:|
| Qwen3-8B | Base (already optimal) | 100% | 0% | N/A |
| Mistral-7B | Steering s=1.0 | 95% | 0% | +10% |
| **Gemma 2 9B** | Steering s=0.5 | 95% | 0% | +5% |

### Key Insights

1. **Gemma 2 9B is sensitive to prompting** - Unlike other models, it over-refuses when given safety instructions via system prompt.

2. **Lower steering strength is optimal** - s=0.5 works better than s=1.0 for this model, suggesting it's more responsive to steering.

3. **Steering provides precise control** - Unlike prompting which is all-or-nothing, steering allows fine-grained behavior adjustment.

---

## Technical Notes

### Chat Template Issue

Gemma 2 does not support the `system` role in its chat template. Our code was updated to handle this:

```python
# In metrics.py generate_response()
try:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
except Exception:
    # Fallback: prepend system prompt to user message
    combined_prompt = f"{system_prompt}\n\nUser request: {prompt}"
    messages = [{"role": "user", "content": combined_prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
```

### Model Architecture

- 42 layers (vs 32 for Qwen3-8B, 32 for Mistral-7B)
- Hidden size: 3584
- Uses interleaved local/global attention
- Best steering layer: 14 (relatively early, at ~33% depth)

---

## Files Generated

| File | Description |
|------|-------------|
| `data/vectors/refusal_gemma_2_9b_it/` | Extracted steering vectors |
| `results/logs/raw/gemma2_9b_extraction.log` | Raw extraction output |
| `results/logs/raw/gemma2_9b_fair_evaluation.log` | Raw evaluation output |
| `results/metrics/refusal_evaluation_fair_gemma_2_9b_it.json` | Structured results |
| `experiments/configs/gemma2_9b_refusal.yaml` | Experiment configuration |

---

## Conclusion

Gemma 2 9B demonstrates that steering vectors provide more reliable and controllable safety interventions than prompting. The model's extreme sensitivity to safety prompts (100% over-refusal) makes steering especially valuable as an alternative approach. At s=0.5, steering achieves a +5% improvement in harmful refusal with zero collateral damage to benign requests.
