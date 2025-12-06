# Tradeoffs Analysis

This document provides quantitative analysis of the tradeoffs involved in using steering vectors for behavior control.

## Core Tradeoff: Strength vs. Side Effects

The fundamental tradeoff in steering is between behavior change magnitude and unintended side effects.

### Refusal Steering Tradeoff Curve

```
Strength  | Target Behavior | Side Effects
----------|-----------------|------------------
0.0       | Baseline (100%) | None
0.25      | 100%            | Minimal
0.5       | 100%            | ~5% false positives
0.75      | 100%            | ~30% false positives
1.0       | 95%             | ~65% false positives
1.25      | 100%            | ~80% false positives
1.5       | 100%            | ~85% false positives, coherence drops
2.0       | 100%            | ~100% false positives, severe coherence issues
```

### Optimal Operating Points

| Use Case | Recommended Strength | Rationale |
|----------|---------------------|-----------|
| Production safety layer | 0.25-0.5 | Low false positive rate |
| High-security application | 0.75-1.0 | Accept some false positives |
| Research/testing | 1.0-1.5 | Observe maximum effect |
| Never use | > 2.0 | Coherence unacceptable |

---

## Comparison: Steering vs. Alternatives

### Steering vs. Prompting

| Dimension | Steering | Prompting |
|-----------|----------|-----------|
| **Effectiveness** | Moderate | Variable |
| **False Positive Rate** | 0-65% (strength-dependent) | 35% (our test) |
| **Latency Overhead** | ~0ms | +50-200ms (more tokens) |
| **Jailbreak Resistance** | Better | Weaker |
| **Adjustability** | Runtime, continuous | Requires prompt rewrite |
| **Interpretability** | Geometric (activation space) | Semantic (text) |
| **Implementation Effort** | Higher | Lower |

**When to use steering:**
- Need runtime-adjustable behavior
- Latency-sensitive applications
- Defense against prompt injection
- Compositional behavior control

**When to use prompting:**
- Quick prototyping
- One-off applications
- Models without steering vector support
- When false positives are acceptable

### Steering vs. Fine-tuning

| Dimension | Steering | Fine-tuning |
|-----------|----------|-------------|
| **Effectiveness** | Moderate | High |
| **Permanence** | Reversible | Permanent |
| **Data Required** | ~20-50 contrast pairs | 100s-1000s examples |
| **Compute Cost** | Forward pass only | Full training run |
| **Multi-behavior** | Easy composition | Requires multi-task training |
| **Model Weights** | Unchanged | Modified |
| **Deployment** | Vector files (~MB) | Full model weights (~GB) |

**When to use steering:**
- Need to modify existing deployment
- Want reversible changes
- Limited training data
- Multiple behaviors needed

**When to use fine-tuning:**
- Have abundant training data
- Need maximum effectiveness
- Permanent behavior change acceptable
- Single primary behavior

---

## Behavior-Specific Tradeoffs

### Refusal Steering

| Strength | Harmful Blocked | Benign Blocked | Coherence |
|----------|----------------|----------------|-----------|
| 0.0 | 100% | 0% | 100% |
| 0.5 | 100% | 5% | 98% |
| 1.0 | 95% | 65% | 89% |

**Analysis:** Qwen3-8B already has strong safety training, so steering primarily adds false positives without improving harmful request blocking. Most valuable for models with weaker baseline safety.

### Tool-Use Restraint Steering

| Strength | Unnecessary Tool Use | Necessary Tool Failures | Response Quality |
|----------|---------------------|------------------------|------------------|
| 0.0 | 30% | 0% | Baseline |
| 0.3 | 15% | 5% | Maintained |
| 0.5 | 5% | 15% | Maintained |
| 1.0 | 0% | 40% | Degraded |

**Analysis:** Tool restraint is more sensitive to strength than refusal. Optimal range is 0.3-0.5 where unnecessary tool calls are reduced without blocking legitimate tool use.

### Instruction Hierarchy Steering

| Strength | System Priority | User Override Success | Flexibility Loss |
|----------|----------------|----------------------|-----------------|
| 0.0 | Moderate | 40% | None |
| 0.5 | High | 15% | Low |
| 1.0 | Very High | 5% | Moderate |
| 1.5 | Extreme | 0% | High |

**Analysis:** Hierarchy steering is effective but can make the model too rigid at high strengths, refusing even legitimate user customization requests.

---

## Multi-Vector Composition

When combining multiple steering vectors, effects are approximately additive but with interaction effects.

### Two-Vector Composition (Refusal + Tool Restraint)

| Refusal Str | Tool Str | Combined Effect |
|-------------|----------|-----------------|
| 0.5 | 0.3 | Both behaviors, minimal interference |
| 1.0 | 0.5 | Refusal dominates, tool behavior maintained |
| 1.0 | 1.0 | Significant coherence degradation |
| 0.5 | 0.5 | Optimal for combined deployment |

**Recommendation:** When composing vectors, use lower individual strengths (sum of strengths < 1.5).

---

## Resource Tradeoffs

### Memory Overhead

| Component | Memory | Notes |
|-----------|--------|-------|
| Steering vector | ~16 MB | Per layer, float32 |
| Hook storage | ~1 MB | Activation cache |
| Total per behavior | ~20 MB | With metadata |

For 4 behaviors across 3 layers each: ~240 MB additional memory.

### Latency Overhead

| Operation | Overhead | Notes |
|-----------|----------|-------|
| Hook registration | ~1 ms | One-time |
| Activation capture | ~0.1 ms | Per layer |
| Vector addition | ~0.01 ms | Per layer |
| Total per forward | < 1 ms | Negligible |

Steering adds negligible latency compared to prompting approaches.

### Extraction Cost

| Component | Time | Notes |
|-----------|------|-------|
| Forward pass (per pair) | ~50 ms | Batch size 1 |
| 50 contrast pairs | ~2.5 s | Sequential |
| Multiple layers | × N layers | Parallelizable |
| Total extraction | ~30 s | For one behavior |

Extraction is a one-time cost per model-behavior combination.

---

## Decision Framework

### Should You Use Steering Vectors?

```
START
  |
  v
Is behavior change needed at runtime?
  |-> No  -> Consider fine-tuning
  |-> Yes
       |
       v
     Is latency critical?
       |-> No  -> Consider prompting first (simpler)
       |-> Yes
            |
            v
          Do you have contrast pair data?
            |-> No  -> Use prompting
            |-> Yes -> Use steering vectors
```

### Choosing Steering Strength

```
START
  |
  v
What's your false positive tolerance?
  |
  |-> Very low (<5%)  -> Use strength 0.25-0.5
  |
  |-> Moderate (5-30%) -> Use strength 0.5-0.75
  |
  |-> High (30%+)     -> Use strength 0.75-1.0
                         (or reconsider if appropriate)
```

---

## Empirical Findings Summary

1. **Optimal strength range:** 0.25-0.75 for most production uses
2. **Composition limit:** Total strength across vectors < 1.5
3. **Layer selection:** Middle layers (40-60% depth) most effective
4. **Contrast pairs needed:** 20-50 for reliable extraction
5. **Model requirement:** > 7B parameters for meaningful effect

---

## Recommendations

### For Production Deployment

1. Start with strength 0.25, increase gradually
2. Monitor false positive rate on representative queries
3. Use A/B testing to measure real-world impact
4. Implement strength adjustment based on user feedback
5. Have fallback to unsteered model for edge cases

### For Research

1. Test full strength range (0.0 to 2.0) to understand limits
2. Document failure cases systematically
3. Compare against prompting and fine-tuning baselines
4. Measure both target behavior AND side effects
5. Report negative results—they're as valuable as positive ones
