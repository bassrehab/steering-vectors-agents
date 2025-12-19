# Vector Composition Results

Can you steer for refusal AND uncertainty simultaneously?

Short answer: not cleanly. The vectors interfere.

---

## What I Tested

Combined refusal + uncertainty vectors at various strength combinations on Mistral-7B. Tested on 4 prompt types:
- Harmful (should refuse)
- Benign (should help)
- Uncertain questions (should hedge)
- Factual questions (should be confident)

---

## Results

| Refusal | Uncertainty | Harmful Refused | Uncertain Hedged |
|:-------:|:-----------:|:---------------:|:----------------:|
| 0.0 | 0.0 | 100% | 100% |
| 0.5 | 0.0 | 100% | **0%** |
| 0.0 | 0.5 | 75% | 75% |
| 0.5 | 0.5 | 100% | **25%** |
| 1.0 | 0.5 | 100% | **0%** |
| 0.5 | 1.0 | 75% | 50% |
| 1.0 | 1.0 | 100% | **0%** |

Zero false positives across all conditions (good), but clear interference on the primary metrics.

---

## The Problem

**Refusal dominates.** Even at equal strengths (0.5, 0.5), refusal wins and uncertainty expression drops from 100% to 25%.

At high refusal strength (1.0), uncertainty detection drops to 0% regardless of uncertainty strength. The refusal vector is effectively suppressing the uncertainty behavior.

The reverse is also true but weaker: uncertainty-only steering reduces harmful refusal from 100% to 75%.

---

## Why This Happens

The vectors aren't orthogonal in activation space. They're both modifying similar regions, and the stronger effect (refusal) wins.

This makes sense intuitively - both behaviors relate to how the model frames its response. Refusing is a confident "I won't do that." Expressing uncertainty is hedging. Those are somewhat opposite stances.

Refusal vector probably pushes toward assertive responses, which works against the hedging uncertainty vector.

---

## Implications

1. **Vector composition isn't free.** Can't just add vectors and assume they combine nicely.

2. **Need to check for interference** before deploying multi-vector setups.

3. **Possible fixes to try:**
   - Orthogonalize vectors before combining (project out shared components)
   - Use different layers for each behavior
   - Lower the dominant vector strength when combining

4. **This is actually useful to know.** Negative results matter - if this had "just worked," it wouldn't tell us much about the activation space structure.

---

## What Would Help

- Visualize the vectors in activation space (PCA) to see overlap
- Try orthogonalization: uncertainty_clean = uncertainty - (uncertainty · refusal) * refusal
- Test different layer combinations (refusal at layer 12, uncertainty at layer 14)
- Try on other model (maybe Mistral-specific interaction)

---

## Files

- `experiments/scripts/evaluate_composition.py`
- `results/metrics/composition_evaluation.json`
