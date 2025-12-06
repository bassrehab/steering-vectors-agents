# Conclusions

## Project Summary

This project explored using steering vectors—activation-level interventions extracted via Contrastive Activation Addition (CAA)—to control agent behaviors at inference time. We implemented a complete pipeline from vector extraction to LangChain agent integration, evaluated against prompting baselines, and documented both successes and limitations.

## Key Findings

### 1. Steering Vectors Work, But Context Matters

Steering vectors reliably modify model behavior in the intended direction. However, effectiveness depends heavily on:

- **Baseline model behavior:** Already safety-tuned models (like Qwen3-8B) leave little room for improvement in refusal behavior
- **Steering strength:** The useful range is narrow (0.25-1.0); beyond this, side effects dominate
- **Behavior type:** Some behaviors (tool restraint, instruction hierarchy) are more amenable to steering than others

### 2. The Strength-Quality Tradeoff is Fundamental

There is no free lunch. Increasing steering strength increases the target behavior but also:

- Increases false positives (over-refusal, excessive caution)
- Decreases output coherence
- Reduces model flexibility

This tradeoff is not a bug but an inherent property of the technique.

### 3. Comparison to Baselines

| Method | Harmful Refusal | Benign Refusal | Notes |
|--------|-----------------|----------------|-------|
| Baseline | 100% | 0% | Qwen3-8B already safety-tuned |
| Prompting | 100% | 35% | High false positive rate |
| Steering s=0.5 | 100% | ~5% | Maintains baseline quality |
| Steering s=1.0 | 95% | ~65% | Over-refusal begins |

**Key insight:** For Qwen3-8B specifically, steering adds limited value for refusal since the baseline is already strong. The technique would show more dramatic improvement on models with weaker safety training.

### 4. Runtime Adjustability is Valuable

Unlike fine-tuning or fixed prompts, steering vectors can be adjusted at runtime:

```python
# Dial up safety for sensitive contexts
agent.set_strength("refusal", 0.8)

# Dial down for trusted users
agent.set_strength("refusal", 0.2)
```

This enables adaptive safety levels without model reloading or prompt modification.

### 5. Multi-Vector Composition Works

Multiple steering vectors can be applied simultaneously with independent strengths:

```python
injector = ActivationInjector(
    model,
    vectors=[refusal_vector, tool_restraint_vector, hierarchy_vector],
    strengths=[0.5, 0.3, 0.4],
)
```

Effects are approximately additive, enabling compositional behavior control.

## What Steering Vectors Are Good For

1. **Runtime safety adjustment** — Vary caution level based on context
2. **Defense in depth** — Additional layer against prompt injection
3. **Weaker-safety models** — More impact on models without strong RLHF
4. **Multi-behavior control** — Combine refusal, tool restraint, hierarchy independently
5. **Low-latency applications** — No additional token generation overhead

## What Steering Vectors Are Not Good For

1. **Creating new behaviors** — Only amplifies existing tendencies
2. **Replacing safety training** — Complements, doesn't replace RLHF
3. **Adversarial robustness** — Sophisticated attacks can still succeed
4. **Very high behavior change** — Coherence degrades before reaching extreme changes
5. **Cross-model transfer** — Vectors are model-specific

## Honest Assessment

### What Worked

- Clean activation extraction with PyTorch hooks
- Reliable vector injection during generation
- LangChain integration for practical agent use
- Systematic evaluation against baselines
- Documented failure modes

### What Didn't Work As Expected

- Qwen3-8B's strong baseline left little room for improvement
- High-strength steering degraded quality faster than expected
- Some adversarial prompts still bypassed steering

### What We'd Do Differently

1. **Start with weaker-safety models** to demonstrate larger effect sizes
2. **Focus on tool restraint earlier** — more room for improvement there
3. **Build adaptive strength selection** into the core library
4. **More diverse contrast pairs** to handle tokenization variation

## Implications for Agent Safety

Steering vectors offer a useful tool in the agent safety toolkit, but should be understood correctly:

**They are a dial, not a switch.** You can increase or decrease behaviors along a continuum, but you cannot guarantee absolute outcomes.

**They complement, not replace, other techniques.** Use alongside:
- Input filtering
- Output monitoring
- Prompt engineering
- Model fine-tuning
- Human oversight

**They work best when you know what you want.** The technique requires clear definition of target behavior via contrast pairs.

## Future Directions

### Technical Extensions

1. **Adaptive strength selection** — Automatically adjust based on input classification
2. **Multi-layer steering** — Apply vectors across multiple layers for stronger effect
3. **Learned composition** — Optimize multi-vector weights for specific objectives
4. **Cross-model transfer** — Investigate when vectors transfer between similar models

### Research Questions

1. Can steering vectors improve calibration/uncertainty expression?
2. How do steering vectors interact with chain-of-thought reasoning?
3. What's the minimum contrast pair set size for reliable extraction?
4. Can we extract vectors for more abstract behaviors (honesty, helpfulness)?

### Application Areas

1. **Enterprise deployments** — Customizable safety levels per user tier
2. **Research tools** — Interpretability via steering direction analysis
3. **Red teaming** — Using negative steering to find vulnerabilities
4. **Education** — Teaching models with adjustable scaffolding

## Final Thoughts

Steering vectors are one tool among many for controlling AI behavior. Their strength lies in runtime adjustability, low latency overhead, and compositional nature. Their weakness is the fundamental tradeoff between behavior change and output quality.

For applied AI research, the value isn't in claiming steering vectors solve AI safety—they don't. The value is in:

1. Understanding when they're useful vs. when alternatives are better
2. Documenting failure modes honestly
3. Providing practical implementations that others can build on
4. Contributing to the broader toolkit for AI control

This project demonstrates that steering vectors can be a practical component of agent behavior control, with clear caveats about their limitations.

---

## Acknowledgments

This work builds on foundational research by:

- Zou et al. (2023) — Representation Engineering
- Turner et al. (2023) — Activation Addition
- Anthropic's interpretability research

---

## Citation

If you use this work, please cite:

```bibtex
@software{steering_vectors_agents,
  author = {Mitra, Subhadip},
  title = {Steering Vectors for Agent Behavior Control},
  year = {2025},
  url = {https://github.com/bassrehab/steering-vectors-agents}
}
```
