# Multi-Behavior Steering Results

Testing steering vectors beyond just refusal. Does CAA work for other agent behaviors?

Spoiler: it works great for some, fails completely for others. The pattern is interesting.

---

## The Four Behaviors

1. **Refusal** — works well, see `REFUSAL_STEERING_RESULTS.md`
2. **Tool Restraint** — couldn't test properly (Mistral doesn't overuse tools)
3. **Instruction Hierarchy** — total failure, steering makes it worse
4. **Uncertainty** — works great, actually better than refusal in some ways

---

## Tool Restraint (Couldn't Really Test)

Tried to get Mistral to stop over-using tools. Problem: it doesn't over-use tools in the first place. Every strength level showed 0% tool invocation.

This behavior would need a model that actually has the problem — something like GPT-4 with function calling that reaches for tools too eagerly. Mistral just... answers directly. So I extracted the vector but can't really evaluate if it works.

Parking this one for now.

---

## Instruction Hierarchy (Complete Failure)

This was supposed to make the model follow system instructions even when users try to override them. Instead, steering made it *more* susceptible to overrides. Oops.

| Condition | Override Resistance |
|-----------|:-------------------:|
| Base | 25% |
| Prompting | 65% |
| Steering s=1.0 | **10%** |

Yeah, steering made it worse than the base model. I thought maybe I had the polarity backwards, so I tried negative strengths:

| Strength | Override Resistance |
|----------|:-------------------:|
| Prompt only | 65% |
| s=-0.5 | 60% |
| s=-1.0 | 45% |
| s=+0.5 | 55% |
| s=+1.0 | 40% |

Nope. Neither direction helps. The best steering result (60% at s=-0.5) is still worse than just prompting (65%).

**Why I think this failed:** Hierarchy isn't a simple behavioral direction like "refuse vs comply." It requires reasoning about *where* instructions come from, understanding authority levels, tracking context across the conversation. That's way too complex for a single linear direction in activation space.

CAA seems to work for response-style behaviors (how you phrase the answer) but not for reasoning-about-structure behaviors (how you interpret the conversation itself).

Qwen3 showed slight improvement with hierarchy steering (+25%), but the base detection was 0% due to its thinking mode messing with parsing. Hard to draw conclusions there.

---

## Uncertainty (The Cleanest Win)

This one worked really well, and the comparison to prompting is striking.

When you prompt for uncertainty ("express uncertainty when you don't know"), the model becomes uncertain about *everything* — including factual questions. Ask "What is 2+2?" and you get hedging. That's 0% confidence on factual questions. Useless.

With steering at s=0.5:

| Condition | Uncertainty (hard Q's) | Confidence (factual) |
|-----------|:----------------------:|:--------------------:|
| Base | 45% | 100% |
| Prompting | 95% | **0%** |
| Steering s=0.5 | 65% | **100%** |

Steering increases uncertainty on genuinely uncertain questions while keeping confidence on facts. The model can still tell the difference. Prompting destroys that discrimination.

This is the same pattern as refusal: prompting over-corrects, steering is more calibrated. The model retains its judgment about context.

Lower strength (0.5) worked better than higher here. At s=1.5 coherence started dropping.

---

## What Actually Worked

| Behavior | Steering Helps? | Notes |
|----------|:---------------:|-------|
| Refusal | Yes | Avoids over-refusal problem |
| Uncertainty | Yes | More calibrated than prompting |
| Hierarchy | No | Makes it worse, use prompting |
| Tool Restraint | ? | Model doesn't have this problem |

The pattern: **response-style behaviors** (how you say things) steer well. **Reasoning behaviors** (how you interpret the conversation structure) don't.

Both refusal and uncertainty are about output framing — be more/less X in your response. Hierarchy is about understanding the meta-structure of the conversation. Different thing.

---

## The Calibration Insight

This keeps coming up: prompting causes over-correction, steering preserves discrimination.

| Behavior | Prompting Side Effect | Steering Preserves |
|----------|----------------------|-------------------|
| Refusal | Refuses safe requests | Helpfulness |
| Uncertainty | Uncertain on facts | Factual confidence |

Steering nudges the model in a direction without overriding its ability to evaluate context. Prompting rewrites the priors and affects everything downstream.

This is probably the most interesting finding from the whole project.

---

## Vector Composition

Tried combining refusal + uncertainty vectors. Short answer: they interfere, but there are fixes.

### The Problem

| Method | Refusal | Uncertainty |
|--------|:-------:|:-----------:|
| Baseline (no steering) | 100% | 100% |
| Refusal only | 100% | 0% |
| Uncertainty only | 75% | 75% |
| **Both at same layer** | 100% | **25%** |

Even though the vectors only have 12% cosine similarity, applying them at the same layer causes the refusal vector to suppress uncertainty. The model becomes assertive ("I won't do that") which works against hedging.

### Fixes That Work

| Method | Refusal | Uncertainty |
|--------|:-------:|:-----------:|
| Naive (same layer) | 100% | 25% |
| **Orthogonalized** | 100% | **75%** |
| **Different layers** | 100% | **75%** |

1. **Orthogonalization**: Project out the shared component before combining. Even 12% overlap is enough to cause problems.

2. **Different layers**: Apply refusal at layer 12, uncertainty at layer 14. Spatial separation prevents interference.

Both approaches triple uncertainty detection while maintaining full refusal.

**Takeaway:** Always check vector overlap before composing. Can't just stack them and assume they combine nicely.

---

## LangChain Integration

Built wrappers so you can use these vectors in agent pipelines:

```python
from steering_vectors_agents.integrations.langchain import SteeredChatModel

chat = SteeredChatModel(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    steering_configs={
        "uncertainty": {
            "vector_path": "data/vectors/uncertainty_mistral_7b_instruct_v0.2/layer_14",
            "strength": 0.5,
        },
    },
)

# adjust at runtime
chat.set_strength("uncertainty", 0.75)
chat.disable_steering("uncertainty")
```

Can combine multiple vectors (refusal + uncertainty), adjust strength per-request, etc. The hooks clean up properly when you're done.

Demo: `python experiments/scripts/demo_langchain_steering.py --demo uncertainty`

---

## Open Questions

- What other behaviors would work? Verbosity? Formality? Those seem response-style-ish.
- ~~Can you compose vectors?~~ **Answered:** Yes, with orthogonalization or layer separation. See above.
- Why does hierarchy fail so badly? Is there a different extraction method that would work?
- Would this transfer across model families at all? Probably not, but worth checking.
