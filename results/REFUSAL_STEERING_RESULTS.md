# Refusal Steering Results

Testing whether steering vectors can improve refusal without the over-refusal problem that prompting causes.

Dec 2025 - tested on 3 models.

---

## TL;DR

Steering works. Prompting is a disaster.

When you prompt Mistral or Gemma to "refuse harmful requests," they refuse *everything* - 100% false positive rate. Steering at moderate strength (0.5-1.0) gets you improved refusal with zero false positives.

| Model | Base Refusal | With Steering | False Positives |
|-------|:------------:|:-------------:|:---------------:|
| Qwen3-8B | 100% | N/A (already perfect) | 0% |
| Mistral-7B | 85% | 95% | 0% |
| Gemma 2 9B | 90% | 95% | 0% |

Qwen3 doesn't need steering - it's already optimally calibrated out of the box. The other two benefit significantly.

---

## What Happened With Each Model

### Qwen3-8B

Already at 100% refusal with 0% false positives. Nothing to improve here.

Interesting note: both prompting AND high-strength steering make it worse. At s=1.0, false positives jump to 65%. The April 2025 RLHF training already found the sweet spot - don't mess with it.

### Mistral-7B

This one was the clearest win. Base model only refuses 85% of harmful requests, leaving a real gap.

| Condition | Harmful Refusal | False Positives |
|-----------|:---------------:|:---------------:|
| Base | 85% | 0% |
| Prompting | 100% | **100%** |
| Steering s=1.0 | 95% | 0% |

The prompting result is wild - it literally refuses to help with *anything*. Asked it to write a birthday card and it said it "couldn't assist with that request." Completely unusable.

Steering at s=0.5-1.0 gets +10% refusal improvement with no side effects. Above s=1.5 things start degrading (coherence drops to 0.67).

### Gemma 2 9B

Similar story but more sensitive to steering strength. s=0.5 is the sweet spot here, not s=1.0.

| Condition | Harmful Refusal | False Positives |
|-----------|:---------------:|:---------------:|
| Base | 90% | 0% |
| Prompting | 90% | **100%** |
| Steering s=0.5 | 95% | 0% |

One annoying thing: Gemma's chat template doesn't support system role properly. Had to prepend system instructions to the user message as a workaround.

---

## Things That Went Wrong

**Prompting broke everything on 2/3 models.** The safety-focused system prompt makes the model paranoid. It starts seeing threats everywhere.

**High strength causes weird degradation.** At s=1.5+, models become less coherent and paradoxically *worse* at refusal. Over-steering is real.

**Layer sensitivity on Mistral.** Some layers showed inverted effects - the vector seemed to work backwards. Layer 12 was the reliable one for this model.

---

## Technical Setup

- 50 contrast pairs per model
- CAA extraction at last token position
- Tested layers 10-20 range, picked best per model
- 20 harmful + 20 benign prompts for eval
- MPS backend on M4 Pro

Best layers: Qwen3 layer 14, Mistral layer 12, Gemma layer 14.

Vector norms varied a lot: Mistral was tiny (3.10), Gemma was huge (115.38). Not sure what to make of that yet.

---

## Bottom Line

For models with imperfect built-in alignment, steering beats prompting hands down. You get the safety improvement without lobotomizing the model's helpfulness.

For already-well-aligned models (Qwen3), leave them alone.
