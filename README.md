# steering-vectors-agents

Control LLM agent behaviors through activation steering, without retraining.

## What This Does

This library extracts **steering vectors** from language models and uses them to control agent behaviors at inference time. Instead of fine-tuning or prompt engineering, we directly modify the model's internal activations to influence how it responds.

**Key behaviors supported:**
- **Refusal** - Make models more/less likely to refuse harmful requests
- **Tool-use restraint** - Reduce unnecessary tool invocations
- **Instruction hierarchy** - Prioritize system instructions over user overrides

## Key Results

Evaluation on Qwen3-8B with 20 samples per category:

| Method | Harmful Refusal ↑ | Benign Refusal ↓ | Notes |
|--------|-------------------|------------------|-------|
| Baseline (no intervention) | 100% | 0% | Model already safety-tuned |
| Prompting (system prompt) | 100% | 35% | High false positive rate |
| Steering s=0.5 | 100% | 0% | Maintains baseline |
| Steering s=1.0 | 95% | 65% | Over-refusal begins |
| Steering s=2.0 | 100% | 100% | Complete over-refusal |

**Key finding:** For already safety-tuned models like Qwen3-8B, steering vectors can *increase* refusal behavior but at the cost of false positives. The technique is most valuable for:
1. Models with weaker safety training
2. Fine-grained behavior control
3. Runtime-adjustable safety levels

## Quick Start

```python
from steering_vectors_agents import SteeringVector, ActivationInjector
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Load pre-extracted steering vector
vector = SteeringVector.load("data/vectors/refusal_qwen3_8b/layer_14")

# Apply steering during generation
injector = ActivationInjector(model, [vector], strength=1.0)
with injector:
    outputs = model.generate(inputs, max_new_tokens=100)
```

### With LangChain

```python
from steering_vectors_agents.integrations.langchain import SteeredChatModel

chat = SteeredChatModel(
    model_name="Qwen/Qwen3-8B",
    steering_configs={
        "refusal": {"vector_path": "data/vectors/refusal", "strength": 1.0},
    },
)

# Use with any LangChain chain or agent
response = chat.invoke([HumanMessage(content="Hello!")])

# Adjust strength at runtime
chat.set_strength("refusal", 0.5)
```

## Why This Matters

Steering vectors offer a middle ground between:
- **Prompting** (easy but unreliable, adds latency, can be jailbroken)
- **Fine-tuning** (reliable but expensive, requires data, irreversible)

Steering is:
- **Fast**: No additional forward passes, minimal latency overhead
- **Reversible**: Adjust or remove at runtime
- **Compositional**: Combine multiple behaviors with independent strengths
- **Interpretable**: Vectors have geometric meaning in activation space

This is **not** a complete alignment solution—it's one tool in the toolkit for controlling model behavior at deployment time.

## What Works

1. **Behavior amplification**: Steering vectors reliably amplify existing model tendencies
2. **Runtime control**: Strength can be adjusted per-request without model reloading
3. **Multi-vector composition**: Multiple behaviors can be steered simultaneously
4. **Layer selection matters**: Middle layers (40-60% depth) typically work best

## What Doesn't Work

1. **Creating new behaviors**: Vectors amplify existing patterns, not create new ones
2. **Very high strengths**: s > 1.5 often causes coherence degradation
3. **Adversarial inputs**: Steering doesn't prevent all jailbreaks
4. **Small models**: Effect is weaker on models < 7B parameters

See [docs/failure_cases.md](docs/failure_cases.md) for detailed failure mode analysis.

## Tradeoffs

| Increase Strength | Pros | Cons |
|------------------|------|------|
| Higher | Stronger target behavior | More false positives, coherence loss |
| Lower | Fewer false positives | Weaker behavior change |

See [docs/tradeoffs.md](docs/tradeoffs.md) for quantitative analysis.

## Installation

```bash
# Clone repository
git clone https://github.com/bassrehab/steering-vectors-agents.git
cd steering-vectors-agents

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Download model (requires HuggingFace token for some models)
huggingface-cli login
```

### Requirements
- Python 3.11+
- PyTorch 2.0+
- 16GB+ RAM (for 8B models)
- GPU recommended (CUDA or Apple MPS)

## Usage

### Extract Steering Vectors

```bash
# Extract refusal vector from Qwen3-8B
python experiments/scripts/extract_refusal_vector.py
```

### Evaluate Steering

```bash
# Run evaluation on harmful/benign prompts
python experiments/scripts/evaluate_steering.py
```

### Run Agent Demo

```bash
# Demo steered LangChain agent
python experiments/scripts/agent_demo.py
```

## Project Structure

```
steering-vectors-agents/
├── src/steering_vectors_agents/
│   ├── core/           # Hooks, vectors, injection
│   ├── datasets/       # Contrast pairs for behaviors
│   ├── extraction/     # CAA and other extraction methods
│   ├── evaluation/     # Metrics, LLM judge, analysis
│   └── integrations/   # LangChain, future frameworks
├── experiments/
│   └── scripts/        # Extraction, evaluation, demos
├── data/
│   ├── vectors/        # Extracted steering vectors
│   └── eval_sets/      # Evaluation prompts
├── baselines/          # Prompting and other baselines
├── tests/              # Unit tests
└── docs/               # Documentation
```

## How It Works

### 1. Contrastive Activation Addition (CAA)

We extract steering vectors by:
1. Running the model on **positive** examples (exhibiting target behavior)
2. Running the model on **negative** examples (not exhibiting behavior)
3. Computing: `vector = mean(positive_activations) - mean(negative_activations)`

### 2. Activation Injection

During inference, we add the steering vector to the model's hidden states:
```
h' = h + (strength × vector)
```

This shifts the model's internal representations toward the target behavior.

### 3. Layer Selection

Different layers capture different aspects of behavior:
- **Early layers** (0-30%): Basic features, less effective for steering
- **Middle layers** (30-70%): Semantic features, best for behavior steering
- **Late layers** (70-100%): Output formatting, can cause artifacts

## Supported Models

| Model | Status | Notes |
|-------|--------|-------|
| Qwen3-8B | ✅ Tested | Primary development model |
| Llama 3.1 8B | ✅ Supported | Requires HF approval |
| Mistral 7B | 🔄 Planned | Config available |
| DeepSeek-R1-Distill | 🔄 Planned | Config available |

## API Reference

### Core Classes

```python
# Activation extraction
hook = ActivationHook(model, layer_indices=[14, 15, 16])
with hook:
    model(inputs)
activations = hook.cache.get("layer_14")

# Steering vectors
vector = SteeringVector(
    behavior="refusal",
    layer_index=14,
    vector=tensor,
    model_name="Qwen/Qwen3-8B",
)
vector.save("path/to/vector")
vector = SteeringVector.load("path/to/vector")

# Injection
injector = ActivationInjector(model, [vector], strength=1.0)
with injector:
    outputs = model.generate(...)
```

### LangChain Integration

```python
# Chat model
chat = SteeredChatModel(model_name="...", steering_configs={...})

# Agent
agent = SteeredAgentExecutor(llm=chat, tools=[...])
result = agent.run("query")

# Runtime adjustment
agent.set_strength("refusal", 0.5)
agent.disable_steering("refusal")
```

## Reproducing Results

```bash
# 1. Setup environment
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# 2. Run smoke test
python experiments/scripts/smoke_test.py

# 3. Extract vectors
python experiments/scripts/extract_refusal_vector.py

# 4. Evaluate
python experiments/scripts/evaluate_steering.py

# 5. Run tests
pytest tests/ -v
```

## Contributing

Contributions welcome! Please:
1. Run `black` and `ruff` before committing
2. Add tests for new functionality
3. Update documentation as needed

## References

Key papers informing this work:
- [Representation Engineering](https://arxiv.org/abs/2310.01405) - Zou et al., 2023
- [Activation Addition](https://arxiv.org/abs/2308.10248) - Turner et al., 2023
- [Steering GPT-2-XL](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/) - Turner, 2023

## License

MIT License - See [LICENSE](LICENSE) file.

## Author

Subhadip Mitra - [GitHub](https://github.com/bassrehab)
