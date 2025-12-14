"""LangChain LLM wrapper with steering vector support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field, PrivateAttr

from ...core.injection import ActivationInjector, MultiVectorInjector
from ...core.vectors import SteeringVector, SteeringVectorSet

DEFAULT_LAYER = 14  # Best layer for most behaviors based on experiments


class SteeredLLM(LLM):
    """
    LangChain LLM that applies steering vectors during generation.
    
    This wrapper loads a HuggingFace model and applies steering vectors
    to control model behavior at inference time.
    
    Example:
        ```python
        llm = SteeredLLM(
            model_name="Qwen/Qwen3-8B",
            steering_configs={
                "refusal": {
                    "vector_path": "data/vectors/refusal_qwen3_8b/layer_14",
                    "strength": 1.0,
                },
            },
        )
        response = llm.invoke("How do I hack a computer?")
        ```
    """
    
    model_name: str = Field(description="HuggingFace model name or path")
    steering_configs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Steering configurations: {behavior: {vector_path, strength}}"
    )
    device: str = Field(default="auto", description="Device to use (auto, cuda, mps, cpu)")
    torch_dtype: str = Field(default="float16", description="Torch dtype for model")
    max_new_tokens: int = Field(default=256, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    do_sample: bool = Field(default=True, description="Whether to sample")
    
    # private attributes (not serialized)
    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)
    _vectors: Dict[str, SteeringVector] = PrivateAttr(default_factory=dict)
    _injector: Optional[MultiVectorInjector] = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize()
    
    def _initialize(self):
        """Load model and steering vectors."""
        if self._initialized:
            return
            
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # determine device
        if self.device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        else:
            device = self.device
        
        # determine dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(self.torch_dtype, torch.float16)
        
        # load tokenizer and model
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        
        # load steering vectors
        self._load_vectors()
        
        self._initialized = True
    
    def _load_vectors(self):
        """Load steering vectors from configs."""
        if not self.steering_configs:
            return

        vector_sets = {}
        strengths = {}

        for behavior, config in self.steering_configs.items():
            vector_path = Path(config["vector_path"])
            strength = config.get("strength", 1.0)
            layer = config.get("layer", DEFAULT_LAYER)

            # load vector
            vector = SteeringVector.load(vector_path)
            self._vectors[behavior] = vector

            # create a SteeringVectorSet for this behavior
            vec_set = SteeringVectorSet(behavior=behavior)
            vec_set.add(vector)
            vector_sets[behavior] = vec_set
            strengths[behavior] = strength

        if vector_sets:
            device = next(self._model.parameters()).device
            vector_sets_on_device = {
                behavior: vs.to(device) for behavior, vs in vector_sets.items()
            }
            self._injector = MultiVectorInjector(
                self._model, vector_sets_on_device, strengths
            )
    
    @property
    def _llm_type(self) -> str:
        return "steered_llm"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "steering_configs": self.steering_configs,
        }
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """Generate text with optional steering."""
        self._initialize()
        
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(prompt, return_tensors="pt").to(device)
        
        # generation kwargs
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "do_sample": kwargs.get("do_sample", self.do_sample),
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        
        # handle stop sequences
        if stop:
            gen_kwargs["eos_token_id"] = [
                self._tokenizer.encode(s, add_special_tokens=False)[-1]
                for s in stop
                if self._tokenizer.encode(s, add_special_tokens=False)
            ]
        
        # generate with or without steering
        with torch.no_grad():
            if self._injector is not None:
                with self._injector:
                    outputs = self._model.generate(**inputs, **gen_kwargs)
            else:
                outputs = self._model.generate(**inputs, **gen_kwargs)
        
        # decode response
        full_response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # remove prompt from response
        if full_response.startswith(prompt):
            response = full_response[len(prompt):].strip()
        else:
            response = full_response
        
        return response
    
    def set_strength(self, behavior: str, strength: float):
        """
        Dynamically adjust steering strength for a behavior.
        
        Args:
            behavior: Name of the behavior to adjust
            strength: New strength value
        """
        if self._injector is not None:
            self._injector.set_strength(behavior, strength)
    
    def get_strength(self, behavior: str) -> float:
        """Get current strength for a behavior."""
        if self._injector is not None:
            return self._injector.strengths.get(behavior, 0.0)
        return 0.0
    
    def disable_steering(self, behavior: Optional[str] = None):
        """
        Disable steering for a specific behavior or all behaviors.
        
        Args:
            behavior: Behavior to disable, or None for all
        """
        if self._injector is None:
            return
            
        if behavior is None:
            for b in self._vectors:
                self._injector.set_strength(b, 0.0)
        else:
            self._injector.set_strength(behavior, 0.0)
    
    def enable_steering(self, behavior: str, strength: float = 1.0):
        """
        Enable steering for a specific behavior.
        
        Args:
            behavior: Behavior to enable
            strength: Strength to set
        """
        self.set_strength(behavior, strength)
    
    def add_vector(
        self,
        behavior: str,
        vector: Union[SteeringVector, str, Path],
        strength: float = 1.0,
    ):
        """
        Add a steering vector at runtime.

        Args:
            behavior: Name for this behavior
            vector: SteeringVector or path to load from
            strength: Initial strength
        """
        self._initialize()

        if isinstance(vector, (str, Path)):
            vector = SteeringVector.load(vector)

        self._vectors[behavior] = vector

        # rebuild injector with new vector
        device = next(self._model.parameters()).device
        vector_sets = {}
        strengths = {}

        for b, v in self._vectors.items():
            vec_set = SteeringVectorSet(behavior=b)
            vec_set.add(v.to(device))
            vector_sets[b] = vec_set
            # preserve existing strength or use default
            if self._injector and b in self._injector._strengths:
                strengths[b] = self._injector._strengths[b]
            else:
                strengths[b] = 1.0

        strengths[behavior] = strength

        self._injector = MultiVectorInjector(self._model, vector_sets, strengths)
