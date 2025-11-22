"""LangChain Chat Model wrapper with steering vector support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

from ...core.injection import MultiVectorInjector
from ...core.vectors import SteeringVector


class SteeredChatModel(BaseChatModel):
    """
    LangChain Chat Model that applies steering vectors during generation.
    
    This wrapper provides chat-style interactions with steering vector support,
    properly handling system messages, conversation history, and chat templates.
    
    Example:
        ```python
        chat = SteeredChatModel(
            model_name="Qwen/Qwen3-8B",
            steering_configs={
                "refusal": {
                    "vector_path": "data/vectors/refusal_qwen3_8b/layer_14",
                    "strength": 1.0,
                },
            },
        )
        
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="How do I hack a computer?"),
        ]
        response = chat.invoke(messages)
        ```
    """
    
    model_name: str = Field(description="HuggingFace model name or path")
    steering_configs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Steering configurations: {behavior: {vector_path, strength}}"
    )
    device: str = Field(default="auto", description="Device to use")
    torch_dtype: str = Field(default="float16", description="Torch dtype")
    max_new_tokens: int = Field(default=256, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    do_sample: bool = Field(default=True, description="Whether to sample")
    
    # private attributes
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
            
        vectors = []
        strengths = {}
        
        for behavior, config in self.steering_configs.items():
            vector_path = Path(config["vector_path"])
            strength = config.get("strength", 1.0)
            
            vector = SteeringVector.load(vector_path)
            self._vectors[behavior] = vector
            vectors.append(vector)
            strengths[behavior] = strength
        
        if vectors:
            device = next(self._model.parameters()).device
            vectors_on_device = [v.to(device) for v in vectors]
            self._injector = MultiVectorInjector(
                self._model, vectors_on_device, strengths
            )
    
    @property
    def _llm_type(self) -> str:
        return "steered_chat_model"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "steering_configs": self.steering_configs,
        }
    
    def _convert_messages_to_chat_format(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, str]]:
        """Convert LangChain messages to HuggingFace chat format."""
        chat_messages = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                chat_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                chat_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                chat_messages.append({"role": "assistant", "content": msg.content})
            else:
                # fallback for other message types
                chat_messages.append({"role": "user", "content": str(msg.content)})
        
        return chat_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """Generate a response for the given messages."""
        self._initialize()
        
        device = next(self._model.parameters()).device
        
        # convert to chat format and apply template
        chat_messages = self._convert_messages_to_chat_format(messages)
        prompt = self._tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(device)
        
        # generation kwargs
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "do_sample": kwargs.get("do_sample", self.do_sample),
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        
        # generate with or without steering
        with torch.no_grad():
            if self._injector is not None:
                with self._injector:
                    outputs = self._model.generate(**inputs, **gen_kwargs)
            else:
                outputs = self._model.generate(**inputs, **gen_kwargs)
        
        # decode and extract assistant response
        full_response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # extract just the assistant's response
        if "assistant" in full_response.lower():
            response = full_response.split("assistant")[-1].strip()
        else:
            # fallback: remove the prompt
            response = full_response[len(prompt):].strip() if prompt in full_response else full_response
        
        # create chat result
        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])
    
    def set_strength(self, behavior: str, strength: float):
        """Dynamically adjust steering strength."""
        if self._injector is not None:
            self._injector.set_strength(behavior, strength)
    
    def get_strength(self, behavior: str) -> float:
        """Get current strength for a behavior."""
        if self._injector is not None:
            return self._injector.strengths.get(behavior, 0.0)
        return 0.0
    
    def disable_steering(self, behavior: Optional[str] = None):
        """Disable steering for a specific behavior or all."""
        if self._injector is None:
            return
        if behavior is None:
            for b in self._vectors:
                self._injector.set_strength(b, 0.0)
        else:
            self._injector.set_strength(behavior, 0.0)
    
    def enable_steering(self, behavior: str, strength: float = 1.0):
        """Enable steering for a specific behavior."""
        self.set_strength(behavior, strength)
    
    def add_vector(
        self,
        behavior: str,
        vector: Union[SteeringVector, str, Path],
        strength: float = 1.0,
    ):
        """Add a steering vector at runtime."""
        self._initialize()
        
        if isinstance(vector, (str, Path)):
            vector = SteeringVector.load(vector)
        
        self._vectors[behavior] = vector
        
        device = next(self._model.parameters()).device
        vectors = [v.to(device) for v in self._vectors.values()]
        strengths = {
            b: self._injector.strengths.get(b, 1.0) if self._injector else 1.0
            for b in self._vectors
        }
        strengths[behavior] = strength
        
        self._injector = MultiVectorInjector(self._model, vectors, strengths)
