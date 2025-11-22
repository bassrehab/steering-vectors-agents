"""LangChain Agent wrapper with steering vector support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from .steered_chat import SteeredChatModel
from .steered_llm import SteeredLLM


class SteeredAgentExecutor:
    """
    Agent executor that uses a steered language model.
    
    This wrapper provides a simple way to create agents with steering
    vector support, allowing runtime control of agent behaviors.
    
    Example:
        ```python
        from langchain.tools import Tool
        
        # Create steered chat model
        chat = SteeredChatModel(
            model_name="Qwen/Qwen3-8B",
            steering_configs={
                "refusal": {
                    "vector_path": "data/vectors/refusal",
                    "strength": 1.0,
                },
            },
        )
        
        # Define tools
        tools = [
            Tool(name="Calculator", func=calc_fn, description="Math"),
        ]
        
        # Create agent
        agent = SteeredAgentExecutor(chat, tools)
        
        # Run agent
        result = agent.run("What is 2+2?")
        
        # Adjust steering at runtime
        agent.set_strength("refusal", 0.5)
        ```
    """
    
    def __init__(
        self,
        llm: Union[SteeredLLM, SteeredChatModel, BaseLanguageModel],
        tools: List[BaseTool],
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize the steered agent executor.
        
        Args:
            llm: Language model (preferably SteeredLLM or SteeredChatModel)
            tools: List of tools available to the agent
            system_prompt: Optional system prompt for the agent
            max_iterations: Maximum number of agent iterations
            verbose: Whether to print intermediate steps
        """
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.tools_list = tools
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # build system prompt with tool descriptions
        self.system_prompt = system_prompt or self._build_default_system_prompt()
    
    def _build_default_system_prompt(self) -> str:
        """Build a default ReAct-style system prompt."""
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools_list
        ])
        
        return f"""You are a helpful AI assistant with access to tools.

Available tools:
{tool_descriptions}

To use a tool, respond with:
Thought: [your reasoning about what to do]
Action: [tool name]
Action Input: [input to the tool]

After receiving the tool result, continue reasoning or provide your final answer.

To provide a final answer without using tools, respond with:
Thought: [your reasoning]
Final Answer: [your answer to the user]

Always think step by step and use tools only when necessary.
"""
    
    def _parse_agent_output(self, output: str) -> Union[AgentAction, AgentFinish]:
        """Parse the agent's output to determine the next action."""
        output = output.strip()
        
        # check for final answer
        if "Final Answer:" in output:
            answer_start = output.find("Final Answer:") + len("Final Answer:")
            answer = output[answer_start:].strip()
            return AgentFinish(return_values={"output": answer}, log=output)
        
        # check for action
        if "Action:" in output and "Action Input:" in output:
            action_start = output.find("Action:") + len("Action:")
            action_end = output.find("Action Input:")
            action = output[action_start:action_end].strip()
            
            input_start = output.find("Action Input:") + len("Action Input:")
            action_input = output[input_start:].strip()
            
            # clean up action name
            action = action.split("\n")[0].strip()
            
            return AgentAction(tool=action, tool_input=action_input, log=output)
        
        # default to final answer if no clear action
        return AgentFinish(return_values={"output": output}, log=output)
    
    def _format_tool_result(self, tool_name: str, result: str) -> str:
        """Format a tool result for the conversation."""
        return f"Observation: {result}\n"
    
    def run(
        self,
        input: str,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> str:
        """
        Run the agent on an input.
        
        Args:
            input: User input/query
            chat_history: Optional conversation history
            
        Returns:
            Agent's final response
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        
        # build messages
        messages = [SystemMessage(content=self.system_prompt)]
        
        if chat_history:
            messages.extend(chat_history)
        
        messages.append(HumanMessage(content=input))
        
        # agent loop
        for i in range(self.max_iterations):
            if self.verbose:
                print(f"\n--- Iteration {i+1} ---")
            
            # get LLM response
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(messages)
                if hasattr(response, 'content'):
                    output = response.content
                else:
                    output = str(response)
            else:
                # fallback for non-langchain LLMs
                prompt = "\n".join([
                    f"{m.type}: {m.content}" for m in messages
                ])
                output = self.llm(prompt)
            
            if self.verbose:
                print(f"Agent output: {output[:200]}...")
            
            # parse output
            result = self._parse_agent_output(output)
            
            if isinstance(result, AgentFinish):
                return result.return_values["output"]
            
            # execute tool
            tool_name = result.tool
            tool_input = result.tool_input
            
            if tool_name in self.tools:
                try:
                    tool_result = self.tools[tool_name].run(tool_input)
                except Exception as e:
                    tool_result = f"Error: {str(e)}"
            else:
                tool_result = f"Error: Tool '{tool_name}' not found."
            
            if self.verbose:
                print(f"Tool {tool_name} result: {tool_result}")
            
            # add to messages
            messages.append(AIMessage(content=output))
            messages.append(HumanMessage(content=self._format_tool_result(tool_name, tool_result)))
        
        # max iterations reached
        return "I was unable to complete the task within the allowed number of steps."
    
    def set_strength(self, behavior: str, strength: float):
        """
        Set steering strength for a behavior.
        
        Args:
            behavior: Name of the behavior
            strength: New strength value
        """
        if hasattr(self.llm, 'set_strength'):
            self.llm.set_strength(behavior, strength)
    
    def get_strength(self, behavior: str) -> float:
        """Get current steering strength for a behavior."""
        if hasattr(self.llm, 'get_strength'):
            return self.llm.get_strength(behavior)
        return 0.0
    
    def disable_steering(self, behavior: Optional[str] = None):
        """Disable steering for a behavior or all behaviors."""
        if hasattr(self.llm, 'disable_steering'):
            self.llm.disable_steering(behavior)
    
    def enable_steering(self, behavior: str, strength: float = 1.0):
        """Enable steering for a behavior."""
        if hasattr(self.llm, 'enable_steering'):
            self.llm.enable_steering(behavior, strength)


def create_steered_agent(
    model_name: str = "Qwen/Qwen3-8B",
    steering_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
    device: str = "auto",
    verbose: bool = False,
) -> SteeredAgentExecutor:
    """
    Convenience function to create a steered agent.
    
    Args:
        model_name: HuggingFace model name
        steering_configs: Steering vector configurations
        tools: Tools for the agent
        system_prompt: Optional system prompt
        device: Device to use
        verbose: Verbose output
        
    Returns:
        Configured SteeredAgentExecutor
    """
    chat = SteeredChatModel(
        model_name=model_name,
        steering_configs=steering_configs or {},
        device=device,
    )
    
    return SteeredAgentExecutor(
        llm=chat,
        tools=tools or [],
        system_prompt=system_prompt,
        verbose=verbose,
    )


__all__ = ["SteeredAgentExecutor", "create_steered_agent"]
