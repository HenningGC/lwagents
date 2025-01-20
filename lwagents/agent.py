from .tools import Tool
from .memory import Memory
from abc import ABC, abstractmethod
from typing_extensions import Self, override
from pydantic import BaseModel
from typing import Optional, Dict
import json


class InvalidAgent(Exception):
    pass

class Agent(ABC):
    def __init__(self, tools: list[Tool]):
        self.tools = None
        if tools:
            self.tools = {type(tool).__name__: tool for tool in tools}

    @abstractmethod
    def action(self, current_node: str):
        """Make a decision based on the current node."""
        pass

    def use_tool(self, tool_name: str, *args, **kwargs):
        if tool_name in self.tools:
            return self.tools[tool_name].execute(*args, **kwargs)
        raise ValueError(f"Tool {tool_name} not found!")
    

class LLMAgentResponse(BaseModel):
    role: str  # e.g., "assistant" or "user"
    content: str  # The actual message content
    tool_used: Optional[str] = None  # Optional: Tool used during execution

class LLMAgent(Agent):
    def __init__(self, llm_model, tools=None, memory: Optional[Memory] = None):
        super().__init__(tools)
        self.llm_model = llm_model
        self.memory = memory or Memory()

    @override
    def action(self, prompt: str, *args, **kwargs):
        response = self.llm_model.generate(messages=prompt, tools = self.tools, *args, **kwargs)
        tool_calls = response.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = self.tools[function_name]
                function_args = json.loads(tool_call.function.arguments)
                if function_args:
                    function_response = function_to_call.execute(**function_args)
                else:
                    function_response = function_to_call.execute()
                tool_response_content = {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                
                result = LLMAgentResponse(
                    role="tool",
                    content=str(tool_response_content.get("content","")),
                    tool_used=str(tool_response_content.get("function_name",None))
                )
                return result
        
        result = LLMAgentResponse(
                    role="assistant",
                    content=response.content,
                    tool_used=None
                )
        if self.memory:
            self.memory.add_entry(result)

                
        return result
