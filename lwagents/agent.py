from .tools import Tool
from abc import ABC, abstractmethod
from typing_extensions import Self, override
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

class LLMAgent(Agent):
    def __init__(self, llm_model, tools=None):
        super().__init__(tools)
        self.llm_model = llm_model

    @override
    def action(self, prompt: str):
        response = self.llm_model.generate(messages=prompt, tools = self.tools)
        tool_calls = response.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = self.tools[function_name]
                function_args = json.loads(tool_call.function.arguments)
                if function_args:
                    function_response = function_to_call.execute(function_args)
                else:
                    function_response = function_to_call.execute()
                response = {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                return response
                
        return response.content
