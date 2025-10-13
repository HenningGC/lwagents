from .tools import Tool
from .state import AgentState, get_global_agent_state, State
from abc import ABC, abstractmethod
from typing_extensions import Self, override
from pydantic import BaseModel
from typing import Optional, Dict
import json
from .messages import LLMAgentResponse, LLMAgentRequest, LLMEntry


class InvalidAgent(Exception):
    pass


class Agent(ABC):
    def __init__(self, name: str, tools: list[Tool], state: AgentState):
        self.tools = None
        self.state = state
        self.name = name
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

    def update_global_state(self, name, entry, **kwargs):
        """Update the global agent state with information about this agent's action."""
        global_state = get_global_agent_state()
        global_state.update_state(
            agent_name=name, agent_kind=type(self).__name__, entry=entry, **kwargs
        )


class LLMAgent(Agent):
    def __init__(
        self,
        name: str,
        llm_model,
        tools=None,
        state: Optional[AgentState] = AgentState(),
    ):
        super().__init__(name=name, tools=tools, state=state)
        self.llm_model = llm_model

    @override
    def action(self, prompt: str, state_entry: Optional[dict] = {}, *args, **kwargs):
        request = LLMAgentRequest(content=prompt)
        response = self.llm_model.generate(
            messages=prompt, tools=self.tools, *args, **kwargs
        )
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
                    content=str(tool_response_content.get("content", "")),
                    tool_used=str(tool_response_content.get("function_name", None)),
                )
        else:
            result = LLMAgentResponse(
                role="assistant", content=response.content, tool_used=None
            )

        entry = LLMEntry(AgentRequest=request, AgentResponse=result)

        # Update both local and global state
        self.update_state(request=request, response=result, **state_entry)
        self.update_global_state(name=self.name, entry=entry)

        return result

    def update_state(self, *args, **kwargs):
        self.state.update_state(*args, **kwargs)
