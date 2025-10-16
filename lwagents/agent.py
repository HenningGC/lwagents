import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from pydantic import BaseModel
from typing_extensions import Self, override

from .messages import LLMAgentRequest, LLMAgentResponse, LLMEntry, LLMToolResponse
from .state import AgentState, State, get_global_agent_state
from .tools import Tool, ToolUtility, ToolsExecutionResults


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
        tools: List = None,
        model_name: str = None,
        state: Optional[AgentState] = AgentState(),
    ):
        super().__init__(name=name, tools=tools, state=state)
        self.llm_model = llm_model
        self.model_name = model_name

    @override
    def action(
        self,
        prompt: List[Dict[str, str]],
        state_entry: Optional[dict] = {},
        use_model: str = None,
        system: str = None,
        *args,
        **kwargs,
    ):
        request = LLMAgentRequest(content=prompt)
        if not use_model:
            use_model = self.model_name
        if not use_model:
            raise ValueError(
                "Model name must be specified either in agent or action call!"
            )
        response = self.llm_model.generate(
            model_name=use_model,
            prompt=prompt,
            tools=self.tools,
            system=system,
            *args,
            **kwargs,
        )
        result = None
        if type(response) == LLMToolResponse:
            tool_execution_results = ToolUtility.execute_from_response(
                tool_response=response, tools=self.tools
            )
            tool_results = []
            if tool_execution_results:
                for tool_execution_result in tool_execution_results.results:
                    tool_response_content = {
                        "tool_call_id": tool_execution_result.id,
                        "name": tool_execution_result.name,
                        "content": tool_execution_result.content,
                    }
                    tool_results.append(tool_response_content)

                result = LLMAgentResponse(
                    role="tool",
                    content=response.content,
                    tools_used=[tr["name"] for tr in tool_results],
                )
            else:
                result = LLMAgentResponse(
                    role="assistant", content=None, tool_used=None
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
