import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from typing_extensions import Literal, Self, override

from lwagents.graph import Graph

from .messages import LLMAgentResponse, LLMToolResponse
from .state import AgentState, GraphState, State, get_global_agent_state
from .tools import Tool, ToolUtility, ToolsExecutionResults
from ._types import ExecutionMode, StepResult


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

    def update_global_state(self, name, action_result, **kwargs):
        """Update the global agent state with information about this agent's action."""
        global_state = get_global_agent_state()
        global_state.update_state(
            agent_name=name,
            agent_kind=type(self).__name__,
            action_result=action_result,
            **kwargs,
        )


class LLMAgent(Agent):
    def __init__(
        self,
        name: str,
        llm_model,
        tools: list[Tool] = [],
        state: Optional[AgentState] = AgentState(),
    ):
        super().__init__(name=name, tools=tools, state=state)
        self.llm_model = llm_model

    @override
    def action(
        self,
        state_entry: Optional[dict] = {},
        model_params: Dict[str, Any] = {},
    ):
        response = self.llm_model.generate(
            tools=self.tools,
            model_params=model_params,
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
                    content=str([tr["content"] for tr in tool_results]),
                    tools_used=[tr["name"] for tr in tool_results],
                )
            else:
                result = LLMAgentResponse(
                    role="assistant", content=None, tools_used=None
                )
        else:
            result = LLMAgentResponse(
                role="assistant", content=response.content, tools_used=None
            )

        # Update both local and global state
        self.update_state(response=result, **state_entry)
        self.update_global_state(name=self.name, action_result=result.content)

        return result
    
    def multi_step_action(
        self,
        steps: List[Dict[str, Any]] = [],
        model_params: Dict[str, Any] = {},
        mode: ExecutionMode = ExecutionMode.CHAIN,
        graph: Graph = None,
        *args,
        **kwargs
    ) -> List[LLMAgentResponse]:
        responses = []
        if mode == ExecutionMode.CHAIN:
            for step in steps:
                response = self.action(
                    state_entry=step.get("state_entry", {}),
                    model_params={**model_params, **step.get("model_params", {})},
                )
                responses.append(StepResult(step=step.get("step", ""), response=response))
        elif mode == ExecutionMode.GRAPH:
            if graph is None:
                raise ValueError("Graph must be provided for GRAPH execution mode.")
            if steps:
                Warning("Steps are ignored in GRAPH execution mode.")
            graph.run(
                start_node=graph.get_start_node(),
                streaming=kwargs.get("streaming", False),
            )
        else:
            raise ValueError(f"Invalid execution mode: {mode}")
        
        return responses

    def update_state(self, *args, **kwargs):
        self.state.update_state(*args, **kwargs)
