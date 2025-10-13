# from models import Message, History
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, List, Optional, Sequence, TypedDict

from typing_extensions import Self, override

if TYPE_CHECKING:
    from .agent import Agent


class InvalidSchemaError(Exception):
    pass


class InvalidAgent(Exception):
    pass


# Abstract Base Class
class State(ABC):
    def __init__(self, initial_history=None):
        self.history = initial_history or []
        self.last_update = None

    @abstractmethod
    def update_state(self, action: str) -> None:
        pass

    @override
    def print_history(self) -> None:
        """
        Prints the State execution history in a more human-readable format.

        Args:
            history (list): The history log from the graph's execution.
        """
        for step in self.history:
            print(step)

    def update_state(self) -> None:
        pass

    def enforce_schema(self, state_entry: dict) -> None:
        if self.history and self.history[-1].keys() != state_entry.keys():
            raise InvalidSchemaError(
                f"State entry keys do not match the schema. Expected keys: {self.history[-1].keys()}, Got keys: {state_entry.keys()}"
            )

    def get_history(self) -> List:
        return self.history

    def get_last_entry(self) -> dict:
        return self.history[-1]


# Concrete Implementation
class AgentState(State):
    """
    Agent-specific state that extends the base State class.

    Args:
        agent (Agent, optional): The current agent associated with the state.
        initial_history (list, optional): The initial history for the state.
    """

    def __init__(self, initial_history: Optional[List] = []):
        super().__init__(initial_history=initial_history)
        self.history = initial_history or []

    @property
    def current_agent(self) -> Optional["Agent"]:
        return self._agent

    @current_agent.setter
    def current_agent(self, agent: "Agent") -> None:
        if isinstance(agent, Agent):
            self._agent = agent
        else:
            raise InvalidAgent(
                f"Agent {agent} must be of type Agent, got {type(agent)} instead."
            )

    @override
    def print_history(self) -> None:
        return super().print_history()

    @override
    def update_state(self, enforce_schema: Optional[bool] = False, **kwargs) -> None:
        """
        Updates the agent state with a new log entry.
        """
        state_entry = {**kwargs}
        if enforce_schema:
            self.enforce_schema(state_entry)
        self.history.append(state_entry)
        self.last_update = state_entry


class GraphState(State):

    def __init__(self, initial_history: Optional[List] = []):
        super().__init__(initial_history=initial_history)

    def print_history(self) -> None:
        """
        Prints the State execution history in a more human-readable format.

        Args:
            history (list): The history log from the graph's execution.
        """
        print("\State Execution History")
        print("=" * 50)
        for step in self.history:
            print(f"Step {step['step_number']}:")
            print(f"  Node Name     : {step['node_name']}")
            print(f"  Node Kind     : {step['node_kind'].name}")
            print(f"  Command Result: {step['command_result']}")
            if step["transition"]:
                edge_name, next_node = step["transition"]
                print(f"  Transition    : via Edge '{edge_name}' to Node '{next_node}'")
            else:
                print("  Transition    : None")
            additional_params = {
                k: v
                for k, v in step.items()
                if k
                not in {
                    "step_number",
                    "node_name",
                    "node_kind",
                    "command_result",
                    "transition",
                }
            }
            if additional_params:
                print("  Additional Parameters:")
                for key, value in additional_params.items():
                    print(f"    {key}: {value}")
            print("-" * 50)

    @override
    def update_state(
        self,
        step_number,
        node_name,
        node_kind,
        command_result,
        transition,
        enforce_schema: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """
        Updates the agent state with a new log entry.
        """
        state_entry = {
            "step_number": step_number,
            "node_name": node_name,
            "node_kind": node_kind,
            "command_result": command_result,
            "transition": transition,
            **kwargs,
        }
        if enforce_schema:
            self.enforce_schema(state_entry)
        self.history.append(state_entry)


class GlobalAgentState(State):
    def __init__(self, initial_history=None):
        super().__init__(initial_history)

    def print_history(self) -> None:
        return super().print_history()

    @override
    def update_state(
        self,
        agent_name,
        agent_kind,
        entry,
        enforce_schema: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """
        Updates the agent state with a new log entry.
        """
        state_entry = {
            "agent_name": agent_name,
            "agent_kind": agent_kind,
            "entry": entry,
            **kwargs,
        }
        if enforce_schema:
            self.enforce_schema(state_entry)
        self.history.append(state_entry)

    @override
    def get_last_entry(self, inner_content=False) -> dict:
        if not self.history:
            raise IndexError("History is empty")
        if inner_content:
            return {
                "request": self.history[-1]["entry"].AgentRequest.content,
                "response": self.history[-1]["entry"].AgentResponse.content,
            }
        return self.history[-1]


_global_agent_state = GlobalAgentState()


def get_global_agent_state() -> GlobalAgentState:
    """
    Returns the global agent state instance.
    This allows agents to access and update the global state.
    """
    return _global_agent_state


def reset_global_agent_state() -> None:
    """
    Resets the global agent state to a fresh instance.
    Useful for testing or when you want to clear the global history.
    """
    global _global_agent_state
    _global_agent_state = GlobalAgentState()
