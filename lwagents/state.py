#from models import Message, History
from .agent import Agent, InvalidAgent
from typing import TypedDict, Annotated, Sequence, List, Optional
from typing_extensions import Self, override
from abc import ABC, abstractmethod

# Abstract Base Class
class State(ABC):
    def __init__(self):
        self.history = None

    @abstractmethod
    def update_state(self, action: str) -> None:
        pass

    @property
    @abstractmethod
    def current_agent(self) -> Optional["Agent"]:
        pass

    @current_agent.setter
    @abstractmethod
    def current_agent(self, agent: "Agent") -> None:
        pass

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
            if step['transition']:
                edge_name, next_node = step['transition']
                print(f"  Transition    : via Edge '{edge_name}' to Node '{next_node}'")
            else:
                print("  Transition    : None")
            print("-" * 50)

# Concrete Implementation
class MainState(State):
    def __init__(self, agent: Agent = None, messages: Optional[List] = []):
        self._agent = agent
        self.history = messages or []
        self.last_update = None

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

    def update_state(self, step_number, node_name, node_kind, command_result, transition, **kwargs) -> None:
        log_entry = log_entry = {
            "step_number": step_number,
            "node_name": node_name,
            "node_kind": node_kind,
            "command_result": command_result,
            "transition": transition,
            **kwargs}
        self.history.append(log_entry)
        self.last_update = log_entry


        

