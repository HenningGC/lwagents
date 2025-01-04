#from models import Message, History
from agent import Agent, InvalidAgent
from typing import TypedDict, Annotated, Sequence, List, Optional
from typing_extensions import Self, override
from abc import ABC, abstractmethod

# Abstract Base Class
class State(ABC):
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

# Concrete Implementation
class MainState(State):
    def __init__(self, messages: Optional[List] = []):
        self._agent = None
        self.history = messages or []

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

    def update_state(self, action: str) -> None:
        self.history.append(action)


        

