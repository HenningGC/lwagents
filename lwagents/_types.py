from pydantic import BaseModel
from typing import Any, List, Optional
from enum import Enum

from lwagents.messages import LLMAgentResponse

class ExecutionMode(str, Enum):
    CHAIN = "chain"
    GRAPH = "graph"

class StepResult(BaseModel):
    step: str
    response: LLMAgentResponse