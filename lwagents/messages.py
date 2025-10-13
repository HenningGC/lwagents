from typing import Dict, List, Optional

from pydantic import BaseModel


class LLMAgentResponse(BaseModel):
    role: str  # e.g., "assistant" or "user"
    content: str  # The actual message content
    tool_used: Optional[str] = None  # Optional: Tool used during execution


class LLMAgentRequest(BaseModel):
    content: List[Dict[str, str]]


class LLMEntry(BaseModel):
    AgentRequest: LLMAgentRequest
    AgentResponse: LLMAgentResponse
