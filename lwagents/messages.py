from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from anthropic import types as anthropic_types
from openai import types as openai_types


class LLMAgentResponse(BaseModel):
    role: str  # e.g., "assistant" or "user"
    content: str  # The actual message content
    tools_used: Optional[List[str]] = None  # Optional: Tool used during execution


class LLMAgentRequest(BaseModel):
    content: List[Dict[str, str]]


class LLMEntry(BaseModel):
    AgentRequest: LLMAgentRequest
    AgentResponse: LLMAgentResponse


class GPTResponse(BaseModel):
    response_message: str
    @property
    def content(self):
        return self.response_message
    
class GPTToolResponse(BaseModel):
    tool_response: Any


class AnthropicResponse(BaseModel):
    response_message: anthropic_types.Message
    @property
    def content(self):
        return self.response_message.content
    
class AnthropicToolResponse(BaseModel):
    tool_response: anthropic_types.ToolUseBlock

class LLMResponse(BaseModel):
    # when accessed determine if GPT or Anthropic response and then return content accordingly
    response: GPTResponse | AnthropicResponse
    @property
    def content(self):
        if isinstance(self.response, GPTResponse):
            return self.response.content
        elif isinstance(self.response, AnthropicResponse):
            return self.response.content[0].text if type(self.response.content[0]) != anthropic_types.ToolUseBlock else str(self.response.content)
        else:
            return None

class LLMToolResponse(BaseModel):
    results: GPTToolResponse | AnthropicToolResponse