from .messages import *


class AgentLogEntry:
    content: List[List[LLMAgentRequest, LLMAgentResponse]]
