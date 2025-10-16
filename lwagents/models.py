import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Protocol

import openai
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from pydantic import BaseModel
from typing_extensions import Self, override
import json
from .tools import ToolUtility

from .messages import (
    AnthropicToolResponse,
    GPTResponse,
    AnthropicResponse,
    GPTResponse,
    GPTToolResponse,
    LLMResponse,
    LLMToolResponse,
)

# -------------------------------
# 1. The LLMModel interface
# -------------------------------


class LLMModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text given a prompt."""
        pass


# ------------------------------------
# 2. A Protocol for Model Loaders
# ------------------------------------
class ModelLoader(Protocol):
    def load_model(self) -> Any:
        """Load and return the internal model object."""
        pass


# ---------------------------------
# 3. Base class for LLM models
# ---------------------------------
class BaseLLMModel(LLMModel):
    """
    An abstract base class to share common functionality
    among various LLM model implementations.
    """

    def __init__(self, model: ModelLoader):
        self._model = model

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """
        Concrete subclasses must implement their own generate method,
        """
        pass


# ------------------------------------
# 4. Concrete model loader classes
# ------------------------------------


class ModelLoader:

    @staticmethod
    def load_model(model_type: str, api_key: str, *args, **kwargs) -> OpenAI:

        if model_type == "openai":
            return OpenAI(api_key=api_key, *args, **kwargs)
        elif model_type == "deepseek":
            return OpenAI(
                api_key=api_key, base_url="https://api.deepseek.com", *args, **kwargs
            )
        elif model_type == "anthropic":
            return anthropic.Anthropic(api_key=api_key, *args, **kwargs)


# ----------------------------------
# 5. Concrete model implementations
# ----------------------------------


class GPTModel(BaseLLMModel):
    @override
    def generate(
        self,
        model_name: str,
        prompt: List[Dict[str, str]] | None = None,
        structure: BaseModel | None = None,
        tools: Dict[str, callable] | None = None,
        system: str | None = None,
        *args,
        **kwargs,
    ):
        """
        Generates a response using the LLM, dynamically integrating tools.

        Args:
            model_name (str): The name of the LLM model.
            messages (List[Dict[str, str]]): The conversation messages.
            tools (List[BaseTool]): A list of tools to integrate into the LLM.

        Returns:
            str: The model's response or tool execution result.
        """
        # try:
        if tools and structure:
            raise Warning(
                "Tool calling with structured output is currently incompatible!"
            )

        if structure:
            completion = self._model.responses.parse(
                model=model_name, messages=prompt, text_format=structure, **kwargs
            )
            return LLMResponse(
                response=GPTResponse(response_message=completion.choices[0].message)
            )
        if tools:
            openai_tools = ToolUtility.get_tools_info_gpt(tools)
            completion = self._model.responses.create(
                model=model_name,
                instructions=system,
                input=prompt,
                tools=openai_tools if tools else None,
                tool_choice="required",
                **kwargs,
            )
            # Return the full completion object for tool execution
            return LLMToolResponse(
                results=GPTToolResponse(
                    tool_response=completion, content=completion.output_text
                )
            )

        else:
            completion = self._model.responses.create(
                model=model_name,
                instructions=system,
                input=prompt,
                **kwargs,
            )

            return LLMResponse(
                response=GPTResponse(response_message=completion.output_text)
            )
        # except Exception as e:
        #     return f"Error: {str(e)}"


class DeepSeekModel(GPTModel):
    pass


class AnthropicModel(BaseLLMModel):
    @override
    def generate(
        self,
        model_name: str,
        prompt: List[Dict[str, str]] | None = None,
        tools: Dict[str, callable] | None = None,
        system: str = None,
        *args,
        **kwargs,
    ):

        if kwargs.get("structure"):
            raise Warning("Structured output is currently incompatible with Anthropic!")

        if tools:
            anthropic_tools = ToolUtility.get_tools_info_anthropic(tools=tools)
            message = self._model.messages.create(
                model=model_name,
                system=system,
                messages=prompt,
                tools=anthropic_tools,
                max_tokens=kwargs.get("max_tokens", 200),
                *args,
                **kwargs,
            )
            return LLMToolResponse(
                results=AnthropicToolResponse(tool_response=message, content="")
            )
        else:
            message = self._model.messages.create(
                model=model_name,
                system=system,
                messages=prompt,
                max_tokens=kwargs.get("max_tokens", 200),
                *args,
                **kwargs,
            )

            return LLMResponse(response=AnthropicResponse(response_message=message))


# -------------------------------------------------
# 6. LLMFactory to create model instances on demand
# -------------------------------------------------


def create_model(model_type: str, *args, **kwargs) -> LLMModel:

    loader = ModelLoader.load_model(model_type=model_type, *args, **kwargs)

    if model_type == "openai":
        return GPTModel(loader)
    elif model_type == "deepseek":
        return DeepSeekModel(loader)
    elif model_type == "anthropic":
        return AnthropicModel(loader)
