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


class CustomModelError(Exception):
    pass


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
    def generate(self) -> str:
        """
        Concrete subclasses must implement their own generate method,
        """
        pass


# ------------------------------------
# 4. Concrete model loader classes
# ------------------------------------


class ModelLoader:

    @staticmethod
    def load_model(
        model_type: str, instance_params: dict, custom_implementation: Any
    ) -> OpenAI:

        if model_type == "openai":
            return OpenAI(**instance_params)
        elif model_type == "deepseek":
            return OpenAI(**instance_params, base_url="https://api.deepseek.ai/v1")
        elif model_type == "anthropic":
            return anthropic.Anthropic(**instance_params)
        elif model_type == "custom":
            return custom_implementation(**instance_params)


# ----------------------------------
# 5. Concrete model implementations
# ----------------------------------


class GPTModel(BaseLLMModel):
    @override
    def generate(
        self,
        tools: Dict[str, callable] | None = None,
        model_params: Dict[str, Any] = {},
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
        if tools and model_params.get("structure"):
            raise Warning(
                "Tool calling with structured output is currently incompatible!"
            )

        if model_params.get("structure"):
            completion = self._model.responses.parse(
                model=model_params.get("model"),
                messages=model_params.get("prompt"),
                text_format=model_params.get("structure"),
                **model_params,
            )
            return LLMResponse(
                response=GPTResponse(response_message=completion.choices[0].message)
            )
        if tools:
            openai_tools = ToolUtility.get_tools_info_gpt(tools)
            completion = self._model.responses.create(
                tools=openai_tools,
                **model_params,
            )
            # Return the full completion object for tool execution
            return LLMToolResponse(
                results=GPTToolResponse(
                    tool_response=completion, content=completion.output_text
                )
            )

        else:
            completion = self._model.responses.create(
                **model_params,
            )

            return LLMResponse(
                response=GPTResponse(response_message=completion.output_text)
            )


class DeepSeekModel(GPTModel):
    pass


class AnthropicModel(BaseLLMModel):
    @override
    def generate(
        self,
        tools: Dict[str, callable] | None = None,
        model_params: Dict[str, Any] = {},
    ):

        if model_params.get("structure"):
            raise Warning("Structured output is currently incompatible with Anthropic!")

        if tools:
            anthropic_tools = ToolUtility.get_tools_info_anthropic(tools=tools)
            message = self._model.messages.create(
                tools=anthropic_tools,
                **model_params,
            )
            return LLMToolResponse(
                results=AnthropicToolResponse(tool_response=message, content="")
            )
        else:
            message = self._model.messages.create(
                **model_params,
            )

            return LLMResponse(response=AnthropicResponse(response_message=message))

# -------------------------------------------------
# 6. LLMFactory to create model instances on demand
# -------------------------------------------------


def create_model(model_type: str, *args, **kwargs) -> LLMModel:

    custom_model = kwargs.get("custom_model")
    custom_implementation = kwargs.get("custom_implementation")
    loader = ModelLoader.load_model(
        model_type=model_type,
        custom_implementation=custom_implementation,
        *args,
        **kwargs,
    )

    if model_type == "openai":
        return GPTModel(loader)
    elif model_type == "deepseek":
        return DeepSeekModel(loader)
    elif model_type == "anthropic":
        return AnthropicModel(loader)
    elif model_type == "custom":
        if custom_model is None or not custom_implementation:
            raise CustomModelError(
                "custom_model and custom_implementation must be provided for custom model type"
            )
        return custom_model(loader)
