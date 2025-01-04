from openai import OpenAI
import openai
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Any, Protocol, List, Dict
from typing_extensions import Self, override
import os
from dotenv import load_dotenv


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

class GPTModelLoader:

    def load_model(api_key, *args, **kwargs)->OpenAI:
        
        return OpenAI(api_key= api_key, *args)

class LLamaModelLoader:
    def __init__(self, model_path: str):
        self._model_path = model_path

    def load_model(self):
        # Pseudocode for loading a LLaMA model
        print(f"[LLamaModelLoader] Loading LLaMA model from {self._model_path}...")
        # return LLamaModelObject(...loaded from path...)
        return "LLamaModelObject"

# ----------------------------------
# 5. Concrete model implementations
# ----------------------------------

class GPTModel(BaseLLMModel):
    @override
    def generate(self, model_name: str = "gpt-4o-mini", messages: List[Dict[str, str]] | None = None, structure: BaseModel | None = None, tools: Dict[str,callable] | None = None) -> str:
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
            raise Warning("Tool calling with structured output is currently incompatible!")
        openai_tools = []
        if tools:
            for tool in tools or []:
                func = tools[tool]
                tool_schema = func.schema
                openai_tool = openai.pydantic_function_tool(tool_schema)
                openai_tools.append(openai_tool)
            
        if structure:
            completion = self._model.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=structure
        )
        else:
            if tools:
                completion = self._model.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="required"
                )

            else:
                completion = self._model.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages)

            
        return completion.choices[0].message#.content
        # except Exception as e:
        #     return f"Error: {str(e)}"

class LLamaModel(BaseLLMModel):
    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        print("[LLamaModel] Generating text...")
        # Pseudocode for calling the actual LLaMA model:
        # output = self._model.generate(prompt, max_tokens)
        # return output
        return f"LLaMA response to '{prompt[:20]}...' with max_tokens={max_tokens}"


# -------------------------------------------------
# 6. LLMFactory to create model instances on demand
# -------------------------------------------------

class LLMFactory:
    def create_model(self, model_type: str, *args, **kwargs) -> LLMModel:
        if model_type.lower() == "gpt":
            loader = GPTModelLoader.load_model(api_key=kwargs['openai_api_key'], *args, **kwargs)
            return GPTModel(loader)
        elif model_type.lower() == "llama":
            loader = LLamaModelLoader(kwargs['model_path'])
            return LLamaModel(loader)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        

class BaseMessage(BaseModel):
    message: str

class GPTMessage(BaseMessage):
    pass

class History(BaseModel):
    messages: List[BaseMessage]