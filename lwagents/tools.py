import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, get_type_hints
import json

from pydantic import BaseModel, Field
import openai

from lwagents.messages import (
    AnthropicResponse,
    AnthropicToolResponse,
    GPTResponse,
    GPTToolResponse,
)


class ToolExecutionError(Exception):
    pass


class ToolExecutionResult(BaseModel):
    id: str
    name: str
    content: Any


class ToolsExecutionResults(BaseModel):
    results: List[ToolExecutionResult]


class BaseTool(ABC):
    @abstractmethod
    def execute(self, *args, **kwargs):
        """Execute the tool with the given arguments."""
        pass


def Tool(func):
    """
    Decorator to create a tool based on a function.
    Automatically generates a schema for the tool using Pydantic.
    """
    # Generate the schema dynamically
    signature = inspect.signature(func)
    schema_dict = {}
    annotations = {}
    for param_name, param in signature.parameters.items():
        # Ensure every parameter has a type annotation
        param_type = (
            param.annotation if param.annotation is not inspect.Parameter.empty else Any
        )

        # Determine the default value or make it required
        if param.default is not inspect.Parameter.empty:
            schema_dict[param_name] = Field(default=param.default)
        else:
            schema_dict[param_name] = Field(...)

        # Add to annotations for Pydantic
        annotations[param_name] = param_type

    # Dynamically create a Pydantic model for the tool's schema
    ToolSchema = type(
        f"{func.__name__}",
        (BaseModel,),
        {
            "__annotations__": annotations,  # Explicitly define type annotations
            **schema_dict,  # Include Field definitions
        },
    )

    # Define the tool class
    class FunctionTool(BaseTool):
        schema = ToolSchema

        def __init__(self):
            self._function = func

        def execute(self, **kwargs):
            # Validate input arguments using the generated schema
            validated_args = self.schema(**kwargs)
            return self._function(**validated_args.dict())

    FunctionTool.__name__ = (
        func.__name__
    )  # Name the class after the function for clarity

    return FunctionTool()


class ToolUtility:
    @staticmethod
    def get_tools_info_gpt(tools: Dict[str, callable]) -> List[Dict]:
        """
        Extracts and returns the schema information of the provided tools.

        Args:
            tools (Dict[str, callable]): A dictionary of tool names to their callable implementations.

        Returns:
            List[Dict]: A list of tool schemas in OpenAI-compatible format.
        """
        model_tools = []
        for tool in tools:
            func = tools[tool]
            tool_schema = func.schema
            model_tool = openai.pydantic_function_tool(tool_schema)

            # Extract the function definition from the pydantic tool format
            function_def = model_tool["function"]

            # Transform properties to remove 'title' and 'additionalProperties'
            simplified_properties = {}
            for param_name, param_schema in function_def["parameters"][
                "properties"
            ].items():
                simplified_properties[param_name] = {"type": param_schema["type"]}
                # Add description if available (using title as description if present)
                if "title" in param_schema and param_schema["title"] != param_name:
                    simplified_properties[param_name]["description"] = param_schema[
                        "title"
                    ]

            # Create simplified tool format
            simplified_tool = {
                "type": "function",
                "name": function_def["name"],
                "description": function_def.get("description", function_def["name"]),
                "parameters": {
                    "type": "object",
                    "properties": simplified_properties,
                    "required": function_def["parameters"].get("required", []),
                },
            }

            model_tools.append(simplified_tool)
        return model_tools

    @staticmethod
    def get_tools_info_anthropic(tools: Dict[str, callable]) -> List[BaseModel]:
        """
        Extracts and returns the schema information of the provided tools.

        Args:
            tools (Dict[str, callable]): A dictionary of tool names to their callable implementations.

        Returns:
            List[BaseModel]: A list of Pydantic models representing the tools' schemas.
        """
        model_tools = []
        for tool in tools:
            func = tools[tool]
            tool_schema = func.schema
            model_tool = openai.pydantic_function_tool(tool_schema)

            model_tool_function = model_tool["function"]
            model_tool_function_params = model_tool_function["parameters"]["properties"]

            # Transform properties to remove 'title' and keep only type and description
            anthropic_properties = {}
            for param_name, param_schema in model_tool_function_params.items():
                anthropic_properties[param_name] = {
                    "type": param_schema["type"],
                }
                # Add description if it exists (title can serve as description)
                if "title" in param_schema:
                    anthropic_properties[param_name]["description"] = param_schema[
                        "title"
                    ]

            anthropic_tool = {
                "name": model_tool_function["name"],
                "description": model_tool_function.get(
                    "description", model_tool_function["name"]
                ),
                "input_schema": {
                    "type": model_tool_function["parameters"]["type"],
                    "properties": anthropic_properties,
                    "required": model_tool_function["parameters"].get("required", []),
                },
            }

            model_tools.append(anthropic_tool)
        return model_tools

    @classmethod
    def execute_from_response(cls, tool_response: Any, tools: dict) -> Any:
        if type(tool_response.results) == GPTToolResponse:
            return cls.execute_gpt_tools_from_response(
                response=tool_response.results, tools=tools
            )
        elif type(tool_response.results) == AnthropicToolResponse:
            return cls.execute_anthropic_tools_from_response(
                response=tool_response.results, tools=tools
            )
        else:
            raise ValueError("Unsupported response type")

    @classmethod
    def execute_gpt_tools_from_response(cls, response: Any, tools: dict) -> Any:
        """
        Execute tools from OpenAI Responses API format.

        Args:
            response: LLMResponse object containing the OpenAI response
            tools: Dictionary of tool names to tool instances

        Returns:
            ToolsExecutionResults with executed tool results or None
        """
        tool_results = []

        # Handle new OpenAI Responses API format (response.output)
        if hasattr(response, "tool_response"):
            # This is an LLMResponse wrapping a GPTResponse
            # We need to check if it's a string (text response) or has output items
            response_content = response.tool_response
            # Check if it's an OpenAI Response object with output
            if hasattr(response_content, "output"):
                for item in response_content.output:
                    if hasattr(item, "type") and item.type == "function_call":
                        tool_call = item
                        tool_name = tool_call.name
                        tool_args = json.loads(tool_call.arguments)

                        if tool_name in tools:
                            if tool_args:
                                function_response = tools[tool_name].execute(
                                    **tool_args
                                )
                            else:
                                function_response = tools[tool_name].execute()

                            tool_result = ToolExecutionResult(
                                id=tool_call.id,
                                name=tool_name,
                                content=function_response,
                            )
                            tool_results.append(tool_result)
                        else:
                            raise ToolExecutionError(f"Tool {tool_name} not found!")

                return (
                    ToolsExecutionResults(results=tool_results)
                    if tool_results
                    else None
                )

        # Handle legacy format (response.tool_calls) for backwards compatibility
        # if hasattr(response, 'tool_calls') and response.tool_calls:
        #     for tool_call in response.tool_calls:
        #         tool_name = tool_call.function.name
        #         tool_args = json.loads(tool_call.function.arguments)
        #         if tool_name in tools:
        #             if tool_args:
        #                 tool_results.append(tools[tool_name].execute(**tool_args))
        #             else:
        #                 tool_results.append(tools[tool_name].execute())
        #         else:
        #             raise ToolExecutionError(f"Tool {tool_name} not found!")
        #     return ToolsExecutionResults(results=tool_results)

        # return None

    @classmethod
    def execute_anthropic_tools_from_response(cls, response: Any, tools: dict) -> Any:
        tool_results = []
        if response.tool_response.stop_reason == "tool_use":
            for c in response.content:
                if c.type == "tool_use":
                    tool_name = c.name
                    tool_args = c.input

                    if tool_name in tools:
                        if tool_args:
                            function_response = tools[tool_name].execute(**tool_args)
                            tool_result = ToolExecutionResult(
                                id=c.id,
                                name=tool_name,
                                content=function_response,
                            )
                            tool_results.append(tool_result)
                        else:
                            function_response = tools[tool_name].execute()
                            tool_result = ToolExecutionResult(
                                id=c.id,
                                name=tool_name,
                                content=function_response,
                            )
                            tool_results.append(tool_result)
                    else:
                        raise ToolExecutionError(f"Tool {tool_name} not found!")

            return ToolsExecutionResults(results=tool_results)
        else:
            return None
