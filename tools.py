from abc import ABC, abstractmethod
import inspect
from pydantic import BaseModel, Field
from typing import Any


class BaseTool(ABC):
    @abstractmethod
    def execute(self, *args, **kwargs):
        """Execute the tool with the given arguments."""
        pass

from typing import Any
from pydantic import BaseModel, Field
import inspect

from typing import Any
from pydantic import BaseModel, Field
import inspect

from typing import Any
from pydantic import BaseModel, Field
import inspect

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
        param_type = param.annotation if param.annotation is not inspect.Parameter.empty else Any

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
            **schema_dict,                   # Include Field definitions
        }
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
    FunctionTool.__name__ = func.__name__  # Name the class after the function for clarity

    return FunctionTool()

def create_tool_class(tool_name: str, params: dict):
    """
    Dynamically create a Pydantic model for a tool.
    
    Args:
        tool_name (str): The name of the tool.
        params (dict): The parameters the tool expects, with names as keys and types as values.
    
    Returns:
        BaseModel: A dynamically created Pydantic model class.
    """
    return type(
        tool_name,
        (BaseModel,),
        params
    )

