from abc import ABC, abstractmethod

class BaseTool(ABC):
    @abstractmethod
    def execute(self, *args, **kwargs):
        """Execute the tool with the given arguments."""
        pass

def Tool(func):
    class FunctionTool(BaseTool):
        def __init__(self):
            self._function = func

        def execute(self, *args, **kwargs):
            return self._function(*args, **kwargs)

    FunctionTool.__name__ = func.__name__  # Name the class after the function for clarity
    return FunctionTool()