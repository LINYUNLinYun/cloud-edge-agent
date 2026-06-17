"""Base tool interface and registry for the plugin-based tool system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result returned by a tool execution."""

    output: str
    success: bool = True
    error: str | None = None


class BaseTool(ABC):
    """Abstract tool that can be invoked by the agent.

    Every tool must declare a name, description, and parameter schema.
    """

    name: str
    description: str
    parameters: dict  # JSON Schema for the tool's parameters

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Run the tool with the given keyword arguments.

        Returns:
            ToolResult with output string and success flag.
        """
        ...
