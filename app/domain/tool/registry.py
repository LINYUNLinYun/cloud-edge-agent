"""Tool registry — register, discover, and execute tools by name."""

from app.core.exceptions.exceptions import ToolException
from app.core.logger.logger import get_logger
from app.domain.tool.base import BaseTool, ToolResult

logger = get_logger(__name__)


class ToolRegistry:
    """Central registry for all available tools.

    Usage:
        registry = ToolRegistry()
        registry.register(SearchTool())
        result = await registry.execute("search", query="hello")
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if tool.name in self._tools:
            logger.warning("tool_overridden", tool=tool.name)
        self._tools[tool.name] = tool
        logger.info("tool_registered", tool=tool.name)

    def get(self, name: str) -> BaseTool:
        """Retrieve a tool by name."""
        if name not in self._tools:
            raise ToolException(f"Tool '{name}' not found", tool_name=name)
        return self._tools[name]

    def list_tools(self) -> list[dict]:
        """Return metadata for all registered tools (for LLM tool-calling)."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(name)
        logger.info("tool_execute", tool=name, args=list(kwargs.keys()))
        return await tool.execute(**kwargs)
