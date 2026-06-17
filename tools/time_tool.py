"""Time tool — return current date/time information."""

from datetime import UTC, datetime

from app.domain.tool.base import BaseTool, ToolResult


class TimeTool(BaseTool):
    """Get the current date and time."""

    name = "time"
    description = "Get the current date and time in UTC."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self, **kwargs) -> ToolResult:
        """Return current UTC time."""
        now = datetime.now(UTC)
        return ToolResult(
            output=now.strftime("%Y-%m-%d %H:%M:%S UTC")
        )
