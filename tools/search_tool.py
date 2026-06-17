"""Web search tool — uses httpx to query a search API."""

import httpx

from app.core.logger.logger import get_logger
from app.domain.tool.base import BaseTool, ToolResult

logger = get_logger(__name__)


class SearchTool(BaseTool):
    """Search the web for information."""

    name = "search"
    description = "Search the web for current information on a topic."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str = "", engine: str = "duckduckgo") -> None:
        self._api_key = api_key
        self._engine = engine

    async def execute(self, query: str = "", **kwargs) -> ToolResult:
        """Execute a web search."""
        if not query:
            return ToolResult(output="", success=False, error="Empty query")

        try:
            # DuckDuckGo instant answer API (no key needed)
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_redirect": "1"},
                )
                data = response.json()

            abstract = data.get("AbstractText", "")
            if abstract:
                return ToolResult(output=abstract)

            # Fallback to related topics
            topics = data.get("RelatedTopics", [])
            if topics:
                results = [
                    t.get("Text", "") for t in topics[:3] if t.get("Text")
                ]
                return ToolResult(output="\n".join(results))

            return ToolResult(output=f"No results found for: {query}")

        except Exception as exc:
            logger.error("search_failed", query=query, error=str(exc))
            return ToolResult(output="", success=False, error=str(exc))
