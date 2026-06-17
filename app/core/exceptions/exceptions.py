"""Unified exception hierarchy for the application."""


class BaseAppException(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


class LLMException(BaseAppException):
    """Errors from LLM client operations."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        details: dict | None = None,
    ) -> None:
        self.provider = provider
        super().__init__(message, details)


class ToolException(BaseAppException):
    """Errors from tool execution."""

    def __init__(
        self, message: str, tool_name: str = "", details: dict | None = None
    ) -> None:
        self.tool_name = tool_name
        super().__init__(message, details)


class RAGException(BaseAppException):
    """Errors from RAG pipeline."""

    pass


class MemoryException(BaseAppException):
    """Errors from memory operations."""

    pass


class PrivacyException(BaseAppException):
    """Errors from privacy engine."""

    pass


class PolicyException(BaseAppException):
    """Errors from policy / routing decisions."""

    pass
