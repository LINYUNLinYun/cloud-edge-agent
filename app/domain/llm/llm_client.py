"""Abstract LLM client interface — all LLM providers implement this."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class LLMMessage:
    """A single message in a conversation."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: str | None = None


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""

    content: str
    model: str
    usage: "TokenUsage | None" = None
    finish_reason: str | None = None
    raw: dict | None = None


@dataclass
class TokenUsage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""

    delta: str
    finish_reason: str | None = None


class LLMClient(ABC):
    """Unified LLM client interface.

    All LLM providers (Ollama, DeepSeek, Qwen, etc.) implement this.
    Domain / Agent code depends ONLY on this interface.
    """

    provider: str
    model_name: str

    @abstractmethod
    async def invoke(self, messages: list[LLMMessage]) -> LLMResponse:
        """Send a conversation and return the full response.

        Args:
            messages: conversation history in OpenAI message format.

        Returns:
            LLMResponse with generated text and metadata.
        """
        ...

    @abstractmethod
    async def stream_invoke(
        self, messages: list[LLMMessage]
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response chunk by chunk.

        Args:
            messages: conversation history.

        Yields:
            StreamChunk objects as they arrive.
        """
        ...

    @abstractmethod
    async def think(self, messages: list[LLMMessage]) -> LLMResponse:
        """Extended-thinking / chain-of-thought invocation.

        May include scratchpad / reasoning trace in the response.
        Default implementation delegates to invoke(); providers may override.
        """
        ...

    @abstractmethod
    async def embedding(self, text: str) -> list[float]:
        """Compute an embedding vector for the given text.

        Args:
            text: input text to embed.

        Returns:
            Float vector.
        """
        ...
