"""OpenAI-compatible LLM client — works with Ollama, DeepSeek, Qwen, etc."""

import time
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.core.exceptions.exceptions import LLMException
from app.core.logger.logger import get_logger
from app.domain.llm.llm_client import (
    LLMClient,
    LLMMessage,
    LLMResponse,
    StreamChunk,
    TokenUsage,
)

logger = get_logger(__name__)


class OpenAICompatibleClient(LLMClient):
    """LLM client using the OpenAI-compatible chat/completions API.

    Works with: Ollama (local), DeepSeek, Qwen, OpenAI, vLLM, etc.
    Only difference between providers is base_url + api_key + model_name.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    async def invoke(self, messages: list[LLMMessage]) -> LLMResponse:
        """Send a conversation and return the full response."""
        openai_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        start_time = time.monotonic()
        try:
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,  # type: ignore[arg-type]
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            logger.error(
                "llm_invoke_failed", provider=self.provider, error=str(exc)
            )
            raise LLMException(
                f"LLM invocation failed: {exc}", provider=self.provider
            ) from exc

        elapsed_ms = (time.monotonic() - start_time) * 1000
        choice = response.choices[0]
        usage = (
            TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
            if response.usage
            else None
        )

        logger.info(
            "llm_invoke",
            provider=self.provider,
            model=self.model_name,
            latency_ms=round(elapsed_ms, 1),
            tokens=usage.total_tokens if usage else None,
        )

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=usage,
            finish_reason=choice.finish_reason,
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    async def stream_invoke(
        self, messages: list[LLMMessage]
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response chunk by chunk."""
        openai_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        try:
            stream = await self._client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,  # type: ignore[arg-type]
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield StreamChunk(
                        delta=delta.content,
                        finish_reason=chunk.choices[0].finish_reason,
                    )
        except Exception as exc:
            logger.error("llm_stream_failed", provider=self.provider, error=str(exc))
            raise LLMException(
                f"LLM streaming failed: {exc}", provider=self.provider
            ) from exc

    async def think(self, messages: list[LLMMessage]) -> LLMResponse:
        """Extended-thinking invocation — prepends a thinking prompt."""
        think_prefix = LLMMessage(
            role="system",
            content=(
                "Think step by step before answering. "
                "Show your reasoning in <thinking> tags."
            ),
        )
        return await self.invoke([think_prefix] + messages)

    async def embedding(self, text: str) -> list[float]:
        """Compute an embedding vector using the model's embedding endpoint."""
        try:
            response = await self._client.embeddings.create(
                model=self.model_name,
                input=text,
            )
            return response.data[0].embedding
        except Exception as exc:
            logger.error("llm_embed_failed", provider=self.provider, error=str(exc))
            raise LLMException(
                f"Embedding failed: {exc}", provider=self.provider
            ) from exc
