"""Local LLM client using HuggingFace Transformers.

Runs models locally on GPU (or CPU fallback).
No API key required — fully offline inference.
"""

import time
from collections.abc import AsyncIterator

from app.core.logger.logger import get_logger
from app.domain.llm.llm_client import (
    LLMClient,
    LLMMessage,
    LLMResponse,
    StreamChunk,
    TokenUsage,
)

logger = get_logger(__name__)

# Default model for local inference
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class LocalLLMClient(LLMClient):
    """LLM client that runs models locally using HuggingFace Transformers.

    Supports any HuggingFace model that fits in GPU memory.
    For RTX 5060 (8GB), models up to ~7B parameters work well with quantization.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        load_in_4bit: bool = False,
    ) -> None:
        """Initialize local LLM client.

        Args:
            model_name: HuggingFace model name or path.
            device: Device to use ('auto', 'cuda', 'cpu').
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            load_in_4bit: Use 4-bit quantization (saves VRAM).
        """
        self.provider = "local"
        self.model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._load_in_4bit = load_in_4bit
        self._device = device
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        """Lazily load the model and tokenizer."""
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                logger.info("local_llm_loading", model=self.model_name)

                # Load tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )

                # Load model with optional quantization
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16,
                }

                if self._load_in_4bit:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )

                if self._device == "auto":
                    model_kwargs["device_map"] = "auto"
                else:
                    model_kwargs["device_map"] = self._device

                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs,
                )

                logger.info(
                    "local_llm_loaded",
                    model=self.model_name,
                    device=str(self._model.device),
                    dtype=str(next(self._model.parameters()).dtype),
                )

            except Exception as exc:
                logger.error("local_llm_load_failed", error=str(exc))
                raise

    def _convert_messages(self, messages: list[LLMMessage]) -> str:
        """Convert LLM messages to chat template format."""
        self._ensure_model()

        # Use tokenizer's chat template if available
        if hasattr(self._tokenizer, "apply_chat_template"):
            chat_messages = [
                {"role": m.role, "content": m.content}
                for m in messages
            ]
            return self._tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback: simple concatenation
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    async def invoke(self, messages: list[LLMMessage]) -> LLMResponse:
        """Generate a response from the local model."""
        import torch

        self._ensure_model()
        start_time = time.monotonic()

        # Prepare input
        prompt = self._convert_messages(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature if self._temperature > 0 else None,
                do_sample=self._temperature > 0,
                top_p=0.9 if self._temperature > 0 else None,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Calculate token usage
        prompt_tokens = inputs["input_ids"].shape[1]
        completion_tokens = len(new_tokens)

        logger.info(
            "local_llm_invoke",
            model=self.model_name,
            latency_ms=round(elapsed_ms, 1),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return LLMResponse(
            content=response_text.strip(),
            model=self.model_name,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def stream_invoke(
        self, messages: list[LLMMessage]
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response from the local model."""
        # For simplicity, use non-streaming and yield in chunks
        response = await self.invoke(messages)
        chunk_size = 10  # characters per chunk
        for i in range(0, len(response.content), chunk_size):
            yield StreamChunk(
                delta=response.content[i:i+chunk_size],
                finish_reason="stop" if i + chunk_size >= len(response.content) else None,
            )

    async def think(self, messages: list[LLMMessage]) -> LLMResponse:
        """Extended thinking invocation."""
        # Add thinking prompt prefix
        think_prefix = LLMMessage(
            role="system",
            content="Think step by step before answering.",
        )
        return await self.invoke([think_prefix] + messages)

    async def embedding(self, text: str) -> list[float]:
        """Not supported for local LLM — use MiniLMEmbedder instead."""
        raise NotImplementedError(
            "LocalLLMClient does not support embeddings. "
            "Use MiniLMEmbedder for embeddings."
        )
