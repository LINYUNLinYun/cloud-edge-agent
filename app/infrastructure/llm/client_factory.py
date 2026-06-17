"""Factory to create LLMClient instances from settings."""

from app.core.config.settings import CloudLLMSettings, EdgeLLMSettings
from app.domain.llm.llm_client import LLMClient
from app.infrastructure.llm.openai_compatible_client import OpenAICompatibleClient


def create_edge_llm_client(settings: EdgeLLMSettings) -> LLMClient:
    """Create the local/edge LLM client (Ollama or vLLM)."""
    return OpenAICompatibleClient(
        provider=settings.provider,
        model_name=settings.model_name,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def create_cloud_llm_client(settings: CloudLLMSettings) -> LLMClient:
    """Create the cloud LLM client (DeepSeek, Qwen, etc.)."""
    return OpenAICompatibleClient(
        provider=settings.provider,
        model_name=settings.model_name,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
