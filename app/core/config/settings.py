"""Application settings loaded from environment variables."""

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EdgeLLMSettings(BaseSettings):
    """Edge (local) LLM configuration."""

    model_config = {"env_prefix": "EDGE_LLM_"}

    provider: str = Field(default="ollama", description="Local LLM provider")
    base_url: str = Field(
        default="http://localhost:11434/v1", description="Ollama OpenAI-compatible URL"
    )
    model_name: str = Field(
        default="qwen2.5:7b-instruct", description="Local model name"
    )
    api_key: str = Field(default="ollama", description="API key (ollama uses dummy key)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)


class CloudLLMSettings(BaseSettings):
    """Cloud LLM configuration."""

    model_config = {"env_prefix": "CLOUD_LLM_"}

    provider: str = Field(default="deepseek", description="Cloud LLM provider")
    base_url: str = Field(
        default="https://api.deepseek.com/v1", description="Cloud API base URL"
    )
    model_name: str = Field(default="deepseek-chat", description="Cloud model name")
    api_key: str = Field(default="", description="Cloud API key")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)


class PrivacySettings(BaseSettings):
    """Privacy engine configuration."""

    model_config = {"env_prefix": "PRIVACY_"}

    default_epsilon: float = Field(
        default=8.0, description="Default privacy budget (ε) per session"
    )
    sanitize_cost_epsilon: float = Field(
        default=0.3, description="ε cost for sanitize-cloud-restore mode"
    )
    sketch_refine_cost_epsilon: float = Field(
        default=1.5, description="ε cost for sketch-refine mode"
    )
    slm_model: str = Field(
        default="qwen2.5:1.5b", description="SLM for privacy judgment"
    )
    enable_ner: bool = Field(default=True, description="Enable NER-based detection")


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""

    model_config = {"env_prefix": "VECTOR_STORE_"}

    provider: str = Field(default="qdrant", description="Vector DB provider")
    host: str = Field(default="localhost", description="Qdrant host")
    port: int = Field(default=6333, description="Qdrant port")
    collection_name: str = Field(
        default="agent_memory", description="Default collection"
    )


class Settings(BaseSettings):
    """Root application settings."""

    model_config = {"env_prefix": "APP_", "env_file": ".env", "extra": "ignore"}

    app_name: str = "CloudEdgeAgent"
    debug: bool = Field(default=False)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    host: str = Field(default="0.0.0.0")  # noqa: S104
    port: int = Field(default=8000)

    edge_llm: EdgeLLMSettings = Field(default_factory=EdgeLLMSettings)
    cloud_llm: CloudLLMSettings = Field(default_factory=CloudLLMSettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)


def get_settings() -> Settings:
    """Create settings instance (cached via lru_cache if desired)."""
    return Settings()
