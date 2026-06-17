"""Security utilities: API key validation, request signing, etc."""

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.core.config.settings import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    """Verify the request carries a valid API key (for external-facing endpoints)."""
    settings = get_settings()
    if settings.debug:
        return "debug"
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    # TODO: implement proper key validation against a stored key / DB
    return api_key
