"""Run the development server.

Usage:
    python scripts/run.py
"""

import uvicorn

from app.core.config.settings import get_settings


def main() -> None:
    """Start uvicorn with settings from environment."""
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.value.lower(),
    )


if __name__ == "__main__":
    main()
