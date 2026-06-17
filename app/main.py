"""FastAPI application entrypoint.

Usage:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dependencies.deps import create_components
from app.api.routers import chat, health
from app.core.config.settings import get_settings
from app.core.logger.logger import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — initialize components on startup."""
    settings = get_settings()
    setup_logging(settings.log_level)

    components = create_components()
    app.state.components = components

    yield

    # Cleanup (close connections, flush caches, etc.)
    components.cache.clear()


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Privacy-First Cloud-Edge Collaborative AI Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router)
    app.include_router(chat.router)

    return app


app = create_app()
