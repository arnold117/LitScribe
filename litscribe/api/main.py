"""FastAPI application factory."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from litscribe.api.routes.reviews import router as reviews_router
from litscribe.api.routes.memory import router as memory_router
from litscribe.api.routes.sessions import router as sessions_router
from litscribe.api.websocket import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: initialize singletons on startup, close on shutdown.

    When LITSCRIBE_TESTING=1 the lifespan is a no-op so that unit tests can
    instantiate the app without a real database or LLM credentials.
    """
    if os.getenv("LITSCRIBE_TESTING") != "1":
        from litscribe.api.deps import init_app, shutdown_app
        await init_app()
        try:
            yield
        finally:
            await shutdown_app()
    else:
        yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LitScribe",
        description="Self-evolving multi-agent literature review engine",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(reviews_router)
    app.include_router(memory_router)
    app.include_router(sessions_router)
    app.include_router(ws_router)

    return app


def run() -> None:
    """Entry point for the litscribe-server CLI script."""
    import uvicorn

    uvicorn.run("litscribe.api.main:create_app", factory=True, host="0.0.0.0", port=8000)
