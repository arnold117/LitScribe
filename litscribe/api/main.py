"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from litscribe.api.routes.reviews import router as reviews_router
from litscribe.api.routes.memory import router as memory_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LitScribe",
        description="Self-evolving multi-agent literature review engine",
        version="2.0.0",
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

    return app


def run() -> None:
    """Entry point for the litscribe-server CLI script."""
    import uvicorn

    uvicorn.run("litscribe.api.main:create_app", factory=True, host="0.0.0.0", port=8000)
