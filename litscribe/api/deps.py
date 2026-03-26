"""FastAPI dependency injection helpers."""
from __future__ import annotations

from functools import lru_cache

from litscribe.config import Config
from litscribe.store.unified import UnifiedStore
from litscribe.evolution.memory_manager import MemoryManager
from litscribe.llm.router import LLMRouter


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return a cached Config singleton."""
    cfg = Config()
    cfg.ensure_directories()
    return cfg


_store: UnifiedStore | None = None
_memory: MemoryManager | None = None
_llm: LLMRouter | None = None


async def init_app() -> None:
    """Initialize all application singletons (store, memory, LLM router).

    Called during the FastAPI lifespan startup event.
    """
    global _store, _memory, _llm
    cfg = get_config()
    _store = UnifiedStore(db_path=cfg.db_path, chroma_path=cfg.chroma_path)
    await _store.initialize()
    _memory = MemoryManager(
        db_path=cfg.db_path,
        chroma_path=cfg.chroma_path,
        skills_dir=cfg.skills_dir,
    )
    await _memory.initialize()
    _llm = LLMRouter(cfg)


async def shutdown_app() -> None:
    """Cleanly shut down all application singletons.

    Called during the FastAPI lifespan shutdown event.
    """
    if _store:
        await _store.close()
    if _memory:
        await _memory.close()


def get_store() -> UnifiedStore:
    """Return the UnifiedStore singleton (must be called after init_app)."""
    assert _store is not None, "UnifiedStore not initialized — call init_app() first"
    return _store


def get_memory() -> MemoryManager:
    """Return the MemoryManager singleton (must be called after init_app)."""
    assert _memory is not None, "MemoryManager not initialized — call init_app() first"
    return _memory


def get_llm() -> LLMRouter:
    """Return the LLMRouter singleton (must be called after init_app)."""
    assert _llm is not None, "LLMRouter not initialized — call init_app() first"
    return _llm
