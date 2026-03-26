"""FastAPI dependency injection helpers."""
from __future__ import annotations

from functools import lru_cache

from litscribe.config import Config


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return a cached Config singleton."""
    return Config()
