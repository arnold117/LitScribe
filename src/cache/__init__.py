"""LitScribe Cache System.

This package provides SQLite-based caching for:
- Paper metadata
- PDF downloads
- Parsed documents
- Search results (with TTL)
- LLM responses
- Command history
"""

from cache.database import CacheDatabase, get_cache_db, init_cache
from cache.paper_cache import PaperCache
from cache.search_cache import SearchCache
from cache.pdf_cache import PDFCache
from cache.parse_cache import ParseCache
from cache.cached_tools import CachedTools, get_cached_tools

__all__ = [
    "CacheDatabase",
    "get_cache_db",
    "init_cache",
    "PaperCache",
    "SearchCache",
    "PDFCache",
    "ParseCache",
    "CachedTools",
    "get_cached_tools",
]
