"""LitScribe Cache System.

This package provides SQLite-based caching for:
- Paper metadata
- PDF downloads
- Parsed documents
- Search results (with TTL)
- LLM responses
- Command history
- Failed papers retry queue
- GraphRAG entities and communities (Phase 7.5)
"""

from cache.database import CacheDatabase, get_cache_db, init_cache
from cache.paper_cache import PaperCache
from cache.search_cache import SearchCache
from cache.pdf_cache import PDFCache
from cache.parse_cache import ParseCache
from cache.cached_tools import CachedTools, get_cached_tools
from cache.failed_papers import (
    record_failed_paper,
    get_pending_retries,
    mark_resolved,
    get_failure_stats,
)
from cache.graphrag_cache import GraphRAGCache, get_graphrag_cache

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
    # Failed papers queue
    "record_failed_paper",
    "get_pending_retries",
    "mark_resolved",
    "get_failure_stats",
    # GraphRAG cache
    "GraphRAGCache",
    "get_graphrag_cache",
]
