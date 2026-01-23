"""Aggregators for unified search across multiple sources."""

from aggregators.unified_search import (
    UnifiedSearchAggregator,
    search_all_sources,
)
from aggregators.deduplicator import (
    deduplicate_papers,
    rank_papers,
    title_similarity,
    are_same_paper,
)

__all__ = [
    "UnifiedSearchAggregator",
    "search_all_sources",
    "deduplicate_papers",
    "rank_papers",
    "title_similarity",
    "are_same_paper",
]
