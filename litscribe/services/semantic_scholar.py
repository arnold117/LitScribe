"""Semantic Scholar search service stub. Port from src/services/semantic_scholar.py."""
from __future__ import annotations

from litscribe.models.paper import Paper


class SemanticScholarService:
    """Semantic Scholar search service. Port from src/services/semantic_scholar.py."""

    source_name = "semantic_scholar"

    async def search(self, query: str, max_results: int = 10, **filters) -> list[Paper]:
        raise NotImplementedError("Port from src/services/semantic_scholar.py")
