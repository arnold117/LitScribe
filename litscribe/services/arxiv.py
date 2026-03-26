"""arXiv search service stub. Port from src/services/arxiv.py."""
from __future__ import annotations

from litscribe.models.paper import Paper


class ArxivService:
    """arXiv search service. Port from src/services/arxiv.py."""

    source_name = "arxiv"

    async def search(self, query: str, max_results: int = 10, **filters) -> list[Paper]:
        raise NotImplementedError("Port from src/services/arxiv.py")
