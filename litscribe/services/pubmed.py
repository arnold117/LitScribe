"""PubMed search service stub. Port from src/services/pubmed.py."""
from __future__ import annotations

from litscribe.models.paper import Paper


class PubMedService:
    """PubMed search service. Port from src/services/pubmed.py."""

    source_name = "pubmed"

    async def search(self, query: str, max_results: int = 10, **filters) -> list[Paper]:
        raise NotImplementedError("Port from src/services/pubmed.py")
