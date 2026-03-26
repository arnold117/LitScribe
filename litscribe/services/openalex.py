"""OpenAlex search service stub. Port from src/services/openalex.py."""
from __future__ import annotations

from litscribe.models.paper import Paper


class OpenAlexService:
    """OpenAlex search service. Port from src/services/openalex.py."""

    source_name = "openalex"

    async def search(self, query: str, max_results: int = 10, **filters) -> list[Paper]:
        raise NotImplementedError("Port from src/services/openalex.py")
