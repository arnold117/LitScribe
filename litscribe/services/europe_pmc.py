"""Europe PMC search service stub. Port from src/services/europe_pmc.py."""
from __future__ import annotations

from litscribe.models.paper import Paper


class EuropePMCService:
    """Europe PMC search service. Port from src/services/europe_pmc.py."""

    source_name = "europe_pmc"

    async def search(self, query: str, max_results: int = 10, **filters) -> list[Paper]:
        raise NotImplementedError("Port from src/services/europe_pmc.py")
