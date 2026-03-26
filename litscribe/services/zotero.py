"""Zotero search service stub. Port from src/services/zotero.py."""
from __future__ import annotations

from litscribe.models.paper import Paper


class ZoteroService:
    """Zotero personal library search service. Port from src/services/zotero.py."""

    source_name = "zotero"

    async def search(self, query: str, max_results: int = 10, **filters) -> list[Paper]:
        raise NotImplementedError("Port from src/services/zotero.py")
