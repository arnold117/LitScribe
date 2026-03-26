"""Zotero personal library search service via pyzotero."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from litscribe.models.paper import Paper

logger = logging.getLogger(__name__)


def _item_to_paper(item: Dict[str, Any]) -> Paper:
    """Convert a Zotero item dict (pyzotero response) to a Paper."""
    data = item.get("data", item)  # pyzotero wraps in {"data": {...}}

    # Build author list: "lastName firstName"
    authors = [
        (c.get("lastName", "") + " " + c.get("firstName", "")).strip()
        for c in data.get("creators", [])
        if c.get("creatorType") == "author" or "lastName" in c or "firstName" in c
    ]
    authors = [a for a in authors if a]  # drop empty strings

    # Extract year from date string (e.g. "2024", "2024-01-15", "January 2024")
    year = 0
    date_str = data.get("date") or ""
    if date_str:
        import re
        m = re.search(r"\b(\d{4})\b", date_str)
        if m:
            year = int(m.group(1))

    key = data.get("key", "")
    doi = data.get("DOI", "") or ""

    return Paper(
        paper_id=f"zotero:{key}",
        title=data.get("title", ""),
        authors=authors,
        abstract=data.get("abstractNote", "") or "",
        year=year,
        sources={"zotero": key},
        venue=data.get("publicationTitle", ""),
        doi=doi,
    )


class ZoteroService:
    """Search the user's personal Zotero library via pyzotero.

    Credentials are read from environment variables if not supplied:
        ZOTERO_API_KEY, ZOTERO_LIBRARY_ID, ZOTERO_LIBRARY_TYPE
    """

    source_name = "zotero"

    def __init__(
        self,
        api_key: str = "",
        library_id: str = "",
        library_type: str = "user",
    ) -> None:
        self._api_key = api_key or os.environ.get("ZOTERO_API_KEY", "")
        self._library_id = library_id or os.environ.get("ZOTERO_LIBRARY_ID", "")
        self._library_type = library_type or os.environ.get("ZOTERO_LIBRARY_TYPE", "user")

    def _make_client(self):
        from pyzotero import zotero  # lazy import — optional dependency

        return zotero.Zotero(
            library_id=self._library_id,
            library_type=self._library_type,
            api_key=self._api_key,
        )

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **filters,
    ) -> list[Paper]:
        """Search the Zotero library for items matching *query*.

        pyzotero is synchronous; we run it in an executor to avoid
        blocking the event loop.

        Supported filters:
            collection (str): Zotero collection key to search within
            item_type (str): Zotero item type (e.g. "journalArticle")
        """
        collection: Optional[str] = filters.get("collection")
        item_type: Optional[str] = filters.get("item_type")

        def _do_search():
            zot = self._make_client()
            params: Dict[str, Any] = {"q": query, "limit": max_results}
            if item_type:
                params["itemType"] = item_type
            if collection:
                items = zot.collection_items(collection, **params)
            else:
                items = zot.items(**params)
            # Filter out attachments and notes
            return [
                i for i in items
                if (i.get("data", i)).get("itemType") not in ("attachment", "note")
            ]

        loop = asyncio.get_event_loop()
        try:
            items = await loop.run_in_executor(None, _do_search)
        except Exception as exc:
            logger.warning("Zotero search failed: %s", exc)
            return []

        return [_item_to_paper(i) for i in items]
