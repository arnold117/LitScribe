"""PubMed search service — wraps Bio.Entrez with the v2 SearchService protocol."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from Bio import Entrez, Medline

from litscribe.models.paper import Paper

logger = logging.getLogger(__name__)


class PubMedService:
    """PubMed search service implementing the SearchService protocol."""

    source_name = "pubmed"

    def __init__(
        self,
        ncbi_email: Optional[str] = None,
        ncbi_api_key: Optional[str] = None,
    ) -> None:
        email = ncbi_email or os.environ.get("NCBI_EMAIL", "")
        api_key = ncbi_api_key or os.environ.get("NCBI_API_KEY", "")

        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key

        if not email:
            logger.warning(
                "NCBI_EMAIL not set — PubMed searches may fail or be rate-limited. "
                "Set NCBI_EMAIL in .env or pass ncbi_email= to PubMedService()."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_records(self, id_list: list[str]) -> list[dict]:
        """Fetch Medline records for the given PubMed ID list (blocking)."""
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(id_list),
            rettype="medline",
            retmode="text",
        )
        records = list(Medline.parse(handle))
        handle.close()
        return records

    @staticmethod
    def _record_to_paper(record: dict) -> Paper:
        """Convert a Medline record dict to a v2 Paper model."""
        pmid = record.get("PMID", "")

        # Extract year from DP field (e.g. "2024 Jan" → 2024)
        dp = record.get("DP", "")
        year = int(dp[:4]) if dp and dp[:4].isdigit() else 0

        # Extract DOI from AID list (e.g. ["10.1038/test [doi]"])
        doi = next(
            (aid.split(" ")[0] for aid in record.get("AID", []) if "[doi]" in aid),
            "",
        )

        return Paper(
            paper_id=f"pmid:{pmid}",
            title=record.get("TI", ""),
            authors=record.get("AU", []),
            abstract=record.get("AB", ""),
            year=year,
            sources={"pubmed": pmid},
            venue=record.get("SO", ""),
            doi=doi,
        )

    # ------------------------------------------------------------------
    # SearchService protocol
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **filters,
    ) -> list[Paper]:
        """
        Search PubMed via Entrez esearch + efetch and return v2 Paper objects.

        Supported filters (passed as keyword arguments):
            sort        (str)       "relevance" (default) or "date"
            min_date    (str)       e.g. "2020" or "2020/01/01"
            max_date    (str)       e.g. "2024" or "2024/12/31"
            article_types (list)   PubMed publication types, e.g. ["Review"]
        """
        max_results = min(max_results, 100)
        sort_order = "relevance" if filters.get("sort", "relevance") == "relevance" else "pub_date"
        article_types: list[str] = filters.get("article_types") or []
        min_date: Optional[str] = filters.get("min_date")
        max_date: Optional[str] = filters.get("max_date")

        # Append article type filter to query string
        if article_types:
            type_filter = " OR ".join(f"{at}[pt]" for at in article_types)
            query = f"({query}) AND ({type_filter})"

        search_params: dict = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": sort_order,
        }
        if min_date:
            search_params["mindate"] = min_date
        if max_date:
            search_params["maxdate"] = max_date
        if min_date or max_date:
            search_params["datetype"] = "pdat"

        loop = asyncio.get_event_loop()

        # --- esearch ---
        def do_search():
            handle = Entrez.esearch(**search_params)
            record = Entrez.read(handle)
            handle.close()
            return record

        search_record = await loop.run_in_executor(None, do_search)
        id_list: list[str] = search_record.get("IdList", [])

        if not id_list:
            return []

        # --- efetch + parse ---
        records = await loop.run_in_executor(None, self._fetch_records, id_list)

        return [self._record_to_paper(r) for r in records]
