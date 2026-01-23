"""Unified search aggregator for multiple academic sources."""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from aggregators.deduplicator import deduplicate_papers, rank_papers
from models.unified_paper import UnifiedPaper


def _arxiv_to_unified(paper_dict: dict) -> UnifiedPaper:
    """Convert arXiv paper dict to UnifiedPaper."""
    # Parse year from published date
    year = None
    if paper_dict.get("published"):
        try:
            pub_date = datetime.fromisoformat(paper_dict["published"].replace("Z", "+00:00"))
            year = pub_date.year
        except (ValueError, AttributeError):
            pass

    return UnifiedPaper(
        title=paper_dict.get("title", ""),
        authors=paper_dict.get("authors", []),
        abstract=paper_dict.get("abstract", ""),
        year=year or 0,
        sources={"arxiv": paper_dict.get("arxiv_id", "")},
        venue=paper_dict.get("journal_ref", ""),
        citations=paper_dict.get("citations") or 0,
        pdf_urls=[paper_dict["pdf_url"]] if paper_dict.get("pdf_url") else [],
        doi=paper_dict.get("doi"),
        arxiv_id=paper_dict.get("arxiv_id"),
        categories=paper_dict.get("categories", []),
        comment=paper_dict.get("comment"),
        journal_ref=paper_dict.get("journal_ref"),
    )


def _pubmed_to_unified(article_dict: dict) -> UnifiedPaper:
    """Convert PubMed article dict to UnifiedPaper."""
    # Parse year from publication_date
    year = None
    if article_dict.get("publication_date"):
        try:
            pub_date = datetime.fromisoformat(article_dict["publication_date"])
            year = pub_date.year
        except (ValueError, AttributeError):
            pass

    return UnifiedPaper(
        title=article_dict.get("title", ""),
        authors=article_dict.get("authors", []),
        abstract=article_dict.get("abstract", ""),
        year=year or 0,
        sources={"pubmed": article_dict.get("pmid", "")},
        venue=article_dict.get("journal", ""),
        citations=article_dict.get("citations") or 0,
        pdf_urls=[],  # PubMed doesn't provide direct PDF links
        doi=article_dict.get("doi"),
        pmid=article_dict.get("pmid"),
        pmc_id=article_dict.get("pmc_id"),
        mesh_terms=article_dict.get("mesh_terms", []),
        keywords=article_dict.get("keywords", []),
        url=article_dict.get("url"),
    )


def _zotero_to_unified(item_dict: dict) -> UnifiedPaper:
    """Convert Zotero item dict to UnifiedPaper."""
    # Parse year from date field
    year = None
    if item_dict.get("date"):
        try:
            # Zotero dates can be various formats
            date_str = item_dict["date"]
            # Try to extract year
            import re
            year_match = re.search(r"(\d{4})", date_str)
            if year_match:
                year = int(year_match.group(1))
        except (ValueError, AttributeError):
            pass

    # Extract author names from creators
    authors = []
    for creator in item_dict.get("creators", []):
        if creator.get("creatorType") == "author":
            name = f"{creator.get('firstName', '')} {creator.get('lastName', '')}".strip()
            if name:
                authors.append(name)

    return UnifiedPaper(
        title=item_dict.get("title", ""),
        authors=authors,
        abstract=item_dict.get("abstract", ""),
        year=year or 0,
        sources={"zotero": item_dict.get("key", "")},
        venue=item_dict.get("publication_title", ""),
        citations=0,  # Zotero doesn't track citations
        pdf_urls=[],  # Would need separate call to get PDF path
        doi=item_dict.get("doi"),
        url=item_dict.get("url"),
    )


class UnifiedSearchAggregator:
    """
    Aggregates search results from multiple academic sources.

    Supports arXiv, PubMed, and Zotero (local library).
    """

    def __init__(self):
        """Initialize the aggregator with MCP clients."""
        self._arxiv_search = None
        self._pubmed_search = None
        self._zotero_search = None
        self._initialized = False

    async def _lazy_init(self):
        """Lazily initialize MCP server functions."""
        if self._initialized:
            return

        # Import MCP server functions
        try:
            from mcp_servers.arxiv_server import search_papers as arxiv_search
            self._arxiv_search = arxiv_search
        except ImportError:
            print("Warning: arXiv MCP server not available")

        try:
            from mcp_servers.pubmed_server import search_pubmed as pubmed_search
            self._pubmed_search = pubmed_search
        except ImportError:
            print("Warning: PubMed MCP server not available")

        try:
            from mcp_servers.zotero_server import search_items as zotero_search
            self._zotero_search = zotero_search
        except ImportError:
            print("Warning: Zotero MCP server not available")

        self._initialized = True

    async def search_arxiv(
        self,
        query: str,
        max_results: int = 20,
    ) -> List[UnifiedPaper]:
        """Search arXiv and return unified papers."""
        await self._lazy_init()

        if not self._arxiv_search:
            return []

        try:
            result = await self._arxiv_search(query=query, max_results=max_results)
            papers = result.get("papers", [])
            return [_arxiv_to_unified(p) for p in papers]
        except Exception as e:
            print(f"arXiv search error: {e}")
            return []

    async def search_pubmed(
        self,
        query: str,
        max_results: int = 20,
    ) -> List[UnifiedPaper]:
        """Search PubMed and return unified papers."""
        await self._lazy_init()

        if not self._pubmed_search:
            return []

        try:
            result = await self._pubmed_search(query=query, max_results=max_results)
            articles = result.get("articles", [])
            return [_pubmed_to_unified(a) for a in articles]
        except Exception as e:
            print(f"PubMed search error: {e}")
            return []

    async def search_zotero(
        self,
        query: str,
        limit: int = 25,
    ) -> List[UnifiedPaper]:
        """Search Zotero library and return unified papers."""
        await self._lazy_init()

        if not self._zotero_search:
            return []

        try:
            result = await self._zotero_search(query=query, limit=limit)
            items = result.get("items", [])
            return [_zotero_to_unified(item) for item in items]
        except Exception as e:
            print(f"Zotero search error: {e}")
            return []

    async def search_all(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_per_source: int = 20,
        deduplicate: bool = True,
        sort_by: str = "relevance",
    ) -> Dict[str, Any]:
        """
        Search all specified sources and return unified, deduplicated results.

        Args:
            query: Search query string
            sources: List of sources to search ("arxiv", "pubmed", "zotero")
                     Default: ["arxiv", "pubmed"]
            max_per_source: Maximum results per source
            deduplicate: Whether to merge duplicate papers
            sort_by: Ranking criteria ("relevance", "citations", "year", "completeness")

        Returns:
            Dictionary with search results and metadata
        """
        if sources is None:
            sources = ["arxiv", "pubmed"]

        await self._lazy_init()

        # Build list of search tasks
        tasks = []
        source_names = []

        if "arxiv" in sources:
            tasks.append(self.search_arxiv(query, max_per_source))
            source_names.append("arxiv")

        if "pubmed" in sources:
            tasks.append(self.search_pubmed(query, max_per_source))
            source_names.append("pubmed")

        if "zotero" in sources:
            tasks.append(self.search_zotero(query, max_per_source))
            source_names.append("zotero")

        # Execute searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all papers
        all_papers = []
        source_counts = {}

        for source_name, result in zip(source_names, results):
            if isinstance(result, Exception):
                print(f"Error from {source_name}: {result}")
                source_counts[source_name] = 0
            else:
                source_counts[source_name] = len(result)
                all_papers.extend(result)

        # Calculate initial relevance scores (based on position in results)
        for i, paper in enumerate(all_papers):
            # Higher score for earlier results within each source
            paper.relevance_score = 1.0 - (i / max(len(all_papers), 1)) * 0.5
            paper.calculate_completeness_score()

        # Deduplicate if requested
        if deduplicate:
            all_papers = deduplicate_papers(all_papers)

        # Sort results
        all_papers = rank_papers(all_papers, sort_by)

        return {
            "query": query,
            "sources_searched": source_names,
            "source_counts": source_counts,
            "total_before_dedup": sum(source_counts.values()),
            "total_after_dedup": len(all_papers),
            "papers": [p.to_dict() for p in all_papers],
        }


# Convenience function for direct use
async def search_all_sources(
    query: str,
    sources: Optional[List[str]] = None,
    max_per_source: int = 20,
    deduplicate: bool = True,
    sort_by: str = "relevance",
) -> Dict[str, Any]:
    """
    Convenience function to search all sources.

    Args:
        query: Search query string
        sources: List of sources to search ("arxiv", "pubmed", "zotero")
        max_per_source: Maximum results per source
        deduplicate: Whether to merge duplicate papers
        sort_by: Ranking criteria

    Returns:
        Dictionary with search results
    """
    aggregator = UnifiedSearchAggregator()
    return await aggregator.search_all(
        query=query,
        sources=sources,
        max_per_source=max_per_source,
        deduplicate=deduplicate,
        sort_by=sort_by,
    )
