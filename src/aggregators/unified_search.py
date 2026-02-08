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


def _semantic_scholar_to_unified(paper_dict: dict) -> UnifiedPaper:
    """Convert Semantic Scholar paper dict to UnifiedPaper."""
    return UnifiedPaper(
        title=paper_dict.get("title", ""),
        authors=paper_dict.get("authors", []),
        abstract=paper_dict.get("abstract", ""),
        year=paper_dict.get("year") or 0,
        sources={"semantic_scholar": paper_dict.get("paper_id", "")},
        venue=paper_dict.get("venue", ""),
        citations=paper_dict.get("citation_count") or 0,
        pdf_urls=[paper_dict["pdf_url"]] if paper_dict.get("pdf_url") else [],
        doi=paper_dict.get("doi"),
        arxiv_id=paper_dict.get("arxiv_id"),
        pmid=paper_dict.get("pmid"),
        scholar_id=paper_dict.get("paper_id"),
        keywords=paper_dict.get("fields_of_study", []),
        url=paper_dict.get("url"),
    )


def _compute_keyword_relevance(papers: List["UnifiedPaper"], research_question: str) -> None:
    """Compute keyword-based relevance scores for papers.

    Replaces positional scoring with semantic keyword matching.
    Score breakdown: title match=0.4, abstract match=0.4, keyword match=0.2.

    Args:
        papers: List of UnifiedPaper objects (modified in place)
        research_question: The original research question
    """
    # Extract meaningful keywords from research question
    stopwords = {
        "the", "a", "an", "in", "on", "of", "for", "and", "or", "to", "is",
        "are", "was", "were", "what", "how", "which", "that", "this", "with",
        "from", "by", "at", "its", "their", "have", "has", "been", "be",
        "do", "does", "did", "will", "can", "could", "would", "should",
        "about", "into", "through", "during", "before", "after", "between",
        "latest", "recent", "advances", "methods", "approaches", "review",
        "survey", "applications", "based",
    }
    words = research_question.lower().split()
    keywords = [w.strip("?.,!\"'()") for w in words if len(w) >= 3 and w.lower() not in stopwords]

    if not keywords:
        # Fallback: give all papers a neutral score
        for paper in papers:
            paper.relevance_score = 0.5
        return

    for paper in papers:
        title = (paper.title or "").lower()
        abstract = (paper.abstract or "").lower()
        paper_keywords = " ".join(getattr(paper, "keywords", None) or []).lower()

        # Calculate match ratios
        title_matches = sum(1 for kw in keywords if kw in title)
        abstract_matches = sum(1 for kw in keywords if kw in abstract)
        keyword_matches = sum(1 for kw in keywords if kw in paper_keywords)

        n = len(keywords)
        title_score = min(title_matches / n, 1.0) * 0.4
        abstract_score = min(abstract_matches / n, 1.0) * 0.4
        kw_score = min(keyword_matches / n, 1.0) * 0.2

        paper.relevance_score = title_score + abstract_score + kw_score


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
        self._semantic_scholar_search = None
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

        try:
            from mcp_servers.semantic_scholar_server import search_papers as semantic_scholar_search
            self._semantic_scholar_search = semantic_scholar_search
        except ImportError:
            print("Warning: Semantic Scholar MCP server not available")

        self._initialized = True

    async def search_arxiv(
        self,
        query: str,
        max_results: int = 20,
        category: Optional[str] = None,
    ) -> List[UnifiedPaper]:
        """Search arXiv and return unified papers.

        Args:
            query: Search query
            max_results: Max results
            category: arXiv category filter (e.g. "q-bio.BM")
        """
        await self._lazy_init()

        if not self._arxiv_search:
            return []

        try:
            result = await self._arxiv_search(
                query=query, max_results=max_results, category=category,
            )
            papers = result.get("papers", [])
            return [_arxiv_to_unified(p) for p in papers]
        except Exception as e:
            print(f"arXiv search error: {e}")
            return []

    async def search_pubmed(
        self,
        query: str,
        max_results: int = 20,
        mesh_terms: Optional[List[str]] = None,
    ) -> List[UnifiedPaper]:
        """Search PubMed and return unified papers.

        Args:
            query: Search query
            max_results: Max results
            mesh_terms: MeSH terms to append to query for domain filtering
        """
        await self._lazy_init()

        if not self._pubmed_search:
            return []

        try:
            # Append MeSH terms to query for domain filtering
            filtered_query = query
            if mesh_terms:
                mesh_filter = " AND ".join(f"{m}[MeSH]" for m in mesh_terms[:3])
                filtered_query = f"({query}) AND ({mesh_filter})"

            result = await self._pubmed_search(query=filtered_query, max_results=max_results)
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

    async def search_semantic_scholar(
        self,
        query: str,
        max_results: int = 20,
        fields_of_study: Optional[List[str]] = None,
    ) -> List[UnifiedPaper]:
        """Search Semantic Scholar and return unified papers.

        Args:
            query: Search query
            max_results: Max results
            fields_of_study: Field of study filter (e.g. ["Biology"])
        """
        await self._lazy_init()

        if not self._semantic_scholar_search:
            return []

        try:
            result = await self._semantic_scholar_search(
                query=query, limit=max_results, fields_of_study=fields_of_study,
            )
            papers = result.get("papers", [])
            return [_semantic_scholar_to_unified(p) for p in papers]
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
            return []

    async def search_all(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_per_source: int = 20,
        deduplicate: bool = True,
        sort_by: str = "relevance",
        arxiv_categories: Optional[List[str]] = None,
        s2_fields: Optional[List[str]] = None,
        pubmed_mesh: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Search all specified sources and return unified, deduplicated results.

        Args:
            query: Search query string
            sources: List of sources to search ("arxiv", "pubmed", "zotero", "semantic_scholar")
                     Default: ["arxiv", "pubmed"]
            max_per_source: Maximum results per source
            deduplicate: Whether to merge duplicate papers
            sort_by: Ranking criteria ("relevance", "citations", "year", "completeness")
            arxiv_categories: arXiv category filters (e.g. ["q-bio.BM"])
            s2_fields: Semantic Scholar field filters (e.g. ["Biology"])
            pubmed_mesh: PubMed MeSH term filters (e.g. ["Alkaloids"])

        Returns:
            Dictionary with search results and metadata
        """
        if sources is None:
            sources = ["arxiv", "semantic_scholar", "pubmed"]

        await self._lazy_init()

        # Build list of search tasks with domain filters
        tasks = []
        source_names = []

        if "arxiv" in sources:
            # Use first arXiv category as filter (API supports single category)
            category = arxiv_categories[0] if arxiv_categories else None
            tasks.append(self.search_arxiv(query, max_per_source, category=category))
            source_names.append("arxiv")

        if "pubmed" in sources:
            tasks.append(self.search_pubmed(query, max_per_source, mesh_terms=pubmed_mesh))
            source_names.append("pubmed")

        if "zotero" in sources:
            tasks.append(self.search_zotero(query, max_per_source))
            source_names.append("zotero")

        if "semantic_scholar" in sources:
            tasks.append(self.search_semantic_scholar(
                query, max_per_source, fields_of_study=s2_fields,
            ))
            source_names.append("semantic_scholar")

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

        # Compute keyword-based relevance scores instead of positional
        _compute_keyword_relevance(all_papers, query)
        for paper in all_papers:
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
    arxiv_categories: Optional[List[str]] = None,
    s2_fields: Optional[List[str]] = None,
    pubmed_mesh: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to search all sources.

    Args:
        query: Search query string
        sources: List of sources to search ("arxiv", "pubmed", "zotero", "semantic_scholar")
        max_per_source: Maximum results per source
        deduplicate: Whether to merge duplicate papers
        sort_by: Ranking criteria
        arxiv_categories: arXiv category filters
        s2_fields: Semantic Scholar field filters
        pubmed_mesh: PubMed MeSH term filters

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
        arxiv_categories=arxiv_categories,
        s2_fields=s2_fields,
        pubmed_mesh=pubmed_mesh,
    )


async def search_local_first(
    query: str,
    sources: Optional[List[str]] = None,
    max_per_source: int = 20,
    zotero_collection: Optional[str] = None,
    deduplicate: bool = True,
    sort_by: str = "relevance",
) -> Dict[str, Any]:
    """Search with local-first strategy: Zotero library first, then external APIs.

    This function always searches Zotero first (if configured), then searches
    the specified external sources. Zotero results are prioritized during dedup.

    Args:
        query: Search query string
        sources: External sources to search ("arxiv", "pubmed", "semantic_scholar")
        max_per_source: Maximum results per source
        zotero_collection: Specific Zotero collection to search (optional)
        deduplicate: Whether to merge duplicate papers
        sort_by: Ranking criteria

    Returns:
        Dictionary with search results including Zotero origin tracking
    """
    aggregator = UnifiedSearchAggregator()

    # Search Zotero first
    zotero_papers = await aggregator.search_zotero(query, limit=max_per_source)
    zotero_count = len(zotero_papers)

    # Then search external sources (excluding "zotero" to avoid double-search)
    external_sources = [s for s in (sources or []) if s != "zotero"]
    external_result = await aggregator.search_all(
        query=query,
        sources=external_sources or None,
        max_per_source=max_per_source,
        deduplicate=deduplicate,
        sort_by=sort_by,
    )

    # Merge: Zotero papers first for dedup priority
    all_papers = zotero_papers
    external_papers_raw = external_result.get("papers", [])
    for p in external_papers_raw:
        if isinstance(p, dict):
            all_papers.append(p)
        else:
            all_papers.append(p)

    # Deduplicate
    if deduplicate:
        all_papers = deduplicate_papers(all_papers)

    all_papers = rank_papers(all_papers, sort_by)

    source_counts = external_result.get("source_counts", {})
    source_counts["zotero"] = zotero_count

    return {
        "query": query,
        "sources_searched": external_result.get("sources_searched", []) + (["zotero"] if zotero_count > 0 else []),
        "source_counts": source_counts,
        "total_before_dedup": external_result.get("total_before_dedup", 0) + zotero_count,
        "total_after_dedup": len(all_papers),
        "from_zotero": zotero_count,
        "papers": [p.to_dict() if hasattr(p, "to_dict") else p for p in all_papers],
    }
