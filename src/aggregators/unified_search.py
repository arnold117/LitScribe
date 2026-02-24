"""Unified search aggregator for multiple academic sources."""

import asyncio
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from aggregators.deduplicator import deduplicate_papers, rank_papers
from models.unified_paper import UnifiedPaper

# Shared English stopwords: sklearn's 318 words + academic domain terms
ACADEMIC_STOPWORDS_EN: set = ENGLISH_STOP_WORDS | {
    "latest", "recent", "advances", "methods", "approaches", "review",
    "survey", "applications", "based", "using", "novel", "new", "study",
    "analysis", "paper", "research", "results", "method", "approach",
    "effect", "effects", "role", "via", "two", "one", "high", "low",
    "different", "specific", "related", "associated", "potential", "used",
    "use",
}

logger = logging.getLogger(__name__)

# Limit concurrent arXiv requests to avoid 503 rate limiting + retry storms
_arxiv_semaphore = asyncio.Semaphore(1)


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
        title=paper_dict.get("title") or "",
        authors=paper_dict.get("authors") or [],
        abstract=paper_dict.get("abstract") or "",
        year=year or 0,
        sources={"arxiv": paper_dict.get("arxiv_id") or ""},
        venue=paper_dict.get("journal_ref") or "",
        citations=paper_dict.get("citations") or 0,
        pdf_urls=[paper_dict["pdf_url"]] if paper_dict.get("pdf_url") else [],
        doi=paper_dict.get("doi"),
        arxiv_id=paper_dict.get("arxiv_id"),
        categories=paper_dict.get("categories") or [],
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
        title=article_dict.get("title") or "",
        authors=article_dict.get("authors") or [],
        abstract=article_dict.get("abstract") or "",
        year=year or 0,
        sources={"pubmed": article_dict.get("pmid") or ""},
        venue=article_dict.get("journal") or "",
        citations=article_dict.get("citations") or 0,
        pdf_urls=[],  # PubMed doesn't provide direct PDF links
        doi=article_dict.get("doi"),
        pmid=article_dict.get("pmid"),
        pmc_id=article_dict.get("pmc_id"),
        mesh_terms=article_dict.get("mesh_terms") or [],
        keywords=article_dict.get("keywords") or [],
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
        title=paper_dict.get("title") or "",
        authors=paper_dict.get("authors") or [],
        abstract=paper_dict.get("abstract") or "",
        year=paper_dict.get("year") or 0,
        sources={"semantic_scholar": paper_dict.get("paper_id") or ""},
        venue=paper_dict.get("venue") or "",
        citations=paper_dict.get("citation_count") or 0,
        pdf_urls=[paper_dict["pdf_url"]] if paper_dict.get("pdf_url") else [],
        doi=paper_dict.get("doi"),
        arxiv_id=paper_dict.get("arxiv_id"),
        pmid=paper_dict.get("pmid"),
        scholar_id=paper_dict.get("paper_id"),
        keywords=paper_dict.get("fields_of_study") or [],
        url=paper_dict.get("url"),
    )


def _openalex_to_unified(paper_dict: dict) -> UnifiedPaper:
    """Convert OpenAlex paper dict to UnifiedPaper."""
    return UnifiedPaper(
        title=paper_dict.get("title") or "",
        authors=paper_dict.get("authors") or [],
        abstract=paper_dict.get("abstract") or "",
        year=paper_dict.get("year") or 0,
        sources={"openalex": paper_dict.get("paper_id") or ""},
        venue=paper_dict.get("venue") or "",
        citations=paper_dict.get("citation_count") or 0,
        pdf_urls=[paper_dict["pdf_url"]] if paper_dict.get("pdf_url") else [],
        doi=paper_dict.get("doi"),
        pmid=paper_dict.get("pmid"),
        keywords=paper_dict.get("fields_of_study") or [],
        url=paper_dict.get("url"),
    )


def _europe_pmc_to_unified(paper_dict: dict) -> UnifiedPaper:
    """Convert Europe PMC paper dict to UnifiedPaper."""
    return UnifiedPaper(
        title=paper_dict.get("title") or "",
        authors=paper_dict.get("authors") or [],
        abstract=paper_dict.get("abstract") or "",
        year=paper_dict.get("year") or 0,
        sources={"europe_pmc": paper_dict.get("paper_id") or ""},
        venue=paper_dict.get("venue") or "",
        citations=paper_dict.get("citation_count") or 0,
        pdf_urls=[paper_dict["pdf_url"]] if paper_dict.get("pdf_url") else [],
        doi=paper_dict.get("doi"),
        pmid=paper_dict.get("pmid"),
        pmc_id=paper_dict.get("pmc_id"),
        keywords=[],
        url=paper_dict.get("url"),
    )


def _has_cjk(text: str) -> bool:
    """Check if text contains CJK (Chinese/Japanese/Korean) characters."""
    return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text))


def _extract_cjk_keywords(text: str) -> List[str]:
    """Extract CJK keyword tokens from text using jieba segmentation.

    Uses jieba for proper Chinese word segmentation, filters stopwords
    and single-character tokens.
    """
    cjk_stopwords = {
        "的", "了", "在", "是", "和", "与", "或", "及", "对", "从",
        "到", "为", "被", "把", "上", "下", "中", "等", "其", "这",
        "那", "有", "无", "不", "也", "都", "很", "更", "最", "可",
        "能", "会", "将", "所", "以", "之", "而", "但", "如", "若",
        "则", "已", "因", "于", "用", "并", "个", "各", "些", "种",
        "类", "进行", "研究", "分析", "方法", "基于", "通过", "关于",
        "综述", "探讨", "概述", "最新", "相关",
    }
    import jieba
    # Cut in search mode for finer granularity (e.g. "生物合成" → "生物", "合成", "生物合成")
    tokens = jieba.lcut_for_search(text)

    # Filter: keep tokens ≥2 chars, skip stopwords
    keywords = [t for t in tokens if len(t) >= 2 and t not in cjk_stopwords]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


def _match_keyword(keyword: str, text: str) -> bool:
    """Match a keyword against text, CJK-aware.

    CJK keywords use substring matching (no word boundaries in CJK).
    Latin keywords use word-boundary regex matching.
    """
    if _has_cjk(keyword):
        return keyword in text
    return bool(re.search(r'\b' + re.escape(keyword) + r'\b', text))


def _compute_keyword_relevance(papers: List["UnifiedPaper"], research_question: str) -> None:
    """Compute keyword-based relevance scores for papers.

    Replaces positional scoring with semantic keyword matching.
    Score breakdown: title match=0.4, abstract match=0.4, keyword match=0.2.
    Supports both English (word-boundary) and CJK (substring) matching.

    Args:
        papers: List of UnifiedPaper objects (modified in place)
        research_question: The original research question
    """
    is_cjk = _has_cjk(research_question)

    if is_cjk:
        # CJK path: character n-gram extraction
        keywords = _extract_cjk_keywords(research_question)
        # Also extract any English keywords mixed in
        en_words = re.findall(r'[a-zA-Z]{3,}', research_question.lower())
        keywords.extend(w for w in en_words if w not in ACADEMIC_STOPWORDS_EN)
    else:
        # English path: word-level extraction
        words = research_question.lower().split()
        keywords = [w.strip("?.,!\"'()") for w in words if len(w) >= 3 and w.lower() not in ACADEMIC_STOPWORDS_EN]

    if not keywords:
        # Fallback: give all papers a neutral score
        for paper in papers:
            paper.relevance_score = 0.5
        return

    # Check if keywords are exclusively CJK (no English terms mixed in).
    # Pure CJK keywords can't match English paper text, so we'll need a floor.
    cjk_only = is_cjk and not any(
        all(ord(c) < 128 for c in kw) for kw in keywords
    )

    for paper in papers:
        title = (paper.title or "").lower()
        abstract = (paper.abstract or "").lower()
        paper_keywords = " ".join(getattr(paper, "keywords", None) or []).lower()

        # Calculate match ratios (CJK-aware matching)
        title_matches = sum(1 for kw in keywords if _match_keyword(kw, title))
        abstract_matches = sum(1 for kw in keywords if _match_keyword(kw, abstract))
        keyword_matches = sum(1 for kw in keywords if _match_keyword(kw, paper_keywords))

        n = len(keywords)
        title_score = min(title_matches / n, 1.0) * 0.4
        abstract_score = min(abstract_matches / n, 1.0) * 0.4
        kw_score = min(keyword_matches / n, 1.0) * 0.2

        score = title_score + abstract_score + kw_score

        # CJK cross-language floor: when the query is purely CJK and papers are
        # in English, keyword matching returns ~0 for every paper, which causes
        # the MIN_RELEVANCE filter to kill all results.  Set a neutral floor so
        # the LLM paper-selection step decides relevance instead.
        if cjk_only and score < 0.35:
            score = max(score, 0.35)

        # Review/survey papers get relevance bonus — foundational for introductions
        if any(w in title for w in ("review", "survey", "meta-analysis", "overview", "综述")):
            score = min(score + 0.12, 1.0)

        # Highly cited papers get relevance bonus — likely foundational literature
        if getattr(paper, "citations", 0) >= 100:
            score = min(score + 0.08, 1.0)

        paper.relevance_score = score


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
        self._openalex_search = None
        self._europe_pmc_search = None
        self._initialized = False

    async def _lazy_init(self):
        """Lazily initialize MCP server functions."""
        if self._initialized:
            return

        # Import service functions
        try:
            from services.arxiv import search_papers as arxiv_search
            self._arxiv_search = arxiv_search
        except ImportError:
            logger.warning("arXiv service not available")

        try:
            from services.pubmed import search_pubmed as pubmed_search
            self._pubmed_search = pubmed_search
        except ImportError:
            logger.warning("PubMed service not available")

        try:
            from services.zotero import search_items as zotero_search
            self._zotero_search = zotero_search
        except ImportError:
            logger.warning("Zotero service not available")

        try:
            from services.semantic_scholar import search_papers as semantic_scholar_search
            self._semantic_scholar_search = semantic_scholar_search
        except ImportError:
            logger.warning("Semantic Scholar service not available")

        try:
            from services.openalex import search_papers as openalex_search
            self._openalex_search = openalex_search
        except ImportError:
            logger.warning("OpenAlex service not available")

        try:
            from services.europe_pmc import search_papers as europe_pmc_search
            self._europe_pmc_search = europe_pmc_search
        except ImportError:
            logger.warning("Europe PMC service not available")

        self._initialized = True

    async def search_arxiv(
        self,
        query: str,
        max_results: int = 20,
        category: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> List[UnifiedPaper]:
        """Search arXiv and return unified papers.

        Args:
            query: Search query
            max_results: Max results
            category: arXiv category filter (e.g. "q-bio.BM")
            year_from: Filter papers from this year (inclusive)
            year_to: Filter papers until this year (inclusive)
        """
        await self._lazy_init()

        if not self._arxiv_search:
            return []

        try:
            # Serialize arXiv requests to avoid 503 rate-limit → retry storms
            async with _arxiv_semaphore:
                result = await self._arxiv_search(
                    query=query, max_results=max_results, category=category,
                    year_from=year_from, year_to=year_to,
                )
            papers = result.get("papers", [])
            return [_arxiv_to_unified(p) for p in papers]
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return []

    async def search_pubmed(
        self,
        query: str,
        max_results: int = 20,
        mesh_terms: Optional[List[str]] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        article_types: Optional[List[str]] = None,
    ) -> List[UnifiedPaper]:
        """Search PubMed and return unified papers.

        Args:
            query: Search query
            max_results: Max results
            mesh_terms: MeSH terms to append to query for domain filtering
            year_from: Filter papers from this year (inclusive)
            year_to: Filter papers until this year (inclusive)
            article_types: PubMed publication type filter (e.g. ["Review"])
        """
        await self._lazy_init()

        if not self._pubmed_search:
            return []

        try:
            # Append MeSH terms to query for domain filtering
            filtered_query = query
            if mesh_terms:
                if len(mesh_terms) >= 3:
                    # Require primary MeSH term AND at least one secondary
                    primary = f"{mesh_terms[0]}[MeSH]"
                    secondary = " OR ".join(f"{m}[MeSH]" for m in mesh_terms[1:3])
                    mesh_filter = f"{primary} AND ({secondary})"
                else:
                    mesh_filter = " OR ".join(f"{m}[MeSH]" for m in mesh_terms[:3])
                filtered_query = f"({query}) AND ({mesh_filter})"

            # Build date params
            min_date = str(year_from) if year_from else None
            max_date = str(year_to) if year_to else None

            result = await self._pubmed_search(
                query=filtered_query, max_results=max_results,
                min_date=min_date, max_date=max_date,
                article_types=article_types,
            )
            articles = result.get("articles", [])
            return [_pubmed_to_unified(a) for a in articles]
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
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
            logger.error(f"Zotero search error: {e}")
            return []

    async def search_semantic_scholar(
        self,
        query: str,
        max_results: int = 20,
        fields_of_study: Optional[List[str]] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        min_citations: Optional[int] = None,
    ) -> List[UnifiedPaper]:
        """Search Semantic Scholar and return unified papers.

        Args:
            query: Search query
            max_results: Max results
            fields_of_study: Field of study filter (e.g. ["Biology"])
            year_from: Filter papers from this year (inclusive)
            year_to: Filter papers until this year (inclusive)
            min_citations: Only return papers with >= this many citations
        """
        await self._lazy_init()

        if not self._semantic_scholar_search:
            return []

        try:
            # Build year range string for S2 API (e.g. "2020-2026")
            # S2 API requires "YYYY-YYYY" format, not trailing/leading dash.
            year_str = None
            if year_from and year_to:
                year_str = f"{year_from}-{year_to}"
            elif year_from:
                year_str = f"{year_from}-2099"
            elif year_to:
                year_str = f"1900-{year_to}"

            result = await self._semantic_scholar_search(
                query=query, limit=max_results, fields_of_study=fields_of_study,
                year=year_str, min_citations=min_citations,
            )
            papers = result.get("papers", [])
            return [_semantic_scholar_to_unified(p) for p in papers]
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            return []

    async def search_openalex(
        self,
        query: str,
        max_results: int = 20,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        min_citations: Optional[int] = None,
    ) -> List[UnifiedPaper]:
        """Search OpenAlex and return unified papers."""
        await self._lazy_init()

        if not self._openalex_search:
            return []

        try:
            result = await self._openalex_search(
                query=query, max_results=max_results,
                year_from=year_from, year_to=year_to,
                min_citations=min_citations,
            )
            papers = result.get("papers", [])
            return [_openalex_to_unified(p) for p in papers]
        except Exception as e:
            logger.error(f"OpenAlex search error: {e}")
            return []

    async def search_europe_pmc(
        self,
        query: str,
        max_results: int = 20,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        min_citations: Optional[int] = None,
    ) -> List[UnifiedPaper]:
        """Search Europe PMC and return unified papers."""
        await self._lazy_init()

        if not self._europe_pmc_search:
            return []

        try:
            result = await self._europe_pmc_search(
                query=query, max_results=max_results,
                year_from=year_from, year_to=year_to,
                min_citations=min_citations,
            )
            papers = result.get("papers", [])
            return [_europe_pmc_to_unified(p) for p in papers]
        except Exception as e:
            logger.error(f"Europe PMC search error: {e}")
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
        research_question: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        article_types: Optional[List[str]] = None,
        min_citations: Optional[int] = None,
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
            research_question: Original research question for semantic scoring
            year_from: Filter papers from this year (inclusive)
            year_to: Filter papers until this year (inclusive)
            article_types: PubMed publication type filter (e.g. ["Review"])
            min_citations: S2 min citation count filter

        Returns:
            Dictionary with search results and metadata
        """
        if sources is None:
            sources = ["arxiv", "semantic_scholar", "pubmed", "openalex", "europe_pmc"]

        await self._lazy_init()

        # Build list of search tasks with domain filters
        tasks = []
        source_names = []

        if "arxiv" in sources:
            # Use first arXiv category as filter (API supports single category)
            category = arxiv_categories[0] if arxiv_categories else None
            tasks.append(self.search_arxiv(
                query, max_per_source, category=category,
                year_from=year_from, year_to=year_to,
            ))
            source_names.append("arxiv")

        if "pubmed" in sources:
            tasks.append(self.search_pubmed(
                query, max_per_source, mesh_terms=pubmed_mesh,
                year_from=year_from, year_to=year_to,
                article_types=article_types,
            ))
            source_names.append("pubmed")

        if "zotero" in sources:
            tasks.append(self.search_zotero(query, max_per_source))
            source_names.append("zotero")

        if "semantic_scholar" in sources:
            tasks.append(self.search_semantic_scholar(
                query, max_per_source, fields_of_study=s2_fields,
                year_from=year_from, year_to=year_to,
                min_citations=min_citations,
            ))
            source_names.append("semantic_scholar")

        if "openalex" in sources:
            tasks.append(self.search_openalex(
                query, max_per_source,
                year_from=year_from, year_to=year_to,
                min_citations=min_citations,
            ))
            source_names.append("openalex")

        if "europe_pmc" in sources:
            tasks.append(self.search_europe_pmc(
                query, max_per_source,
                year_from=year_from, year_to=year_to,
                min_citations=min_citations,
            ))
            source_names.append("europe_pmc")

        # Execute searches in parallel with per-source timeout
        # Prevents one slow source (e.g. arXiv retrying) from blocking all results
        async def _with_timeout(coro, source_name, timeout=45):
            try:
                return await asyncio.wait_for(coro, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"{source_name} search timed out after {timeout}s")
                return []

        timed_tasks = [
            _with_timeout(task, name) for task, name in zip(tasks, source_names)
        ]
        results = await asyncio.gather(*timed_tasks, return_exceptions=True)

        # Collect all papers
        all_papers = []
        source_counts = {}

        for source_name, result in zip(source_names, results):
            if isinstance(result, Exception):
                logger.error(f"Search error from {source_name}: {result}")
                source_counts[source_name] = 0
            elif result is None:
                logger.warning(f"{source_name} returned None")
                source_counts[source_name] = 0
            else:
                source_counts[source_name] = len(result)
                all_papers.extend(result)
                logger.info(f"{source_name}: {len(result)} papers found")

        # Filter relaxation: retry without domain filters when a source returns 0
        retry_tasks = []
        retry_names = []
        arxiv_had_filter = arxiv_categories and "arxiv" in sources
        s2_had_filter = s2_fields and "semantic_scholar" in sources
        pubmed_had_filter = pubmed_mesh and "pubmed" in sources

        if arxiv_had_filter and source_counts.get("arxiv", 0) == 0:
            logger.info("arxiv: 0 results with category filter, retrying without filter")
            retry_tasks.append(self.search_arxiv(
                query, max_per_source, category=None,
                year_from=year_from, year_to=year_to,
            ))
            retry_names.append("arxiv")

        if s2_had_filter and source_counts.get("semantic_scholar", 0) == 0:
            logger.info("semantic_scholar: 0 results with field filter, retrying without filter")
            retry_tasks.append(self.search_semantic_scholar(
                query, max_per_source, fields_of_study=None,
                year_from=year_from, year_to=year_to,
                min_citations=min_citations,
            ))
            retry_names.append("semantic_scholar")

        if pubmed_had_filter and source_counts.get("pubmed", 0) == 0:
            logger.info("pubmed: 0 results with MeSH filter, retrying without filter")
            retry_tasks.append(self.search_pubmed(
                query, max_per_source, mesh_terms=None,
                year_from=year_from, year_to=year_to,
                article_types=article_types,
            ))
            retry_names.append("pubmed")

        if retry_tasks:
            retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
            for rname, rresult in zip(retry_names, retry_results):
                if isinstance(rresult, Exception) or rresult is None:
                    continue
                if len(rresult) > 0:
                    source_counts[rname] = len(rresult)
                    all_papers.extend(rresult)
                    logger.info(f"{rname} (retry no filter): {len(rresult)} papers found")

        # Compute relevance scores: semantic (embedding) with keyword fallback
        _score_text = research_question or query
        try:
            from embeddings.semantic_scorer import compute_semantic_relevance

            compute_semantic_relevance(all_papers, _score_text)
        except Exception as _sem_err:
            logger.debug(f"Semantic scoring unavailable, using keyword: {_sem_err}")
            _compute_keyword_relevance(all_papers, _score_text)
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
    research_question: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    article_types: Optional[List[str]] = None,
    min_citations: Optional[int] = None,
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
        research_question: Original research question for semantic relevance scoring
        year_from: Filter papers from this year (inclusive)
        year_to: Filter papers until this year (inclusive)
        article_types: PubMed publication type filter
        min_citations: S2 min citation count filter

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
        research_question=research_question,
        year_from=year_from,
        year_to=year_to,
        article_types=article_types,
        min_citations=min_citations,
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
