"""Discovery Agent for LitScribe.

This agent is responsible for:
1. Query expansion - generating diverse search queries from the research question
2. Multi-source search - searching arXiv, Semantic Scholar, PubMed, etc.
3. Paper selection - ranking and selecting the most relevant papers
4. Snowball sampling - finding related papers through citations and references

Supports local-first search with SQLite caching (Phase 6.5).
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.errors import AgentError, ErrorType, LLMError
from agents.prompts import (
    ABSTRACT_SCREENING_PROMPT,
    PAPER_SELECTION_PROMPT,
    QUERY_EXPANSION_PROMPT,
    format_papers_for_prompt,
)
from agents.state import LitScribeState, SearchResult, TIER_CONFIG
from agents.tools import (
    call_llm,
    call_llm_for_json,
    extract_json,
    get_paper_citations,
    get_paper_references,
    get_recommendations,
    unified_search,
)
from cache.cached_tools import CachedTools, get_cached_tools

logger = logging.getLogger(__name__)


async def expand_queries(
    research_question: str,
    model: Optional[str] = None,
    domain_hint: str = "",
    tracker=None,
) -> List[str]:
    """Expand a research question into multiple search queries.

    Uses LLM to generate diverse queries that cover different aspects
    of the research topic, constrained to the detected domain.

    Args:
        research_question: The original research question
        model: LLM model to use (optional)
        domain_hint: Detected research domain for constraining queries

    Returns:
        List of expanded search queries
    """
    prompt = QUERY_EXPANSION_PROMPT.format(
        research_question=research_question,
        domain_hint=domain_hint or "General",
    )

    try:
        queries = await call_llm_for_json(prompt, model=model, temperature=0.4, max_tokens=500, tracker=tracker, agent_name="discovery")

        if not isinstance(queries, list):
            raise ValueError("Expected JSON array")

        # Include original question as first query only if it's in English
        # (non-English queries perform poorly on English academic databases)
        is_english = all(ord(c) < 128 for c in research_question if c.isalpha())
        if is_english:
            all_queries = [research_question] + queries
        else:
            all_queries = queries if queries else [research_question]
        logger.info(f"Expanded '{research_question}' into {len(all_queries)} queries")

        return all_queries

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse query expansion response: {e}")
        # Fall back to original question and simple variations
        # If non-English, use generic English terms from the question
        return [
            research_question,
            f"{research_question} review",
            f"{research_question} survey",
        ]
    except LLMError as e:
        logger.error(f"LLM error during query expansion: {e}")
        return [research_question]


async def search_all_sources(
    queries: List[str],
    sources: List[str],
    max_per_source: int = 20,
    cached_tools: Optional[CachedTools] = None,
    arxiv_categories: Optional[List[str]] = None,
    s2_fields: Optional[List[str]] = None,
    pubmed_mesh: Optional[List[str]] = None,
    zotero_collection: Optional[str] = None,
) -> Dict[str, Any]:
    """Search multiple sources with multiple queries.

    Supports local-first search with caching. When cache is enabled:
    1. Check local cache for each source
    2. Only fetch from sources not in cache
    3. Cache new results for future use

    Args:
        queries: List of search queries
        sources: List of sources to search
        max_per_source: Maximum results per source per query
        cached_tools: CachedTools instance for local-first search (optional)
        arxiv_categories: arXiv category filters (e.g. ["q-bio.BM"])
        s2_fields: Semantic Scholar field filters (e.g. ["Biology"])
        pubmed_mesh: PubMed MeSH term filters (e.g. ["Alkaloids"])
        zotero_collection: Zotero collection key to search (None = entire library)

    Returns:
        Combined search results with deduplication
    """
    all_papers = []
    source_counts = {source: 0 for source in sources}
    if "zotero" not in source_counts:
        source_counts["zotero"] = 0
    errors = []
    from_cache_count = 0
    from_zotero_count = 0

    # Search with each query in parallel (use all queries — tier system controls volume upstream)
    queries_to_search = queries[:12]  # Safety cap at 12 to avoid API flooding
    max_per_query = max(max_per_source // max(len(queries_to_search), 1), 5)

    async def search_single_query(query: str):
        """Search a single query across all sources."""
        try:
            if cached_tools and cached_tools.cache_enabled:
                result = await cached_tools.search_with_cache(
                    query=query,
                    sources=sources,
                    max_per_source=max_per_query,
                    zotero_collection=zotero_collection,
                    arxiv_categories=arxiv_categories,
                    s2_fields=s2_fields,
                    pubmed_mesh=pubmed_mesh,
                )
            else:
                result = await unified_search(
                    query=query,
                    sources=sources,
                    max_per_source=max_per_query,
                    deduplicate=True,
                )
            return {"success": True, "result": result, "query": query}
        except AgentError as e:
            logger.warning(f"Search failed for query '{query}': {e}")
            return {"success": False, "error": str(e), "query": query}
        except Exception as e:
            logger.warning(f"Unexpected error searching '{query}': {e}")
            return {"success": False, "error": str(e), "query": query}

    # Run all queries in parallel
    search_results = await asyncio.gather(
        *[search_single_query(q) for q in queries_to_search],
        return_exceptions=True,
    )

    # Process results
    for res in search_results:
        if isinstance(res, Exception):
            errors.append(str(res))
            continue
        if not res.get("success"):
            errors.append(res.get("error", "Unknown error"))
            continue

        result = res["result"]
        from_cache_count += result.get("from_cache", 0)
        from_zotero_count += result.get("from_zotero", 0)
        papers = result.get("papers", [])
        all_papers.extend(papers)

        # Aggregate source counts from search result (not per-paper "source" field,
        # since UnifiedPaper uses "sources" dict, not a single "source" string)
        result_counts = result.get("source_counts", {})
        for src, count in result_counts.items():
            if src in source_counts:
                source_counts[src] += count
            else:
                source_counts[src] = count

    # Deduplicate by paper_id or title similarity
    seen_ids = set()
    seen_titles = set()
    unique_papers = []

    for paper in all_papers:
        paper_id = paper.get("paper_id") or paper.get("arxiv_id") or paper.get("doi")
        title = paper.get("title", "").lower().strip()

        if paper_id and paper_id in seen_ids:
            continue
        if title and title in seen_titles:
            continue

        if paper_id:
            seen_ids.add(paper_id)
        if title:
            seen_titles.add(title)
        unique_papers.append(paper)

    # Ensure Zotero count is reflected in source_counts
    if from_zotero_count > 0 and source_counts.get("zotero", 0) == 0:
        source_counts["zotero"] = from_zotero_count

    origin_parts = []
    if from_cache_count > 0:
        origin_parts.append(f"{from_cache_count} from cache")
    if from_zotero_count > 0:
        origin_parts.append(f"{from_zotero_count} from Zotero")
    origin_info = f" ({', '.join(origin_parts)})" if origin_parts else ""
    logger.info(f"Found {len(unique_papers)} unique papers from {len(all_papers)} total results{origin_info}")

    return {
        "papers": unique_papers,
        "source_counts": source_counts,
        "total_found": len(unique_papers),
        "errors": errors,
        "from_cache": from_cache_count,
        "from_zotero": from_zotero_count,
    }


async def select_papers(
    papers: List[Dict[str, Any]],
    research_question: str,
    max_papers: int = 10,
    model: Optional[str] = None,
    domain_hint: str = "",
    tracker=None,
    sub_topics: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Select the most relevant papers for the literature review.

    Uses LLM to evaluate and rank papers based on relevance,
    impact, recency, and diversity.

    Args:
        papers: List of candidate papers
        research_question: The research question
        max_papers: Maximum number of papers to select
        model: LLM model to use
        domain_hint: Research domain for filtering
        sub_topics: Sub-topic list from research plan

    Returns:
        Selected papers in ranked order
    """
    # Filter out very low relevance papers regardless of count
    MIN_RELEVANCE = 0.35
    pre_filter_count = len(papers)
    papers = [p for p in papers if (p.get("relevance_score") or 0) >= MIN_RELEVANCE]
    if len(papers) < pre_filter_count:
        logger.info(f"Filtered {pre_filter_count - len(papers)} papers below relevance threshold ({MIN_RELEVANCE})")

    if len(papers) <= max_papers:
        logger.info(f"Only {len(papers)} papers found, using all")
        return papers

    # Format papers for prompt
    papers_list = format_papers_for_prompt(papers)

    # Format sub-topics section if available
    sub_topics_section = ""
    if sub_topics:
        topic_names = [t.get("name", "") for t in sub_topics if t.get("selected", True)]
        if topic_names:
            sub_topics_section = "Sub-topics to cover:\n" + "\n".join(f"- {name}" for name in topic_names)

    prompt = PAPER_SELECTION_PROMPT.format(
        research_question=research_question,
        domain_hint=domain_hint or "General",
        sub_topics_section=sub_topics_section,
        total_papers=len(papers),
        papers_list=papers_list,
        max_papers=max_papers,
    )

    try:
        selected_ids = await call_llm_for_json(prompt, model=model, temperature=0.3, max_tokens=500, tracker=tracker, agent_name="discovery")

        if not isinstance(selected_ids, list):
            raise ValueError("Expected JSON array of paper IDs")

        # Match selected IDs to papers
        id_to_paper = {}
        for paper in papers:
            paper_id = paper.get("paper_id") or paper.get("arxiv_id") or paper.get("doi")
            if paper_id:
                id_to_paper[paper_id] = paper

        selected_papers = []
        for pid in selected_ids[:max_papers]:
            if pid in id_to_paper:
                selected_papers.append(id_to_paper[pid])

        # If not enough papers selected, add more by citation count
        if len(selected_papers) < max_papers:
            remaining = [p for p in papers if p not in selected_papers]
            remaining.sort(key=lambda x: x.get("citations", 0) or 0, reverse=True)
            selected_papers.extend(remaining[: max_papers - len(selected_papers)])

        logger.info(f"Selected {len(selected_papers)} papers from {len(papers)} candidates")
        return selected_papers

    except Exception as e:
        logger.warning(f"Paper selection failed: {e}, falling back to relevance-based ranking")
        # Fall back to relevance_score first, then citations as tiebreaker
        sorted_papers = sorted(
            papers,
            key=lambda x: (x.get("relevance_score", 0) or 0, x.get("citations", 0) or 0),
            reverse=True,
        )
        return sorted_papers[:max_papers]


def _paper_matches_keywords(paper: Dict[str, Any], keywords: List[str], min_matches: int = 2) -> bool:
    """Check if a paper's title/abstract contains enough keywords from the research question.

    Args:
        paper: Paper dict with title and abstract
        keywords: Keywords extracted from research question
        min_matches: Minimum number of keyword matches required

    Returns:
        True if paper matches enough keywords
    """
    text = ((paper.get("title") or "") + " " + (paper.get("abstract") or "")).lower()
    matches = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text))
    return matches >= min_matches


def _extract_keywords(research_question: str) -> List[str]:
    """Extract meaningful keywords from research question (skip stopwords)."""
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
    # Keep words >= 3 chars that aren't stopwords
    return [w.strip("?.,!\"'()") for w in words if len(w) >= 3 and w.lower() not in stopwords]


def _extract_queries_from_papers(
    papers: List[Dict[str, Any]],
    research_question: str,
    max_queries: int = 3,
) -> List[str]:
    """Extract additional search queries from initial search results.

    Analyzes titles and keywords of top papers to discover domain-specific
    terms not in the original query, then builds new queries from them.
    No LLM call needed — purely statistical keyword extraction.

    Args:
        papers: Papers from initial search
        research_question: Original research question
        max_queries: Maximum queries to generate

    Returns:
        List of new search queries derived from paper keywords
    """
    from collections import Counter

    # Existing keywords from the question (to avoid duplicating)
    question_keywords = set(_extract_keywords(research_question))

    stopwords = {
        "the", "a", "an", "in", "on", "of", "for", "and", "or", "to", "is",
        "are", "was", "were", "what", "how", "which", "that", "this", "with",
        "from", "by", "at", "its", "their", "have", "has", "been", "be",
        "do", "does", "did", "will", "can", "could", "would", "should",
        "about", "into", "through", "during", "before", "after", "between",
        "using", "based", "novel", "new", "study", "analysis", "paper",
        "research", "results", "method", "approach", "review", "effect",
        "effects", "role", "via", "two", "one", "high", "low", "different",
        "specific", "related", "associated", "potential", "used", "use",
    }

    # Count terms from titles and keywords of top papers
    term_counter = Counter()
    top_papers = sorted(papers, key=lambda p: p.get("relevance_score", 0) or 0, reverse=True)[:15]

    for paper in top_papers:
        # Extract from title
        title = (paper.get("title") or "").lower()
        for word in title.split():
            word = word.strip(".,;:!?()[]\"'-")
            if len(word) >= 4 and word not in stopwords and word not in question_keywords:
                term_counter[word] += 1

        # Extract from fields_of_study / keywords
        fields = paper.get("fields_of_study") or paper.get("keywords") or []
        if isinstance(fields, str):
            fields = [fields]
        for field in fields:
            for word in field.lower().split():
                word = word.strip(".,;:!?()[]\"'-")
                if len(word) >= 4 and word not in stopwords and word not in question_keywords:
                    term_counter[word] += 1

    # Get the most frequent novel terms (appearing in 3+ papers)
    frequent_terms = [term for term, count in term_counter.most_common(20) if count >= 2]

    if not frequent_terms:
        return []

    # Build queries by combining question keywords with novel terms
    queries = []
    q_kw_list = list(question_keywords)[:3]  # Top question keywords
    q_base = " ".join(q_kw_list) if q_kw_list else research_question[:50]

    for term in frequent_terms[:max_queries]:
        query = f"{q_base} {term}"
        queries.append(query)

    logger.info(f"Paper keyword extraction: {len(queries)} additional queries from {len(frequent_terms)} novel terms")
    return queries


async def snowball_sampling(
    seed_papers: List[Dict[str, Any]],
    max_additional: int = 5,
    direction: str = "both",
    research_question: str = "",
    max_rounds: int = 2,
) -> List[Dict[str, Any]]:
    """Find additional papers through multi-round citation network expansion.

    Round 1: Expand from top-3 seed papers (citations + references).
    Round 2: Expand from best papers found in round 1 + co-citation analysis.

    Co-citation: papers referenced by >=2 seed papers are likely core literature.

    Args:
        seed_papers: Initial set of papers to expand from
        max_additional: Maximum additional papers to add
        direction: "citations", "references", or "both"
        research_question: Research question for keyword-based relevance filtering
        max_rounds: Number of snowball rounds (default: 2)

    Returns:
        List of additional papers found through snowball sampling
    """
    additional_papers = []
    seen_ids = {
        p.get("paper_id") or p.get("arxiv_id") or p.get("doi")
        for p in seed_papers
    }

    # Extract keywords for relevance validation (relaxed matching for snowball)
    keywords = _extract_keywords(research_question) if research_question else []
    snowball_min_matches = 2 if len(keywords) >= 4 else 1

    current_seeds = sorted(
        seed_papers,
        key=lambda x: x.get("citations", 0) or 0,
        reverse=True,
    )[:5]

    for round_num in range(max_rounds):
        round_papers = []

        for paper in current_seeds:
            paper_id = paper.get("paper_id") or paper.get("arxiv_id") or paper.get("doi")
            if not paper_id:
                continue

            try:
                # Get citations (papers citing this one)
                if direction in ("citations", "both"):
                    citations_result = await get_paper_citations(paper_id, limit=10)
                    for cited_paper in (citations_result or {}).get("citations", [])[:7]:
                        cited_id = cited_paper.get("paper_id") or cited_paper.get("paperId")
                        if cited_id and cited_id not in seen_ids:
                            if keywords and not _paper_matches_keywords(cited_paper, keywords, min_matches=snowball_min_matches):
                                continue
                            seen_ids.add(cited_id)
                            round_papers.append(cited_paper)

                # Get references (papers this one cites)
                if direction in ("references", "both"):
                    refs_result = await get_paper_references(paper_id, limit=10)
                    for ref_paper in (refs_result or {}).get("references", [])[:7]:
                        ref_id = ref_paper.get("paper_id") or ref_paper.get("paperId")
                        if ref_id and ref_id not in seen_ids:
                            if keywords and not _paper_matches_keywords(ref_paper, keywords, min_matches=snowball_min_matches):
                                continue
                            seen_ids.add(ref_id)
                            round_papers.append(ref_paper)

            except Exception as e:
                logger.warning(f"Snowball round {round_num + 1} failed for {paper_id}: {e}")
                continue

        # Co-citation analysis on round 1: find papers referenced by multiple seeds
        if round_num == 0 and len(current_seeds) >= 2:
            try:
                co_cited = await _extract_common_references(
                    current_seeds, seen_ids, keywords, snowball_min_matches,
                )
                round_papers.extend(co_cited)
                if co_cited:
                    logger.info(f"Co-citation analysis found {len(co_cited)} commonly referenced papers")
            except Exception as e:
                logger.warning(f"Co-citation analysis failed: {e}")

        additional_papers.extend(round_papers)
        logger.info(f"Snowball round {round_num + 1}: found {len(round_papers)} papers")

        if len(additional_papers) >= max_additional:
            break

        # Prepare seeds for next round: best papers from this round
        if round_papers:
            current_seeds = sorted(
                round_papers,
                key=lambda x: x.get("citations", 0) or x.get("citationCount", 0) or 0,
                reverse=True,
            )[:3]
        else:
            break  # No new papers found, stop

    logger.info(f"Snowball sampling found {len(additional_papers)} additional papers over {min(round_num + 1, max_rounds)} rounds")
    return additional_papers[:max_additional]


async def _extract_common_references(
    seed_papers: List[Dict[str, Any]],
    seen_ids: set,
    keywords: List[str],
    min_matches: int,
    min_co_citations: int = 2,
) -> List[Dict[str, Any]]:
    """Find papers commonly referenced by multiple seed papers.

    If seeds A, B, C all cite paper X, then X is likely core literature.

    Args:
        seed_papers: Papers to analyze references from
        seen_ids: Already-seen paper IDs to skip (will be modified)
        keywords: Keywords for relevance filtering
        min_matches: Minimum keyword matches required
        min_co_citations: Minimum number of seeds that must cite the paper

    Returns:
        List of commonly referenced papers
    """
    from collections import Counter

    ref_count: Counter = Counter()
    ref_info: Dict[str, Dict[str, Any]] = {}

    for seed in seed_papers:
        seed_id = seed.get("paper_id") or seed.get("arxiv_id") or seed.get("doi")
        if not seed_id:
            continue

        try:
            refs_result = await get_paper_references(seed_id, limit=20)
            for ref in (refs_result or {}).get("references", []):
                ref_id = ref.get("paper_id") or ref.get("paperId")
                if ref_id and ref_id not in seen_ids:
                    ref_count[ref_id] += 1
                    ref_info[ref_id] = ref
        except Exception as e:
            logger.debug(f"Co-citation: failed to get references for {seed_id}: {e}")

    # Return papers cited by >= min_co_citations seeds
    co_cited = []
    for pid, count in ref_count.items():
        if count >= min_co_citations and pid not in seen_ids:
            paper = ref_info[pid]
            # Still validate keywords
            if keywords and not _paper_matches_keywords(paper, keywords, min_matches=max(1, min_matches - 1)):
                continue
            seen_ids.add(pid)
            co_cited.append(paper)

    return co_cited


async def screen_papers_by_abstract(
    papers: List[Dict[str, Any]],
    research_question: str,
    domain_hint: str = "",
    batch_size: int = 10,
    tracker=None,
) -> List[Dict[str, Any]]:
    """Screen papers by abstract using a lightweight LLM call.

    Quickly filters obviously irrelevant papers before expensive
    critical_reading analysis. Uses batch LLM calls for efficiency.

    Args:
        papers: Papers to screen (each must have title and abstract)
        research_question: The research question
        domain_hint: Research domain for context
        batch_size: Number of papers per LLM call
        tracker: Token tracker

    Returns:
        Papers that passed screening (classified as relevant)
    """
    if not papers:
        return papers

    relevant_papers = []
    paper_id_map = {}

    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]

        # Format batch for prompt
        batch_lines = []
        for paper in batch:
            pid = paper.get("paper_id") or paper.get("arxiv_id") or paper.get("doi") or f"paper_{i}"
            paper_id_map[pid] = paper
            title = paper.get("title", "Unknown")
            abstract = (paper.get("abstract") or "")[:500]
            batch_lines.append(f"[{pid}] {title}\nAbstract: {abstract}\n")

        papers_batch = "\n".join(batch_lines)

        prompt = ABSTRACT_SCREENING_PROMPT.format(
            research_question=research_question,
            domain_hint=domain_hint or "General",
            papers_batch=papers_batch,
        )

        try:
            response = await call_llm(
                prompt, temperature=0.2, max_tokens=800,
                tracker=tracker, agent_name="discovery",
            )
            results = extract_json(response)

            if not isinstance(results, list):
                # If parsing fails, keep all papers in this batch
                logger.warning("Abstract screening: failed to parse response, keeping all papers in batch")
                relevant_papers.extend(batch)
                continue

            # Collect relevant paper IDs
            relevant_ids = set()
            for entry in results:
                if entry.get("relevant", True):  # Default to relevant if unclear
                    relevant_ids.add(entry.get("paper_id", ""))

            # Match back to papers
            batch_relevant = []
            batch_removed = []
            for paper in batch:
                pid = paper.get("paper_id") or paper.get("arxiv_id") or paper.get("doi") or ""
                if pid in relevant_ids:
                    batch_relevant.append(paper)
                else:
                    batch_removed.append(paper.get("title", "Unknown"))

            if batch_removed:
                logger.info(
                    f"Abstract screening: removed {len(batch_removed)} papers from batch: "
                    + "; ".join(batch_removed[:3])
                )

            relevant_papers.extend(batch_relevant)

        except Exception as e:
            logger.warning(f"Abstract screening batch failed: {e}, keeping all papers in batch")
            relevant_papers.extend(batch)

    removed_count = len(papers) - len(relevant_papers)
    if removed_count > 0:
        logger.info(f"Abstract screening: {removed_count}/{len(papers)} papers removed, {len(relevant_papers)} passed")
    else:
        logger.info(f"Abstract screening: all {len(papers)} papers passed")

    return relevant_papers


async def discovery_agent(state: LitScribeState) -> Dict[str, Any]:
    """Main entry point for the Discovery Agent.

    This function is called by the LangGraph workflow to execute
    the discovery phase of the literature review.

    Supports local-first search with SQLite caching when cache_enabled=True.

    Args:
        state: Current workflow state

    Returns:
        State updates with search results and selected papers
    """
    research_question = state["research_question"]
    sources = state.get("sources", ["arxiv", "semantic_scholar", "pubmed"])
    max_papers = state.get("max_papers", 10)
    cache_enabled = state.get("cache_enabled", True)
    errors = list(state.get("errors", []))
    from utils.token_tracker import get_tracker
    tracker = get_tracker()

    research_plan = state.get("research_plan")
    domain_hint = state.get("domain_hint", "")

    # Extract domain filters from research plan (skip if ablation disabled)
    arxiv_categories = None
    s2_fields = None
    pubmed_mesh = None
    disable_domain_filter = state.get("disable_domain_filter", False)
    disable_snowball = state.get("disable_snowball", False)
    zotero_collection_raw = state.get("zotero_collection")
    zotero_collection = None  # Will be resolved to a key below

    # Resolve zotero_collection: name → key (if needed)
    if zotero_collection_raw:
        from cache.cached_tools import resolve_zotero_collection, _zotero_available
        if _zotero_available():
            zotero_collection = await resolve_zotero_collection(zotero_collection_raw)
            if zotero_collection:
                logger.info(f"Zotero collection: {zotero_collection_raw} → key {zotero_collection}")
            else:
                logger.warning(f"Could not resolve Zotero collection '{zotero_collection_raw}', searching entire library")

    if research_plan and not disable_domain_filter:
        domain_hint = domain_hint or research_plan.get("domain_hint", "")
        arxiv_categories = research_plan.get("arxiv_categories") or None
        s2_fields = research_plan.get("s2_fields") or None
        pubmed_mesh = research_plan.get("pubmed_mesh") or None
    elif disable_domain_filter:
        logger.info("Domain filtering disabled (ablation mode)")

    logger.info(f"Discovery Agent starting for: {research_question}")
    logger.info(f"Cache enabled: {cache_enabled}, domain: {domain_hint or 'undetected'}")
    if arxiv_categories:
        logger.info(f"arXiv categories: {arxiv_categories}")
    if s2_fields:
        logger.info(f"S2 fields: {s2_fields}")
    if pubmed_mesh:
        logger.info(f"PubMed MeSH: {pubmed_mesh}")

    # Local-files-only mode: skip online search (Emergency Fix Step 6)
    local_files = state.get("local_files", [])
    if not sources and local_files:
        logger.info("Local-files-only mode: skipping online search")
        return {
            "search_results": SearchResult(
                query=research_question,
                expanded_queries=[],
                papers=[],
                source_counts={},
                total_found=0,
                search_timestamp=datetime.now().isoformat(),
            ),
            "papers_to_analyze": [],
            "errors": errors,
            "current_agent": "critical_reading",
        }

    # Initialize cached tools if caching is enabled
    cached_tools = get_cached_tools(cache_enabled=cache_enabled) if cache_enabled else None

    try:
        # Get tier configuration for search parameters
        review_tier = state.get("review_tier", "standard")
        tier_cfg = TIER_CONFIG.get(review_tier, TIER_CONFIG["standard"])
        use_per_subtopic = tier_cfg.get("per_subtopic_search", False)
        max_per_src = tier_cfg.get("max_per_source", 15)
        max_q_per_topic = tier_cfg.get("max_queries_per_topic", 3)

        logger.info(f"Discovery using tier '{review_tier}': per_subtopic={use_per_subtopic}, max_per_source={max_per_src}")

        # Step 1: Build queries — per-sub-topic for standard/comprehensive, flat for quick
        selected_topics = []
        if research_plan and research_plan.get("sub_topics"):
            selected_topics = [t for t in research_plan["sub_topics"] if t.get("selected", True)]

        if use_per_subtopic and selected_topics:
            # Per-sub-topic search: each topic gets its own search round
            all_papers = []
            all_source_counts: Dict[str, int] = {}
            expanded_queries = []
            existing_ids: set = set()
            existing_titles: set = set()

            for topic in selected_topics:
                topic_queries = topic.get("custom_queries", [])[:max_q_per_topic]
                if not topic_queries:
                    topic_queries = [topic["name"]]

                expanded_queries.extend(topic_queries)
                logger.info(f"Searching sub-topic '{topic['name']}' with {len(topic_queries)} queries")

                topic_results = await search_all_sources(
                    queries=topic_queries,
                    sources=sources,
                    max_per_source=max_per_src,
                    cached_tools=cached_tools,
                    arxiv_categories=arxiv_categories,
                    s2_fields=s2_fields,
                    pubmed_mesh=pubmed_mesh,
                    zotero_collection=zotero_collection,
                )

                if topic_results.get("errors"):
                    errors.extend(topic_results["errors"])

                # Merge results, dedup across topics
                for p in topic_results.get("papers", []):
                    pid = p.get("paper_id") or p.get("arxiv_id") or p.get("doi")
                    ptitle = (p.get("title") or "").lower().strip()
                    if (not pid or pid not in existing_ids) and (not ptitle or ptitle not in existing_titles):
                        all_papers.append(p)
                        if pid:
                            existing_ids.add(pid)
                        if ptitle:
                            existing_titles.add(ptitle)

                # Merge source counts
                for src, count in topic_results.get("source_counts", {}).items():
                    all_source_counts[src] = all_source_counts.get(src, 0) + count

            # Also run LLM query expansion for additional coverage
            llm_queries = await expand_queries(research_question, domain_hint=domain_hint, tracker=tracker)
            novel_llm = [q for q in llm_queries if q not in expanded_queries]
            if novel_llm:
                llm_results = await search_all_sources(
                    queries=novel_llm[:4],
                    sources=sources,
                    max_per_source=max_per_src,
                    cached_tools=cached_tools,
                    arxiv_categories=arxiv_categories,
                    s2_fields=s2_fields,
                    pubmed_mesh=pubmed_mesh,
                    zotero_collection=zotero_collection,
                )
                for p in llm_results.get("papers", []):
                    pid = p.get("paper_id") or p.get("arxiv_id") or p.get("doi")
                    ptitle = (p.get("title") or "").lower().strip()
                    if (not pid or pid not in existing_ids) and (not ptitle or ptitle not in existing_titles):
                        all_papers.append(p)
                        if pid:
                            existing_ids.add(pid)
                        if ptitle:
                            existing_titles.add(ptitle)
                expanded_queries.extend(novel_llm[:4])
                for src, count in llm_results.get("source_counts", {}).items():
                    all_source_counts[src] = all_source_counts.get(src, 0) + count

            papers = all_papers
            search_results = {"papers": papers, "source_counts": all_source_counts, "errors": []}
            logger.info(f"Per-sub-topic search: {len(papers)} unique papers from {len(selected_topics)} topics")

        else:
            # Quick tier or no plan: flat search (original behavior)
            plan_queries = []
            for topic in selected_topics:
                queries_for_topic = topic.get("custom_queries", [])
                priority = topic.get("priority", 0.5)
                if priority >= 0.8:
                    plan_queries.extend(queries_for_topic[:3])
                elif priority >= 0.5:
                    plan_queries.extend(queries_for_topic[:2])
                else:
                    plan_queries.extend(queries_for_topic[:1])

            if plan_queries:
                llm_queries = await expand_queries(research_question, domain_hint=domain_hint, tracker=tracker)
                expanded_queries = plan_queries + [q for q in llm_queries if q not in plan_queries]
                logger.info(f"Using {len(plan_queries)} plan queries + {len(llm_queries)} expanded queries")
            else:
                expanded_queries = await expand_queries(research_question, domain_hint=domain_hint, tracker=tracker)

            search_results = await search_all_sources(
                queries=expanded_queries,
                sources=sources,
                max_per_source=max_per_src,
                cached_tools=cached_tools,
                arxiv_categories=arxiv_categories,
                s2_fields=s2_fields,
                pubmed_mesh=pubmed_mesh,
                zotero_collection=zotero_collection,
            )

            if search_results.get("errors"):
                errors.extend(search_results["errors"])

            papers = search_results.get("papers", [])

        # Step 1b: Append additional_queries from self-review loop-back (if any)
        additional_queries = state.get("additional_queries", [])
        if additional_queries:
            new_queries = [q for q in additional_queries if q not in expanded_queries]
            if new_queries:
                expanded_queries.extend(new_queries)
                logger.info(f"Added {len(new_queries)} additional queries from self-review loop-back")
                # Search additional queries
                extra_results = await search_all_sources(
                    queries=new_queries,
                    sources=sources,
                    max_per_source=max_per_src,
                    cached_tools=cached_tools,
                    arxiv_categories=arxiv_categories,
                    s2_fields=s2_fields,
                    pubmed_mesh=pubmed_mesh,
                    zotero_collection=zotero_collection,
                )
                existing_ids_set = {p.get("paper_id") or p.get("arxiv_id") or p.get("doi") for p in papers}
                for p in extra_results.get("papers", []):
                    pid = p.get("paper_id") or p.get("arxiv_id") or p.get("doi")
                    if not pid or pid not in existing_ids_set:
                        papers.append(p)
                        if pid:
                            existing_ids_set.add(pid)

        if not papers:
            error_msg = "No papers found for the research question"
            logger.warning(error_msg)
            errors.append(error_msg)
            return {
                "search_results": SearchResult(
                    query=research_question,
                    expanded_queries=expanded_queries,
                    papers=[],
                    source_counts={},
                    total_found=0,
                    search_timestamp=datetime.now().isoformat(),
                ),
                "papers_to_analyze": [],
                "errors": errors,
                "current_agent": "complete",
            }

        # Step 2b: Second-round search using keywords extracted from initial results
        keyword_queries = _extract_queries_from_papers(papers, research_question, max_queries=5)
        if keyword_queries:
            novel_queries = [q for q in keyword_queries if q not in expanded_queries]
            if novel_queries:
                round2_results = await search_all_sources(
                    queries=novel_queries,
                    sources=sources,
                    max_per_source=10,
                    cached_tools=cached_tools,
                    arxiv_categories=arxiv_categories,
                    s2_fields=s2_fields,
                    pubmed_mesh=pubmed_mesh,
                    zotero_collection=zotero_collection,
                )
                round2_papers = round2_results.get("papers", [])
                if round2_papers:
                    existing_ids = {p.get("paper_id") or p.get("arxiv_id") or p.get("doi") for p in papers}
                    existing_titles = {(p.get("title") or "").lower().strip() for p in papers}
                    new_count = 0
                    for p in round2_papers:
                        pid = p.get("paper_id") or p.get("arxiv_id") or p.get("doi")
                        ptitle = (p.get("title") or "").lower().strip()
                        if (pid and pid not in existing_ids) and (ptitle and ptitle not in existing_titles):
                            papers.append(p)
                            if pid:
                                existing_ids.add(pid)
                            if ptitle:
                                existing_titles.add(ptitle)
                            new_count += 1
                    logger.info(f"Second-round search: {new_count} new papers from {len(novel_queries)} keyword queries")
                    expanded_queries.extend(novel_queries)

                    for src, count in round2_results.get("source_counts", {}).items():
                        current = search_results.get("source_counts", {}).get(src, 0)
                        search_results.setdefault("source_counts", {})[src] = current + count

        # Step 3: Select best papers
        plan_sub_topics = research_plan.get("sub_topics") if research_plan else None
        selected_papers = await select_papers(
            papers=papers,
            research_question=research_question,
            max_papers=max_papers,
            domain_hint=domain_hint,
            tracker=tracker,
            sub_topics=plan_sub_topics,
        )

        # Step 4: Optional snowball sampling if we need more papers (tier-aware)
        snowball_rounds = tier_cfg.get("snowball_rounds", 2)
        snowball_seeds = tier_cfg.get("snowball_seeds", 3)
        if len(selected_papers) < max_papers and len(selected_papers) > 0 and not disable_snowball:
            additional = await snowball_sampling(
                seed_papers=selected_papers,
                max_additional=max(max_papers - len(selected_papers), 10),
                research_question=research_question,
                max_rounds=snowball_rounds,
            )
            # Re-select from combined pool
            all_candidates = selected_papers + additional
            selected_papers = await select_papers(
                papers=all_candidates,
                research_question=research_question,
                max_papers=max_papers,
                domain_hint=domain_hint,
                tracker=tracker,
                sub_topics=plan_sub_topics,
            )

        # Step 5: Abstract screening — lightweight LLM check before critical_reading
        if len(selected_papers) > 3:
            pre_screen_count = len(selected_papers)
            selected_papers = await screen_papers_by_abstract(
                papers=selected_papers,
                research_question=research_question,
                domain_hint=domain_hint,
                tracker=tracker,
            )
            screened_out = pre_screen_count - len(selected_papers)
            if screened_out > 0:
                logger.info(f"Abstract screening removed {screened_out} papers before critical_reading")

        # Build SearchResult
        search_result = SearchResult(
            query=research_question,
            expanded_queries=expanded_queries,
            papers=papers,  # All papers found
            source_counts=search_results.get("source_counts", {}),
            total_found=len(papers),
            search_timestamp=datetime.now().isoformat(),
        )

        logger.info(
            f"Discovery complete: {len(selected_papers)} papers selected "
            f"from {len(papers)} found"
        )

        # Auto-save selected papers to Zotero collection (if specified)
        if zotero_collection and selected_papers:
            try:
                from services.zotero import save_papers_to_collection
                save_result = await save_papers_to_collection(
                    papers=selected_papers,
                    collection_key=zotero_collection,
                )
                saved = save_result.get("saved", 0)
                failed = save_result.get("failed", 0)
                skipped = save_result.get("skipped", 0)
                logger.info(
                    f"Zotero auto-save to collection {zotero_collection}: "
                    f"{saved} saved, {failed} failed, {skipped} skipped"
                )
            except Exception as e:
                logger.warning(f"Zotero auto-save failed (non-blocking): {e}")

        return {
            "search_results": search_result,
            "papers_to_analyze": selected_papers,
            "errors": errors,
            "current_agent": "critical_reading",  # Next stage
        }

    except Exception as e:
        import traceback
        error_msg = f"Discovery Agent failed: {e}"
        logger.error(error_msg)
        logger.error(f"Discovery Agent traceback:\n{traceback.format_exc()}")
        errors.append(error_msg)
        return {
            "errors": errors,
            "current_agent": "complete",  # End on fatal error
        }


# Export for use in graph.py
__all__ = [
    "discovery_agent",
    "expand_queries",
    "search_all_sources",
    "select_papers",
    "snowball_sampling",
]
