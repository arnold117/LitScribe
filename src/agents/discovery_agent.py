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
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.errors import AgentError, ErrorType, LLMError
from agents.prompts import (
    PAPER_SELECTION_PROMPT,
    QUERY_EXPANSION_PROMPT,
    format_papers_for_prompt,
)
from agents.state import LitScribeState, SearchResult
from agents.tools import (
    call_llm,
    get_paper_citations,
    get_paper_references,
    get_recommendations,
    unified_search,
)
from cache.cached_tools import CachedTools, get_cached_tools

logger = logging.getLogger(__name__)


async def expand_queries(research_question: str, model: Optional[str] = None) -> List[str]:
    """Expand a research question into multiple search queries.

    Uses LLM to generate diverse queries that cover different aspects
    of the research topic.

    Args:
        research_question: The original research question
        model: LLM model to use (optional)

    Returns:
        List of expanded search queries
    """
    prompt = QUERY_EXPANSION_PROMPT.format(research_question=research_question)

    try:
        response = await call_llm(prompt, model=model, temperature=0.7, max_tokens=500)

        # Parse JSON array from response
        # Handle possible markdown code blocks
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        queries = json.loads(response)

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

    # Search with each query in parallel (limit to top 3 queries to avoid rate limits)
    queries_to_search = queries[:3]
    max_per_query = max_per_source // len(queries_to_search)

    async def search_single_query(query: str):
        """Search a single query across all sources."""
        try:
            if cached_tools and cached_tools.cache_enabled:
                result = await cached_tools.search_with_cache(
                    query=query,
                    sources=sources,
                    max_per_source=max_per_query,
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

        # Update source counts
        for paper in papers:
            source = paper.get("source", "unknown")
            if source in source_counts:
                source_counts[source] += 1

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
) -> List[Dict[str, Any]]:
    """Select the most relevant papers for the literature review.

    Uses LLM to evaluate and rank papers based on relevance,
    impact, recency, and diversity.

    Args:
        papers: List of candidate papers
        research_question: The research question
        max_papers: Maximum number of papers to select
        model: LLM model to use

    Returns:
        Selected papers in ranked order
    """
    if len(papers) <= max_papers:
        logger.info(f"Only {len(papers)} papers found, using all")
        return papers

    # Format papers for prompt
    papers_list = format_papers_for_prompt(papers)

    prompt = PAPER_SELECTION_PROMPT.format(
        research_question=research_question,
        total_papers=len(papers),
        papers_list=papers_list,
        max_papers=max_papers,
    )

    try:
        response = await call_llm(prompt, model=model, temperature=0.3, max_tokens=500)

        # Parse JSON array of paper IDs
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        selected_ids = json.loads(response)

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
            remaining.sort(key=lambda x: x.get("citations", 0), reverse=True)
            selected_papers.extend(remaining[: max_papers - len(selected_papers)])

        logger.info(f"Selected {len(selected_papers)} papers from {len(papers)} candidates")
        return selected_papers

    except (json.JSONDecodeError, LLMError) as e:
        logger.warning(f"Paper selection failed: {e}, falling back to citation-based ranking")
        # Fall back to simple citation-based ranking
        sorted_papers = sorted(papers, key=lambda x: x.get("citations", 0), reverse=True)
        return sorted_papers[:max_papers]


async def snowball_sampling(
    seed_papers: List[Dict[str, Any]],
    max_additional: int = 5,
    direction: str = "both",
) -> List[Dict[str, Any]]:
    """Find additional papers through citation network.

    Performs forward (citations) and backward (references) snowball sampling
    to discover related papers not found in initial search.

    Args:
        seed_papers: Initial set of papers to expand from
        max_additional: Maximum additional papers to add
        direction: "citations", "references", or "both"

    Returns:
        List of additional papers found through snowball sampling
    """
    additional_papers = []
    seen_ids = {
        p.get("paper_id") or p.get("arxiv_id") or p.get("doi")
        for p in seed_papers
    }

    # Sample from top cited papers to avoid too many API calls
    top_seeds = sorted(
        seed_papers,
        key=lambda x: x.get("citations", 0),
        reverse=True
    )[:3]

    for paper in top_seeds:
        paper_id = paper.get("paper_id") or paper.get("arxiv_id") or paper.get("doi")
        if not paper_id:
            continue

        try:
            # Get citations (papers citing this one)
            if direction in ("citations", "both"):
                citations_result = await get_paper_citations(paper_id, limit=5)
                for cited_paper in citations_result.get("citations", [])[:3]:
                    cited_id = cited_paper.get("paper_id") or cited_paper.get("paperId")
                    if cited_id and cited_id not in seen_ids:
                        seen_ids.add(cited_id)
                        additional_papers.append(cited_paper)

            # Get references (papers this one cites)
            if direction in ("references", "both"):
                refs_result = await get_paper_references(paper_id, limit=5)
                for ref_paper in refs_result.get("references", [])[:3]:
                    ref_id = ref_paper.get("paper_id") or ref_paper.get("paperId")
                    if ref_id and ref_id not in seen_ids:
                        seen_ids.add(ref_id)
                        additional_papers.append(ref_paper)

            if len(additional_papers) >= max_additional:
                break

        except AgentError as e:
            logger.warning(f"Snowball sampling failed for {paper_id}: {e}")
            continue

    logger.info(f"Snowball sampling found {len(additional_papers)} additional papers")
    return additional_papers[:max_additional]


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

    research_plan = state.get("research_plan")

    logger.info(f"Discovery Agent starting for: {research_question}")
    logger.info(f"Cache enabled: {cache_enabled}")

    # Initialize cached tools if caching is enabled
    cached_tools = get_cached_tools(cache_enabled=cache_enabled) if cache_enabled else None

    try:
        # Step 1: Expand queries (use research plan sub-topics if available)
        if research_plan and research_plan.get("sub_topics"):
            # Use sub-topic queries from planning agent
            plan_queries = []
            for topic in research_plan["sub_topics"]:
                if topic.get("selected", True):
                    plan_queries.extend(topic.get("custom_queries", []))
            if plan_queries:
                # Combine plan queries with LLM-expanded queries
                llm_queries = await expand_queries(research_question)
                expanded_queries = plan_queries + [q for q in llm_queries if q not in plan_queries]
                logger.info(f"Using {len(plan_queries)} plan queries + {len(llm_queries)} expanded queries")
            else:
                expanded_queries = await expand_queries(research_question)
        else:
            expanded_queries = await expand_queries(research_question)

        # Step 2: Search all sources (with caching if enabled)
        search_results = await search_all_sources(
            queries=expanded_queries,
            sources=sources,
            max_per_source=20,
            cached_tools=cached_tools,
        )

        if search_results.get("errors"):
            errors.extend(search_results["errors"])

        papers = search_results.get("papers", [])

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
                "current_agent": "complete",  # End if no papers found
            }

        # Step 3: Select best papers
        selected_papers = await select_papers(
            papers=papers,
            research_question=research_question,
            max_papers=max_papers,
        )

        # Step 4: Optional snowball sampling if we need more papers
        if len(selected_papers) < max_papers and len(selected_papers) > 0:
            additional = await snowball_sampling(
                seed_papers=selected_papers,
                max_additional=max_papers - len(selected_papers),
            )
            # Re-select from combined pool
            all_candidates = selected_papers + additional
            selected_papers = await select_papers(
                papers=all_candidates,
                research_question=research_question,
                max_papers=max_papers,
            )

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

        return {
            "search_results": search_result,
            "papers_to_analyze": selected_papers,
            "errors": errors,
            "current_agent": "critical_reading",  # Next stage
        }

    except Exception as e:
        error_msg = f"Discovery Agent failed: {e}"
        logger.error(error_msg)
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
