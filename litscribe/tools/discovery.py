from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from litscribe.errors import LLMError
from litscribe.models.paper import Paper
from litscribe.prompts.discovery import (
    ABSTRACT_SCREENING_PROMPT,
    PAPER_SELECTION_PROMPT,
    QUERY_EXPANSION_PROMPT,
)
from litscribe.prompts.utils import format_papers_for_prompt
from litscribe.services.base import dedup_papers

logger = logging.getLogger(__name__)


def _msg(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


async def expand_queries(
    router,
    research_question: str,
    domain: str = "General",
) -> list[str]:
    prompt = QUERY_EXPANSION_PROMPT.format(
        research_question=research_question,
        domain_hint=domain,
    )
    expanded = await router.call_json(_msg(prompt), task_type="query_expansion")

    queries: list[str] = []
    if isinstance(expanded, dict):
        for dim in [
            "core_methodology", "application_domain", "review_meta",
            "recent_advances", "cross_disciplinary", "synonyms_nomenclature",
        ]:
            v = expanded.get(dim, [])
            if isinstance(v, list):
                queries.extend(v)
            elif isinstance(v, str):
                queries.append(v)
    elif isinstance(expanded, list):
        queries = expanded

    all_queries = [research_question] + queries
    seen: set[str] = set()
    deduped = []
    for q in all_queries:
        q_lower = q.strip().lower()
        if q_lower not in seen:
            seen.add(q_lower)
            deduped.append(q.strip())

    logger.info(f"Expanded into {len(deduped)} queries")
    return deduped[:12]


async def search_all_sources(
    queries: list[str],
    config,
    max_per_source: int = 10,
) -> list[Paper]:
    from litscribe.services.arxiv import ArxivService
    from litscribe.services.openalex import OpenAlexService
    from litscribe.services.europe_pmc import EuropePMCService

    services = [ArxivService(), OpenAlexService(), EuropePMCService()]

    try:
        from litscribe.services.semantic_scholar import SemanticScholarService
        s2_key = getattr(config, "s2_api_key", None) or getattr(
            getattr(config, "services", None), "s2_api_key", None
        )
        services.append(SemanticScholarService(api_key=s2_key))
    except Exception:
        pass

    try:
        from litscribe.services.pubmed import PubMedService
        ncbi_email = getattr(config, "ncbi_email", None) or getattr(
            getattr(config, "services", None), "ncbi_email", None
        )
        if ncbi_email:
            services.append(PubMedService(email=ncbi_email))
    except Exception:
        pass

    all_papers: list[Paper] = []
    queries_to_use = queries[:8]
    max_per_query = max(max_per_source // max(len(queries_to_use), 1), 5)

    async def _search_one(svc, query: str) -> list[Paper]:
        try:
            return await svc.search(query, max_results=max_per_query)
        except Exception as e:
            logger.warning(f"{svc.source_name} failed for '{query[:40]}': {e}")
            return []

    tasks = [
        _search_one(svc, q)
        for svc in services
        for q in queries_to_use
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, list):
            all_papers.extend(r)

    deduped = dedup_papers(all_papers)
    logger.info(
        f"Search complete: {len(all_papers)} raw → {len(deduped)} deduped "
        f"from {len(services)} sources × {len(queries_to_use)} queries"
    )
    return deduped


async def select_papers(
    router,
    papers: list[Paper],
    research_question: str,
    domain: str,
    max_papers: int = 40,
) -> list[Paper]:
    if len(papers) <= max_papers:
        return papers

    papers_list = format_papers_for_prompt(
        [p.model_dump() for p in papers], max_chars=15000
    )

    prompt = PAPER_SELECTION_PROMPT.format(
        research_question=research_question,
        domain_hint=domain,
        sub_topics_section="",
        total_papers=len(papers),
        papers_list=papers_list,
        max_papers=max_papers,
    )

    try:
        selected_ids = await router.call_json(_msg(prompt), task_type="planning")
        if isinstance(selected_ids, list):
            id_set = set(str(sid) for sid in selected_ids)
            selected = [p for p in papers if p.paper_id in id_set]
            if selected:
                logger.info(f"LLM selected {len(selected)}/{len(papers)} papers")
                return selected
    except Exception as e:
        logger.warning(f"Paper selection LLM failed: {e}, returning top {max_papers}")

    return papers[:max_papers]


async def discover_papers(
    research_question: str,
    domain: str,
    config,
    router,
    max_papers: int = 40,
    extra_queries: list[str] | None = None,
) -> dict[str, Any]:
    queries = await expand_queries(router, research_question, domain)
    if extra_queries:
        queries.extend(extra_queries)

    papers = await search_all_sources(queries, config)
    selected = await select_papers(router, papers, research_question, domain, max_papers)

    return {
        "papers": selected,
        "total_found": len(papers),
        "total_selected": len(selected),
        "queries_used": len(queries),
    }
