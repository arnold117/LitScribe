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
        svc_cfg = getattr(config, "services", None)
        ncbi_email = getattr(svc_cfg, "ncbi_email", None) or getattr(config, "ncbi_email", None)
        ncbi_key = getattr(svc_cfg, "ncbi_api_key", None) or getattr(config, "ncbi_api_key", None)
        if ncbi_email:
            services.append(PubMedService(ncbi_email=ncbi_email, ncbi_api_key=ncbi_key))
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


async def domain_filter(
    router,
    papers: list[Paper],
    research_question: str,
    domain: str,
) -> list[Paper]:
    if len(papers) <= 5:
        return papers

    batch_size = 10
    kept: list[Paper] = []

    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        batch_text = "\n".join(
            f'- paper_id: {p.paper_id}, title: "{p.title}", abstract: "{(p.abstract or "")[:200]}"'
            for p in batch
        )
        prompt = ABSTRACT_SCREENING_PROMPT.format(
            research_question=research_question,
            domain_hint=domain,
            papers_batch=batch_text,
        )
        try:
            results = await router.call_json(_msg(prompt), task_type="planning")
            if isinstance(results, list):
                relevant_ids = {
                    r.get("paper_id") for r in results
                    if isinstance(r, dict) and r.get("relevant", False)
                }
                kept.extend(p for p in batch if p.paper_id in relevant_ids)
            else:
                kept.extend(batch)
        except Exception as e:
            logger.warning(f"Domain filter failed for batch: {e}, keeping all")
            kept.extend(batch)

    if not kept:
        logger.warning("Domain filter removed all papers, keeping original")
        return papers

    # Second pass: stricter check on borderline papers if still too many
    if len(kept) > 30:
        logger.info(f"Domain filter pass 2: {len(kept)} papers still large, running strict filter")
        batch_text = "\n".join(
            f'- paper_id: {p.paper_id}, title: "{p.title}", abstract: "{(p.abstract or "")[:150]}"'
            for p in kept[:30]
        )
        strict_prompt = (
            f"You are filtering papers for a VERY SPECIFIC literature review.\n\n"
            f"Research Question: {research_question}\n"
            f"Domain: {domain}\n\n"
            f"For each paper, answer: is this paper's PRIMARY topic directly about "
            f'"{research_question}"? Not just mentioning it — it must be a central focus.\n\n'
            f"Papers:\n{batch_text}\n\n"
            f"Output JSON array: [{{\"paper_id\": \"...\", \"relevant\": true/false}}]"
        )
        try:
            results2 = await router.call_json(_msg(strict_prompt), task_type="planning")
            if isinstance(results2, list):
                strict_ids = {
                    r.get("paper_id") for r in results2
                    if isinstance(r, dict) and r.get("relevant", False)
                }
                strict_kept = [p for p in kept if p.paper_id in strict_ids]
                if len(strict_kept) >= 5:
                    kept = strict_kept
                    logger.info(f"Domain filter pass 2: {len(kept)} papers after strict filter")
        except Exception as e:
            logger.warning(f"Domain filter pass 2 failed: {e}")

    logger.info(f"Domain filter: {len(papers)} → {len(kept)} papers")
    return kept


async def snowball_sample(
    papers: list[Paper],
    config,
    max_extra: int = 10,
) -> list[Paper]:
    try:
        from litscribe.services.semantic_scholar import SemanticScholarService
        s2_key = getattr(config, "s2_api_key", None) or getattr(
            getattr(config, "services", None), "s2_api_key", None
        )
        s2 = SemanticScholarService(api_key=s2_key)
    except Exception:
        return papers

    seed_ids = [p.paper_id for p in papers if p.paper_id.startswith("s2:") or ":" not in p.paper_id]
    if not seed_ids:
        return papers

    cited_papers: list[Paper] = []
    existing_ids = {p.paper_id for p in papers}

    for seed_id in seed_ids[:5]:
        try:
            refs = await s2.search(f"references:{seed_id}", max_results=5)
            for p in refs:
                if p.paper_id not in existing_ids:
                    existing_ids.add(p.paper_id)
                    cited_papers.append(p)
                    if len(cited_papers) >= max_extra:
                        break
        except Exception:
            continue
        if len(cited_papers) >= max_extra:
            break

    if cited_papers:
        logger.info(f"Snowball: added {len(cited_papers)} papers from references")
    return papers + cited_papers


async def discover_papers(
    research_question: str,
    domain: str,
    config,
    router,
    max_papers: int = 40,
    extra_queries: list[str] | None = None,
    disable_domain_filter: bool = False,
    enable_snowball: bool = True,
) -> dict[str, Any]:
    queries = await expand_queries(router, research_question, domain)
    if extra_queries:
        queries.extend(extra_queries)

    papers = await search_all_sources(queries, config)

    if not disable_domain_filter and len(papers) > 10:
        papers = await domain_filter(router, papers, research_question, domain)

    if enable_snowball and len(papers) >= 3:
        papers = await snowball_sample(papers, config, max_extra=min(10, max_papers // 4))

    papers = dedup_papers(papers)
    selected = await select_papers(router, papers, research_question, domain, max_papers)

    return {
        "papers": selected,
        "total_found": len(papers),
        "total_selected": len(selected),
        "queries_used": len(queries),
    }
