from __future__ import annotations

import asyncio
import logging

from litscribe.models.paper import Paper
from litscribe.services.base import dedup_papers

logger = logging.getLogger(__name__)

_search_cache: dict[str, list[Paper]] = {}


async def search_all_sources(
    queries: list[str],
    config,
    max_per_source: int = 10,
    domain: str = "",
) -> list[Paper]:
    from litscribe.services.openalex import OpenAlexService
    from litscribe.services.europe_pmc import EuropePMCService

    services = [OpenAlexService(), EuropePMCService()]

    bio_domains = {"biology", "medicine", "chemistry", "biochemistry", "pharmacology", "bioengineering"}
    cs_domains = {"computer science", "mathematics", "physics", "engineering"}

    if not domain or domain.lower() in cs_domains:
        try:
            from litscribe.services.arxiv import ArxivService
            services.append(ArxivService())
        except Exception as _e:
            logger.debug(f"Silent error: {_e}")

    try:
        from litscribe.services.semantic_scholar import SemanticScholarService
        svc_cfg = getattr(config, "services", None)
        s2_key = getattr(svc_cfg, "semantic_scholar_api_key", None) if svc_cfg else None
        services.append(SemanticScholarService(api_key=s2_key))
    except Exception as _e:
        logger.debug(f"Silent error: {_e}")

    # CrossRef for broader coverage (especially Chinese literature)
    try:
        from litscribe.services.cnki import CNKIService
        services.append(CNKIService())
    except Exception as _e:
        logger.debug(f"Silent error: {_e}")

    try:
        from litscribe.services.pubmed import PubMedService
        svc_cfg = getattr(config, "services", None)
        ncbi_email = getattr(svc_cfg, "ncbi_email", None) if svc_cfg else None
        ncbi_key = getattr(svc_cfg, "ncbi_api_key", None) if svc_cfg else None
        if ncbi_email:
            services.append(PubMedService(ncbi_email=ncbi_email, ncbi_api_key=ncbi_key))
    except Exception as _e:
        logger.debug(f"Silent error: {_e}")

    queries_to_use = queries[:8]
    max_per_query = max(max_per_source // max(len(queries_to_use), 1), 5)

    async def _search_source(svc) -> list[Paper]:
        svc_papers: list[Paper] = []
        consecutive_fails = 0
        for query in queries_to_use:
            cache_key = f"{svc.source_name}:{query}"
            if cache_key in _search_cache:
                svc_papers.extend(_search_cache[cache_key])
                continue

            ok = False
            for attempt in range(2):
                try:
                    papers = await asyncio.wait_for(
                        svc.search(query, max_results=max_per_query),
                        timeout=15.0,
                    )
                    _search_cache[cache_key] = papers
                    svc_papers.extend(papers)
                    consecutive_fails = 0
                    ok = True
                    break
                except asyncio.TimeoutError:
                    logger.warning(f"{svc.source_name} timed out for '{query[:40]}' (attempt {attempt+1})")
                except Exception as e:
                    logger.warning(f"{svc.source_name} failed for '{query[:40]}' (attempt {attempt+1}): {e}")
                if attempt == 0:
                    await asyncio.sleep(1)

            if not ok:
                consecutive_fails += 1
                if consecutive_fails >= 2:
                    logger.warning(f"{svc.source_name}: 2 consecutive failures, skipping remaining")
                    break
        return svc_papers

    sem = asyncio.Semaphore(3)

    async def _guarded_search(svc) -> list[Paper]:
        async with sem:
            papers = await _search_source(svc)
            logger.info(f"  {svc.source_name}: {len(papers)} papers")
            return papers

    results = await asyncio.gather(
        *[_guarded_search(svc) for svc in services],
        return_exceptions=True,
    )

    all_papers: list[Paper] = []
    for r in results:
        if isinstance(r, list):
            all_papers.extend(r)

    deduped = dedup_papers(all_papers)
    logger.info(f"Search: {len(all_papers)} raw → {len(deduped)} deduped from {len(services)} sources × {len(queries_to_use)} queries")
    return deduped
