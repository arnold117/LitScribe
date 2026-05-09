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
        except Exception:
            pass

    try:
        from litscribe.services.semantic_scholar import SemanticScholarService
        svc_cfg = getattr(config, "services", None)
        s2_key = getattr(svc_cfg, "semantic_scholar_api_key", None) if svc_cfg else None
        services.append(SemanticScholarService(api_key=s2_key))
    except Exception:
        pass

    try:
        from litscribe.services.pubmed import PubMedService
        svc_cfg = getattr(config, "services", None)
        ncbi_email = getattr(svc_cfg, "ncbi_email", None) if svc_cfg else None
        ncbi_key = getattr(svc_cfg, "ncbi_api_key", None) if svc_cfg else None
        if ncbi_email:
            services.append(PubMedService(ncbi_email=ncbi_email, ncbi_api_key=ncbi_key))
    except Exception:
        pass

    queries_to_use = queries[:12]
    max_per_query = max(max_per_source // max(len(queries_to_use), 1), 5)

    async def _search_one(svc, query: str) -> list[Paper]:
        cache_key = f"{svc.source_name}:{query}"
        if cache_key in _search_cache:
            return _search_cache[cache_key]
        try:
            papers = await asyncio.wait_for(
                svc.search(query, max_results=max_per_query),
                timeout=15.0,
            )
            _search_cache[cache_key] = papers
            return papers
        except asyncio.TimeoutError:
            logger.warning(f"{svc.source_name} timed out for '{query[:40]}'")
            return []
        except Exception as e:
            logger.warning(f"{svc.source_name} failed for '{query[:40]}': {e}")
            return []

    tasks = [_search_one(svc, q) for svc in services for q in queries_to_use]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_papers: list[Paper] = []
    for r in results:
        if isinstance(r, list):
            all_papers.extend(r)

    deduped = dedup_papers(all_papers)
    logger.info(f"Search: {len(all_papers)} raw → {len(deduped)} deduped from {len(services)} sources × {len(queries_to_use)} queries")
    return deduped
