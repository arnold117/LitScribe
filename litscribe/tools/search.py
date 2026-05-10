from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import aiohttp

from litscribe.models.paper import Paper
from litscribe.services.base import dedup_papers

logger = logging.getLogger(__name__)

_search_cache: dict[str, list[Paper]] = {}

# Shared session with conservative connection limits
_shared_session: aiohttp.ClientSession | None = None
_shared_connector: aiohttp.TCPConnector | None = None


async def _get_session() -> aiohttp.ClientSession:
    global _shared_session, _shared_connector
    if _shared_session is None or _shared_session.closed:
        _shared_connector = aiohttp.TCPConnector(
            limit=6,
            limit_per_host=2,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        _shared_session = aiohttp.ClientSession(connector=_shared_connector)
    return _shared_session


def _patch_service_session(svc, session: aiohttp.ClientSession):
    """Monkey-patch a service to use the shared session instead of creating its own."""
    svc._shared_session = session


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

    queries_to_use = queries[:8]
    max_per_query = max(max_per_source // max(len(queries_to_use), 1), 5)

    # Search sources in parallel, but queries within each source are sequential.
    # This avoids multiple sessions per source while still searching sources concurrently.
    async def _search_source(svc) -> list[Paper]:
        svc_papers: list[Paper] = []
        for query in queries_to_use:
            cache_key = f"{svc.source_name}:{query}"
            if cache_key in _search_cache:
                svc_papers.extend(_search_cache[cache_key])
                continue
            try:
                papers = await asyncio.wait_for(
                    svc.search(query, max_results=max_per_query),
                    timeout=15.0,
                )
                _search_cache[cache_key] = papers
                svc_papers.extend(papers)
            except asyncio.TimeoutError:
                logger.warning(f"{svc.source_name} timed out for '{query[:40]}'")
            except Exception as e:
                logger.warning(f"{svc.source_name} failed for '{query[:40]}': {e}")
                break
        return svc_papers

    # Run sources in parallel (max 3 concurrent sources, each with sequential queries)
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
