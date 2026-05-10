from __future__ import annotations

import logging

import aiohttp

logger = logging.getLogger(__name__)

UNPAYWALL_API = "https://api.unpaywall.org/v2"


async def get_open_access_url(doi: str, email: str = "") -> str | None:
    if not doi:
        return None

    url = f"{UNPAYWALL_API}/{doi}?email={email}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

                best = data.get("best_oa_location")
                if best:
                    pdf_url = best.get("url_for_pdf") or best.get("url")
                    if pdf_url:
                        return pdf_url

                for loc in data.get("oa_locations", []):
                    pdf_url = loc.get("url_for_pdf") or loc.get("url")
                    if pdf_url:
                        return pdf_url
    except Exception as e:
        logger.debug(f"Unpaywall lookup failed for {doi}: {e}")

    return None


async def enrich_pdf_urls(papers, email: str = "") -> int:
    import asyncio

    sem = asyncio.Semaphore(3)
    added = 0

    async def _check_one(paper):
        nonlocal added
        if paper.pdf_urls:
            return
        if not paper.doi:
            return
        async with sem:
            url = await get_open_access_url(paper.doi, email)
            if url:
                paper.pdf_urls.append(url)
                added += 1

    await asyncio.gather(*[_check_one(p) for p in papers], return_exceptions=True)
    if added:
        logger.info(f"Unpaywall: enriched {added} papers with OA PDF URLs")
    return added
