"""Unpaywall API client for legal open access PDF discovery.

Unpaywall (https://unpaywall.org) aggregates open access locations for
academic papers by DOI. It covers ~30% of scholarly articles with free
PDF links from repositories, preprint servers, and publisher OA pages.

API: GET https://api.unpaywall.org/v2/{doi}?email={email}
Rate limit: No strict limit, but we add a polite delay.
"""

import asyncio
import logging
from typing import Optional

import aiohttp

from utils.config import Config

logger = logging.getLogger(__name__)

# Polite delay between requests
_POLITE_DELAY = 0.5  # seconds


async def get_oa_pdf_url(doi: str, email: str = "") -> Optional[str]:
    """Query Unpaywall API for an open access PDF URL.

    Fallback chain for OA locations:
    1. best_oa_location.url_for_pdf (direct PDF)
    2. Any oa_location with url_for_pdf
    3. best_oa_location.url_for_landing_page (if host_type == "repository")

    Args:
        doi: Digital Object Identifier of the paper
        email: Contact email (required by Unpaywall TOS).
               Falls back to UNPAYWALL_EMAIL, then NCBI_EMAIL from config.

    Returns:
        URL string for the OA PDF, or None if not available.
    """
    if not doi:
        return None

    # Resolve email
    if not email:
        email = getattr(Config, "UNPAYWALL_EMAIL", "") or Config.NCBI_EMAIL
    if not email:
        logger.debug("Unpaywall skipped: no email configured (set UNPAYWALL_EMAIL or NCBI_EMAIL)")
        return None

    # Clean DOI
    doi = doi.strip()
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    elif doi.startswith("http://dx.doi.org/"):
        doi = doi[len("http://dx.doi.org/"):]

    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 404:
                    logger.debug(f"Unpaywall: no record for DOI {doi}")
                    return None
                if resp.status != 200:
                    logger.debug(f"Unpaywall: HTTP {resp.status} for DOI {doi}")
                    return None

                data = await resp.json()

        # Strategy 1: best_oa_location.url_for_pdf
        best = data.get("best_oa_location") or {}
        if best.get("url_for_pdf"):
            logger.info(f"Unpaywall: found PDF for {doi} via best_oa_location")
            await asyncio.sleep(_POLITE_DELAY)
            return best["url_for_pdf"]

        # Strategy 2: any oa_location with url_for_pdf
        for loc in data.get("oa_locations", []):
            if loc.get("url_for_pdf"):
                logger.info(f"Unpaywall: found PDF for {doi} via oa_locations")
                await asyncio.sleep(_POLITE_DELAY)
                return loc["url_for_pdf"]

        # Strategy 3: landing page from repository (may have PDF link)
        if best.get("url_for_landing_page") and best.get("host_type") == "repository":
            logger.info(f"Unpaywall: found repository landing page for {doi}")
            await asyncio.sleep(_POLITE_DELAY)
            return best["url_for_landing_page"]

        logger.debug(f"Unpaywall: no OA PDF found for DOI {doi}")
        await asyncio.sleep(_POLITE_DELAY)
        return None

    except asyncio.TimeoutError:
        logger.debug(f"Unpaywall: timeout for DOI {doi}")
        return None
    except Exception as e:
        logger.debug(f"Unpaywall: error for DOI {doi}: {e}")
        return None
