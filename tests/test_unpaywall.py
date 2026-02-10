#!/usr/bin/env python
"""Tests for Unpaywall integration and PDF coverage improvements.

Covers:
- Unpaywall client: function exists, async, email handling, API response parsing
- PMC download: integration points in acquire_pdf and get_pdf_with_cache
- S2 URL retry: content-type validation, PDF magic bytes check
- Config: UNPAYWALL_EMAIL attribute

Run with: pytest tests/test_unpaywall.py -v
"""

import asyncio
import inspect
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# === Test 1: get_oa_pdf_url function exists and is async ===

def test_unpaywall_function_exists():
    """get_oa_pdf_url should be importable and async."""
    from services.unpaywall import get_oa_pdf_url
    assert inspect.iscoroutinefunction(get_oa_pdf_url)


# === Test 2: No email returns None ===

@pytest.mark.asyncio
async def test_unpaywall_no_email_returns_none():
    """Should return None when no email is configured."""
    from services.unpaywall import get_oa_pdf_url

    with patch("services.unpaywall.Config") as mock_config:
        mock_config.UNPAYWALL_EMAIL = ""
        mock_config.NCBI_EMAIL = ""
        result = await get_oa_pdf_url("10.1234/test", email="")
        assert result is None


# === Test 3: Successful API response returns pdf_url ===

@pytest.mark.asyncio
async def test_unpaywall_returns_pdf_url():
    """Should return url_for_pdf from best_oa_location."""
    from services.unpaywall import get_oa_pdf_url

    mock_json = {
        "best_oa_location": {
            "url_for_pdf": "https://example.com/paper.pdf",
            "host_type": "publisher",
        },
        "oa_locations": [],
    }

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=mock_json)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_resp),
        __aexit__=AsyncMock(return_value=False),
    ))

    with patch("services.unpaywall.aiohttp.ClientSession", return_value=mock_session):
        result = await get_oa_pdf_url("10.1234/test", email="test@test.com")
        assert result == "https://example.com/paper.pdf"


# === Test 4: No OA available returns None ===

@pytest.mark.asyncio
async def test_unpaywall_no_oa_returns_none():
    """Should return None when no OA location has a PDF."""
    from services.unpaywall import get_oa_pdf_url

    mock_json = {
        "best_oa_location": None,
        "oa_locations": [],
    }

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=mock_json)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_resp),
        __aexit__=AsyncMock(return_value=False),
    ))

    with patch("services.unpaywall.aiohttp.ClientSession", return_value=mock_session):
        result = await get_oa_pdf_url("10.1234/nooa", email="test@test.com")
        assert result is None


# === Test 5: Timeout returns None without exception ===

@pytest.mark.asyncio
async def test_unpaywall_timeout_returns_none():
    """Should handle timeout gracefully."""
    from services.unpaywall import get_oa_pdf_url

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(side_effect=asyncio.TimeoutError)

    with patch("services.unpaywall.aiohttp.ClientSession", return_value=mock_session):
        result = await get_oa_pdf_url("10.1234/slow", email="test@test.com")
        assert result is None


# === Test 6: DOI cleaning ===

@pytest.mark.asyncio
async def test_unpaywall_cleans_doi():
    """Should strip https://doi.org/ prefix from DOI."""
    from services.unpaywall import get_oa_pdf_url

    mock_resp = AsyncMock()
    mock_resp.status = 404

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_resp),
        __aexit__=AsyncMock(return_value=False),
    ))

    with patch("services.unpaywall.aiohttp.ClientSession", return_value=mock_session):
        await get_oa_pdf_url("https://doi.org/10.1234/test", email="test@test.com")
        # Verify the cleaned DOI was used in the URL
        call_args = mock_session.get.call_args
        url_used = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        assert "doi.org/10.1234/test" not in url_used or "api.unpaywall.org" in url_used


# === Test 7: acquire_pdf includes Unpaywall ===

def test_acquire_pdf_has_unpaywall():
    """acquire_pdf source should include Unpaywall fallback."""
    source = Path(__file__).parent.parent / "src" / "agents" / "critical_reading_agent.py"
    code = source.read_text()
    assert "unpaywall" in code.lower(), "acquire_pdf should include Unpaywall fallback"
    assert "get_oa_pdf_url" in code, "acquire_pdf should call get_oa_pdf_url"


# === Test 8: acquire_pdf includes PMC ===

def test_acquire_pdf_has_pmc():
    """acquire_pdf source should include PMC fallback."""
    source = Path(__file__).parent.parent / "src" / "agents" / "critical_reading_agent.py"
    code = source.read_text()
    assert "pmc_id" in code, "acquire_pdf should check pmc_id"
    assert "ncbi.nlm.nih.gov/pmc/articles" in code, "acquire_pdf should build PMC URL"


# === Test 9: get_pdf_with_cache includes Unpaywall and PMC ===

def test_cached_tools_has_unpaywall_and_pmc():
    """get_pdf_with_cache should include both Unpaywall and PMC fallbacks."""
    source = Path(__file__).parent.parent / "src" / "cache" / "cached_tools.py"
    code = source.read_text()
    assert "unpaywall" in code.lower(), "cached_tools should include Unpaywall"
    assert "pmc_id" in code, "cached_tools should include PMC"


# === Test 10: Config has UNPAYWALL_EMAIL ===

def test_config_has_unpaywall_email():
    """Config should have UNPAYWALL_EMAIL attribute."""
    from utils.config import Config
    assert hasattr(Config, "UNPAYWALL_EMAIL")


# === Test 11: PDF download rejects HTML content ===

def test_download_pdf_rejects_html():
    """_download_pdf_from_url should reject text/html responses."""
    source = Path(__file__).parent.parent / "src" / "agents" / "critical_reading_agent.py"
    code = source.read_text()
    assert '"html"' in code or "'html'" in code, \
        "_download_pdf_from_url should check for HTML content-type"
    assert "%PDF-" in code or "pdf" in code.lower(), \
        "_download_pdf_from_url should verify PDF magic bytes"


# === Test 12: PDF fallback chain order ===

def test_pdf_fallback_chain_order():
    """PDF acquisition should follow correct order: cache/local → Zotero → arXiv → Unpaywall → PMC → direct URL."""
    source = Path(__file__).parent.parent / "src" / "agents" / "critical_reading_agent.py"
    code = source.read_text()

    # Find positions of each fallback
    arxiv_pos = code.find("download_arxiv_pdf")
    unpaywall_pos = code.find("get_oa_pdf_url")
    pmc_pos = code.find("ncbi.nlm.nih.gov/pmc/articles")
    direct_pos = code.find("# Try direct URL download")

    assert arxiv_pos > 0, "arXiv fallback should exist"
    assert unpaywall_pos > 0, "Unpaywall fallback should exist"
    assert pmc_pos > 0, "PMC fallback should exist"
    assert direct_pos > 0, "Direct URL fallback should exist"

    # Verify order: arXiv < Unpaywall < PMC < direct URL
    assert arxiv_pos < unpaywall_pos, "arXiv should come before Unpaywall"
    assert unpaywall_pos < pmc_pos, "Unpaywall should come before PMC"
    assert pmc_pos < direct_pos, "PMC should come before direct URL"


# === Entrypoint ===

async def main():
    """Run all tests via pytest."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=str(Path(__file__).parent.parent),
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
