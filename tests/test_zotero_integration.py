#!/usr/bin/env python
"""Tests for Zotero integration: collection resolution, state wiring, auto-save logic.

Run with: pytest tests/test_zotero_integration.py -v
"""

import asyncio
import inspect
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# === Test 1: resolve_zotero_collection key detection ===

@pytest.mark.asyncio
async def test_resolve_collection_key_format():
    """8-char alphanumeric strings should be treated as Zotero collection keys."""
    from cache.cached_tools import resolve_zotero_collection, _zotero_available

    # Mock _zotero_available to return True
    with patch("cache.cached_tools._zotero_available", return_value=True):
        # 8-char alphanumeric → return directly as key
        result = await resolve_zotero_collection("ABC123XY")
        assert result == "ABC123XY"

        result = await resolve_zotero_collection("a1b2c3d4")
        assert result == "a1b2c3d4"


@pytest.mark.asyncio
async def test_resolve_collection_name_format():
    """Non-key strings should be looked up or created as collection names."""
    from cache.cached_tools import resolve_zotero_collection

    with patch("cache.cached_tools._zotero_available", return_value=True):
        mock_result = {"key": "NEWKEY12", "name": "My Review", "created": False}
        with patch("services.zotero.create_or_get_collection", new_callable=AsyncMock, return_value=mock_result):
            result = await resolve_zotero_collection("My Review")
            assert result == "NEWKEY12"


@pytest.mark.asyncio
async def test_resolve_collection_name_creates_new():
    """If collection name doesn't exist, it should be created."""
    from cache.cached_tools import resolve_zotero_collection

    with patch("cache.cached_tools._zotero_available", return_value=True):
        mock_result = {"key": "CREAT3D1", "name": "New Collection", "created": True}
        with patch("services.zotero.create_or_get_collection", new_callable=AsyncMock, return_value=mock_result):
            result = await resolve_zotero_collection("New Collection")
            assert result == "CREAT3D1"


@pytest.mark.asyncio
async def test_resolve_collection_not_key_if_wrong_length():
    """Strings that are alphanumeric but not exactly 8 chars should be treated as names."""
    from cache.cached_tools import resolve_zotero_collection

    with patch("cache.cached_tools._zotero_available", return_value=True):
        # 7 chars — not a key
        mock_result = {"key": "LOOKUP01", "name": "SHORT12", "created": False}
        with patch("services.zotero.create_or_get_collection", new_callable=AsyncMock, return_value=mock_result):
            result = await resolve_zotero_collection("SHORT12")
            assert result == "LOOKUP01"

        # 9 chars — not a key
        mock_result = {"key": "LOOKUP02", "name": "TOOLONG12", "created": False}
        with patch("services.zotero.create_or_get_collection", new_callable=AsyncMock, return_value=mock_result):
            result = await resolve_zotero_collection("TOOLONG12")
            assert result == "LOOKUP02"


@pytest.mark.asyncio
async def test_resolve_collection_not_key_if_special_chars():
    """8-char strings with special characters should be treated as names."""
    from cache.cached_tools import resolve_zotero_collection

    with patch("cache.cached_tools._zotero_available", return_value=True):
        mock_result = {"key": "LOOKUP03", "name": "My Coll!", "created": False}
        with patch("services.zotero.create_or_get_collection", new_callable=AsyncMock, return_value=mock_result):
            result = await resolve_zotero_collection("My Coll!")
            assert result == "LOOKUP03"


@pytest.mark.asyncio
async def test_resolve_collection_zotero_unavailable():
    """Should return None when Zotero is not configured."""
    from cache.cached_tools import resolve_zotero_collection

    with patch("cache.cached_tools._zotero_available", return_value=False):
        result = await resolve_zotero_collection("ABC123XY")
        assert result is None


@pytest.mark.asyncio
async def test_resolve_collection_api_error():
    """Should return None and not crash when Zotero API fails."""
    from cache.cached_tools import resolve_zotero_collection

    with patch("cache.cached_tools._zotero_available", return_value=True):
        with patch("services.zotero.create_or_get_collection", new_callable=AsyncMock, side_effect=Exception("API error")):
            result = await resolve_zotero_collection("Failing Collection")
            assert result is None


@pytest.mark.asyncio
async def test_resolve_collection_api_returns_error():
    """Should return None when API returns an error dict."""
    from cache.cached_tools import resolve_zotero_collection

    with patch("cache.cached_tools._zotero_available", return_value=True):
        mock_result = {"error": "Permission denied"}
        with patch("services.zotero.create_or_get_collection", new_callable=AsyncMock, return_value=mock_result):
            result = await resolve_zotero_collection("No Access")
            assert result is None


# === Test 2: _zotero_available ===

def test_zotero_available_both_set():
    """Should return True when both ZOTERO_API_KEY and ZOTERO_LIBRARY_ID are set."""
    from cache.cached_tools import _zotero_available

    with patch("cache.cached_tools.Config") as mock_config:
        mock_config.ZOTERO_API_KEY = "test_key"
        mock_config.ZOTERO_LIBRARY_ID = "123456"
        assert _zotero_available() is True


def test_zotero_available_missing_key():
    """Should return False when ZOTERO_API_KEY is empty."""
    from cache.cached_tools import _zotero_available

    with patch("cache.cached_tools.Config") as mock_config:
        mock_config.ZOTERO_API_KEY = ""
        mock_config.ZOTERO_LIBRARY_ID = "123456"
        assert _zotero_available() is False


def test_zotero_available_missing_library():
    """Should return False when ZOTERO_LIBRARY_ID is empty."""
    from cache.cached_tools import _zotero_available

    with patch("cache.cached_tools.Config") as mock_config:
        mock_config.ZOTERO_API_KEY = "test_key"
        mock_config.ZOTERO_LIBRARY_ID = ""
        assert _zotero_available() is False


# === Test 3: _zotero_item_to_paper conversion ===

def test_zotero_item_to_paper_basic():
    """Should correctly convert a Zotero item dict to unified paper format."""
    from cache.cached_tools import _zotero_item_to_paper

    item = {
        "key": "ABCD1234",
        "title": "Test Paper",
        "abstract": "An abstract",
        "date": "2023-05-15",
        "doi": "10.1234/test",
        "url": "https://example.com",
        "publication_title": "Nature",
        "creators": [
            {"creatorType": "author", "firstName": "John", "lastName": "Smith"},
            {"creatorType": "author", "firstName": "Jane", "lastName": "Doe"},
        ],
    }

    paper = _zotero_item_to_paper(item)
    assert paper["title"] == "Test Paper"
    assert paper["abstract"] == "An abstract"
    assert paper["year"] == 2023
    assert paper["paper_id"] == "10.1234/test"
    assert paper["source"] == "zotero"
    assert paper["zotero_key"] == "ABCD1234"
    assert paper["authors"] == ["John Smith", "Jane Doe"]
    assert paper["venue"] == "Nature"


def test_zotero_item_to_paper_no_doi():
    """Should use zotero:key as paper_id when DOI is missing."""
    from cache.cached_tools import _zotero_item_to_paper

    item = {
        "key": "XYZ78901",
        "title": "No DOI Paper",
        "date": "2021",
        "creators": [],
    }

    paper = _zotero_item_to_paper(item)
    assert paper["paper_id"] == "zotero:XYZ78901"
    assert paper["year"] == 2021
    assert paper["authors"] == []


def test_zotero_item_to_paper_no_date():
    """Should handle missing date gracefully."""
    from cache.cached_tools import _zotero_item_to_paper

    item = {"key": "NODATE01", "title": "No Date", "creators": []}
    paper = _zotero_item_to_paper(item)
    assert paper["year"] == 0


# === Test 4: State and function signatures ===

def test_state_has_zotero_collection_field():
    """LitScribeState should have zotero_collection field."""
    from agents.state import create_initial_state

    # Default: None
    state = create_initial_state(research_question="test")
    assert "zotero_collection" in state
    assert state["zotero_collection"] is None

    # Explicit value
    state2 = create_initial_state(
        research_question="test",
        zotero_collection="ABC123XY",
    )
    assert state2["zotero_collection"] == "ABC123XY"


def test_run_literature_review_has_zotero_collection_param():
    """run_literature_review should accept zotero_collection parameter."""
    from agents.graph import run_literature_review

    sig = inspect.signature(run_literature_review)
    params = list(sig.parameters.keys())
    assert "zotero_collection" in params


def test_search_all_sources_has_zotero_collection_param():
    """search_all_sources should accept zotero_collection parameter."""
    from agents.discovery_agent import search_all_sources

    sig = inspect.signature(search_all_sources)
    params = list(sig.parameters.keys())
    assert "zotero_collection" in params


# === Test 5: save_papers_to_collection function exists ===

def test_save_papers_to_collection_exists():
    """services.zotero should have save_papers_to_collection function."""
    from services.zotero import save_papers_to_collection
    assert callable(save_papers_to_collection)

    sig = inspect.signature(save_papers_to_collection)
    params = list(sig.parameters.keys())
    assert "papers" in params
    assert "collection_key" in params


# === Entrypoint for direct execution ===

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
