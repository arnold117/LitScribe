#!/usr/bin/env python
"""Tests for arXiv rate limiting and singleton client.

Covers:
- _get_client returns singleton
- _wait_cooldown enforces minimum interval
- _ARXIV_COOLDOWN is >= 3 seconds
- Client has num_retries >= 3
- All search functions use _wait_cooldown + _get_client
- unified_search has arXiv semaphore

Run with: pytest tests/test_arxiv_ratelimit.py -v
"""

import sys
import time
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# === Test 1: _get_client returns singleton ===

def test_get_client_singleton():
    """_get_client should return the same client instance on repeated calls."""
    from services.arxiv import _get_client
    client1 = _get_client()
    client2 = _get_client()
    assert client1 is client2


# === Test 2: Client has proper retry settings ===

def test_client_retry_settings():
    """Singleton client should have num_retries >= 3 and delay_seconds >= 5."""
    from services.arxiv import _get_client
    client = _get_client()
    assert client.num_retries >= 3
    assert client.delay_seconds >= 5


# === Test 3: _ARXIV_COOLDOWN is at least 3 seconds ===

def test_cooldown_value():
    """_ARXIV_COOLDOWN should be at least 3 seconds."""
    from services.arxiv import _ARXIV_COOLDOWN
    assert _ARXIV_COOLDOWN >= 3.0


# === Test 4: _wait_cooldown enforces minimum interval ===

def test_wait_cooldown_enforces_interval():
    """_wait_cooldown should sleep if called too quickly."""
    import services.arxiv as arxiv_mod

    # Set last request time to now
    arxiv_mod._last_request_time = time.monotonic()

    start = time.monotonic()
    arxiv_mod._wait_cooldown()
    elapsed = time.monotonic() - start

    # Should have waited at least ~3 seconds (minus small tolerance)
    assert elapsed >= arxiv_mod._ARXIV_COOLDOWN - 0.2


# === Test 5: _wait_cooldown is a no-op when enough time has passed ===

def test_wait_cooldown_no_wait():
    """_wait_cooldown should not sleep if enough time has already passed."""
    import services.arxiv as arxiv_mod

    # Set last request time to long ago
    arxiv_mod._last_request_time = time.monotonic() - 100

    start = time.monotonic()
    arxiv_mod._wait_cooldown()
    elapsed = time.monotonic() - start

    assert elapsed < 0.1  # Should be nearly instant


# === Test 6: All search functions use _wait_cooldown ===

def test_all_functions_use_cooldown():
    """All arXiv search functions should call _wait_cooldown."""
    source = Path(__file__).parent.parent / "src" / "services" / "arxiv.py"
    code = source.read_text()

    # Count _wait_cooldown calls (should be in every do_search/do_fetch/do_download)
    cooldown_count = code.count("_wait_cooldown()")
    # search_papers, get_paper_metadata, download_pdf, get_paper_by_doi,
    # get_recent_papers, search_by_author, batch_get_papers = 7 functions
    assert cooldown_count >= 7, f"Expected >= 7 _wait_cooldown calls, got {cooldown_count}"


# === Test 7: All functions use _get_client (no local client creation) ===

def test_all_functions_use_singleton():
    """No function should create arxiv.Client() directly — all should use _get_client."""
    source = Path(__file__).parent.parent / "src" / "services" / "arxiv.py"
    code = source.read_text()

    # _get_client definition itself creates arxiv.Client, that's expected
    # But no other line should have "arxiv.Client("
    lines = code.split("\n")
    direct_client_lines = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if "arxiv.Client(" in stripped and "_get_client" not in stripped and "def _get_client" not in stripped:
            # Skip the line inside _get_client body
            if "_arxiv_client = arxiv.Client(" not in stripped:
                direct_client_lines.append((i, stripped))

    assert len(direct_client_lines) == 0, (
        f"Found direct arxiv.Client() creation outside _get_client: {direct_client_lines}"
    )


# === Test 8: unified_search has arXiv semaphore ===

def test_unified_search_has_semaphore():
    """unified_search should serialize arXiv requests with a semaphore."""
    source = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
    code = source.read_text()
    assert "_arxiv_semaphore" in code
    assert "Semaphore(1)" in code


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
    import asyncio
    sys.exit(asyncio.run(main()))
