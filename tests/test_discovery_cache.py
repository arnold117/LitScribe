#!/usr/bin/env python
"""Test script for discovery agent cache integration.

Run with: python tests/test_discovery_cache.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def test_cached_tools_import():
    """Test that CachedTools can be imported in discovery_agent."""
    print("=" * 60)
    print("Test 1: Import CachedTools in discovery_agent")
    print("=" * 60)

    try:
        from agents.discovery_agent import (
            discovery_agent,
            search_all_sources,
            expand_queries,
        )
        from cache.cached_tools import CachedTools, get_cached_tools

        print("Import successful")

        # Create cached tools instance
        cached_tools = get_cached_tools(cache_enabled=True)
        print(f"CachedTools created, cache_enabled={cached_tools.cache_enabled}")

        print("PASS: Import and instantiation")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_state_with_cache():
    """Test state creation with cache_enabled parameter."""
    print("\n" + "=" * 60)
    print("Test 2: State with cache_enabled parameter")
    print("=" * 60)

    try:
        from agents.state import create_initial_state, LitScribeState

        # Test with cache enabled (default)
        state1 = create_initial_state(
            research_question="test question",
            max_papers=5,
        )
        print(f"Default cache_enabled: {state1.get('cache_enabled')}")

        # Test with cache disabled
        state2 = create_initial_state(
            research_question="test question",
            cache_enabled=False,
        )
        print(f"Explicit cache_enabled=False: {state2.get('cache_enabled')}")

        if state1.get("cache_enabled") != True:
            print("FAIL: Default should be True")
            return False

        if state2.get("cache_enabled") != False:
            print("FAIL: Explicit False not working")
            return False

        print("PASS: State with cache_enabled")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_search_with_cache_mock():
    """Test search_all_sources with cache parameter (mock)."""
    print("\n" + "=" * 60)
    print("Test 3: search_all_sources with cache parameter")
    print("=" * 60)

    try:
        from agents.discovery_agent import search_all_sources
        from cache.cached_tools import CachedTools

        # Test without cache (should work normally)
        # Note: This won't actually search without MCP servers running
        # Just verifying the function signature works
        print("Function signature accepts cached_tools parameter: OK")

        # Check function signature
        import inspect
        sig = inspect.signature(search_all_sources)
        params = list(sig.parameters.keys())
        print(f"Parameters: {params}")

        if "cached_tools" not in params:
            print("FAIL: cached_tools parameter not found")
            return False

        print("PASS: search_all_sources signature")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_cache_stats():
    """Test cache stats retrieval."""
    print("\n" + "=" * 60)
    print("Test 4: Cache stats retrieval")
    print("=" * 60)

    try:
        from cache.cached_tools import get_cached_tools

        cached_tools = get_cached_tools(cache_enabled=True)
        stats = cached_tools.get_cache_stats()

        print(f"Cache stats: {stats}")
        print(f"  - cache_enabled: {stats.get('cache_enabled')}")
        print(f"  - papers_count: {stats.get('papers_count', 'N/A')}")
        print(f"  - search_cache_count: {stats.get('search_cache_count', 'N/A')}")

        if not stats.get("cache_enabled"):
            print("FAIL: Cache should be enabled")
            return False

        print("PASS: Cache stats retrieval")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("Discovery Agent Cache Integration Tests")
    print("=" * 60)

    results = []

    results.append(("Import CachedTools", await test_cached_tools_import()))
    results.append(("State with cache", await test_state_with_cache()))
    results.append(("search_all_sources signature", await test_search_with_cache_mock()))
    results.append(("Cache stats", await test_cache_stats()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
