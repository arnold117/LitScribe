#!/usr/bin/env python
"""Test script for critical reading agent cache integration.

Run with: python tests/test_critical_reading_cache.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def test_cached_tools_import():
    """Test that CachedTools can be imported in critical_reading_agent."""
    print("=" * 60)
    print("Test 1: Import CachedTools in critical_reading_agent")
    print("=" * 60)

    try:
        from agents.critical_reading_agent import (
            critical_reading_agent,
            acquire_pdf,
            parse_paper_pdf,
            analyze_single_paper,
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


async def test_acquire_pdf_signature():
    """Test acquire_pdf accepts cached_tools parameter."""
    print("\n" + "=" * 60)
    print("Test 2: acquire_pdf signature")
    print("=" * 60)

    try:
        from agents.critical_reading_agent import acquire_pdf
        import inspect

        sig = inspect.signature(acquire_pdf)
        params = list(sig.parameters.keys())
        print(f"Parameters: {params}")

        if "cached_tools" not in params:
            print("FAIL: cached_tools parameter not found")
            return False

        print("PASS: acquire_pdf signature")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_parse_paper_pdf_signature():
    """Test parse_paper_pdf accepts cached_tools parameter."""
    print("\n" + "=" * 60)
    print("Test 3: parse_paper_pdf signature")
    print("=" * 60)

    try:
        from agents.critical_reading_agent import parse_paper_pdf
        import inspect

        sig = inspect.signature(parse_paper_pdf)
        params = list(sig.parameters.keys())
        print(f"Parameters: {params}")

        if "cached_tools" not in params:
            print("FAIL: cached_tools parameter not found")
            return False

        if "paper_id" not in params:
            print("FAIL: paper_id parameter not found")
            return False

        print("PASS: parse_paper_pdf signature")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_analyze_single_paper_signature():
    """Test analyze_single_paper accepts cached_tools parameter."""
    print("\n" + "=" * 60)
    print("Test 4: analyze_single_paper signature")
    print("=" * 60)

    try:
        from agents.critical_reading_agent import analyze_single_paper
        import inspect

        sig = inspect.signature(analyze_single_paper)
        params = list(sig.parameters.keys())
        print(f"Parameters: {params}")

        if "cached_tools" not in params:
            print("FAIL: cached_tools parameter not found")
            return False

        print("PASS: analyze_single_paper signature")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_pdf_cache_operations():
    """Test PDF cache operations."""
    print("\n" + "=" * 60)
    print("Test 5: PDF cache operations")
    print("=" * 60)

    try:
        from cache.cached_tools import get_cached_tools

        cached_tools = get_cached_tools(cache_enabled=True)

        # Test that pdf_cache exists
        if cached_tools.pdf_cache is None:
            print("FAIL: pdf_cache should not be None when enabled")
            return False

        # Test that parse_cache exists
        if cached_tools.parse_cache is None:
            print("FAIL: parse_cache should not be None when enabled")
            return False

        print("pdf_cache initialized: OK")
        print("parse_cache initialized: OK")

        print("PASS: PDF cache operations")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("Critical Reading Agent Cache Integration Tests")
    print("=" * 60)

    results = []

    results.append(("Import CachedTools", await test_cached_tools_import()))
    results.append(("acquire_pdf signature", await test_acquire_pdf_signature()))
    results.append(("parse_paper_pdf signature", await test_parse_paper_pdf_signature()))
    results.append(("analyze_single_paper signature", await test_analyze_single_paper_signature()))
    results.append(("PDF cache operations", await test_pdf_cache_operations()))

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
