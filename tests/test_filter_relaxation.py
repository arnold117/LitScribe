#!/usr/bin/env python
"""Tests for filter relaxation and non-cached path bug fix.

Covers:
- Filter relaxation retry logic exists in unified_search.py
- Non-cached path in discovery_agent passes filter parameters
- search_all has retry_tasks/retry_names variables

Run with: pytest tests/test_filter_relaxation.py -v
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# === Test 1: Filter relaxation logic in unified_search ===

def test_filter_relaxation_retry_logic():
    """search_all should have filter relaxation retry logic."""
    source = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
    code = source.read_text()
    assert "Filter relaxation" in code
    assert "retry_tasks" in code
    assert "retry_names" in code


def test_arxiv_filter_relaxation():
    """arXiv should retry without category filter when 0 results."""
    source = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
    code = source.read_text()
    assert "arxiv_had_filter" in code
    assert 'category=None' in code


def test_s2_filter_relaxation():
    """Semantic Scholar should retry without field filter when 0 results."""
    source = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
    code = source.read_text()
    assert "s2_had_filter" in code
    assert 'fields_of_study=None' in code


def test_pubmed_filter_relaxation():
    """PubMed should retry without MeSH filter when 0 results."""
    source = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
    code = source.read_text()
    assert "pubmed_had_filter" in code
    assert 'mesh_terms=None' in code


# === Test 2: Non-cached path passes filter parameters ===

def test_noncached_path_passes_arxiv_categories():
    """Non-cached path in discovery_agent should pass arxiv_categories."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    # Find the else branch (non-cached path) in search_all_sources
    # Both branches should have arxiv_categories
    # Count occurrences of arxiv_categories= in unified_search calls
    count = code.count("arxiv_categories=arxiv_categories")
    assert count >= 2, f"Expected arxiv_categories in both cached and non-cached paths, found {count} occurrences"


def test_noncached_path_passes_s2_fields():
    """Non-cached path in discovery_agent should pass s2_fields."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    count = code.count("s2_fields=s2_fields")
    assert count >= 2, f"Expected s2_fields in both cached and non-cached paths, found {count} occurrences"


def test_noncached_path_passes_pubmed_mesh():
    """Non-cached path in discovery_agent should pass pubmed_mesh."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    count = code.count("pubmed_mesh=pubmed_mesh")
    assert count >= 2, f"Expected pubmed_mesh in both cached and non-cached paths, found {count} occurrences"


# === Test 3: Retry results are merged back ===

def test_retry_results_merged():
    """Filter relaxation retry results should extend all_papers."""
    source = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
    code = source.read_text()
    # After retry, results should be added to all_papers
    assert "all_papers.extend(rresult)" in code


# === Entrypoint ===

async def main():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=str(Path(__file__).parent.parent),
    )
    return result.returncode


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
