#!/usr/bin/env python
"""Tests for API parameter enhancements.

Covers:
- arXiv search_papers accepts year_from/year_to
- PubMed search_pubmed accepts article_types
- Semantic Scholar search_papers accepts min_citations
- unified_search threads year/article_types/min_citations
- discovery_agent computes year range from TIER_CONFIG

Run with: pytest tests/test_api_parameters.py -v
"""

import inspect
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# === Test 1: arXiv year parameters ===

def test_arxiv_search_accepts_year_params():
    """arXiv search_papers should accept year_from and year_to."""
    from services.arxiv import search_papers
    sig = inspect.signature(search_papers)
    params = list(sig.parameters.keys())
    assert "year_from" in params, "arXiv search should accept year_from"
    assert "year_to" in params, "arXiv search should accept year_to"


def test_arxiv_year_filter_uses_query_syntax():
    """arXiv year filter should use submittedDate query syntax."""
    source = Path(__file__).parent.parent / "src" / "services" / "arxiv.py"
    code = source.read_text()
    assert "submittedDate:" in code, \
        "Should filter by submittedDate in query syntax"


# === Test 2: PubMed article_types parameter ===

def test_pubmed_search_accepts_article_types():
    """PubMed search_pubmed should accept article_types."""
    from services.pubmed import search_pubmed
    sig = inspect.signature(search_pubmed)
    params = list(sig.parameters.keys())
    assert "article_types" in params, "PubMed search should accept article_types"


def test_pubmed_article_types_uses_pt_syntax():
    """PubMed article_types should use [pt] publication type syntax."""
    source = Path(__file__).parent.parent / "src" / "services" / "pubmed.py"
    code = source.read_text()
    assert "[pt]" in code, "Should use [pt] publication type syntax"


# === Test 3: Semantic Scholar min_citations parameter ===

def test_s2_search_accepts_min_citations():
    """S2 search_papers should accept min_citations."""
    from services.semantic_scholar import search_papers
    sig = inspect.signature(search_papers)
    params = list(sig.parameters.keys())
    assert "min_citations" in params, "S2 search should accept min_citations"


def test_s2_min_citations_is_post_filter():
    """S2 min_citations should be a post-filter on citation_count."""
    source = Path(__file__).parent.parent / "src" / "services" / "semantic_scholar.py"
    code = source.read_text()
    assert "citation_count" in code, "Should check citation_count"
    assert "min_citations" in code, "Should have min_citations logic"


def test_s2_fetches_extra_for_post_filter():
    """S2 should fetch extra results when min_citations is set."""
    source = Path(__file__).parent.parent / "src" / "services" / "semantic_scholar.py"
    code = source.read_text()
    assert "fetch_limit" in code, "Should have fetch_limit for over-fetching"


# === Test 4: unified_search threads new parameters ===

def test_unified_search_all_accepts_year_params():
    """search_all() should accept year_from, year_to, article_types, min_citations."""
    from aggregators.unified_search import UnifiedSearchAggregator
    sig = inspect.signature(UnifiedSearchAggregator.search_all)
    params = list(sig.parameters.keys())
    assert "year_from" in params
    assert "year_to" in params
    assert "article_types" in params
    assert "min_citations" in params


def test_unified_search_all_sources_accepts_year_params():
    """search_all_sources() convenience function should accept year params."""
    from aggregators.unified_search import search_all_sources
    sig = inspect.signature(search_all_sources)
    params = list(sig.parameters.keys())
    assert "year_from" in params
    assert "year_to" in params
    assert "article_types" in params
    assert "min_citations" in params


# === Test 5: discovery_agent computes year range ===

def test_discovery_agent_computes_year_range():
    """discovery_agent should compute year_from/year_to from tier config."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    assert "year_lookback" in code, "Should read year_lookback from tier config"
    assert "year_from" in code, "Should compute year_from"
    assert "year_to" in code, "Should compute year_to"


def test_discovery_agent_passes_year_to_search():
    """discovery_agent should pass year_from/year_to to search calls."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    assert "year_from=year_from" in code, "Should pass year_from to search"
    assert "year_to=year_to" in code, "Should pass year_to to search"


# === Test 6: cached_tools threads new parameters ===

def test_cached_tools_accepts_new_params():
    """search_with_cache should accept year_from, year_to, article_types, min_citations."""
    from cache.cached_tools import CachedTools
    sig = inspect.signature(CachedTools.search_with_cache)
    params = list(sig.parameters.keys())
    assert "year_from" in params
    assert "year_to" in params
    assert "article_types" in params
    assert "min_citations" in params


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
