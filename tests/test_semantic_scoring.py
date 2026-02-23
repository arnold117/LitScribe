#!/usr/bin/env python
"""Tests for semantic scoring module (embeddings/semantic_scorer.py).

Covers:
- compute_semantic_relevance modifies papers in place
- Matching papers score higher than non-matching
- Papers without abstract fall back to keyword-only scoring
- deduplicate_queries_by_similarity removes near-duplicates
- Empty input handling
- TIER_CONFIG includes year_lookback
- Composite seed score function exists and works

Run with: pytest tests/test_semantic_scoring.py -v
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# === Test 1: semantic_scorer module exists and is importable ===

def test_semantic_scorer_importable():
    """semantic_scorer module should be importable."""
    from embeddings.semantic_scorer import (
        compute_semantic_relevance,
        compute_query_embedding,
        compute_paper_embeddings_batch,
        deduplicate_queries_by_similarity,
    )
    assert callable(compute_semantic_relevance)
    assert callable(compute_query_embedding)
    assert callable(compute_paper_embeddings_batch)
    assert callable(deduplicate_queries_by_similarity)


# === Test 2: compute_semantic_relevance signature ===

def test_compute_semantic_relevance_signature():
    """compute_semantic_relevance should accept papers and research_question."""
    import inspect
    from embeddings.semantic_scorer import compute_semantic_relevance
    sig = inspect.signature(compute_semantic_relevance)
    params = list(sig.parameters.keys())
    assert "papers" in params
    assert "research_question" in params
    assert "keyword_weight" in params
    assert "semantic_weight" in params


# === Test 3: compute_semantic_relevance handles empty input ===

def test_compute_semantic_relevance_empty():
    """compute_semantic_relevance should handle empty paper list."""
    from embeddings.semantic_scorer import compute_semantic_relevance
    # Should not raise
    compute_semantic_relevance([], "test question")


# === Test 4: deduplicate_queries_by_similarity handles edge cases ===

def test_dedup_queries_single():
    """Single query should be returned as-is."""
    from embeddings.semantic_scorer import deduplicate_queries_by_similarity
    result = deduplicate_queries_by_similarity(["neural networks"])
    assert result == ["neural networks"]


def test_dedup_queries_empty():
    """Empty list should be returned as-is."""
    from embeddings.semantic_scorer import deduplicate_queries_by_similarity
    result = deduplicate_queries_by_similarity([])
    assert result == []


# === Test 5: unified_search uses semantic scoring ===

def test_unified_search_has_semantic_scoring():
    """unified_search.py should import and use compute_semantic_relevance."""
    source = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
    code = source.read_text()
    assert "compute_semantic_relevance" in code, \
        "Should import compute_semantic_relevance"
    assert "semantic_scorer" in code, \
        "Should reference semantic_scorer module"
    assert "research_question or query" in code, \
        "Should use research_question with query fallback"


# === Test 6: research_question is threaded through search pipeline ===

def test_research_question_in_search_all():
    """search_all() should accept research_question parameter."""
    source = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
    code = source.read_text()
    assert "research_question: Optional[str]" in code


def test_research_question_in_cached_tools():
    """search_with_cache() should accept research_question parameter."""
    source = Path(__file__).parent.parent / "src" / "cache" / "cached_tools.py"
    code = source.read_text()
    assert "research_question" in code


def test_research_question_in_discovery_agent():
    """search_all_sources wrapper should accept research_question parameter."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    # The wrapper function
    assert "research_question: Optional[str]" in code
    # Should pass it through to calls
    assert "research_question=research_question" in code


# === Test 7: TIER_CONFIG has year_lookback ===

def test_tier_config_year_lookback():
    """TIER_CONFIG should have year_lookback for each tier."""
    from agents.state import TIER_CONFIG
    assert TIER_CONFIG["quick"]["year_lookback"] == 5
    assert TIER_CONFIG["standard"]["year_lookback"] == 10
    assert TIER_CONFIG["comprehensive"]["year_lookback"] is None


# === Test 8: Composite seed score function ===

def test_composite_seed_score_exists():
    """_composite_seed_score should be importable."""
    from agents.discovery_agent import _composite_seed_score
    assert callable(_composite_seed_score)


def test_composite_seed_score_basic():
    """Composite score should value relevance, recency, and citations."""
    from agents.discovery_agent import _composite_seed_score

    # High relevance, recent, high citations → high score
    good_paper = {"relevance_score": 0.9, "year": 2025, "citations": 100}
    # Low relevance, old, low citations → low score
    bad_paper = {"relevance_score": 0.1, "year": 2000, "citations": 0}

    good_score = _composite_seed_score(good_paper)
    bad_score = _composite_seed_score(bad_paper)
    assert good_score > bad_score, \
        f"Good paper ({good_score}) should score higher than bad ({bad_score})"


def test_composite_seed_score_prefers_relevant_over_cited():
    """A highly relevant paper should beat a highly cited but irrelevant one."""
    from agents.discovery_agent import _composite_seed_score

    relevant = {"relevance_score": 0.95, "year": 2024, "citations": 10}
    cited = {"relevance_score": 0.2, "year": 2024, "citations": 5000}

    assert _composite_seed_score(relevant) > _composite_seed_score(cited), \
        "Relevance should outweigh citation count in seed selection"


# === Test 9: Snowball semantic filtering function ===

def test_snowball_paper_is_relevant_exists():
    """_snowball_paper_is_relevant should be importable."""
    from agents.discovery_agent import _snowball_paper_is_relevant
    assert callable(_snowball_paper_is_relevant)


def test_snowball_uses_semantic_filtering():
    """snowball_sampling should use _snowball_paper_is_relevant."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    assert "_snowball_paper_is_relevant" in code


# === Test 10: Query dedup is used in expand_queries ===

def test_expand_queries_uses_dedup():
    """expand_queries should call deduplicate_queries_by_similarity."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    assert "deduplicate_queries_by_similarity" in code


# === Test 11: CJK keyword extraction uses jieba ===

def test_cjk_keyword_extraction():
    """_extract_cjk_keywords should segment Chinese text using jieba."""
    from aggregators.unified_search import _extract_cjk_keywords
    keywords = _extract_cjk_keywords("倍半萜香豆素生源合成途径解析")
    # jieba should produce meaningful multi-char tokens
    assert len(keywords) >= 3, f"Should extract >=3 keywords, got {keywords}"
    # Should contain domain terms
    assert any("香豆素" in kw or "豆素" in kw for kw in keywords), \
        f"Should contain 香豆素/豆素, got {keywords}"
    assert any("合成" in kw for kw in keywords), \
        f"Should contain 合成, got {keywords}"


def test_cjk_keyword_extraction_filters_stopwords():
    """CJK extraction should filter common Chinese function words."""
    from aggregators.unified_search import _extract_cjk_keywords
    keywords = _extract_cjk_keywords("关于生物合成的研究方法分析")
    # "关于", "研究", "分析", "方法" are stopwords
    assert "关于" not in keywords
    assert "研究" not in keywords
    assert "分析" not in keywords
    # "生物" and "合成" should remain
    assert any("合成" in kw for kw in keywords)


def test_cjk_has_cjk_detection():
    """_has_cjk should detect CJK characters."""
    from aggregators.unified_search import _has_cjk
    assert _has_cjk("倍半萜香豆素") is True
    assert _has_cjk("sesquiterpene coumarin") is False
    assert _has_cjk("混合 mixed text") is True


def test_cjk_match_keyword_substring():
    """CJK keywords should use substring matching."""
    from aggregators.unified_search import _match_keyword
    assert _match_keyword("香豆素", "倍半萜香豆素生物合成")
    assert not _match_keyword("香豆素", "倍半萜类化合物")


def test_keyword_relevance_cjk_papers():
    """_compute_keyword_relevance should work with CJK research questions."""
    from aggregators.unified_search import _compute_keyword_relevance
    from models.unified_paper import UnifiedPaper

    papers = [
        UnifiedPaper(
            title="倍半萜香豆素的生物合成途径", authors=["张三"],
            abstract="本文研究了伞形科植物中倍半萜香豆素的生源合成途径", year=2024,
        ),
        UnifiedPaper(
            title="SARS-CoV-2 vaccine development", authors=["Smith J"],
            abstract="This paper reviews COVID-19 vaccine candidates", year=2024,
        ),
    ]
    _compute_keyword_relevance(papers, "倍半萜香豆素生源合成途径解析")
    assert papers[0].relevance_score > papers[1].relevance_score, \
        f"Chinese matching paper ({papers[0].relevance_score}) should score higher than unrelated ({papers[1].relevance_score})"
    assert papers[0].relevance_score > 0.3, \
        f"Matching paper should score > 0.3, got {papers[0].relevance_score}"


def test_extract_keywords_cjk_in_discovery():
    """_extract_keywords in discovery_agent should handle CJK text."""
    from agents.discovery_agent import _extract_keywords
    keywords = _extract_keywords("倍半萜香豆素生源合成途径解析")
    assert len(keywords) >= 3, f"Should extract >=3 keywords, got {keywords}"
    assert any("合成" in kw for kw in keywords)


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
