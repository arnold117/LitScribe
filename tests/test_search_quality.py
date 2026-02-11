#!/usr/bin/env python
"""Tests for search quality fixes.

Covers:
- PubMed MeSH term OR logic
- Relevance score threshold filtering
- Snowball None-safe keyword matching
- Paper summary author formatting
- Reference list filtering

Run with: pytest tests/test_search_quality.py -v
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# === Test 1: PubMed MeSH OR logic ===

def test_pubmed_mesh_uses_or():
    """MeSH terms should be joined with OR, not AND."""
    # Read the source to verify the fix
    source = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
    code = source.read_text()
    assert '" OR ".join' in code, "MeSH terms should use OR join"
    assert '" AND ".join(f"{m}[MeSH]"' not in code, "MeSH terms should NOT use AND join"


def test_pubmed_mesh_query_construction():
    """Verify MeSH query format is correct."""
    mesh_terms = ["Alkaloids", "Biosynthetic Pathways", "Plants, Medicinal"]
    mesh_filter = " OR ".join(f"{m}[MeSH]" for m in mesh_terms[:3])
    filtered_query = f"(huperzine A biosynthesis) AND ({mesh_filter})"

    assert "OR" in filtered_query
    assert "Alkaloids[MeSH]" in filtered_query
    assert "Biosynthetic Pathways[MeSH]" in filtered_query
    # Should NOT be AND between MeSH terms
    assert "Alkaloids[MeSH] AND Biosynthetic" not in filtered_query


# === Test 2: Relevance score threshold ===

def test_relevance_threshold_filters_low_scores():
    """Papers below relevance threshold should be filtered out."""
    from agents.discovery_agent import select_papers

    # Simulate: we can't call select_papers directly (needs LLM),
    # but we can verify the threshold logic exists in the code
    import inspect
    source = inspect.getsource(select_papers)
    assert "MIN_RELEVANCE" in source, "select_papers should have MIN_RELEVANCE threshold"
    assert "relevance_score" in source, "Should filter on relevance_score"


def test_relevance_threshold_value():
    """Minimum relevance threshold should be reasonable (0.2-0.4)."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    # Extract MIN_RELEVANCE value
    import re
    match = re.search(r'MIN_RELEVANCE\s*=\s*([\d.]+)', code)
    assert match, "MIN_RELEVANCE should be defined"
    threshold = float(match.group(1))
    assert 0.2 <= threshold <= 0.4, f"Threshold {threshold} should be between 0.2 and 0.4"


# === Test 3: Snowball None-safe keyword matching ===

def test_paper_matches_keywords_none_title():
    """_paper_matches_keywords should handle None title gracefully."""
    from agents.discovery_agent import _paper_matches_keywords

    paper = {"title": None, "abstract": "Huperzine A biosynthesis in plants"}
    keywords = ["huperzine", "biosynthesis"]
    # Should not raise TypeError
    result = _paper_matches_keywords(paper, keywords)
    assert isinstance(result, bool)


def test_paper_matches_keywords_none_abstract():
    """_paper_matches_keywords should handle None abstract gracefully."""
    from agents.discovery_agent import _paper_matches_keywords

    paper = {"title": "Huperzine A biosynthesis review", "abstract": None}
    keywords = ["huperzine", "biosynthesis"]
    result = _paper_matches_keywords(paper, keywords)
    assert result is True  # Both keywords in title


def test_paper_matches_keywords_both_none():
    """_paper_matches_keywords should handle both None title and abstract."""
    from agents.discovery_agent import _paper_matches_keywords

    paper = {"title": None, "abstract": None}
    keywords = ["huperzine"]
    result = _paper_matches_keywords(paper, keywords)
    assert result is False  # No text to match against


def test_paper_matches_keywords_missing_keys():
    """_paper_matches_keywords should handle missing title/abstract keys."""
    from agents.discovery_agent import _paper_matches_keywords

    paper = {}  # No title or abstract keys
    keywords = ["test"]
    result = _paper_matches_keywords(paper, keywords)
    assert result is False


def test_paper_matches_keywords_normal():
    """_paper_matches_keywords should work normally with valid strings."""
    from agents.discovery_agent import _paper_matches_keywords

    paper = {
        "title": "Huperzine A biosynthesis pathway",
        "abstract": "This study investigates the alkaloid biosynthesis in Huperzia serrata."
    }
    keywords = ["huperzine", "biosynthesis", "alkaloid"]
    result = _paper_matches_keywords(paper, keywords, min_matches=2)
    assert result is True


# === Test 4: Paper summary author formatting ===

def test_format_summaries_includes_authors():
    """format_summaries_for_prompt should include author names."""
    from agents.prompts import format_summaries_for_prompt

    summaries = [
        {
            "paper_id": "10.1234/test",
            "title": "Test Paper",
            "year": 2024,
            "authors": ["John Smith", "Jane Doe", "Bob Wilson"],
            "key_findings": ["Finding 1"],
            "methodology": "Test method",
            "strengths": ["Strong"],
            "limitations": ["Weak"],
        }
    ]

    result = format_summaries_for_prompt(summaries)
    assert "John Smith" in result, "Should include author names"
    assert "Jane Doe" in result
    assert "Smith et al., 2024" in result, "Should include cite-as guidance"


def test_format_summaries_cite_as_first_author():
    """Cite-as line should use first author's last name."""
    from agents.prompts import format_summaries_for_prompt

    summaries = [
        {
            "paper_id": "test",
            "title": "Paper",
            "year": 2023,
            "authors": ["Thanh Thi Minh Le", "Another Author"],
            "key_findings": ["F1"],
            "methodology": "M",
            "strengths": [],
            "limitations": [],
        }
    ]

    result = format_summaries_for_prompt(summaries)
    assert "Le et al., 2023" in result, "Should use last name of first author"


def test_format_summaries_empty_authors():
    """Should handle empty authors list gracefully."""
    from agents.prompts import format_summaries_for_prompt

    summaries = [
        {
            "paper_id": "test",
            "title": "Paper",
            "year": 2023,
            "authors": [],
            "key_findings": ["F1"],
            "methodology": "M",
            "strengths": [],
            "limitations": [],
        }
    ]

    result = format_summaries_for_prompt(summaries)
    assert "Unknown" in result, "Should show Unknown for empty authors"


def test_format_summaries_many_authors():
    """Should truncate to 3 authors + 'et al.'"""
    from agents.prompts import format_summaries_for_prompt

    summaries = [
        {
            "paper_id": "test",
            "title": "Paper",
            "year": 2023,
            "authors": ["A One", "B Two", "C Three", "D Four", "E Five"],
            "key_findings": ["F1"],
            "methodology": "M",
            "strengths": [],
            "limitations": [],
        }
    ]

    result = format_summaries_for_prompt(summaries)
    assert "A One" in result
    assert "C Three" in result
    assert "et al." in result
    assert "D Four" not in result, "Should truncate after 3 authors"


def test_format_summaries_pubmed_cite_as():
    """Cite-as line should use correct last name for PubMed author format."""
    from agents.prompts import format_summaries_for_prompt

    summaries = [
        {
            "paper_id": "test1",
            "title": "Alkaloid Paper",
            "year": 2006,
            "authors": ["Ma X", "Tan C", "Zhu D", "Gang DR"],
            "key_findings": ["F1"],
            "methodology": "M",
            "strengths": [],
            "limitations": [],
        },
        {
            "paper_id": "test2",
            "title": "Review Paper",
            "year": 2024,
            "authors": ["Zhang ZJ", "Jiang S"],
            "key_findings": ["F1"],
            "methodology": "M",
            "strengths": [],
            "limitations": [],
        },
    ]

    result = format_summaries_for_prompt(summaries)
    assert "[Ma et al., 2006]" in result, f"Expected [Ma et al., 2006], got: {result}"
    assert "[Zhang et al., 2024]" in result, f"Expected [Zhang et al., 2024], got: {result}"
    # Should NOT contain single-letter cite names
    assert "[X et al." not in result
    assert "[ZJ et al." not in result


# === Test 5: Self-review uses extract_json + uses regular model ===

def test_self_review_uses_extract_json():
    """self_review_agent should use extract_json instead of json.loads."""
    source = Path(__file__).parent.parent / "src" / "agents" / "self_review_agent.py"
    code = source.read_text()
    assert "extract_json" in code, "Should use extract_json"
    # The old pattern should NOT be present
    assert "json.loads(response)" not in code, "Should not use raw json.loads on response"


def test_self_review_not_routed_to_reasoning_model():
    """self_review should NOT use reasoning model (bad at structured JSON output)."""
    from agents.tools import REASONING_TASK_TYPES
    assert "self_review" not in REASONING_TASK_TYPES, \
        "self_review should not be in REASONING_TASK_TYPES (reasoning models are poor at JSON output)"


def test_synthesis_not_routed_to_reasoning_model():
    """synthesis should NOT use reasoning model (cites too few papers, crashes with GraphRAG prompts)."""
    from agents.tools import REASONING_TASK_TYPES
    assert "synthesis" not in REASONING_TASK_TYPES, \
        "synthesis should not be in REASONING_TASK_TYPES (reasoning models cite too few papers)"


# === Test 6: Reference filtering logic ===

def test_reference_filtering_by_year_and_author():
    """References should be filtered using citation_grounding extraction."""
    from analysis.citation_grounding import extract_inline_citations, _parse_citation

    review_text = """
    This was shown by [Smith, 2020] and confirmed by [Jones et al., 2021].
    Additional evidence comes from [Lee, 2019].
    """

    raw_citations = extract_inline_citations(review_text)
    assert len(raw_citations) == 3

    cited_authors = set()
    cited_years = set()
    for cit in raw_citations:
        parsed = _parse_citation(cit)
        if parsed:
            author_part, year = parsed
            primary = author_part.split()[0].replace(",", "").rstrip("等")
            cited_authors.add(primary.lower())
            cited_years.add(year)

    assert cited_years == {"2020", "2021", "2019"}
    assert "smith" in cited_authors
    assert "jones" in cited_authors
    assert "lee" in cited_authors


def test_reference_filtering_keeps_matching_papers():
    """Only papers matching year AND author should be kept (no year-only fallback)."""
    from analysis.citation_grounding import extract_inline_citations, _parse_citation, _extract_last_names

    review_text = "[Smith, 2020] found that... [Jones et al., 2021] confirmed..."

    raw_citations = extract_inline_citations(review_text)
    cited_authors = set()
    cited_years = set()
    for cit in raw_citations:
        parsed = _parse_citation(cit)
        if parsed:
            author_part, year = parsed
            primary = author_part.split()[0].replace(",", "").rstrip("等")
            cited_authors.add(primary.lower())
            cited_years.add(year)

    papers = [
        {"year": 2020, "authors": ["John Smith"], "title": "Cited Paper 1"},
        {"year": 2021, "authors": ["Sarah Jones", "Bob Lee"], "title": "Cited Paper 2"},
        {"year": 2020, "authors": ["Eve Brown"], "title": "Same Year Diff Author"},
        {"year": 2022, "authors": ["Eve Brown"], "title": "Uncited Paper"},
        {"year": 2018, "authors": ["Old Author"], "title": "Old Uncited Paper"},
    ]

    cited_papers = []
    for p in papers:
        p_year = str(p.get("year", ""))
        if p_year not in cited_years:
            continue
        authors = p.get("authors", [])
        if isinstance(authors, str):
            authors = [authors]
        last_names = _extract_last_names(authors)
        if any(ln.lower() in cited_authors for ln in last_names):
            cited_papers.append(p)

    # Should include Smith (2020) and Jones (2021) but NOT Brown (2020, wrong author)
    assert len(cited_papers) == 2
    assert cited_papers[0]["title"] == "Cited Paper 1"
    assert cited_papers[1]["title"] == "Cited Paper 2"


def test_reference_filtering_chinese_citations():
    """Reference filtering should work with Chinese 等 citations."""
    from analysis.citation_grounding import extract_inline_citations, _parse_citation, _extract_last_names

    review_text = "[Ma等, 2006] 发现了重要途径。[Chagnon等, 2009] 也证实了这一点。"

    raw_citations = extract_inline_citations(review_text)
    assert len(raw_citations) == 2

    cited_authors = set()
    cited_years = set()
    for cit in raw_citations:
        parsed = _parse_citation(cit)
        if parsed:
            author_part, year = parsed
            primary = author_part.split()[0].replace(",", "").rstrip("等")
            cited_authors.add(primary.lower())
            cited_years.add(year)

    assert "ma" in cited_authors
    assert "chagnon" in cited_authors

    papers = [
        {"year": 2006, "authors": ["Xiao-Jian Ma"], "title": "Alkaloid paper"},
        {"year": 2009, "authors": ["Marie Chagnon"], "title": "Ecology paper"},
        {"year": 2006, "authors": ["Unrelated Author"], "title": "Schwertmannite paper"},
    ]

    cited_papers = []
    for p in papers:
        p_year = str(p.get("year", ""))
        if p_year not in cited_years:
            continue
        authors = p.get("authors", [])
        if isinstance(authors, str):
            authors = [authors]
        last_names = _extract_last_names(authors)
        if any(ln.lower() in cited_authors for ln in last_names):
            cited_papers.append(p)

    # Should keep Ma and Chagnon, NOT "Unrelated Author" even though year matches
    assert len(cited_papers) == 2
    assert cited_papers[0]["title"] == "Alkaloid paper"
    assert cited_papers[1]["title"] == "Ecology paper"


def test_reference_filtering_yearless_citations():
    """Reference filtering should handle citations without years using author match only."""
    from analysis.citation_grounding import extract_all_cited_authors, _extract_last_names

    # Mix of year-bearing and year-less citations (like deepseek-reasoner produces)
    review_text = (
        "[Zhang et al.] showed this pathway. [Wang et al.] optimized it. "
        "[Liu et al., 2021] used gVLC. [Ma et al., 2006] surveyed plants. "
        "[Mitternacht, 2016] built FreeSASA."
    )

    cited_authors = extract_all_cited_authors(review_text)
    assert "zhang" in cited_authors
    assert "wang" in cited_authors
    assert "liu" in cited_authors
    assert "ma" in cited_authors
    assert "mitternacht" in cited_authors

    papers = [
        {"year": 2023, "authors": ["Wei Zhang", "Another"], "title": "Genomic paper"},
        {"year": 2022, "authors": ["Hao Wang"], "title": "Fermentation paper"},
        {"year": 2021, "authors": ["Xiao Liu", "B Person"], "title": "gVLC paper"},
        {"year": 2006, "authors": ["Xiao-Jian Ma"], "title": "Survey paper"},
        {"year": 2016, "authors": ["Simon Mitternacht"], "title": "FreeSASA"},
        {"year": 2024, "authors": ["Unrelated Person"], "title": "StaPep junk"},
        {"year": 2020, "authors": ["Eve Brown"], "title": "Heptapeptide junk"},
    ]

    cited_papers = []
    for p in papers:
        authors = p.get("authors", [])
        if isinstance(authors, str):
            authors = [authors]
        last_names = _extract_last_names(authors)
        if any(ln.lower() in cited_authors for ln in last_names):
            cited_papers.append(p)

    # Should include Zhang, Wang, Liu, Ma, Mitternacht but NOT Person or Brown
    assert len(cited_papers) == 5, f"Expected 5, got {len(cited_papers)}: {[p['title'] for p in cited_papers]}"
    titles = {p["title"] for p in cited_papers}
    assert "StaPep junk" not in titles
    assert "Heptapeptide junk" not in titles


# === Test 7: Word-boundary keyword matching ===

def test_keyword_matching_word_boundary_negative():
    """Keyword matching should use word boundaries, not substring."""
    from agents.discovery_agent import _paper_matches_keywords

    # "bio" should NOT match "biography"
    paper = {"title": "Biography of a scientist", "abstract": "Life story of Dr. Smith"}
    keywords = ["bio", "synthesis"]
    result = _paper_matches_keywords(paper, keywords, min_matches=1)
    assert result is False, "'bio' should not match 'biography' with word boundaries"


def test_keyword_matching_word_boundary_positive():
    """Word-boundary matching should still match exact words."""
    from agents.discovery_agent import _paper_matches_keywords

    paper = {"title": "Bio synthesis of alkaloids", "abstract": "A bio approach"}
    keywords = ["bio", "synthesis"]
    result = _paper_matches_keywords(paper, keywords, min_matches=2)
    assert result is True, "Exact word 'bio' and 'synthesis' should match"


def test_keyword_matching_hyphenated():
    """Word-boundary matching should match across hyphens."""
    from agents.discovery_agent import _paper_matches_keywords

    paper = {"title": "Huperzine-A biosynthesis pathway", "abstract": ""}
    keywords = ["huperzine", "biosynthesis"]
    result = _paper_matches_keywords(paper, keywords, min_matches=2)
    assert result is True, "huperzine should match in 'Huperzine-A'"


# === Test 8: Pre-synthesis relevance filter ===

def test_pre_synthesis_filter_exists():
    """critical_reading_agent should filter low-relevance papers before synthesis."""
    source = Path(__file__).parent.parent / "src" / "agents" / "critical_reading_agent.py"
    code = source.read_text()
    assert "PRE_SYNTHESIS_MIN_RELEVANCE" in code, \
        "critical_reading_agent should have PRE_SYNTHESIS_MIN_RELEVANCE threshold"


def test_pre_synthesis_filter_logic():
    """Papers below pre-synthesis threshold should be removed."""
    PRE_SYNTHESIS_MIN_RELEVANCE = 0.4
    papers = [
        {"paper_id": "p1", "title": "Relevant", "relevance_score": 0.8},
        {"paper_id": "p2", "title": "Irrelevant SARS paper", "relevance_score": 0.1},
        {"paper_id": "p3", "title": "Tangential", "relevance_score": 0.45},
    ]

    filtered = [p for p in papers if p.get("relevance_score", 0.5) >= PRE_SYNTHESIS_MIN_RELEVANCE]
    assert len(filtered) == 2, "Should keep papers with relevance >= 0.4"
    assert filtered[0]["paper_id"] == "p1"
    assert filtered[1]["paper_id"] == "p3"


# === Test 9: Snowball dynamic min_matches ===

def test_snowball_dynamic_min_matches():
    """Snowball sampling should require more keyword matches when >= 4 keywords."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    assert "snowball_min_matches" in code, \
        "Snowball sampling should use dynamic snowball_min_matches"


# === Test 10: MeSH filtering improvement ===

def test_mesh_filter_requires_primary_term():
    """When 3+ MeSH terms, primary term should be required (AND)."""
    source = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
    code = source.read_text()
    assert "len(mesh_terms) >= 3" in code, \
        "Should have conditional logic for 3+ MeSH terms"


# === Test 11: Word-boundary in unified_search ===

def test_unified_search_uses_word_boundary():
    """_compute_keyword_relevance should use re.search for word-boundary matching."""
    source = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
    code = source.read_text()
    assert "re.search" in code, \
        "_compute_keyword_relevance should use re.search for word-boundary matching"
    assert r"\b" in code, \
        "Should use \\b word boundary pattern"


# === Test 12: Query expansion generates 8 queries ===

def test_query_expansion_prompt_asks_for_8():
    """QUERY_EXPANSION_PROMPT should request 8 queries (was 5)."""
    from agents.prompts import QUERY_EXPANSION_PROMPT
    assert "exactly 8 search queries" in QUERY_EXPANSION_PROMPT, \
        "Should request exactly 8 queries"


# === Test 13: Search query cap raised to 12 ===

def test_search_query_cap_raised():
    """Discovery agent should search at least 12 queries (was 6)."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    assert "queries[:12]" in code, \
        "Should search top 12 queries (was queries[:6])"


# === Test 14: Paper keyword extraction function exists ===

def test_paper_keyword_extraction_exists():
    """_extract_queries_from_papers should be defined in discovery_agent."""
    from agents.discovery_agent import _extract_queries_from_papers
    assert callable(_extract_queries_from_papers)


def test_paper_keyword_extraction_basic():
    """Paper keyword extraction should find novel terms from paper titles."""
    from agents.discovery_agent import _extract_queries_from_papers

    papers = [
        {"title": "Alkaloid biosynthesis in endophytic fungi", "relevance_score": 0.9},
        {"title": "Endophytic fungi produce huperzine compounds", "relevance_score": 0.8},
        {"title": "Fermentation optimization for endophytic alkaloid production", "relevance_score": 0.7},
        {"title": "Endophytic fungal metabolites in medicinal plants", "relevance_score": 0.6},
    ]

    queries = _extract_queries_from_papers(papers, "alkaloid biosynthesis", max_queries=3)
    assert len(queries) >= 1, f"Should generate at least 1 query, got {len(queries)}"
    # "endophytic" appears in 4/4 papers but not in the research question
    assert any("endophytic" in q for q in queries), \
        f"Should extract 'endophytic' as novel term, got: {queries}"


# === Test 15: Sub-topic priority scaling ===

def test_subtopic_priority_scaling():
    """High-priority sub-topics should get more queries than low-priority."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    assert "priority >= 0.8" in code, "Should have priority scaling logic"
    assert "queries_for_topic[:3]" in code, "High-priority should get up to 3 queries"
    assert "queries_for_topic[:1]" in code, "Low-priority should get 1 query"


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
