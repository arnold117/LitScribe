#!/usr/bin/env python3
"""Tests for Citation Grounding (Phase 9.5 Step 2).

Tests cover:
- Inline citation extraction from review text
- Author name parsing and matching
- Grounding against analyzed papers
- Edge cases (no citations, no papers, multi-citations)
- Grounding rate calculation
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_extract_simple_citation():
    """Extract a simple [Author, Year] citation."""
    from analysis.citation_grounding import extract_inline_citations

    text = "Recent work [Smith, 2020] shows promising results."
    citations = extract_inline_citations(text)
    assert len(citations) == 1
    assert "Smith, 2020" in citations[0] or "Smith" in citations[0]
    print("PASS: test_extract_simple_citation")


def test_extract_et_al():
    """Extract [Author et al., Year] citation."""
    from analysis.citation_grounding import extract_inline_citations

    text = "As shown by [Smith et al., 2021], the method works."
    citations = extract_inline_citations(text)
    assert len(citations) == 1
    assert "Smith et al." in citations[0]
    print("PASS: test_extract_et_al")


def test_extract_two_authors():
    """Extract [Author & Author, Year] citation."""
    from analysis.citation_grounding import extract_inline_citations

    text = "According to [Smith & Jones, 2019], this is valid."
    citations = extract_inline_citations(text)
    assert len(citations) == 1
    assert "Smith" in citations[0]
    assert "Jones" in citations[0]
    print("PASS: test_extract_two_authors")


def test_extract_multi_citation():
    """Extract multiple citations in one bracket."""
    from analysis.citation_grounding import extract_inline_citations

    text = "Multiple studies [Smith, 2020; Jones, 2021] confirmed this."
    citations = extract_inline_citations(text)
    assert len(citations) == 2
    print("PASS: test_extract_multi_citation")


def test_extract_no_citations():
    """No citations in text returns empty list."""
    from analysis.citation_grounding import extract_inline_citations

    text = "This is plain text without any citations."
    citations = extract_inline_citations(text)
    assert len(citations) == 0
    print("PASS: test_extract_no_citations")


def test_extract_multiple_scattered():
    """Extract citations scattered across text."""
    from analysis.citation_grounding import extract_inline_citations

    text = (
        "Introduction [Smith, 2020] discusses this topic. "
        "Furthermore, [Jones et al., 2019] found similar results. "
        "A recent study [Chen, 2022] confirms the findings."
    )
    citations = extract_inline_citations(text)
    assert len(citations) == 3
    print("PASS: test_extract_multiple_scattered")


def test_grounding_all_matched():
    """All citations should be grounded when papers match."""
    from analysis.citation_grounding import check_citation_grounding

    review_text = "As shown by [Smith, 2020] and [Jones, 2021], the results are clear."
    papers = [
        {"paper_id": "p1", "title": "Deep Learning", "authors": ["John Smith"], "year": 2020},
        {"paper_id": "p2", "title": "NLP Review", "authors": ["Alice Jones"], "year": 2021},
    ]

    result = check_citation_grounding(review_text, papers)
    assert result["grounding_rate"] == 1.0
    assert result["ungrounded_count"] == 0
    assert result["grounded_count"] == 2
    print("PASS: test_grounding_all_matched")


def test_grounding_with_ungrounded():
    """Ungrounded citations should be detected."""
    from analysis.citation_grounding import check_citation_grounding

    # "Fakename" (lowercase 'n') matches regex [A-Z][a-z]+; "FakeName" would not.
    review_text = "According to [Smith, 2020] and [Fakename, 2099], this is true."
    papers = [
        {"paper_id": "p1", "title": "Real Paper", "authors": ["John Smith"], "year": 2020},
    ]

    result = check_citation_grounding(review_text, papers)
    assert result["grounded_count"] == 1
    assert result["ungrounded_count"] == 1
    assert "Fakename, 2099" in result["ungrounded"]
    assert result["grounding_rate"] == 0.5
    print("PASS: test_grounding_with_ungrounded")


def test_grounding_no_papers():
    """All citations ungrounded when no papers provided."""
    from analysis.citation_grounding import check_citation_grounding

    review_text = "Study [Smith, 2020] shows this."
    papers = []

    result = check_citation_grounding(review_text, papers)
    assert result["grounded_count"] == 0
    assert result["ungrounded_count"] == 1
    print("PASS: test_grounding_no_papers")


def test_grounding_no_citations():
    """No citations means 100% grounding rate (vacuously true)."""
    from analysis.citation_grounding import check_citation_grounding

    review_text = "This is plain text."
    papers = [
        {"paper_id": "p1", "title": "Paper", "authors": ["Smith"], "year": 2020},
    ]

    result = check_citation_grounding(review_text, papers)
    assert result["total_citations"] == 0
    assert result["grounding_rate"] == 1.0
    print("PASS: test_grounding_no_citations")


def test_author_format_last_first():
    """Match author in 'Last, First' format."""
    from analysis.citation_grounding import check_citation_grounding

    review_text = "According to [Smith, 2020], results are positive."
    papers = [
        {"paper_id": "p1", "title": "Paper", "authors": ["Smith, J."], "year": 2020},
    ]

    result = check_citation_grounding(review_text, papers)
    assert result["grounded_count"] == 1
    print("PASS: test_author_format_last_first")


def test_author_format_first_last():
    """Match author in 'First Last' format."""
    from analysis.citation_grounding import check_citation_grounding

    review_text = "Study by [Chen, 2022] confirmed this."
    papers = [
        {"paper_id": "p1", "title": "Paper", "authors": ["Wei Chen"], "year": 2022},
    ]

    result = check_citation_grounding(review_text, papers)
    assert result["grounded_count"] == 1
    print("PASS: test_author_format_first_last")


def test_deduplication():
    """Same citation appearing twice should be counted once."""
    from analysis.citation_grounding import check_citation_grounding

    review_text = "[Smith, 2020] shows X. Also [Smith, 2020] confirms Y."
    papers = [
        {"paper_id": "p1", "title": "Paper", "authors": ["Smith"], "year": 2020},
    ]

    result = check_citation_grounding(review_text, papers)
    assert result["total_citations"] == 1  # Deduplicated
    assert result["grounded_count"] == 1
    print("PASS: test_deduplication")


def test_authors_as_string():
    """Handle authors field as a string instead of list."""
    from analysis.citation_grounding import check_citation_grounding

    review_text = "[Lee, 2023] found this."
    papers = [
        {"paper_id": "p1", "title": "Paper", "authors": "Min-Jae Lee", "year": 2023},
    ]

    result = check_citation_grounding(review_text, papers)
    assert result["grounded_count"] == 1
    print("PASS: test_authors_as_string")


def test_return_structure():
    """Verify return dict has all expected keys."""
    from analysis.citation_grounding import check_citation_grounding

    result = check_citation_grounding("No citations.", [])
    expected_keys = {"grounded", "ungrounded", "total_citations",
                     "grounded_count", "ungrounded_count", "grounding_rate"}
    assert expected_keys.issubset(set(result.keys())), \
        f"Missing keys: {expected_keys - set(result.keys())}"
    print("PASS: test_return_structure")


def main():
    tests = [
        test_extract_simple_citation,
        test_extract_et_al,
        test_extract_two_authors,
        test_extract_multi_citation,
        test_extract_no_citations,
        test_extract_multiple_scattered,
        test_grounding_all_matched,
        test_grounding_with_ungrounded,
        test_grounding_no_papers,
        test_grounding_no_citations,
        test_author_format_last_first,
        test_author_format_first_last,
        test_deduplication,
        test_authors_as_string,
        test_return_structure,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
