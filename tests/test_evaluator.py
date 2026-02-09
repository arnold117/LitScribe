#!/usr/bin/env python3
"""Tests for ReviewEvaluator (Phase 9.5 Step 3).

Tests cover:
- Search quality evaluation
- Theme coverage evaluation
- Domain purity evaluation
- Citation grounding evaluation
- Content quality evaluation
- Failure mode detection
- Overall score computation
- Report formatting
"""

import sys
import os

# Add project root and src to path
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "scripts"))


def _make_output(
    papers=None, synthesis=None, self_review=None,
    token_usage=None, citation_grounding=None, errors=None,
):
    """Helper: build a minimal review output dict."""
    return {
        "research_question": "What are the advances in deep learning?",
        "analyzed_papers": papers or [],
        "synthesis": synthesis or {},
        "self_review": self_review,
        "token_usage": token_usage,
        "citation_grounding": citation_grounding,
        "errors": errors or [],
    }


def _make_ground_truth(**overrides):
    """Helper: build a minimal ground truth dict."""
    gt = {
        "id": "test",
        "expected_themes": ["deep learning", "optimization"],
        "must_include_keywords_in_review": ["deep learning", "neural"],
        "domain_keywords": ["machine learning", "neural"],
        "anti_keywords": ["alkaloid", "CRISPR"],
        "min_papers": 3,
        "max_irrelevant_ratio": 0.2,
        "min_citation_grounding_rate": 0.8,
    }
    gt.update(overrides)
    return gt


def test_evaluator_import():
    """ReviewEvaluator should import without errors."""
    from evaluate import ReviewEvaluator
    evaluator = ReviewEvaluator()
    assert evaluator is not None
    print("PASS: test_evaluator_import")


def test_search_quality_basic():
    """Search quality should count keyword hits."""
    from evaluate import ReviewEvaluator

    evaluator = ReviewEvaluator()
    output = _make_output(papers=[
        {"title": "Deep Learning for NLP", "abstract": "A neural network approach"},
        {"title": "Unrelated Paper", "abstract": "This is about cooking"},
    ])
    gt = _make_ground_truth()

    result = evaluator._evaluate_search(output, gt)
    assert result["paper_count"] == 2
    assert result["keyword_hit_rate"] > 0  # At least 1 paper matches
    assert result["keyword_hit_rate"] < 1.0  # Not all match
    print("PASS: test_search_quality_basic")


def test_search_anti_keywords():
    """Anti-keyword hits should be tracked."""
    from evaluate import ReviewEvaluator

    evaluator = ReviewEvaluator()
    output = _make_output(papers=[
        {"title": "Deep Learning", "abstract": "neural network"},
        {"title": "Alkaloid Research", "abstract": "alkaloid biosynthesis in plants"},
    ])
    gt = _make_ground_truth()

    result = evaluator._evaluate_search(output, gt)
    assert result["anti_keyword_hit_rate"] > 0
    print("PASS: test_search_anti_keywords")


def test_theme_coverage_full():
    """Full theme coverage when all themes present."""
    from evaluate import ReviewEvaluator

    evaluator = ReviewEvaluator()
    output = _make_output(synthesis={
        "themes": [
            {"theme": "Deep Learning"},
            {"theme": "Optimization Methods"},
        ],
        "review_text": "This review covers deep learning and optimization techniques.",
    })
    gt = _make_ground_truth(expected_themes=["deep learning", "optimization"])

    result = evaluator._evaluate_themes(output, gt)
    assert result["coverage_rate"] == 1.0
    assert len(result["missed"]) == 0
    print("PASS: test_theme_coverage_full")


def test_theme_coverage_partial():
    """Partial theme coverage when some themes missing."""
    from evaluate import ReviewEvaluator

    evaluator = ReviewEvaluator()
    output = _make_output(synthesis={
        "themes": [{"theme": "Deep Learning"}],
        "review_text": "This review only covers deep learning.",
    })
    gt = _make_ground_truth(expected_themes=["deep learning", "optimization", "generalization"])

    result = evaluator._evaluate_themes(output, gt)
    assert result["coverage_rate"] < 1.0
    assert len(result["missed"]) > 0
    print("PASS: test_theme_coverage_partial")


def test_domain_purity_clean():
    """Domain purity should be 1.0 when no contamination."""
    from evaluate import ReviewEvaluator

    evaluator = ReviewEvaluator()
    output = _make_output(papers=[
        {"title": "Neural Networks", "abstract": "machine learning approach"},
        {"title": "Deep Learning", "abstract": "neural architectures"},
    ])
    gt = _make_ground_truth()

    result = evaluator._evaluate_domain(output, gt)
    assert result["purity_rate"] == 1.0
    assert result["contaminated_count"] == 0
    print("PASS: test_domain_purity_clean")


def test_domain_purity_contaminated():
    """Domain purity should detect contamination."""
    from evaluate import ReviewEvaluator

    evaluator = ReviewEvaluator()
    output = _make_output(papers=[
        {"title": "Neural Networks", "abstract": "machine learning approach"},
        {"title": "Alkaloid Synthesis", "abstract": "alkaloid biosynthesis pathway"},
    ])
    gt = _make_ground_truth()

    result = evaluator._evaluate_domain(output, gt)
    assert result["purity_rate"] < 1.0
    assert result["contaminated_count"] >= 1
    print("PASS: test_domain_purity_contaminated")


def test_grounding_evaluation():
    """Evaluate pre-computed grounding data."""
    from evaluate import ReviewEvaluator

    evaluator = ReviewEvaluator()
    output = _make_output(
        citation_grounding={
            "grounding_rate": 0.9,
            "total_citations": 10,
            "grounded_count": 9,
            "ungrounded": ["FakeName, 2099"],
        }
    )
    gt = _make_ground_truth(min_citation_grounding_rate=0.8)

    result = evaluator._evaluate_grounding(output, gt)
    assert result["grounding_rate"] == 0.9
    assert result["meets_threshold"] is True
    print("PASS: test_grounding_evaluation")


def test_grounding_below_threshold():
    """Grounding below threshold should be flagged."""
    from evaluate import ReviewEvaluator

    evaluator = ReviewEvaluator()
    output = _make_output(
        citation_grounding={
            "grounding_rate": 0.5,
            "total_citations": 4,
            "grounded_count": 2,
            "ungrounded": ["Fake1, 2020", "Fake2, 2021"],
        }
    )
    gt = _make_ground_truth(min_citation_grounding_rate=0.8)

    result = evaluator._evaluate_grounding(output, gt)
    assert result["meets_threshold"] is False
    print("PASS: test_grounding_below_threshold")


def test_content_quality():
    """Content quality should check keyword coverage."""
    from evaluate import ReviewEvaluator

    evaluator = ReviewEvaluator()
    output = _make_output(synthesis={
        "review_text": "Deep learning and neural networks have transformed NLP.",
    })
    gt = _make_ground_truth(
        must_include_keywords_in_review=["deep learning", "neural"]
    )

    result = evaluator._evaluate_content(output, gt)
    assert result["word_count"] > 0
    assert result["keyword_coverage"] == 1.0
    assert len(result["missing_keywords"]) == 0
    print("PASS: test_content_quality")


def test_self_review_extraction():
    """Extract self-review scores from output."""
    from evaluate import ReviewEvaluator

    evaluator = ReviewEvaluator()
    output = _make_output(self_review={
        "overall_score": 0.85,
        "relevance_score": 0.9,
        "coverage_score": 0.8,
        "coherence_score": 0.85,
    })

    result = evaluator._extract_self_review(output)
    assert result["available"] is True
    assert result["overall_score"] == 0.85
    print("PASS: test_self_review_extraction")


def test_self_review_missing():
    """Missing self-review should report as unavailable."""
    from evaluate import ReviewEvaluator

    evaluator = ReviewEvaluator()
    output = _make_output()

    result = evaluator._extract_self_review(output)
    assert result["available"] is False
    print("PASS: test_self_review_missing")


def test_efficiency_with_token_usage():
    """Efficiency should report token usage when available."""
    from evaluate import ReviewEvaluator

    evaluator = ReviewEvaluator()
    output = _make_output(token_usage={
        "total_tokens": 50000,
        "total_calls": 15,
        "estimated_cost_usd": 0.25,
        "elapsed_seconds": 45.3,
    })

    result = evaluator._evaluate_efficiency(output)
    assert result["available"] is True
    assert result["total_tokens"] == 50000
    assert result["estimated_cost_usd"] == 0.25
    print("PASS: test_efficiency_with_token_usage")


def test_failure_search():
    """Detect search failure when not enough papers."""
    from evaluate import ReviewEvaluator

    evaluator = ReviewEvaluator()
    output = _make_output(papers=[{"title": "Only one", "abstract": ""}])
    gt = _make_ground_truth(min_papers=5)

    results = {
        "search": evaluator._evaluate_search(output, gt),
        "theme_coverage": {"missed": []},
        "domain_purity": {"contaminated_count": 0, "total_papers": 1},
        "citation_grounding": {"grounding_rate": 1.0, "meets_threshold": True},
        "self_review": {"available": False},
    }

    failures = evaluator._detect_failures(output, gt, results)
    modes = [f["mode"] for f in failures]
    assert "search_failure" in modes
    print("PASS: test_failure_search")


def test_failure_theme_miss():
    """Detect theme miss failure."""
    from evaluate import ReviewEvaluator

    evaluator = ReviewEvaluator()
    results = {
        "search": {"min_papers_met": True},
        "theme_coverage": {"missed": ["optimization", "scaling"]},
        "domain_purity": {"contaminated_count": 0, "total_papers": 5},
        "citation_grounding": {"grounding_rate": 1.0, "meets_threshold": True},
        "self_review": {"available": False},
    }

    failures = evaluator._detect_failures({}, _make_ground_truth(), results)
    modes = [f["mode"] for f in failures]
    assert "theme_miss" in modes
    print("PASS: test_failure_theme_miss")


def test_failure_hallucination():
    """Detect citation hallucination failure."""
    from evaluate import ReviewEvaluator

    evaluator = ReviewEvaluator()
    results = {
        "search": {"min_papers_met": True},
        "theme_coverage": {"missed": []},
        "domain_purity": {"contaminated_count": 0, "total_papers": 5},
        "citation_grounding": {
            "grounding_rate": 0.5,
            "meets_threshold": False,
            "ungrounded": ["Fake, 2020"],
        },
        "self_review": {"available": False},
    }

    failures = evaluator._detect_failures({}, _make_ground_truth(), results)
    modes = [f["mode"] for f in failures]
    assert "citation_hallucination" in modes
    print("PASS: test_failure_hallucination")


def test_overall_score():
    """Overall score should be a weighted composite."""
    from evaluate import ReviewEvaluator

    evaluator = ReviewEvaluator()
    output = _make_output(
        papers=[
            {"title": "Deep Learning", "abstract": "neural network approach"},
        ],
        synthesis={
            "themes": [{"theme": "deep learning"}],
            "review_text": "This covers deep learning and neural networks.",
        },
        self_review={"overall_score": 0.8, "relevance_score": 0.9,
                     "coverage_score": 0.7, "coherence_score": 0.8},
        citation_grounding={"grounding_rate": 1.0, "total_citations": 5,
                           "grounded_count": 5, "ungrounded": []},
    )
    gt = _make_ground_truth()

    results = evaluator.evaluate(output, gt)
    assert 0.0 <= results["overall_score"] <= 1.0
    assert "search" in results
    assert "theme_coverage" in results
    assert "failure_modes" in results
    print("PASS: test_overall_score")


def test_format_report():
    """Report formatting should produce valid markdown."""
    from evaluate import ReviewEvaluator, format_report

    evaluator = ReviewEvaluator()
    output = _make_output(
        papers=[{"title": "Test", "abstract": "neural network deep learning"}],
        synthesis={"themes": [], "review_text": "Deep learning neural review."},
    )
    gt = _make_ground_truth()

    results = evaluator.evaluate(output, gt)
    report = format_report(results)

    assert isinstance(report, str)
    assert "# Evaluation Report" in report
    assert "Search Quality" in report
    assert "Theme Coverage" in report
    assert "Failure Modes" in report
    print("PASS: test_format_report")


def main():
    tests = [
        test_evaluator_import,
        test_search_quality_basic,
        test_search_anti_keywords,
        test_theme_coverage_full,
        test_theme_coverage_partial,
        test_domain_purity_clean,
        test_domain_purity_contaminated,
        test_grounding_evaluation,
        test_grounding_below_threshold,
        test_content_quality,
        test_self_review_extraction,
        test_self_review_missing,
        test_efficiency_with_token_usage,
        test_failure_search,
        test_failure_theme_miss,
        test_failure_hallucination,
        test_overall_score,
        test_format_report,
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
