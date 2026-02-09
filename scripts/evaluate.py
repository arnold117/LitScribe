#!/usr/bin/env python3
"""Evaluation framework for LitScribe literature reviews.

Compares generated review output against ground truth benchmarks,
computing metrics for search quality, theme coverage, citation
grounding, cost efficiency, and failure modes.

Usage:
    python scripts/evaluate.py --output output/review_*.json \
                               --ground-truth benchmarks/ground_truth/bio_alkaloid_expected.json
    python scripts/evaluate.py --output-dir benchmarks/results/ --all
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class ReviewEvaluator:
    """Evaluates a LitScribe review output against ground truth."""

    def evaluate(
        self,
        output: Dict[str, Any],
        ground_truth: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run all evaluation metrics.

        Args:
            output: The review output JSON (from litscribe review)
            ground_truth: The expected results JSON

        Returns:
            Dict with all metric scores and details
        """
        results = {
            "query_id": ground_truth.get("id", "unknown"),
            "research_question": output.get("research_question", ""),
        }

        # 1. Search quality metrics
        results["search"] = self._evaluate_search(output, ground_truth)

        # 2. Theme coverage
        results["theme_coverage"] = self._evaluate_themes(output, ground_truth)

        # 3. Domain purity
        results["domain_purity"] = self._evaluate_domain(output, ground_truth)

        # 4. Citation grounding
        results["citation_grounding"] = self._evaluate_grounding(output, ground_truth)

        # 5. Review content quality
        results["content_quality"] = self._evaluate_content(output, ground_truth)

        # 6. Self-review score (from LLM)
        results["self_review"] = self._extract_self_review(output)

        # 7. Cost and efficiency
        results["efficiency"] = self._evaluate_efficiency(output)

        # 8. Failure modes
        results["failure_modes"] = self._detect_failures(output, ground_truth, results)

        # Overall composite score
        results["overall_score"] = self._compute_overall(results)

        return results

    def _evaluate_search(
        self, output: Dict, ground_truth: Dict
    ) -> Dict[str, Any]:
        """Evaluate search quality: did we find relevant papers?"""
        analyzed = output.get("analyzed_papers", [])
        if not analyzed:
            return {"paper_count": 0, "keyword_hit_rate": 0.0}

        must_keywords = [
            kw.lower() for kw in ground_truth.get("must_include_keywords_in_review", [])
        ]
        anti_keywords = [
            kw.lower() for kw in ground_truth.get("anti_keywords", [])
        ]

        # Check how many papers have relevant keywords in title/abstract
        keyword_hits = 0
        anti_hits = 0
        for paper in analyzed:
            text = (
                (paper.get("title", "") + " " + paper.get("abstract", ""))
                .lower()
            )
            if any(kw in text for kw in must_keywords):
                keyword_hits += 1
            if any(kw in text for kw in anti_keywords):
                anti_hits += 1

        total = len(analyzed)
        return {
            "paper_count": total,
            "keyword_hit_rate": round(keyword_hits / total, 4) if total else 0.0,
            "anti_keyword_hit_rate": round(anti_hits / total, 4) if total else 0.0,
            "min_papers_met": total >= ground_truth.get("min_papers", 5),
        }

    def _evaluate_themes(
        self, output: Dict, ground_truth: Dict
    ) -> Dict[str, Any]:
        """Evaluate theme coverage against expected themes."""
        synthesis = output.get("synthesis", {})
        if not synthesis:
            return {"coverage_rate": 0.0, "matched": [], "missed": []}

        review_text = synthesis.get("review_text", "").lower()
        themes = synthesis.get("themes", [])
        theme_names = [t.get("theme", "").lower() for t in themes]

        expected = ground_truth.get("expected_themes", [])
        matched = []
        missed = []

        for expected_theme in expected:
            et_lower = expected_theme.lower()
            # Check if theme appears in theme list or review text
            found = any(et_lower in tn for tn in theme_names) or et_lower in review_text
            if found:
                matched.append(expected_theme)
            else:
                missed.append(expected_theme)

        total = len(expected) if expected else 1
        return {
            "coverage_rate": round(len(matched) / total, 4),
            "matched": matched,
            "missed": missed,
            "total_themes_generated": len(themes),
        }

    def _evaluate_domain(
        self, output: Dict, ground_truth: Dict
    ) -> Dict[str, Any]:
        """Evaluate domain purity: no papers from wrong domains?"""
        analyzed = output.get("analyzed_papers", [])
        if not analyzed:
            return {"purity_rate": 1.0}

        domain_keywords = [
            kw.lower() for kw in ground_truth.get("domain_keywords", [])
        ]
        anti_keywords = [
            kw.lower() for kw in ground_truth.get("anti_keywords", [])
        ]

        contaminated = 0
        for paper in analyzed:
            text = (
                (paper.get("title", "") + " " + paper.get("abstract", ""))
                .lower()
            )
            # Paper is contaminated if it matches anti-keywords but not domain keywords
            has_domain = any(kw in text for kw in domain_keywords) if domain_keywords else True
            has_anti = any(kw in text for kw in anti_keywords)
            if has_anti and not has_domain:
                contaminated += 1

        total = len(analyzed)
        return {
            "purity_rate": round(1 - contaminated / total, 4) if total else 1.0,
            "contaminated_count": contaminated,
            "total_papers": total,
        }

    def _evaluate_grounding(
        self, output: Dict, ground_truth: Dict
    ) -> Dict[str, Any]:
        """Evaluate citation grounding rate."""
        grounding = output.get("citation_grounding", {})
        if not grounding:
            # If no grounding data, run it ourselves
            synthesis = output.get("synthesis", {})
            review_text = synthesis.get("review_text", "")
            analyzed = output.get("analyzed_papers", [])
            if review_text and analyzed:
                try:
                    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
                    from analysis.citation_grounding import check_citation_grounding
                    grounding = check_citation_grounding(review_text, analyzed)
                except ImportError:
                    return {"grounding_rate": None, "error": "citation_grounding not available"}

        if not grounding:
            return {"grounding_rate": None}

        min_rate = ground_truth.get("min_citation_grounding_rate", 0.8)
        rate = grounding.get("grounding_rate", 0)
        return {
            "grounding_rate": rate,
            "total_citations": grounding.get("total_citations", 0),
            "grounded_count": grounding.get("grounded_count", 0),
            "ungrounded": grounding.get("ungrounded", []),
            "meets_threshold": rate >= min_rate,
        }

    def _evaluate_content(
        self, output: Dict, ground_truth: Dict
    ) -> Dict[str, Any]:
        """Evaluate review content quality."""
        synthesis = output.get("synthesis", {})
        review_text = synthesis.get("review_text", "")
        if not review_text:
            return {"word_count": 0, "keyword_coverage": 0.0}

        word_count = len(review_text.split())
        review_lower = review_text.lower()

        must_keywords = ground_truth.get("must_include_keywords_in_review", [])
        found = [kw for kw in must_keywords if kw.lower() in review_lower]
        missing = [kw for kw in must_keywords if kw.lower() not in review_lower]

        return {
            "word_count": word_count,
            "keyword_coverage": round(len(found) / len(must_keywords), 4) if must_keywords else 1.0,
            "found_keywords": found,
            "missing_keywords": missing,
            "has_references": "reference" in review_lower or "citation" in review_lower,
        }

    def _extract_self_review(self, output: Dict) -> Dict[str, Any]:
        """Extract self-review scores from output."""
        sr = output.get("self_review", {})
        if not sr:
            return {"available": False}
        return {
            "available": True,
            "overall_score": sr.get("overall_score", 0),
            "relevance_score": sr.get("relevance_score", 0),
            "coverage_score": sr.get("coverage_score", 0),
            "coherence_score": sr.get("coherence_score", 0),
        }

    def _evaluate_efficiency(self, output: Dict) -> Dict[str, Any]:
        """Evaluate cost and time efficiency."""
        token_usage = output.get("token_usage", {})
        if not token_usage:
            return {"available": False}
        return {
            "available": True,
            "total_tokens": token_usage.get("total_tokens", 0),
            "total_calls": token_usage.get("total_calls", 0),
            "estimated_cost_usd": token_usage.get("estimated_cost_usd", 0),
            "elapsed_seconds": token_usage.get("elapsed_seconds", 0),
        }

    def _detect_failures(
        self, output: Dict, ground_truth: Dict, results: Dict
    ) -> List[Dict[str, str]]:
        """Detect failure modes in the review."""
        failures = []

        # Search failure: not enough papers
        search = results.get("search", {})
        if not search.get("min_papers_met", True):
            failures.append({
                "mode": "search_failure",
                "detail": f"Only {search.get('paper_count', 0)} papers found, "
                          f"minimum {ground_truth.get('min_papers', 5)} required",
            })

        # Domain contamination
        domain = results.get("domain_purity", {})
        if domain.get("contaminated_count", 0) > 0:
            max_ratio = ground_truth.get("max_irrelevant_ratio", 0.2)
            total = domain.get("total_papers", 1)
            ratio = domain["contaminated_count"] / total
            if ratio > max_ratio:
                failures.append({
                    "mode": "domain_contamination",
                    "detail": f"{domain['contaminated_count']}/{total} papers "
                              f"from wrong domain ({ratio:.0%} > {max_ratio:.0%} threshold)",
                })

        # Citation hallucination
        grounding = results.get("citation_grounding", {})
        if grounding.get("grounding_rate") is not None and not grounding.get("meets_threshold", True):
            ungrounded = grounding.get("ungrounded", [])
            failures.append({
                "mode": "citation_hallucination",
                "detail": f"Grounding rate {grounding['grounding_rate']:.0%}, "
                          f"{len(ungrounded)} ungrounded: {ungrounded[:3]}",
            })

        # Theme miss
        theme = results.get("theme_coverage", {})
        if theme.get("missed"):
            failures.append({
                "mode": "theme_miss",
                "detail": f"Missed themes: {theme['missed']}",
            })

        # Self-review low score
        sr = results.get("self_review", {})
        if sr.get("available") and sr.get("overall_score", 1.0) < 0.6:
            failures.append({
                "mode": "quality_failure",
                "detail": f"Self-review score {sr['overall_score']:.2f} < 0.6 threshold",
            })

        return failures

    def _compute_overall(self, results: Dict) -> float:
        """Compute weighted overall score."""
        scores = []
        weights = []

        # Search keyword hit rate (weight 0.15)
        search = results.get("search", {})
        if search.get("keyword_hit_rate") is not None:
            scores.append(search["keyword_hit_rate"])
            weights.append(0.15)

        # Theme coverage (weight 0.20)
        theme = results.get("theme_coverage", {})
        if theme.get("coverage_rate") is not None:
            scores.append(theme["coverage_rate"])
            weights.append(0.20)

        # Domain purity (weight 0.15)
        domain = results.get("domain_purity", {})
        if domain.get("purity_rate") is not None:
            scores.append(domain["purity_rate"])
            weights.append(0.15)

        # Citation grounding (weight 0.20)
        grounding = results.get("citation_grounding", {})
        if grounding.get("grounding_rate") is not None:
            scores.append(grounding["grounding_rate"])
            weights.append(0.20)

        # Content keyword coverage (weight 0.15)
        content = results.get("content_quality", {})
        if content.get("keyword_coverage") is not None:
            scores.append(content["keyword_coverage"])
            weights.append(0.15)

        # Self-review (weight 0.15)
        sr = results.get("self_review", {})
        if sr.get("available") and sr.get("overall_score") is not None:
            scores.append(sr["overall_score"])
            weights.append(0.15)

        if not scores:
            return 0.0

        total_weight = sum(weights)
        weighted = sum(s * w for s, w in zip(scores, weights))
        return round(weighted / total_weight, 4)


def format_report(results: Dict[str, Any]) -> str:
    """Format evaluation results as a markdown report."""
    lines = []
    lines.append(f"# Evaluation Report: {results.get('query_id', 'unknown')}")
    lines.append(f"\n**Research Question**: {results.get('research_question', 'N/A')}")
    lines.append(f"\n**Overall Score**: {results.get('overall_score', 0):.2%}")

    # Search
    s = results.get("search", {})
    lines.append("\n## Search Quality")
    lines.append(f"- Papers found: {s.get('paper_count', 0)}")
    lines.append(f"- Keyword hit rate: {s.get('keyword_hit_rate', 0):.0%}")
    lines.append(f"- Anti-keyword hit rate: {s.get('anti_keyword_hit_rate', 0):.0%}")

    # Themes
    t = results.get("theme_coverage", {})
    lines.append("\n## Theme Coverage")
    lines.append(f"- Coverage rate: {t.get('coverage_rate', 0):.0%}")
    lines.append(f"- Matched: {t.get('matched', [])}")
    lines.append(f"- Missed: {t.get('missed', [])}")

    # Domain
    d = results.get("domain_purity", {})
    lines.append("\n## Domain Purity")
    lines.append(f"- Purity rate: {d.get('purity_rate', 0):.0%}")
    lines.append(f"- Contaminated papers: {d.get('contaminated_count', 0)}")

    # Grounding
    g = results.get("citation_grounding", {})
    lines.append("\n## Citation Grounding")
    if g.get("grounding_rate") is not None:
        lines.append(f"- Grounding rate: {g['grounding_rate']:.0%}")
        lines.append(f"- Grounded: {g.get('grounded_count', 0)}/{g.get('total_citations', 0)}")
        if g.get("ungrounded"):
            lines.append(f"- Ungrounded: {g['ungrounded']}")
    else:
        lines.append("- Not available")

    # Content
    c = results.get("content_quality", {})
    lines.append("\n## Content Quality")
    lines.append(f"- Word count: {c.get('word_count', 0)}")
    lines.append(f"- Keyword coverage: {c.get('keyword_coverage', 0):.0%}")
    if c.get("missing_keywords"):
        lines.append(f"- Missing keywords: {c['missing_keywords']}")

    # Self-review
    sr = results.get("self_review", {})
    lines.append("\n## Self-Review Scores")
    if sr.get("available"):
        lines.append(f"- Overall: {sr.get('overall_score', 0):.2f}")
        lines.append(f"- Relevance: {sr.get('relevance_score', 0):.2f}")
        lines.append(f"- Coverage: {sr.get('coverage_score', 0):.2f}")
        lines.append(f"- Coherence: {sr.get('coherence_score', 0):.2f}")
    else:
        lines.append("- Not available")

    # Efficiency
    e = results.get("efficiency", {})
    lines.append("\n## Efficiency")
    if e.get("available"):
        lines.append(f"- Total tokens: {e.get('total_tokens', 0):,}")
        lines.append(f"- LLM calls: {e.get('total_calls', 0)}")
        lines.append(f"- Estimated cost: ${e.get('estimated_cost_usd', 0):.4f}")
        lines.append(f"- Elapsed time: {e.get('elapsed_seconds', 0):.1f}s")
    else:
        lines.append("- Not available")

    # Failures
    failures = results.get("failure_modes", [])
    lines.append("\n## Failure Modes")
    if failures:
        for f in failures:
            lines.append(f"- **{f['mode']}**: {f['detail']}")
    else:
        lines.append("- No failures detected")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LitScribe review output")
    parser.add_argument("--output", "-o", type=str, help="Path to review output JSON")
    parser.add_argument("--ground-truth", "-g", type=str, help="Path to ground truth JSON")
    parser.add_argument("--save", "-s", type=str, help="Save report to file")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of markdown")
    args = parser.parse_args()

    if not args.output or not args.ground_truth:
        parser.error("Both --output and --ground-truth are required")

    # Load files
    with open(args.output, "r", encoding="utf-8") as f:
        output = json.load(f)
    with open(args.ground_truth, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Evaluate
    evaluator = ReviewEvaluator()
    results = evaluator.evaluate(output, ground_truth)

    # Output
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        report = format_report(results)
        print(report)

    # Save if requested
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            if args.json:
                json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                f.write(format_report(results))
        print(f"\nReport saved to: {save_path}")


if __name__ == "__main__":
    main()
