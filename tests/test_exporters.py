#!/usr/bin/env python
"""Test script for LitScribe exporters.

Run with: python tests/test_exporters.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_bibtex_imports():
    """Test that BibTeX exporter imports work."""
    print("=" * 60)
    print("Test 1: BibTeX exporter imports")
    print("=" * 60)

    try:
        from exporters.bibtex_exporter import (
            BibTeXEntry,
            BibTeXExporter,
            escape_bibtex,
            generate_bibtex_entry,
            generate_cite_key,
        )

        print("All BibTeX imports successful")
        print("PASS: BibTeX imports")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_citation_formatter_imports():
    """Test that citation formatter imports work."""
    print("\n" + "=" * 60)
    print("Test 2: Citation formatter imports")
    print("=" * 60)

    try:
        from exporters.citation_formatter import (
            CitationFormatter,
            CitationStyle,
            format_citation,
        )

        print("All citation formatter imports successful")
        print(f"Available styles: {[s.value for s in CitationStyle]}")
        print("PASS: Citation formatter imports")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bibtex_generation():
    """Test BibTeX generation with sample paper."""
    print("\n" + "=" * 60)
    print("Test 3: BibTeX generation")
    print("=" * 60)

    try:
        from exporters.bibtex_exporter import BibTeXExporter, generate_cite_key

        # Sample paper
        paper = {
            "paper_id": "2301.00234",
            "title": "Deep Learning for Natural Language Processing: A Survey",
            "authors": ["John Smith", "Jane Doe", "Bob Johnson"],
            "year": 2023,
            "abstract": "This paper surveys deep learning methods for NLP.",
            "venue": "Proceedings of ACL 2023",
            "citations": 150,
            "source": "arxiv",
            "key_findings": ["Deep learning improves NLP accuracy"],
            "methodology": "Survey methodology",
            "strengths": ["Comprehensive"],
            "limitations": ["Limited scope"],
            "relevance_score": 0.9,
            "pdf_available": True,
        }

        # Test cite key generation
        cite_key = generate_cite_key(paper)
        print(f"Generated cite key: {cite_key}")
        assert "smith" in cite_key.lower(), "Cite key should contain author name"
        assert "2023" in cite_key, "Cite key should contain year"

        # Test BibTeX export
        exporter = BibTeXExporter([paper])
        bibtex = exporter.generate()
        print("\nGenerated BibTeX:")
        print(bibtex[:500] + "..." if len(bibtex) > 500 else bibtex)

        # Verify content
        assert "@" in bibtex, "Should contain @ entry marker"
        assert "title" in bibtex.lower(), "Should contain title field"
        assert "author" in bibtex.lower(), "Should contain author field"
        assert "2023" in bibtex, "Should contain year"

        print("\nPASS: BibTeX generation")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_citation_styles():
    """Test all citation styles."""
    print("\n" + "=" * 60)
    print("Test 4: Citation styles")
    print("=" * 60)

    try:
        from exporters.citation_formatter import CitationFormatter, CitationStyle

        paper = {
            "paper_id": "10.1000/example",
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            "year": 2017,
            "abstract": "Transformer architecture paper.",
            "venue": "Advances in Neural Information Processing Systems",
            "citations": 50000,
            "source": "semantic_scholar",
            "key_findings": ["Transformer outperforms RNNs"],
            "methodology": "Experimental",
            "strengths": ["Novel architecture"],
            "limitations": ["Computational cost"],
            "relevance_score": 1.0,
            "pdf_available": True,
        }

        for style in CitationStyle:
            formatter = CitationFormatter(style)
            citation = formatter.format_paper(paper)
            print(f"\n{style.value.upper()}:")
            print(f"  {citation}")

            # Basic validation
            assert len(citation) > 20, f"{style.value} citation too short"
            assert "2017" in citation or "n.d." in citation, f"{style.value} missing year"

        print("\nPASS: Citation styles")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bibtex_escape():
    """Test BibTeX character escaping."""
    print("\n" + "=" * 60)
    print("Test 5: BibTeX escaping")
    print("=" * 60)

    try:
        from exporters.bibtex_exporter import escape_bibtex

        test_cases = [
            ("Normal text", "Normal text"),
            ("Text with & ampersand", r"Text with \& ampersand"),
            ("100% complete", r"100\% complete"),
            ("Price: $50", r"Price: \$50"),
            ("Item #1", r"Item \#1"),
            ("under_score", r"under\_score"),
        ]

        for input_text, expected in test_cases:
            result = escape_bibtex(input_text)
            print(f"  '{input_text}' -> '{result}'")
            assert result == expected, f"Expected '{expected}', got '{result}'"

        print("\nPASS: BibTeX escaping")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_author_formatting():
    """Test author name formatting for different styles."""
    print("\n" + "=" * 60)
    print("Test 6: Author formatting")
    print("=" * 60)

    try:
        from exporters.citation_formatter import (
            format_authors_apa,
            format_authors_gbt7714,
            format_authors_ieee,
            format_authors_mla,
        )

        authors_1 = ["John Smith"]
        authors_2 = ["John Smith", "Jane Doe"]
        authors_3 = ["John Smith", "Jane Doe", "Bob Johnson"]
        authors_many = [f"Author{i}" for i in range(25)]

        print("\nSingle author:")
        print(f"  APA: {format_authors_apa(authors_1)}")
        print(f"  MLA: {format_authors_mla(authors_1)}")
        print(f"  IEEE: {format_authors_ieee(authors_1)}")
        print(f"  GB/T: {format_authors_gbt7714(authors_1)}")

        print("\nTwo authors:")
        print(f"  APA: {format_authors_apa(authors_2)}")
        print(f"  MLA: {format_authors_mla(authors_2)}")
        print(f"  IEEE: {format_authors_ieee(authors_2)}")

        print("\nThree authors:")
        print(f"  APA: {format_authors_apa(authors_3)}")
        print(f"  MLA: {format_authors_mla(authors_3)}")
        print(f"  IEEE: {format_authors_ieee(authors_3)}")

        print("\nMany authors (25):")
        apa_many = format_authors_apa(authors_many)
        print(f"  APA (truncated): {apa_many[:80]}...")
        assert "..." in apa_many, "APA should truncate many authors"

        mla_many = format_authors_mla(authors_many)
        print(f"  MLA: {mla_many}")
        assert "et al" in mla_many, "MLA should use et al. for 3+ authors"

        gbt_many = format_authors_gbt7714(authors_many)
        print(f"  GB/T: {gbt_many}")
        assert "等" in gbt_many, "GB/T should use 等 for 4+ authors"

        print("\nPASS: Author formatting")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_entry_type_detection():
    """Test BibTeX entry type detection."""
    print("\n" + "=" * 60)
    print("Test 7: Entry type detection")
    print("=" * 60)

    try:
        from exporters.bibtex_exporter import detect_entry_type

        test_cases = [
            ({"venue": "Proceedings of ICML 2023", "source": "arxiv"}, "inproceedings"),
            ({"venue": "Nature", "source": "semantic_scholar"}, "article"),
            ({"venue": "Journal of Machine Learning Research", "source": ""}, "article"),
            ({"venue": "NeurIPS Workshop", "source": ""}, "inproceedings"),
            ({"venue": "", "source": "arxiv"}, "misc"),
            ({"venue": "", "source": "pubmed"}, "article"),
        ]

        for paper, expected in test_cases:
            result = detect_entry_type(paper)
            status = "OK" if result == expected else "FAIL"
            print(f"  {paper.get('venue', 'N/A')[:30]:30} [{paper.get('source', 'N/A'):15}] -> {result:15} {status}")
            assert result == expected, f"Expected '{expected}', got '{result}'"

        print("\nPASS: Entry type detection")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pandoc_imports():
    """Test Pandoc exporter imports."""
    print("\n" + "=" * 60)
    print("Test 8: Pandoc exporter imports")
    print("=" * 60)

    try:
        from exporters.pandoc_exporter import (
            ExportConfig,
            ExportFormat,
            PandocExporter,
            export_review,
            generate_review_markdown,
        )

        print("All Pandoc imports successful")
        print(f"Available formats: {[f.value for f in ExportFormat]}")
        print("PASS: Pandoc imports")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_markdown_generation():
    """Test Markdown generation from state."""
    print("\n" + "=" * 60)
    print("Test 9: Markdown generation")
    print("=" * 60)

    try:
        from exporters.pandoc_exporter import generate_review_markdown
        from exporters.citation_formatter import CitationStyle

        # Create sample state
        state = {
            "research_question": "Deep learning for NLP",
            "analyzed_papers": [
                {
                    "paper_id": "2301.00234",
                    "title": "Attention Is All You Need",
                    "authors": ["Ashish Vaswani", "Noam Shazeer"],
                    "year": 2017,
                    "abstract": "Transformer architecture paper.",
                    "venue": "NeurIPS",
                    "citations": 50000,
                    "source": "arxiv",
                    "key_findings": ["Transformer outperforms RNNs"],
                    "methodology": "Experimental",
                    "strengths": ["Novel architecture"],
                    "limitations": ["Computational cost"],
                    "relevance_score": 1.0,
                    "pdf_available": True,
                },
            ],
            "synthesis": {
                "themes": [
                    {
                        "theme": "Transformer Architecture",
                        "description": "Self-attention mechanisms",
                        "paper_ids": ["2301.00234"],
                        "key_points": ["Attention is all you need"],
                    }
                ],
                "gaps": ["Limited interpretability", "High computational cost"],
                "future_directions": ["More efficient attention"],
                "review_text": "# Introduction\n\nThis review covers deep learning for NLP.\n\n# Main Themes\n\n## Transformer Architecture\n\nThe transformer architecture revolutionized NLP...",
                "word_count": 500,
                "papers_cited": 1,
            },
        }

        # Test English markdown
        md_en = generate_review_markdown(state, CitationStyle.APA, "en")
        print("English Markdown generated")
        print(f"  Length: {len(md_en)} characters")
        assert "---" in md_en, "Should have YAML front matter"
        assert "title:" in md_en, "Should have title"
        assert "# References" in md_en, "Should have references section"
        assert "Vaswani" in md_en, "Should include author"

        # Test Chinese markdown
        md_zh = generate_review_markdown(state, CitationStyle.GB_T_7714, "zh")
        print("Chinese Markdown generated")
        print(f"  Length: {len(md_zh)} characters")
        assert "lang: zh-CN" in md_zh, "Should have Chinese language"
        assert "# 参考文献" in md_zh, "Should have Chinese references header"

        print("\nPASS: Markdown generation")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pandoc_availability():
    """Test Pandoc availability check."""
    print("\n" + "=" * 60)
    print("Test 10: Pandoc availability")
    print("=" * 60)

    try:
        import shutil
        from exporters.pandoc_exporter import PandocExporter

        state = {"research_question": "test", "analyzed_papers": []}
        exporter = PandocExporter(state)

        pandoc_path = shutil.which("pandoc")
        if pandoc_path:
            print(f"Pandoc found at: {pandoc_path}")
            assert exporter._pandoc_available, "Should detect Pandoc"
        else:
            print("Pandoc not installed (optional)")
            assert not exporter._pandoc_available, "Should not detect Pandoc"

        print("PASS: Pandoc availability check")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("LitScribe Exporter Tests")
    print("=" * 60)

    results = []

    results.append(("BibTeX imports", test_bibtex_imports()))
    results.append(("Citation formatter imports", test_citation_formatter_imports()))
    results.append(("BibTeX generation", test_bibtex_generation()))
    results.append(("Citation styles", test_citation_styles()))
    results.append(("BibTeX escaping", test_bibtex_escape()))
    results.append(("Author formatting", test_author_formatting()))
    results.append(("Entry type detection", test_entry_type_detection()))
    results.append(("Pandoc imports", test_pandoc_imports()))
    results.append(("Markdown generation", test_markdown_generation()))
    results.append(("Pandoc availability", test_pandoc_availability()))

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
    sys.exit(main())
