#!/usr/bin/env python
"""Tests for abstract-only analysis enhancement and select_papers prompt optimization.

Covers:
- ABSTRACT_ONLY_ANALYSIS_PROMPT exists with expected fields
- analyze_paper_combined uses abstract-only prompt when parsed_doc is None
- Metadata enrichment (MeSH, fields, keywords)
- select_papers prompt has exclusion criteria and sub_topics support
- select_papers accepts sub_topics parameter

Run with: pytest tests/test_abstract_analysis.py -v
"""

import inspect
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# === Test 1: Abstract-only prompt exists ===

def test_abstract_only_prompt_exists():
    """ABSTRACT_ONLY_ANALYSIS_PROMPT should be defined in prompts.py."""
    from agents.prompts import ABSTRACT_ONLY_ANALYSIS_PROMPT
    assert "ABSTRACT ONLY" in ABSTRACT_ONLY_ANALYSIS_PROMPT or "abstract" in ABSTRACT_ONLY_ANALYSIS_PROMPT.lower()
    assert "{research_question}" in ABSTRACT_ONLY_ANALYSIS_PROMPT
    assert "{abstract}" in ABSTRACT_ONLY_ANALYSIS_PROMPT
    assert "{metadata_section}" in ABSTRACT_ONLY_ANALYSIS_PROMPT
    assert "{venue}" in ABSTRACT_ONLY_ANALYSIS_PROMPT


# === Test 2: Abstract-only prompt warns about limitations ===

def test_abstract_only_prompt_has_limitations_warning():
    """Prompt should warn LLM not to fabricate findings."""
    from agents.prompts import ABSTRACT_ONLY_ANALYSIS_PROMPT
    lower = ABSTRACT_ONLY_ANALYSIS_PROMPT.lower()
    assert "fabricate" in lower or "invent" in lower or "not present" in lower


# === Test 3: analyze_paper_combined imports abstract-only prompt ===

def test_critical_reading_imports_abstract_prompt():
    """critical_reading_agent should import ABSTRACT_ONLY_ANALYSIS_PROMPT."""
    source = Path(__file__).parent.parent / "src" / "agents" / "critical_reading_agent.py"
    code = source.read_text()
    assert "ABSTRACT_ONLY_ANALYSIS_PROMPT" in code


# === Test 4: analyze_paper_combined branches on parsed_doc ===

def test_analyze_paper_combined_branches():
    """analyze_paper_combined should use different prompts for full-text vs abstract-only."""
    source = Path(__file__).parent.parent / "src" / "agents" / "critical_reading_agent.py"
    code = source.read_text()
    # Should have both prompt references in the function
    assert "COMBINED_PAPER_ANALYSIS_PROMPT" in code
    assert "ABSTRACT_ONLY_ANALYSIS_PROMPT" in code
    # Should check for parsed_doc
    assert "if full_text:" in code or "if parsed_doc:" in code


# === Test 5: Metadata enrichment includes MeSH terms ===

def test_metadata_enrichment_includes_mesh():
    """Abstract-only path should include MeSH terms from paper metadata."""
    source = Path(__file__).parent.parent / "src" / "agents" / "critical_reading_agent.py"
    code = source.read_text()
    assert "mesh_terms" in code, "Should include MeSH terms in metadata"
    assert "fields_of_study" in code or "s2_fields" in code, "Should include fields of study"
    assert "keywords" in code, "Should include keywords"


# === Test 6: select_papers prompt has exclusion criteria ===

def test_select_papers_prompt_has_exclusions():
    """PAPER_SELECTION_PROMPT should have explicit exclusion criteria."""
    from agents.prompts import PAPER_SELECTION_PROMPT
    lower = PAPER_SELECTION_PROMPT.lower()
    assert "exclusion" in lower, "Prompt should mention exclusion criteria"
    assert "tangential" in lower or "passing mention" in lower, \
        "Prompt should warn about tangential papers"
    assert "clinical" in lower or "pharmacology" in lower, \
        "Prompt should mention domain-specific exclusions"


# === Test 7: select_papers prompt supports sub_topics ===

def test_select_papers_prompt_has_sub_topics():
    """PAPER_SELECTION_PROMPT should have a sub_topics_section placeholder."""
    from agents.prompts import PAPER_SELECTION_PROMPT
    assert "{sub_topics_section}" in PAPER_SELECTION_PROMPT


# === Test 8: select_papers function accepts sub_topics parameter ===

def test_select_papers_accepts_sub_topics():
    """select_papers should accept a sub_topics parameter."""
    from agents.discovery_agent import select_papers
    sig = inspect.signature(select_papers)
    params = list(sig.parameters.keys())
    assert "sub_topics" in params, "select_papers should accept sub_topics"


# === Test 9: PubMedArticle pmc_id populated from Medline ===

def test_pubmed_article_has_pmc_id():
    """PubMedArticle.from_medline_record should extract PMC ID."""
    from models.pubmed_article import PubMedArticle
    record = {
        "PMID": "12345",
        "TI": "Test Paper",
        "AU": ["Smith J"],
        "AB": "Test abstract",
        "JT": "Test Journal",
        "PMC": "PMC9876543",
    }
    article = PubMedArticle.from_medline_record(record)
    assert article.pmc_id == "PMC9876543"


# === Test 10: UnifiedPaper has pmc_id field ===

def test_unified_paper_has_pmc_id():
    """UnifiedPaper should have pmc_id field."""
    from models.unified_paper import UnifiedPaper
    paper = UnifiedPaper(
        title="Test",
        authors=["Smith"],
        abstract="Test",
        year=2024,
        pmc_id="PMC123",
    )
    assert paper.pmc_id == "PMC123"
    assert paper.to_dict()["pmc_id"] == "PMC123"


# === Entrypoint ===

async def main():
    """Run all tests via pytest."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=str(Path(__file__).parent.parent),
    )
    return result.returncode


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
