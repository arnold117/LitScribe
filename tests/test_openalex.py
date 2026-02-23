#!/usr/bin/env python
"""Tests for OpenAlex integration.

Covers:
- OpenAlex service module importable with correct signature
- _reconstruct_abstract() inverted index → plaintext
- _format_paper() field mapping
- _openalex_to_unified() converter in unified_search
- OpenAlex in default source lists across the codebase
- OpenAlex branch in search_all()

Run with: pytest tests/test_openalex.py -v
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# === Test 1: OpenAlex service importable ===

def test_openalex_importable():
    """OpenAlex service module should be importable."""
    from services.openalex import search_papers
    assert callable(search_papers)


def test_openalex_search_signature():
    """search_papers should accept expected keyword arguments."""
    import inspect
    from services.openalex import search_papers
    sig = inspect.signature(search_papers)
    params = set(sig.parameters.keys())
    assert "query" in params
    assert "max_results" in params
    assert "year_from" in params
    assert "year_to" in params
    assert "min_citations" in params


# === Test 2: Abstract reconstruction ===

def test_reconstruct_abstract_basic():
    """Inverted index should be reconstructed to plaintext."""
    from services.openalex import _reconstruct_abstract
    inverted = {"Hello": [0], "world": [1], "of": [2], "science": [3]}
    assert _reconstruct_abstract(inverted) == "Hello world of science"


def test_reconstruct_abstract_repeated_words():
    """Words appearing at multiple positions should be placed correctly."""
    from services.openalex import _reconstruct_abstract
    inverted = {"the": [0, 2], "cat": [1], "dog": [3]}
    assert _reconstruct_abstract(inverted) == "the cat the dog"


def test_reconstruct_abstract_empty():
    """Empty inverted index should return empty string."""
    from services.openalex import _reconstruct_abstract
    assert _reconstruct_abstract({}) == ""
    assert _reconstruct_abstract(None) == ""


# === Test 3: _format_paper field mapping ===

def test_format_paper_extracts_fields():
    """_format_paper should map OpenAlex work object to standard format."""
    from services.openalex import _format_paper
    work = {
        "id": "https://openalex.org/W12345",
        "display_name": "Test Paper Title",
        "publication_year": 2024,
        "cited_by_count": 42,
        "doi": "https://doi.org/10.1234/test",
        "abstract_inverted_index": {"Abstract": [0], "text": [1]},
        "authorships": [
            {"author": {"display_name": "Alice Smith"}},
            {"author": {"display_name": "Bob Jones"}},
        ],
        "primary_location": {
            "source": {"display_name": "Nature"}
        },
        "locations": [{"pdf_url": "https://example.com/paper.pdf"}],
        "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/12345"},
        "keywords": [{"display_name": "Machine Learning"}],
    }
    result = _format_paper(work)
    assert result["paper_id"] == "W12345"
    assert result["title"] == "Test Paper Title"
    assert result["authors"] == ["Alice Smith", "Bob Jones"]
    assert result["year"] == 2024
    assert result["citation_count"] == 42
    assert result["doi"] == "10.1234/test"
    assert result["abstract"] == "Abstract text"
    assert result["venue"] == "Nature"
    assert result["pdf_url"] == "https://example.com/paper.pdf"
    assert result["pmid"] == "12345"
    assert "Machine Learning" in result["fields_of_study"]


def test_format_paper_handles_missing_fields():
    """_format_paper should handle missing/None fields gracefully."""
    from services.openalex import _format_paper
    work = {"id": "https://openalex.org/W99999"}
    result = _format_paper(work)
    assert result["paper_id"] == "W99999"
    assert result["title"] == ""
    assert result["authors"] == []
    assert result["year"] == 0
    assert result["abstract"] == ""


# === Test 4: _openalex_to_unified converter ===

def test_openalex_to_unified():
    """_openalex_to_unified should produce a valid UnifiedPaper."""
    from aggregators.unified_search import _openalex_to_unified
    paper_dict = {
        "paper_id": "W12345",
        "title": "Test Paper",
        "authors": ["Author A"],
        "abstract": "Some abstract",
        "year": 2024,
        "venue": "Nature",
        "citation_count": 10,
        "pdf_url": "https://example.com/paper.pdf",
        "doi": "10.1234/test",
        "pmid": "12345",
        "fields_of_study": ["Biology"],
    }
    unified = _openalex_to_unified(paper_dict)
    assert unified.title == "Test Paper"
    assert unified.year == 2024
    assert unified.citations == 10
    assert "openalex" in unified.sources
    assert unified.doi == "10.1234/test"
    assert len(unified.pdf_urls) == 1


# === Test 5: OpenAlex in default source lists ===

def test_openalex_in_unified_search_defaults():
    """unified_search search_all() should include openalex in default sources."""
    source = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
    code = source.read_text()
    # Check the default sources line in search_all
    assert '"openalex"' in code


def test_openalex_in_discovery_agent():
    """discovery_agent should include openalex in default sources."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    assert '"openalex"' in code


def test_openalex_in_tools():
    """tools.py should include openalex in default sources."""
    source = Path(__file__).parent.parent / "src" / "agents" / "tools.py"
    code = source.read_text()
    assert '"openalex"' in code


def test_openalex_in_user_config():
    """user_config DEFAULT_USER_CONFIG should include openalex."""
    from utils.user_config import DEFAULT_USER_CONFIG
    assert "openalex" in DEFAULT_USER_CONFIG["sources"]


def test_openalex_in_state_defaults():
    """create_initial_state should include openalex in default sources."""
    from agents.state import create_initial_state
    state = create_initial_state("test question")
    assert "openalex" in state["sources"]


def test_openalex_in_cli():
    """CLI should include openalex in default sources."""
    source = Path(__file__).parent.parent / "src" / "cli" / "litscribe_cli.py"
    code = source.read_text()
    assert '"openalex"' in code


# === Test 6: OpenAlex branch exists in search_all ===

def test_openalex_branch_in_search_all():
    """search_all() should have an 'if openalex in sources' branch."""
    source = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
    code = source.read_text()
    assert '"openalex" in sources' in code
    assert "search_openalex" in code


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
