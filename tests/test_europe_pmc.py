"""Tests for Europe PMC data source integration."""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestEuropePMCImport:
    """Verify Europe PMC service is importable and has correct interface."""

    def test_module_importable(self):
        from services import europe_pmc
        assert europe_pmc is not None

    def test_search_papers_exists(self):
        from services.europe_pmc import search_papers
        import inspect
        assert inspect.iscoroutinefunction(search_papers)

    def test_search_papers_signature(self):
        from services.europe_pmc import search_papers
        import inspect
        sig = inspect.signature(search_papers)
        param_names = list(sig.parameters.keys())
        assert "query" in param_names
        assert "max_results" in param_names
        assert "year_from" in param_names
        assert "year_to" in param_names
        assert "min_citations" in param_names

    def test_format_paper_exists(self):
        from services.europe_pmc import _format_paper
        assert callable(_format_paper)


class TestFormatPaper:
    """Test Europe PMC paper formatting."""

    def test_basic_fields(self):
        from services.europe_pmc import _format_paper

        hit = {
            "pmid": "12345678",
            "pmcid": "PMC9876543",
            "doi": "10.1234/test",
            "title": "Test Paper Title",
            "authorString": "Smith J, Jones A, Brown B",
            "abstractText": "This is the abstract.",
            "citedByCount": 42,
            "pubYear": "2023",
            "journalTitle": "Nature",
            "isOpenAccess": "Y",
        }
        result = _format_paper(hit)

        assert result["paper_id"] == "12345678"
        assert result["title"] == "Test Paper Title"
        assert result["year"] == 2023
        assert result["citation_count"] == 42
        assert result["venue"] == "Nature"
        assert result["doi"] == "10.1234/test"
        assert result["is_open_access"] is True
        assert "PMC9876543" in result["pdf_url"]

    def test_missing_pmid_uses_pmcid(self):
        from services.europe_pmc import _format_paper

        hit = {"pmcid": "PMC123", "title": "Test"}
        result = _format_paper(hit)
        assert result["paper_id"] == "PMC123"

    def test_no_open_access_no_pdf(self):
        from services.europe_pmc import _format_paper

        hit = {"pmid": "123", "pmcid": "PMC456", "isOpenAccess": "N"}
        result = _format_paper(hit)
        assert result["pdf_url"] is None

    def test_empty_fields(self):
        from services.europe_pmc import _format_paper

        hit = {}
        result = _format_paper(hit)
        assert result["title"] == ""
        assert result["authors"] == []
        assert result["year"] == 0
        assert result["citation_count"] == 0


class TestEuropePMCInDefaultSources:
    """Verify europe_pmc is in all default source lists."""

    def test_state_default_sources(self):
        from agents.state import create_initial_state
        state = create_initial_state("test question")
        assert "europe_pmc" in state["sources"]

    def test_user_config_default(self):
        from utils.user_config import DEFAULT_USER_CONFIG
        assert "europe_pmc" in DEFAULT_USER_CONFIG["sources"]

    def test_tools_default(self):
        """tools.py multi_source_search default sources include europe_pmc."""
        src_path = Path(__file__).parent.parent / "src" / "agents" / "tools.py"
        content = src_path.read_text()
        assert "europe_pmc" in content

    def test_discovery_agent_has_europe_pmc(self):
        """Discovery agent source references include europe_pmc."""
        src_path = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
        content = src_path.read_text()
        assert "europe_pmc" in content


class TestEuropePMCInUnifiedSearch:
    """Verify Europe PMC integration in unified search."""

    def test_unified_search_has_europe_pmc_converter(self):
        src_path = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
        content = src_path.read_text()
        assert "_europe_pmc_to_unified" in content

    def test_unified_search_has_europe_pmc_branch(self):
        src_path = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
        content = src_path.read_text()
        assert "search_europe_pmc" in content
        assert '"europe_pmc"' in content
