#!/usr/bin/env python
"""Tests for abstract screening and multi-round snowball with co-citation.

Covers:
- ABSTRACT_SCREENING_PROMPT exists with expected fields
- screen_papers_by_abstract function exists and is async
- screen_papers_by_abstract returns empty list for empty input
- snowball_sampling supports max_rounds parameter
- _extract_common_references function exists
- Co-citation integration in snowball_sampling

Run with: pytest tests/test_screening_snowball.py -v
"""

import inspect
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# === Test 1: Abstract screening prompt exists ===

def test_abstract_screening_prompt_exists():
    """ABSTRACT_SCREENING_PROMPT should be defined in prompts.py."""
    from agents.prompts import ABSTRACT_SCREENING_PROMPT
    assert "{research_question}" in ABSTRACT_SCREENING_PROMPT
    assert "{domain_hint}" in ABSTRACT_SCREENING_PROMPT
    assert "{papers_batch}" in ABSTRACT_SCREENING_PROMPT
    assert "relevant" in ABSTRACT_SCREENING_PROMPT.lower()
    assert "irrelevant" in ABSTRACT_SCREENING_PROMPT.lower()


# === Test 2: Abstract screening prompt has exclusion guidance ===

def test_screening_prompt_has_guidance():
    """Screening prompt should explain what makes a paper irrelevant."""
    from agents.prompts import ABSTRACT_SCREENING_PROMPT
    lower = ABSTRACT_SCREENING_PROMPT.lower()
    assert "different field" in lower or "unrelated" in lower
    assert "passing" in lower or "tangential" in lower


# === Test 3: screen_papers_by_abstract exists and is async ===

def test_screen_papers_by_abstract_exists():
    """screen_papers_by_abstract should be importable and async."""
    from agents.discovery_agent import screen_papers_by_abstract
    assert inspect.iscoroutinefunction(screen_papers_by_abstract)


# === Test 4: screen_papers_by_abstract signature ===

def test_screen_papers_by_abstract_signature():
    """screen_papers_by_abstract should accept expected parameters."""
    from agents.discovery_agent import screen_papers_by_abstract
    sig = inspect.signature(screen_papers_by_abstract)
    params = list(sig.parameters.keys())
    assert "papers" in params
    assert "research_question" in params
    assert "domain_hint" in params
    assert "batch_size" in params


# === Test 5: screen_papers_by_abstract handles empty input ===

@pytest.mark.asyncio
async def test_screen_papers_empty_input():
    """screen_papers_by_abstract should return empty list for empty input."""
    from agents.discovery_agent import screen_papers_by_abstract
    result = await screen_papers_by_abstract([], "test question")
    assert result == []


# === Test 6: discovery_agent calls screen_papers_by_abstract ===

def test_discovery_agent_calls_screening():
    """discovery_agent source should call screen_papers_by_abstract after selection."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    assert "screen_papers_by_abstract" in code
    # The screening step comment should come after the select step comment
    select_pos = code.find("# Step 3: Select best papers")
    screen_pos = code.find("# Step 5: Abstract screening")
    assert select_pos > 0 and screen_pos > 0, "Both step comments should exist"
    assert select_pos < screen_pos, "Screening should come after paper selection"


# === Test 7: snowball_sampling supports max_rounds ===

def test_snowball_has_max_rounds():
    """snowball_sampling should accept max_rounds parameter."""
    from agents.discovery_agent import snowball_sampling
    sig = inspect.signature(snowball_sampling)
    params = sig.parameters
    assert "max_rounds" in params
    assert params["max_rounds"].default == 2


# === Test 8: _extract_common_references exists ===

def test_extract_common_references_exists():
    """_extract_common_references should be importable and async."""
    from agents.discovery_agent import _extract_common_references
    assert inspect.iscoroutinefunction(_extract_common_references)


# === Test 9: _extract_common_references signature ===

def test_extract_common_references_signature():
    """_extract_common_references should accept expected parameters."""
    from agents.discovery_agent import _extract_common_references
    sig = inspect.signature(_extract_common_references)
    params = list(sig.parameters.keys())
    assert "seed_papers" in params
    assert "seen_ids" in params
    assert "min_co_citations" in params


# === Test 10: snowball source includes co-citation ===

def test_snowball_has_co_citation():
    """snowball_sampling should include co-citation analysis."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    assert "_extract_common_references" in code
    assert "co-citation" in code.lower() or "co_citation" in code.lower() or "co_cited" in code


# === Test 11: snowball multi-round uses round papers as next seeds ===

def test_snowball_multi_round_logic():
    """snowball_sampling should use round results as seeds for next round."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    # Should have round loop and current_seeds update
    assert "for round_num in range" in code
    assert "current_seeds = sorted" in code


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
