#!/usr/bin/env python
"""Tests for incremental loop-back, additional_queries auto-consumption, and self-review threshold.

Covers:
- additional_queries field exists in LitScribeState
- create_initial_state sets additional_queries to empty list
- supervisor incremental loop-back keeps high-relevance papers
- supervisor injects additional_queries from self_review
- discovery_agent consumes additional_queries from state
- self-review loop-back threshold is 0.7

Run with: pytest tests/test_loopback.py -v
"""

import inspect
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# === Test 1: additional_queries field exists in LitScribeState ===

def test_state_has_additional_queries():
    """LitScribeState should have additional_queries field."""
    from agents.state import LitScribeState
    hints = LitScribeState.__annotations__
    assert "additional_queries" in hints


# === Test 2: create_initial_state sets additional_queries to empty ===

def test_initial_state_additional_queries_empty():
    """create_initial_state should set additional_queries to []."""
    from agents.state import create_initial_state
    state = create_initial_state("test question")
    assert state["additional_queries"] == []


# === Test 3: supervisor keeps high-relevance papers on loop-back ===

def test_supervisor_incremental_loopback():
    """Supervisor should keep analyzed_papers with relevance_score >= 0.5 on loop-back."""
    source = Path(__file__).parent.parent / "src" / "agents" / "supervisor.py"
    code = source.read_text()
    # Should filter by relevance_score
    assert "relevance_score" in code
    assert ">= 0.5" in code or ">=0.5" in code


# === Test 4: supervisor injects additional_queries ===

def test_supervisor_injects_additional_queries():
    """Supervisor should inject additional_queries from self_review into state."""
    source = Path(__file__).parent.parent / "src" / "agents" / "supervisor.py"
    code = source.read_text()
    assert "additional_queries" in code
    # Should read from self_review assessment
    assert 'self_review.get("additional_queries"' in code


# === Test 5: supervisor preserves parsed_documents for kept papers ===

def test_supervisor_preserves_parsed_documents():
    """Supervisor should preserve parsed_documents for kept papers on loop-back."""
    source = Path(__file__).parent.parent / "src" / "agents" / "supervisor.py"
    code = source.read_text()
    # Should filter parsed_documents by keep_ids
    assert "parsed_documents" in code
    assert "keep_ids" in code


# === Test 6: discovery_agent reads additional_queries from state ===

def test_discovery_reads_additional_queries():
    """discovery_agent should read and use additional_queries from state."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    assert 'state.get("additional_queries"' in code
    assert "additional queries from self-review" in code.lower()


# === Test 7: discovery_agent appends additional_queries to expanded_queries ===

def test_discovery_appends_additional_queries():
    """discovery_agent should append additional_queries to expanded_queries (deduped)."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    # Should deduplicate
    assert "q not in expanded_queries" in code


# === Test 8: self-review threshold is 0.7 ===

def test_self_review_threshold():
    """Self-review loop-back threshold should be 0.7."""
    source = Path(__file__).parent.parent / "src" / "agents" / "self_review_agent.py"
    code = source.read_text()
    assert "< 0.7" in code
    # Should NOT have old threshold
    assert "< 0.6" not in code.split("# Step 4c")[1].split("iteration_count")[0]


# === Test 9: supervisor does NOT clear analyzed_papers to empty on loop-back ===

def test_supervisor_no_full_clear():
    """Supervisor should NOT set analyzed_papers=[] on loop-back (incremental strategy)."""
    source = Path(__file__).parent.parent / "src" / "agents" / "supervisor.py"
    code = source.read_text()
    # The loop-back block should use 'keep' not empty list
    loopback_section = code.split("Incremental loop-back")[1].split("return updates")[0]
    assert '"analyzed_papers"] = []' not in loopback_section
    assert '"analyzed_papers"] = keep' in loopback_section


# === Test 10: supervisor logs incremental strategy ===

def test_supervisor_logs_incremental():
    """Supervisor should log the incremental loop-back strategy."""
    source = Path(__file__).parent.parent / "src" / "agents" / "supervisor.py"
    code = source.read_text()
    assert "Incremental loop-back" in code
    assert "high-relevance papers" in code


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
