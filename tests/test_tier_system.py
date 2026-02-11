#!/usr/bin/env python
"""Tests for review tier system, per-sub-topic search, target_words, and related fixes.

Covers:
- determine_review_tier() returns correct tiers
- calculate_target_words() formula
- TIER_CONFIG has all three tiers with required keys
- create_initial_state sets review_tier and target_words
- Default max_papers is 40 (not 10)
- discovery_agent imports TIER_CONFIG
- Per-sub-topic search logic exists in discovery_agent
- Synthesis uses target_words from state (not hardcoded)
- Supervisor loop-back threshold is 0.7 (matches self_review)
- Snowball min_matches relaxed to 1-2
- review_title fallback in planning_agent
- CLI argparse default is 40

Run with: pytest tests/test_tier_system.py -v
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# === Test 1: determine_review_tier ===

def test_determine_review_tier_quick():
    """<=25 papers should be 'quick' tier."""
    from agents.state import determine_review_tier
    assert determine_review_tier(10) == "quick"
    assert determine_review_tier(25) == "quick"


def test_determine_review_tier_standard():
    """26-60 papers should be 'standard' tier."""
    from agents.state import determine_review_tier
    assert determine_review_tier(26) == "standard"
    assert determine_review_tier(40) == "standard"
    assert determine_review_tier(60) == "standard"


def test_determine_review_tier_comprehensive():
    """>60 papers should be 'comprehensive' tier."""
    from agents.state import determine_review_tier
    assert determine_review_tier(61) == "comprehensive"
    assert determine_review_tier(100) == "comprehensive"
    assert determine_review_tier(200) == "comprehensive"


# === Test 2: calculate_target_words ===

def test_target_words_formula():
    """Target words should follow 1000 + papers * 130."""
    from agents.state import calculate_target_words
    assert calculate_target_words(20) == 1000 + 20 * 130  # 3600
    assert calculate_target_words(40) == 1000 + 40 * 130  # 6200
    assert calculate_target_words(100) == 1000 + 100 * 130  # 14000


def test_target_words_cjk_multiplier():
    """CJK languages should get 1.5x multiplier."""
    from agents.state import calculate_target_words
    en_words = calculate_target_words(40, "en")
    zh_words = calculate_target_words(40, "zh")
    assert zh_words == int(en_words * 1.5)


# === Test 3: TIER_CONFIG ===

def test_tier_config_has_all_tiers():
    """TIER_CONFIG should have quick, standard, comprehensive."""
    from agents.state import TIER_CONFIG
    assert "quick" in TIER_CONFIG
    assert "standard" in TIER_CONFIG
    assert "comprehensive" in TIER_CONFIG


def test_tier_config_quick_flat():
    """Quick tier should NOT use per_subtopic_search."""
    from agents.state import TIER_CONFIG
    assert TIER_CONFIG["quick"]["per_subtopic_search"] is False


def test_tier_config_standard_per_subtopic():
    """Standard tier should use per_subtopic_search."""
    from agents.state import TIER_CONFIG
    assert TIER_CONFIG["standard"]["per_subtopic_search"] is True


def test_tier_config_comprehensive_per_subtopic():
    """Comprehensive tier should use per_subtopic_search."""
    from agents.state import TIER_CONFIG
    assert TIER_CONFIG["comprehensive"]["per_subtopic_search"] is True
    assert TIER_CONFIG["comprehensive"].get("co_citation") is True


# === Test 4: create_initial_state ===

def test_initial_state_has_tier_fields():
    """create_initial_state should set review_tier and target_words."""
    from agents.state import create_initial_state
    state = create_initial_state("test question")
    assert "review_tier" in state
    assert "target_words" in state


def test_initial_state_default_40_papers():
    """Default max_papers should be 40 (standard tier)."""
    from agents.state import create_initial_state
    state = create_initial_state("test question")
    assert state["max_papers"] == 40
    assert state["review_tier"] == "standard"


def test_initial_state_quick_tier():
    """max_papers=20 should set quick tier."""
    from agents.state import create_initial_state
    state = create_initial_state("test question", max_papers=20)
    assert state["review_tier"] == "quick"


def test_initial_state_comprehensive_tier():
    """max_papers=100 should set comprehensive tier."""
    from agents.state import create_initial_state
    state = create_initial_state("test question", max_papers=100)
    assert state["review_tier"] == "comprehensive"


# === Test 5: discovery_agent imports TIER_CONFIG ===

def test_discovery_imports_tier_config():
    """discovery_agent should import TIER_CONFIG from state."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    assert "TIER_CONFIG" in code
    assert "review_tier" in code


# === Test 6: per-sub-topic search logic exists ===

def test_per_subtopic_search_in_discovery():
    """discovery_agent should have per-sub-topic search logic."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    assert "per_subtopic_search" in code
    assert "Searching sub-topic" in code


# === Test 7: synthesis uses target_words from state ===

def test_synthesis_reads_target_words():
    """synthesis_agent should read target_words from state."""
    source = Path(__file__).parent.parent / "src" / "agents" / "synthesis_agent.py"
    code = source.read_text()
    assert 'state.get("target_words"' in code
    # Should NOT have hardcoded target_words=2000 or 2500 in the agent function
    agent_section = code.split("async def synthesis_agent")[1]
    assert "target_words=2000" not in agent_section
    assert "target_words=2500" not in agent_section


# === Test 8: supervisor threshold matches self_review ===

def test_supervisor_loopback_threshold():
    """Supervisor should use coverage < 0.7 + overall < 0.65 (matching self_review_agent)."""
    source = Path(__file__).parent.parent / "src" / "agents" / "supervisor.py"
    code = source.read_text()
    # Phase 5b section uses tightened dual-condition threshold
    phase5b = code.split("Phase 5b")[1].split("return")[0]
    assert "coverage_score" in phase5b
    assert "< 0.7" in phase5b
    assert "< 0.65" in phase5b


# === Test 9: snowball min_matches relaxed ===

def test_snowball_relaxed_matching():
    """Snowball min_matches should be 1-2, not 2-3."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    # Find the snowball_min_matches assignment
    assert "snowball_min_matches = 2 if" in code or "snowball_min_matches = 1" in code
    # Should NOT have the old strict threshold
    assert "snowball_min_matches = 3 if" not in code


# === Test 10: review_title fallback in planning_agent ===

def test_planning_agent_title_fallback():
    """planning_agent should generate fallback title from sub-topics."""
    source = Path(__file__).parent.parent / "src" / "agents" / "planning_agent.py"
    code = source.read_text()
    assert "fallback review_title" in code
    assert "topic_names" in code


# === Test 11: CLI default papers is 40 ===

def test_cli_default_papers():
    """CLI argparse default for --papers should be 40."""
    source = Path(__file__).parent.parent / "src" / "cli" / "litscribe_cli.py"
    code = source.read_text()
    # Find the --papers argument definition
    papers_section = code[code.find('"--papers"'):code.find('"--papers"') + 200]
    assert "default=40" in papers_section


# === Test 12: CLI shows review tier ===

def test_cli_shows_review_tier():
    """CLI should display review tier info."""
    source = Path(__file__).parent.parent / "src" / "cli" / "litscribe_cli.py"
    code = source.read_text()
    assert "Review Scale" in code
    assert "review_tier" in code


# === Test 13: queries[:6] limit removed ===

def test_search_query_limit_raised():
    """search_all_sources should allow more than 6 queries."""
    source = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
    code = source.read_text()
    assert "queries[:6]" not in code
    assert "queries[:12]" in code


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
