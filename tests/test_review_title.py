#!/usr/bin/env python
"""Tests for review_title generation and clarification prompts in Planning Agent.

Covers:
- ResearchPlan has review_title, needs_clarification, clarification_questions fields
- create_initial_state works with new fields (no crash)
- COMPLEXITY_ASSESSMENT_PROMPT includes review_title and clarification fields
- PLAN_REVISION_PROMPT includes review_title
- format_plan_for_user shows review_title
- _build_fallback_plan sets empty defaults for new fields
- CLI uses review_title for markdown heading (not hardcoded "Literature Review:")
- CLI has clarification loop logic

Run with: pytest tests/test_review_title.py -v
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# === Test 1: ResearchPlan has review_title field ===

def test_research_plan_has_review_title():
    """ResearchPlan TypedDict should have review_title field."""
    from agents.state import ResearchPlan
    hints = ResearchPlan.__annotations__
    assert "review_title" in hints


# === Test 2: ResearchPlan has clarification fields ===

def test_research_plan_has_clarification_fields():
    """ResearchPlan should have needs_clarification and clarification_questions."""
    from agents.state import ResearchPlan
    hints = ResearchPlan.__annotations__
    assert "needs_clarification" in hints
    assert "clarification_questions" in hints


# === Test 3: COMPLEXITY_ASSESSMENT_PROMPT includes review_title ===

def test_complexity_prompt_has_review_title():
    """COMPLEXITY_ASSESSMENT_PROMPT should ask LLM to generate review_title."""
    from agents.prompts import COMPLEXITY_ASSESSMENT_PROMPT
    assert "review_title" in COMPLEXITY_ASSESSMENT_PROMPT
    assert "formal academic title" in COMPLEXITY_ASSESSMENT_PROMPT.lower()


# === Test 4: COMPLEXITY_ASSESSMENT_PROMPT includes clarification ===

def test_complexity_prompt_has_clarification():
    """COMPLEXITY_ASSESSMENT_PROMPT should include needs_clarification guidance."""
    from agents.prompts import COMPLEXITY_ASSESSMENT_PROMPT
    assert "needs_clarification" in COMPLEXITY_ASSESSMENT_PROMPT
    assert "clarification_questions" in COMPLEXITY_ASSESSMENT_PROMPT


# === Test 5: PLAN_REVISION_PROMPT includes review_title ===

def test_revision_prompt_has_review_title():
    """PLAN_REVISION_PROMPT should include review_title in output schema."""
    from agents.prompts import PLAN_REVISION_PROMPT
    assert "review_title" in PLAN_REVISION_PROMPT


# === Test 6: format_plan_for_user shows title ===

def test_format_plan_shows_title():
    """format_plan_for_user should display review_title when present."""
    from agents.planning_agent import format_plan_for_user
    from agents.state import ResearchPlan, SubTopic

    plan = ResearchPlan(
        complexity_score=3,
        sub_topics=[SubTopic(
            name="Test Topic",
            description="desc",
            estimated_papers=5,
            priority=0.8,
            custom_queries=["q1"],
            selected=True,
        )],
        scope_estimate="5 papers",
        is_interactive=True,
        confirmed=False,
        domain_hint="Biology",
        arxiv_categories=[],
        s2_fields=[],
        pubmed_mesh=[],
        review_title="CRISPR Knockouts in CHO Cells",
        needs_clarification=False,
        clarification_questions=[],
    )
    formatted = format_plan_for_user(plan)
    assert "CRISPR Knockouts in CHO Cells" in formatted
    assert "Review Title:" in formatted


# === Test 7: format_plan_for_user omits title when empty ===

def test_format_plan_omits_empty_title():
    """format_plan_for_user should not show 'Review Title:' when title is empty."""
    from agents.planning_agent import format_plan_for_user
    from agents.state import ResearchPlan, SubTopic

    plan = ResearchPlan(
        complexity_score=2,
        sub_topics=[SubTopic(
            name="Simple Topic",
            description="",
            estimated_papers=5,
            priority=1.0,
            custom_queries=[],
            selected=True,
        )],
        scope_estimate="5 papers",
        is_interactive=False,
        confirmed=True,
        domain_hint="",
        arxiv_categories=[],
        s2_fields=[],
        pubmed_mesh=[],
        review_title="",
        needs_clarification=False,
        clarification_questions=[],
    )
    formatted = format_plan_for_user(plan)
    assert "Review Title:" not in formatted


# === Test 8: _build_fallback_plan has correct defaults ===

def test_fallback_plan_defaults():
    """_build_fallback_plan should set review_title='', needs_clarification=False."""
    from agents.planning_agent import _build_fallback_plan
    plan = _build_fallback_plan("test question", 10)
    assert plan["review_title"] == ""
    assert plan["needs_clarification"] is False
    assert plan["clarification_questions"] == []


# === Test 9: CLI uses review_title for markdown heading ===

def test_cli_uses_review_title():
    """CLI should use review_title from plan for markdown heading."""
    source = Path(__file__).parent.parent / "src" / "cli" / "litscribe_cli.py"
    code = source.read_text()
    # Should NOT have hardcoded "# Literature Review: {research_question}" anymore
    assert 'f"# Literature Review: {research_question}' not in code
    # Should use plan-derived title
    assert "review_title" in code
    assert "md_title" in code or "resume_title" in code or "refine_title" in code


# === Test 10: CLI has clarification loop ===

def test_cli_has_clarification_loop():
    """CLI should check needs_clarification and prompt user."""
    source = Path(__file__).parent.parent / "src" / "cli" / "litscribe_cli.py"
    code = source.read_text()
    assert "needs_clarification" in code
    assert "clarification_questions" in code
    assert "Clarification Needed" in code


# === Test 11: CLI updates output_path with review_title ===

def test_cli_updates_output_path():
    """CLI should update output_path using review_title for cleaner filenames."""
    source = Path(__file__).parent.parent / "src" / "cli" / "litscribe_cli.py"
    code = source.read_text()
    assert "safe_title" in code
    assert "review_title" in code


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
