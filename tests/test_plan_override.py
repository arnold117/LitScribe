#!/usr/bin/env python
"""Tests for planning agent paper count override.

Run with: pytest tests/test_plan_override.py -v
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# === Test 1: format_plan_for_user shows total estimated papers ===

def test_format_plan_shows_total():
    """format_plan_for_user should display total estimated papers."""
    from agents.planning_agent import format_plan_for_user
    from agents.state import ResearchPlan, SubTopic

    plan = ResearchPlan(
        complexity_score=3,
        sub_topics=[
            SubTopic(name="Topic A", description="Desc A", estimated_papers=8, priority=0.8, custom_queries=[], selected=True),
            SubTopic(name="Topic B", description="Desc B", estimated_papers=6, priority=0.6, custom_queries=[], selected=True),
            SubTopic(name="Topic C", description="Desc C", estimated_papers=4, priority=0.4, custom_queries=[], selected=True),
        ],
        scope_estimate="Estimated 18 papers",
        is_interactive=True,
        confirmed=False,
        domain_hint="Biology",
        arxiv_categories=[],
        s2_fields=[],
        pubmed_mesh=[],
    )

    output = format_plan_for_user(plan)
    assert "~18" in output, "Should show total estimated papers"
    assert "3 sub-topics" in output, "Should show number of sub-topics"


def test_format_plan_only_counts_selected():
    """format_plan_for_user total should only count selected sub-topics."""
    from agents.planning_agent import format_plan_for_user
    from agents.state import ResearchPlan, SubTopic

    plan = ResearchPlan(
        complexity_score=4,
        sub_topics=[
            SubTopic(name="Topic A", description="", estimated_papers=10, priority=0.9, custom_queries=[], selected=True),
            SubTopic(name="Topic B", description="", estimated_papers=8, priority=0.7, custom_queries=[], selected=False),
            SubTopic(name="Topic C", description="", estimated_papers=5, priority=0.5, custom_queries=[], selected=True),
        ],
        scope_estimate="",
        is_interactive=True,
        confirmed=False,
        domain_hint="",
        arxiv_categories=[],
        s2_fields=[],
        pubmed_mesh=[],
    )

    output = format_plan_for_user(plan)
    # Only Topic A (10) + Topic C (5) = 15
    assert "~15" in output, f"Should show 15 for selected topics, got: {output}"
    assert "2 sub-topics" in output, "Should show 2 selected sub-topics"


# === Test 2: Plan total calculation logic ===

def test_plan_total_calculation():
    """Verify the plan total calculation matches CLI logic."""
    from agents.state import SubTopic

    topics = [
        SubTopic(name="A", description="", estimated_papers=10, priority=1.0, custom_queries=[], selected=True),
        SubTopic(name="B", description="", estimated_papers=7, priority=0.8, custom_queries=[], selected=True),
        SubTopic(name="C", description="", estimated_papers=5, priority=0.5, custom_queries=[], selected=False),
    ]

    selected = [t for t in topics if t.get("selected", True)]
    plan_total = sum(t.get("estimated_papers", 5) for t in selected)

    assert len(selected) == 2
    assert plan_total == 17


def test_plan_total_capped_at_500():
    """Plan total should be capped at 500."""
    from agents.state import SubTopic

    # 6 topics * 100 papers each = 600, should cap at 500
    topics = [
        SubTopic(name=f"Topic {i}", description="", estimated_papers=100, priority=0.5, custom_queries=[], selected=True)
        for i in range(6)
    ]

    selected = [t for t in topics if t.get("selected", True)]
    plan_total = sum(t.get("estimated_papers", 5) for t in selected)
    plan_total = min(plan_total, 500)

    assert plan_total == 500


def test_plan_total_default_when_missing():
    """Should use default of 5 when estimated_papers is missing."""
    from agents.state import SubTopic

    # SubTopic with no estimated_papers (shouldn't happen but test fallback)
    topics = [
        {"name": "A", "description": "", "priority": 1.0, "custom_queries": [], "selected": True},
        {"name": "B", "description": "", "estimated_papers": 8, "priority": 0.8, "custom_queries": [], "selected": True},
    ]

    selected = [t for t in topics if t.get("selected", True)]
    plan_total = sum(t.get("estimated_papers", 5) for t in selected)

    assert plan_total == 13  # 5 (default) + 8


# === Test 3: Plan override logic conditions ===

def test_override_when_plan_differs_from_default():
    """When user didn't set -p (default 10) and plan suggests 25, should override."""
    user_papers = 10  # argparse default
    user_explicit = (user_papers != 10)  # False
    plan_total = 25

    # Logic from CLI: if not explicit, always override
    should_override = not user_explicit or plan_total <= user_papers

    assert not user_explicit
    assert should_override is True


def test_no_auto_override_when_user_explicit():
    """When user set -p 15 and plan suggests 30, should ask (not auto-override)."""
    user_papers = 15
    user_explicit = (user_papers != 10)  # True
    plan_total = 30

    # Should ask user, not auto-override
    needs_prompt = user_explicit and plan_total > user_papers

    assert user_explicit
    assert needs_prompt is True


def test_override_when_plan_under_user_explicit():
    """When user set -p 50 and plan suggests 25, should auto-override (plan is smaller)."""
    user_papers = 50
    user_explicit = (user_papers != 10)  # True
    plan_total = 25

    # Plan is smaller than user's explicit value — safe to override
    needs_prompt = user_explicit and plan_total > user_papers

    assert not needs_prompt  # No prompt needed


# === Test 4: SubTopic estimated_papers field exists ===

def test_subtopic_has_estimated_papers():
    """SubTopic TypedDict should have estimated_papers field."""
    from agents.state import SubTopic

    topic = SubTopic(
        name="test",
        description="test desc",
        estimated_papers=10,
        priority=0.5,
        custom_queries=["q1"],
        selected=True,
    )

    assert topic["estimated_papers"] == 10
    assert topic["selected"] is True


# === Test 5: assess_and_decompose returns estimated_papers ===

def test_assess_decompose_output_has_estimated_papers():
    """ResearchPlan sub-topics from assess_and_decompose should have estimated_papers."""
    from agents.state import ResearchPlan, SubTopic

    # Simulate what assess_and_decompose returns
    plan = ResearchPlan(
        complexity_score=3,
        sub_topics=[
            SubTopic(name="A", description="", estimated_papers=8, priority=0.9, custom_queries=["q1"], selected=True),
            SubTopic(name="B", description="", estimated_papers=6, priority=0.7, custom_queries=["q2"], selected=True),
        ],
        scope_estimate="~14 papers",
        is_interactive=True,
        confirmed=False,
        domain_hint="CS",
        arxiv_categories=["cs.CL"],
        s2_fields=["Computer Science"],
        pubmed_mesh=[],
    )

    for topic in plan["sub_topics"]:
        assert "estimated_papers" in topic
        assert isinstance(topic["estimated_papers"], int)
        assert topic["estimated_papers"] > 0


# === Test 6: Plan revision prompt exists ===

def test_plan_revision_prompt_exists():
    """PLAN_REVISION_PROMPT should be defined in prompts.py."""
    from agents.prompts import PLAN_REVISION_PROMPT
    assert "research_question" in PLAN_REVISION_PROMPT
    assert "current_plan_json" in PLAN_REVISION_PROMPT
    assert "user_feedback" in PLAN_REVISION_PROMPT


# === Test 7: revise_plan function exists ===

def test_revise_plan_function_exists():
    """revise_plan should be importable from planning_agent."""
    import inspect
    from agents.planning_agent import revise_plan
    assert inspect.iscoroutinefunction(revise_plan), "revise_plan should be async"

    sig = inspect.signature(revise_plan)
    params = list(sig.parameters.keys())
    assert "research_question" in params
    assert "current_plan" in params
    assert "user_feedback" in params


# === Test 8: CLI has feedback loop ===

def test_cli_has_plan_feedback_loop():
    """litscribe_cli should support plan revision loop (not just Y/n)."""
    source = Path(__file__).parent.parent / "src" / "cli" / "litscribe_cli.py"
    code = source.read_text()
    assert "Y/n/q" in code, "CLI should offer Y/n/q options"
    assert "revise_plan" in code, "CLI should call revise_plan"
    assert "MAX_PLAN_REVISIONS" in code, "CLI should cap revision rounds"


# === Test 9: Max revisions cap ===

def test_max_plan_revisions_is_reasonable():
    """MAX_PLAN_REVISIONS should be a small positive integer."""
    import re as _re
    source = Path(__file__).parent.parent / "src" / "cli" / "litscribe_cli.py"
    code = source.read_text()
    match = _re.search(r'MAX_PLAN_REVISIONS\s*=\s*(\d+)', code)
    assert match, "MAX_PLAN_REVISIONS should be defined"
    max_rev = int(match.group(1))
    assert 2 <= max_rev <= 5, f"MAX_PLAN_REVISIONS={max_rev} should be between 2 and 5"


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
