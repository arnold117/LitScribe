"""Planning Agent for LitScribe (Phase 9.2).

This agent runs before discovery to:
1. Assess research question complexity (1-5)
2. Decompose complex questions into sub-topics
3. Generate targeted search queries per sub-topic
4. Produce a research plan that guides discovery

Simple questions (complexity <= 2) get a minimal plan and proceed directly.
Complex questions (complexity >= 3) get full decomposition.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from agents.errors import LLMError
from agents.prompts import COMPLEXITY_ASSESSMENT_PROMPT
from agents.state import LitScribeState, ResearchPlan, SubTopic
from agents.tools import call_llm

logger = logging.getLogger(__name__)


async def assess_and_decompose(
    research_question: str,
    max_papers: int = 10,
    model: Optional[str] = None,
) -> ResearchPlan:
    """Assess complexity and decompose research question into sub-topics.

    Single LLM call that both rates complexity and produces sub-topics.

    Args:
        research_question: The research question
        max_papers: Max papers budget (used for estimation)
        model: LLM model to use

    Returns:
        ResearchPlan with complexity score and sub-topics
    """
    prompt = COMPLEXITY_ASSESSMENT_PROMPT.format(
        research_question=research_question,
    )

    response = await call_llm(prompt, model=model, temperature=0.3, max_tokens=1500)

    # Parse JSON (same pattern as other agents)
    response = response.strip()
    if response.startswith("```"):
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]
    response = response.strip()

    data = json.loads(response)

    complexity = int(data.get("complexity_score", 2))
    raw_topics = data.get("sub_topics", [])

    # Build SubTopic list
    sub_topics = []
    for t in raw_topics[:6]:  # Cap at 6 sub-topics
        sub_topics.append(SubTopic(
            name=t.get("name", "Unknown"),
            description=t.get("description", ""),
            estimated_papers=int(t.get("estimated_papers", 5)),
            priority=float(t.get("priority", 0.5)),
            custom_queries=t.get("custom_queries", [])[:3],
            selected=True,  # All selected by default
        ))

    # Simple questions: auto-confirm
    is_interactive = complexity >= 3
    confirmed = not is_interactive  # Simple plans auto-confirmed

    return ResearchPlan(
        complexity_score=complexity,
        sub_topics=sub_topics,
        scope_estimate=data.get("scope_estimate", f"Estimated {max_papers} papers"),
        is_interactive=is_interactive,
        confirmed=confirmed,
    )


def _build_fallback_plan(research_question: str, max_papers: int) -> ResearchPlan:
    """Build a minimal fallback plan when LLM fails.

    Creates a single sub-topic covering the whole question.

    Args:
        research_question: The research question
        max_papers: Max papers budget

    Returns:
        Simple ResearchPlan
    """
    return ResearchPlan(
        complexity_score=1,
        sub_topics=[
            SubTopic(
                name=research_question[:80],
                description=research_question,
                estimated_papers=max_papers,
                priority=1.0,
                custom_queries=[],
                selected=True,
            )
        ],
        scope_estimate=f"Estimated {max_papers} papers",
        is_interactive=False,
        confirmed=True,
    )


def format_plan_for_user(plan: ResearchPlan) -> str:
    """Format research plan for display in CLI.

    Args:
        plan: The research plan to format

    Returns:
        Formatted string for terminal display
    """
    lines = []
    lines.append(f"Complexity: {plan['complexity_score']}/5")
    lines.append(f"Scope: {plan['scope_estimate']}")
    lines.append("")

    for i, topic in enumerate(plan["sub_topics"], 1):
        selected = "[x]" if topic["selected"] else "[ ]"
        lines.append(f"  {selected} {i}. {topic['name']} (priority: {topic['priority']:.1f}, ~{topic['estimated_papers']} papers)")
        if topic["description"]:
            lines.append(f"      {topic['description'][:100]}")
        if topic["custom_queries"]:
            lines.append(f"      Queries: {', '.join(topic['custom_queries'][:2])}")

    return "\n".join(lines)


async def planning_agent(state: LitScribeState) -> Dict[str, Any]:
    """Main entry point for the Planning Agent.

    Called by the LangGraph workflow before discovery to assess
    question complexity and decompose into sub-topics.

    Args:
        state: Current workflow state

    Returns:
        State updates with research_plan
    """
    research_question = state["research_question"]
    max_papers = state.get("max_papers", 10)
    errors = list(state.get("errors", []))

    logger.info(f"Planning Agent starting for: {research_question}")

    try:
        plan = await assess_and_decompose(
            research_question=research_question,
            max_papers=max_papers,
        )

        logger.info(
            f"Planning complete: complexity={plan['complexity_score']}, "
            f"sub_topics={len(plan['sub_topics'])}, "
            f"interactive={plan['is_interactive']}"
        )

        return {
            "research_plan": plan,
            "current_agent": "discovery",
        }

    except (json.JSONDecodeError, LLMError) as e:
        error_msg = f"Planning failed: {e}"
        logger.warning(f"{error_msg}, using fallback plan")
        errors.append(error_msg)
        return {
            "research_plan": _build_fallback_plan(research_question, max_papers),
            "errors": errors,
            "current_agent": "discovery",
        }
    except Exception as e:
        error_msg = f"Planning unexpected error: {e}"
        logger.warning(f"{error_msg}, using fallback plan")
        errors.append(error_msg)
        return {
            "research_plan": _build_fallback_plan(research_question, max_papers),
            "errors": errors,
            "current_agent": "discovery",
        }


__all__ = [
    "planning_agent",
    "assess_and_decompose",
    "format_plan_for_user",
]
