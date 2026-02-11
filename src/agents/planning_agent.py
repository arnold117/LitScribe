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
from agents.tools import call_llm, extract_json

logger = logging.getLogger(__name__)


async def assess_and_decompose(
    research_question: str,
    max_papers: int = 10,
    model: Optional[str] = None,
    tracker=None,
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

    response = await call_llm(prompt, model=model, temperature=0.3, max_tokens=1500, tracker=tracker, agent_name="planning")

    data = extract_json(response)

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

    # Ensure review_title is non-empty — generate fallback from sub-topics if LLM omits it
    review_title = data.get("review_title", "").strip()
    if not review_title and sub_topics:
        # Build a title from the first 2-3 sub-topic names
        topic_names = [t["name"] for t in sub_topics[:3]]
        review_title = ": ".join(topic_names[:2]) if len(topic_names) >= 2 else topic_names[0]
        logger.info(f"Generated fallback review_title from sub-topics: {review_title}")

    return ResearchPlan(
        complexity_score=complexity,
        sub_topics=sub_topics,
        scope_estimate=data.get("scope_estimate", f"Estimated {max_papers} papers"),
        is_interactive=is_interactive,
        confirmed=confirmed,
        domain_hint=data.get("domain", ""),
        arxiv_categories=data.get("arxiv_categories", []),
        s2_fields=data.get("s2_fields", []),
        pubmed_mesh=data.get("pubmed_mesh", []),
        review_title=review_title,
        needs_clarification=bool(data.get("needs_clarification", False)),
        clarification_questions=data.get("clarification_questions", []),
    )


async def revise_plan(
    research_question: str,
    current_plan: ResearchPlan,
    user_feedback: str,
    model: Optional[str] = None,
    tracker=None,
) -> ResearchPlan:
    """Revise a research plan based on user feedback.

    Args:
        research_question: The original research question
        current_plan: The current plan to revise
        user_feedback: User's feedback/criticism of the current plan
        model: LLM model to use
        tracker: Token tracker for cost instrumentation

    Returns:
        Revised ResearchPlan
    """
    from agents.prompts import PLAN_REVISION_PROMPT

    plan_json = json.dumps(dict(current_plan), indent=2, ensure_ascii=False)

    prompt = PLAN_REVISION_PROMPT.format(
        research_question=research_question,
        current_plan_json=plan_json,
        user_feedback=user_feedback,
    )

    response = await call_llm(
        prompt, model=model, temperature=0.3, max_tokens=1500,
        tracker=tracker, agent_name="planning",
    )

    data = extract_json(response)

    complexity = int(data.get("complexity_score", current_plan["complexity_score"]))
    raw_topics = data.get("sub_topics", [])

    sub_topics = []
    for t in raw_topics[:6]:
        sub_topics.append(SubTopic(
            name=t.get("name", "Unknown"),
            description=t.get("description", ""),
            estimated_papers=int(t.get("estimated_papers", 5)),
            priority=float(t.get("priority", 0.5)),
            custom_queries=t.get("custom_queries", [])[:3],
            selected=True,
        ))

    # If LLM returns no topics, fall back to current plan's topics
    if not sub_topics:
        sub_topics = list(current_plan["sub_topics"])

    is_interactive = complexity >= 3
    return ResearchPlan(
        complexity_score=complexity,
        sub_topics=sub_topics,
        scope_estimate=data.get("scope_estimate", current_plan["scope_estimate"]),
        is_interactive=is_interactive,
        confirmed=False,
        domain_hint=data.get("domain", current_plan.get("domain_hint", "")),
        arxiv_categories=data.get("arxiv_categories", current_plan.get("arxiv_categories", [])),
        s2_fields=data.get("s2_fields", current_plan.get("s2_fields", [])),
        pubmed_mesh=data.get("pubmed_mesh", current_plan.get("pubmed_mesh", [])),
        review_title=data.get("review_title", current_plan.get("review_title", "")),
        needs_clarification=False,  # Revised plans don't need further clarification
        clarification_questions=[],
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
        domain_hint="",
        arxiv_categories=[],
        s2_fields=[],
        pubmed_mesh=[],
        review_title="",
        needs_clarification=False,
        clarification_questions=[],
    )


def format_plan_for_user(plan: ResearchPlan) -> str:
    """Format research plan for display in CLI.

    Args:
        plan: The research plan to format

    Returns:
        Formatted string for terminal display
    """
    lines = []
    review_title = plan.get("review_title", "")
    if review_title:
        lines.append(f"Review Title: {review_title}")
    lines.append(f"Complexity: {plan['complexity_score']}/5")
    lines.append(f"Scope: {plan['scope_estimate']}")

    selected_topics = [t for t in plan["sub_topics"] if t.get("selected", True)]
    total_estimated = sum(t.get("estimated_papers", 5) for t in selected_topics)
    lines.append(f"Estimated papers: ~{total_estimated} (across {len(selected_topics)} sub-topics)")
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

    from utils.token_tracker import get_tracker
    tracker = get_tracker()
    llm_config = state.get("llm_config", {})
    model = llm_config.get("model")

    logger.info(f"Planning Agent starting for: {research_question}")

    try:
        plan = await assess_and_decompose(
            research_question=research_question,
            max_papers=max_papers,
            model=model,
            tracker=tracker,
        )

        logger.info(
            f"Planning complete: complexity={plan['complexity_score']}, "
            f"sub_topics={len(plan['sub_topics'])}, "
            f"interactive={plan['is_interactive']}"
        )

        return {
            "research_plan": plan,
            "domain_hint": plan.get("domain_hint", ""),
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
