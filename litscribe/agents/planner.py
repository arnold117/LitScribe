"""Planner agent — decomposes a research question into sub-topics."""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Awaitable

from litscribe.models.plan import ResearchPlan, ReviewTier, SubTopic

# How many sub-topics to allow per tier
TIER_SUBTOPIC_LIMITS: dict[ReviewTier, int] = {
    ReviewTier.QUICK: 3,
    ReviewTier.STANDARD: 5,
    ReviewTier.COMPREHENSIVE: 8,
}

PLANNING_PROMPT = """\
You are a research planning assistant. Given a research question, decompose it into
focused sub-topics and identify the academic domain.

Research question: {question}
Review tier: {tier}
Max papers: {max_papers}
Language: {language}
{memory_context_section}

Return ONLY valid JSON with this structure:
{{
  "sub_topics": [
    {{"name": "...", "keywords": ["...", "..."], "estimated_papers": 10}}
  ],
  "domain": "..."
}}

Limit to {max_subtopics} sub-topics. Each sub-topic should have 2-5 keywords.
"""


def _extract_json(raw: str) -> dict:
    """Extract JSON object from LLM response, stripping markdown fences."""
    # Remove markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    return json.loads(cleaned)


def _parse_subtopics(raw_subtopics: Any) -> list[SubTopic]:
    """Handle sub_topics as list-of-dicts or comma-separated string."""
    if isinstance(raw_subtopics, list):
        result = []
        for item in raw_subtopics:
            if isinstance(item, dict):
                result.append(
                    SubTopic(
                        name=item.get("name", ""),
                        keywords=item.get("keywords", []),
                        estimated_papers=int(item.get("estimated_papers", 10)),
                    )
                )
            elif isinstance(item, str) and item.strip():
                result.append(SubTopic(name=item.strip()))
        return result
    elif isinstance(raw_subtopics, str):
        parts = [p.strip() for p in raw_subtopics.split(",") if p.strip()]
        return [SubTopic(name=p) for p in parts]
    return []


def _calculate_target_words(max_papers: int, language: str) -> int:
    base = 1000 + max_papers * 130
    cjk_languages = {"zh", "ja", "ko"}
    if language.lower()[:2] in cjk_languages:
        base = int(base * 1.5)
    return base


async def create_plan(
    question: str,
    tier: ReviewTier,
    max_papers: int,
    language: str,
    llm_call: Callable[..., Awaitable[str]],
    memory_context: str | None = None,
) -> ResearchPlan:
    """Create a ResearchPlan by calling the LLM and parsing its response."""
    max_subtopics = TIER_SUBTOPIC_LIMITS[tier]

    memory_context_section = ""
    if memory_context:
        memory_context_section = f"Memory context:\n{memory_context}\n"

    prompt = PLANNING_PROMPT.format(
        question=question,
        tier=tier.value,
        max_papers=max_papers,
        language=language,
        max_subtopics=max_subtopics,
        memory_context_section=memory_context_section,
    )

    raw = await llm_call(prompt)
    data = _extract_json(raw)

    sub_topics = _parse_subtopics(data.get("sub_topics", []))
    # Enforce tier limit
    sub_topics = sub_topics[:max_subtopics]
    # Ensure at least one sub-topic (fallback to the question itself)
    if not sub_topics:
        sub_topics = [SubTopic(name=question, keywords=[question])]

    domain = data.get("domain", "General")
    target_words = _calculate_target_words(max_papers, language)

    return ResearchPlan(
        question=question,
        sub_topics=sub_topics,
        domain=domain,
        tier=tier,
        max_papers=max_papers,
        language=language,
        target_words=target_words,
    )
