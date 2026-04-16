"""Community summarizer — ported from src/graphrag/summarizer.py.

Generates natural-language summaries for detected communities using an
injected ``llm_call`` callable (same pattern as entity_extractor).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

_COMMUNITY_PROMPT = """Summarize the research theme represented by this community.

**Entities ({entity_count}):**
{entities_block}

**Papers ({paper_count}):**
{papers_block}

Write a concise 2-3 sentence summary covering the main topic, prominent methods, and key findings."""


async def summarize_communities(
    communities: list[dict],
    llm_call: Callable[..., Awaitable[str]],
    *,
    max_concurrent: int = 8,
) -> list[dict]:
    """Generate natural-language summaries for each detected community.

    Each community dict is expected to contain at least ``entity_ids`` and
    ``paper_ids``.  If the caller enriches communities with ``entities``
    (list of entity dicts) and ``papers`` (list of paper dicts) before
    calling, those are used in the prompt for richer summaries.

    Returns the same community dicts with the ``summary`` field populated.
    """
    if not communities:
        return communities

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _summarize_one(community: dict) -> None:
        if community.get("summary"):
            return  # already summarized

        entities_block = _format_entities(community)
        papers_block = _format_papers(community)

        prompt = _COMMUNITY_PROMPT.format(
            entity_count=len(community.get("entity_ids", [])),
            entities_block=entities_block,
            paper_count=len(community.get("paper_ids", [])),
            papers_block=papers_block,
        )

        async with semaphore:
            try:
                summary = await llm_call(prompt)
                community["summary"] = summary.strip()
            except Exception as e:
                logger.error("Failed to summarize community %s: %s",
                             community.get("community_id", "?"), e)
                community["summary"] = _fallback_summary(community)

    # Process level-by-level so children finish before parents
    levels = sorted({c.get("level", 0) for c in communities}, reverse=True)
    for level in levels:
        level_communities = [c for c in communities if c.get("level", 0) == level]
        await asyncio.gather(
            *[_summarize_one(c) for c in level_communities],
            return_exceptions=True,
        )

    return communities


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _format_entities(community: dict) -> str:
    """Build a bullet list of entities for the prompt."""
    # If the caller enriched the community with full entity dicts, use them
    enriched = community.get("entities", [])
    if enriched:
        lines = []
        for e in enriched[:15]:
            etype = e.get("entity_type", "unknown").upper()
            name = e.get("name", "?")
            desc = (e.get("description", "") or "")[:100]
            lines.append(f"- [{etype}] {name}: {desc}" if desc else f"- [{etype}] {name}")
        return "\n".join(lines)

    # Fallback: just list IDs
    ids = community.get("entity_ids", [])
    if not ids:
        return "No entities."
    return ", ".join(ids[:15])


def _format_papers(community: dict) -> str:
    """Build a bullet list of papers for the prompt."""
    enriched = community.get("papers", [])
    if enriched:
        lines = []
        for p in enriched[:10]:
            title = (p.get("title", "?"))[:80]
            year = p.get("year", "")
            lines.append(f"- {title} ({year})" if year else f"- {title}")
        return "\n".join(lines)

    ids = community.get("paper_ids", [])
    if not ids:
        return "No papers."
    return ", ".join(ids[:10])


def _fallback_summary(community: dict) -> str:
    """Deterministic fallback when the LLM call fails."""
    entity_ids = community.get("entity_ids", [])
    return f"Community containing {len(entity_ids)} entities."
