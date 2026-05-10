from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from litscribe.models.analysis import PaperAnalysis
from litscribe.models.paper import Paper

logger = logging.getLogger(__name__)

COMPARISON_PROMPT = """Generate a methodology comparison table for these papers.

Papers:
{papers_summary}

Create a markdown table comparing key aspects across all papers:
| Paper | Method | Dataset/Sample | Key Metric | Main Finding | Limitation |

Output the table in markdown format. Include ALL papers. Be specific with numbers."""


TIMELINE_PROMPT = """Analyze the temporal evolution of research across these papers.

Papers (sorted by year):
{papers_summary}

Generate a research timeline showing how the field evolved:
1. Group papers by phase (early/foundational, development, recent/frontier)
2. For each phase, describe the key advances
3. Show how later papers build on earlier ones

Output as markdown with ## headings for each phase."""


async def generate_comparison_table(
    model: ChatOpenAI,
    papers: list[Paper],
    analyses: list[PaperAnalysis],
    key_map: dict[str, str] | None = None,
) -> str:
    summaries = []
    for p in papers:
        key = key_map.get(p.paper_id, p.paper_id) if key_map else p.paper_id
        a = next((x for x in analyses if x.paper_id == p.paper_id), None)
        method = a.methodology[:100] if a else "N/A"
        findings = "; ".join(a.key_findings[:2]) if a else "N/A"
        summaries.append(f"[@{key}] {p.title} ({p.year}): {method} | Findings: {findings}")

    prompt = COMPARISON_PROMPT.format(papers_summary="\n".join(summaries))

    try:
        result = await model.ainvoke(prompt)
        return result.content.strip()
    except Exception as e:
        logger.warning(f"Comparison table failed: {e}")
        return ""


async def generate_timeline(
    model: ChatOpenAI,
    papers: list[Paper],
    analyses: list[PaperAnalysis],
    key_map: dict[str, str] | None = None,
) -> str:
    sorted_papers = sorted(papers, key=lambda p: p.year or 0)
    summaries = []
    for p in sorted_papers:
        key = key_map.get(p.paper_id, p.paper_id) if key_map else p.paper_id
        a = next((x for x in analyses if x.paper_id == p.paper_id), None)
        findings = a.key_findings[0][:100] if a and a.key_findings else "N/A"
        summaries.append(f"[@{key}] ({p.year}) {p.title}: {findings}")

    prompt = TIMELINE_PROMPT.format(papers_summary="\n".join(summaries))

    try:
        result = await model.ainvoke(prompt)
        return result.content.strip()
    except Exception as e:
        logger.warning(f"Timeline failed: {e}")
        return ""


def classify_temporal_layers(papers: list[Paper]) -> dict[str, list[str]]:
    if not papers:
        return {"foundation": [], "development": [], "frontier": []}

    sorted_papers = sorted(papers, key=lambda p: p.year or 0)
    n = len(sorted_papers)

    if n <= 3:
        return {
            "foundation": [p.paper_id for p in sorted_papers[:1]],
            "development": [p.paper_id for p in sorted_papers[1:-1]],
            "frontier": [p.paper_id for p in sorted_papers[-1:]],
        }

    cut1 = n // 3
    cut2 = 2 * n // 3

    # Also consider citation count — high citations = more foundational
    by_citations = sorted(sorted_papers, key=lambda p: p.citations or 0, reverse=True)

    foundation_ids = set(p.paper_id for p in sorted_papers[:cut1])
    foundation_ids.update(p.paper_id for p in by_citations[:max(2, n // 4)])

    frontier_ids = set(p.paper_id for p in sorted_papers[cut2:])

    return {
        "foundation": [p.paper_id for p in sorted_papers if p.paper_id in foundation_ids],
        "development": [p.paper_id for p in sorted_papers if p.paper_id not in foundation_ids and p.paper_id not in frontier_ids],
        "frontier": [p.paper_id for p in sorted_papers if p.paper_id in frontier_ids],
    }
