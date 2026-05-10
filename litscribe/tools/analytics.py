from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from litscribe.models.paper import Paper
from litscribe.models.analysis import PaperAnalysis
from litscribe.models.review import ReviewOutput

logger = logging.getLogger(__name__)


# ─── Citation Network Graph ────────────────────────────────

def build_citation_network(papers: list[Paper], key_map: dict[str, str] | None = None) -> dict:
    nodes = []
    edges = []

    for p in papers:
        key = key_map.get(p.paper_id, p.paper_id) if key_map else p.paper_id
        nodes.append({
            "id": key,
            "label": f"{p.authors[0].split()[-1] if p.authors else '?'} ({p.year})",
            "title": p.title[:60],
            "year": p.year,
            "citations": p.citations or 0,
            "size": min(30, 8 + (p.citations or 0) // 10),
        })

    # Infer edges from shared keywords/topics
    for i, p1 in enumerate(papers):
        words1 = set(p1.title.lower().split()) | set((p1.abstract or "").lower().split()[:50])
        for j, p2 in enumerate(papers):
            if i >= j:
                continue
            words2 = set(p2.title.lower().split()) | set((p2.abstract or "").lower().split()[:50])
            overlap = len(words1 & words2) - len({"the", "a", "an", "of", "in", "and", "to", "for", "is", "with", "on", "by", "from", "that", "this", "are", "was", "were"})
            if overlap > 5:
                k1 = key_map.get(p1.paper_id, p1.paper_id) if key_map else p1.paper_id
                k2 = key_map.get(p2.paper_id, p2.paper_id) if key_map else p2.paper_id
                edges.append({"from": k1, "to": k2, "weight": overlap})

    return {"nodes": nodes, "edges": edges}


def citation_network_to_mermaid(network: dict) -> str:
    lines = ["graph LR"]
    for node in network["nodes"]:
        lines.append(f'    {node["id"]}["{node["label"]}<br/>{node["title"][:30]}"]')
    for edge in network["edges"]:
        lines.append(f'    {edge["from"]} --- {edge["to"]}')
    return "\n".join(lines)


# ─── Statistical Data Extraction ───────────────────────────

STATS_PROMPT = """Extract quantitative data from these paper analyses for a meta-analysis table.

Papers:
{papers_summary}

For each paper, extract (if available):
- Sample size (n)
- Key metric and its value
- p-value
- Effect size (Cohen's d, OR, HR, RR, etc.)
- Confidence interval

Output JSON array:
[
  {{"paper": "@key", "sample_size": "n=100", "metric": "accuracy", "value": "0.95", "p_value": "p<0.001", "effect_size": "d=0.8", "ci": "95% CI [0.7-0.9]"}}
]

If a value is not available, use "N/R" (not reported)."""


async def extract_statistics(
    model: ChatOpenAI,
    papers: list[Paper],
    analyses: list[PaperAnalysis],
    key_map: dict[str, str] | None = None,
) -> list[dict]:
    summaries = []
    for p in papers:
        key = key_map.get(p.paper_id, p.paper_id) if key_map else p.paper_id
        a = next((x for x in analyses if x.paper_id == p.paper_id), None)
        findings = "; ".join(a.key_findings[:3]) if a else ""
        summaries.append(f"[@{key}] {p.title}: {findings}")

    prompt = STATS_PROMPT.format(papers_summary="\n".join(summaries))

    try:
        result = await model.ainvoke(prompt)
        raw = result.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"Statistics extraction failed: {e}")
        return []


def stats_to_markdown_table(stats: list[dict]) -> str:
    if not stats:
        return ""
    lines = [
        "| Paper | Sample Size | Key Metric | Value | p-value | Effect Size | CI |",
        "|-------|-------------|------------|-------|---------|-------------|-----|",
    ]
    for s in stats:
        lines.append(
            f"| {s.get('paper','')} | {s.get('sample_size','N/R')} | "
            f"{s.get('metric','N/R')} | {s.get('value','N/R')} | "
            f"{s.get('p_value','N/R')} | {s.get('effect_size','N/R')} | "
            f"{s.get('ci','N/R')} |"
        )
    return "\n".join(lines)


# ─── Multi-Review Comparison ───────────────────────────────

MULTI_REVIEW_PROMPT = """Compare these {n} literature reviews and analyze their overlap, complementarity, and contradictions.

{reviews_text}

Output JSON:
{{
  "overlap": ["Topics covered by all reviews"],
  "unique_to_each": [{{"review": 1, "unique_topics": ["topic only in review 1"]}}],
  "contradictions": ["Where reviews disagree"],
  "complementary": ["How reviews complement each other"],
  "combined_gaps": ["Gaps across all reviews"],
  "recommendation": "Which review is strongest and why"
}}"""


async def compare_reviews(
    model: ChatOpenAI,
    reviews: list[str],
) -> dict:
    reviews_text = ""
    for i, r in enumerate(reviews, 1):
        reviews_text += f"\n--- Review {i} (first 1000 chars) ---\n{r[:1000]}\n"

    prompt = MULTI_REVIEW_PROMPT.format(n=len(reviews), reviews_text=reviews_text)

    try:
        result = await model.ainvoke(prompt)
        raw = result.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Multi-review comparison failed: {e}")
        return {"error": str(e)}


# ─── Reading Level Assessment ──────────────────────────────

READABILITY_PROMPT = """Assess the reading difficulty of this literature review.

Review excerpt (first 1500 chars):
{review_text}

Evaluate:
1. Technical vocabulary density
2. Required background knowledge
3. Sentence complexity
4. Assumed familiarity with the field

Output JSON:
{{
  "level": "undergraduate|masters|phd|expert",
  "score": 1-10,
  "reasoning": "Why this level",
  "suggestions_to_simplify": ["How to make it more accessible"],
  "jargon_terms": ["technical terms that need explanation"]
}}"""


async def assess_readability(
    model: ChatOpenAI,
    review_text: str,
) -> dict:
    prompt = READABILITY_PROMPT.format(review_text=review_text[:1500])

    try:
        result = await model.ainvoke(prompt)
        raw = result.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Readability assessment failed: {e}")
        return {"level": "unknown", "score": 5}


# ─── Figure Suggestions ───────────────────────────────────

FIGURE_PROMPT = """Suggest figures and diagrams for this literature review.

Review excerpt:
{review_text}

Papers available: {num_papers}
Themes: {themes}

For each suggested figure, specify:
1. Type (flowchart, bar chart, table, Venn diagram, timeline, network graph, heatmap, etc.)
2. What it shows
3. Where in the review it should go
4. Data source (which papers' data to use)

Output JSON array:
[
  {{"type": "flowchart", "title": "Figure 1: ...", "description": "Shows ...", "placement": "After ## Introduction", "data_source": "Based on [@key1; @key2]"}}
]"""


async def suggest_figures(
    model: ChatOpenAI,
    review_text: str,
    themes: list[str],
    num_papers: int,
) -> list[dict]:
    prompt = FIGURE_PROMPT.format(
        review_text=review_text[:2000],
        num_papers=num_papers,
        themes=", ".join(themes),
    )

    try:
        result = await model.ainvoke(prompt)
        raw = result.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"Figure suggestion failed: {e}")
        return []
