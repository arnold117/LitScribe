from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from litscribe.models.paper import Paper
from litscribe.models.analysis import PaperAnalysis
from litscribe.tools.status import PipelineState

logger = logging.getLogger(__name__)


async def parse_local_papers(
    model: ChatOpenAI,
    paths: list[str],
) -> list[Paper]:
    from litscribe.services.pdf import PDFService
    pdf_svc = PDFService()
    papers = []

    for i, path in enumerate(paths):
        p = Path(path)
        if not p.exists():
            logger.warning(f"File not found: {path}")
            continue

        logger.info(f"Parsing {p.name} ({i+1}/{len(paths)})")

        if p.suffix.lower() == ".pdf":
            try:
                parsed = await pdf_svc.parse(str(p))
                text = parsed.markdown[:3000] if parsed and parsed.markdown else ""
            except Exception as e:
                logger.warning(f"PDF parse failed for {p.name}: {e}")
                text = ""
        elif p.suffix.lower() in (".txt", ".md"):
            text = p.read_text(encoding="utf-8")[:3000]
        elif p.suffix.lower() == ".bib":
            text = p.read_text(encoding="utf-8")
            papers.extend(_parse_bibtex(text))
            continue
        else:
            logger.warning(f"Unsupported file type: {p.suffix}")
            continue

        if text:
            meta = await _extract_metadata(model, text, p.name)
            papers.append(Paper(
                paper_id=f"local:{p.stem}",
                title=meta.get("title", p.stem),
                authors=meta.get("authors", []),
                abstract=meta.get("abstract", text[:500]),
                year=meta.get("year", 2024),
                sources={"local": str(p)},
            ))

    logger.info(f"Parsed {len(papers)} local papers")
    return papers


def _parse_bibtex(bib_text: str) -> list[Paper]:
    entries = re.findall(r'@\w+\{(\w+),\s*(.*?)\n\}', bib_text, re.DOTALL)
    papers = []
    for key, body in entries:
        fields = dict(re.findall(r'(\w+)\s*=\s*\{(.*?)\}', body))
        authors = [a.strip() for a in fields.get("author", "Unknown").split(" and ")]
        papers.append(Paper(
            paper_id=f"bib:{key}",
            title=fields.get("title", key),
            authors=authors,
            abstract=fields.get("abstract", ""),
            year=int(fields.get("year", "2024")),
            sources={"bibtex": key},
            doi=fields.get("doi", ""),
        ))
    return papers


async def _extract_metadata(model: ChatOpenAI, text: str, filename: str) -> dict:
    prompt = (
        f"Extract metadata from this academic paper text.\n\n"
        f"Filename: {filename}\n"
        f"Text (first 1000 chars):\n{text[:1000]}\n\n"
        f"Output JSON: {{\"title\": \"...\", \"authors\": [\"...\"], \"year\": 2024, \"abstract\": \"...\"}}"
    )
    try:
        result = await model.ainvoke(prompt)
        raw = result.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception:
        return {"title": filename, "authors": [], "year": 2024, "abstract": text[:300]}


# ─── Mode A: Draft + References → Improvements ───

DRAFT_REVIEW_PROMPT = """You are an expert academic editor. Analyze this draft literature review and suggest improvements.

Draft:
{draft_text}

Available papers ({num_papers}):
{papers_summary}

Provide:
1. STRENGTHS — what the draft does well
2. WEAKNESSES — specific problems (unsupported claims, missing citations, logical gaps, structural issues)
3. MISSING TOPICS — what should be covered but isn't
4. CITATION ISSUES — claims without citations, wrong citations
5. SUGGESTED ADDITIONS — specific papers from the list that should be cited and where
6. REVISED OUTLINE — improved structure

Output JSON:
{{
  "strengths": ["..."],
  "weaknesses": [{{"issue": "...", "location": "...", "suggestion": "..."}}],
  "missing_topics": ["..."],
  "citation_issues": ["..."],
  "suggested_additions": [{{"paper": "...", "where": "...", "why": "..."}}],
  "revised_outline": ["section 1", "section 2", ...]
}}"""


async def review_draft(
    model: ChatOpenAI,
    draft_text: str,
    papers: list[Paper],
    analyses: list[PaperAnalysis],
) -> dict:
    from litscribe.prompts.utils import format_summaries_for_prompt
    from litscribe.tools.synthesis import _enrich_analyses_with_papers

    enriched = _enrich_analyses_with_papers(analyses, papers)
    summaries = format_summaries_for_prompt(enriched, max_chars=10000)

    prompt = DRAFT_REVIEW_PROMPT.format(
        draft_text=draft_text[:5000],
        num_papers=len(papers),
        papers_summary=summaries,
    )

    try:
        result = await model.ainvoke(prompt)
        raw = result.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Draft review failed: {e}")
        return {"error": str(e)}


# ─── Mode B: References → Outline + Gap Analysis ───

OUTLINE_PROMPT = """You are an expert research strategist. Given these papers, suggest what literature review could be written.

Papers ({num_papers}):
{papers_summary}

Analyze:
1. What common themes emerge?
2. What research question could these papers answer?
3. What's a good structure for a review?
4. What topics are MISSING — what additional papers should be found?

Output JSON:
{{
  "suggested_question": "The research question these papers best address",
  "themes": [{{"name": "...", "papers": ["paper1", "paper2"], "description": "..."}}],
  "proposed_outline": ["## Introduction", "## Theme 1: ...", ...],
  "gaps": ["Missing topic 1", "Missing topic 2"],
  "search_queries": ["query to find missing papers 1", "query 2"]
}}"""


async def suggest_outline(
    model: ChatOpenAI,
    papers: list[Paper],
    analyses: list[PaperAnalysis],
) -> dict:
    from litscribe.prompts.utils import format_summaries_for_prompt
    from litscribe.tools.synthesis import _enrich_analyses_with_papers

    enriched = _enrich_analyses_with_papers(analyses, papers)
    summaries = format_summaries_for_prompt(enriched, max_chars=10000)

    prompt = OUTLINE_PROMPT.format(
        num_papers=len(papers),
        papers_summary=summaries,
    )

    try:
        result = await model.ainvoke(prompt)
        raw = result.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Outline suggestion failed: {e}")
        return {"error": str(e)}


# ─── Mode C: Question + Local Papers → Augmented Review ───

async def augmented_review(
    model: ChatOpenAI,
    config,
    research_question: str,
    local_papers: list[Paper],
    max_additional: int = 10,
    language: str = "en",
    user_instructions: str = "",
) -> str:
    from litscribe.tools.pipeline import run_review
    from litscribe.tools.search import search_all_sources
    from litscribe.services.base import dedup_papers

    state = PipelineState(
        research_question=research_question,
        language=language,
    )

    # Combine local + searched papers
    logger.info(f"Augmented review: {len(local_papers)} local papers + searching {max_additional} more")

    from litscribe.tools.pipeline import step_plan
    await step_plan(model, state)

    queries = [research_question]
    if state.plan:
        for st in state.plan.sub_topics:
            queries.extend(st.keywords[:2])

    searched = await search_all_sources(queries[:4], config, max_per_source=max_additional, domain=state.domain)
    all_papers = dedup_papers(local_papers + searched)
    state.papers = all_papers[:max_additional + len(local_papers)]

    logger.info(f"  Combined: {len(local_papers)} local + {len(searched)} searched = {len(state.papers)} total")

    # Continue pipeline from read step
    from litscribe.tools.pipeline import (
        step_read, step_contradictions, step_synthesize,
        step_debate, step_ground, step_review,
    )

    await step_read(model, state)
    await step_contradictions(model, state)
    await step_synthesize(model, state, user_instructions)
    await step_debate(model, state)
    await step_ground(model, state)
    await step_review(model, state)

    return (
        f"Augmented review: {len(local_papers)} local + {len(searched)} searched papers.\n"
        f"Score: {state.assessment.score:.2f}\n\n"
        f"{state.synthesis.text[:3000]}"
    )
