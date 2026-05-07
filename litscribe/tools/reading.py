from __future__ import annotations

import json
import logging
from typing import Any

from litscribe.models.analysis import PaperAnalysis
from litscribe.models.paper import Paper
from litscribe.prompts.reading import COMBINED_PAPER_ANALYSIS_PROMPT, ABSTRACT_ONLY_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)

BATCH_SIZE = 5


def _msg(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


async def _try_get_full_text(paper: Paper) -> str | None:
    if not paper.pdf_urls:
        return None
    try:
        from litscribe.services.pdf import PDFService
        pdf_svc = PDFService()
        parsed = await pdf_svc.parse(paper.pdf_urls[0])
        if parsed and parsed.markdown and len(parsed.markdown) > 200:
            return parsed.markdown[:8000]
    except Exception as e:
        logger.debug(f"PDF fetch failed for {paper.paper_id}: {e}")
    return None


async def analyze_single_paper(
    router,
    paper: Paper,
    research_question: str,
) -> PaperAnalysis:
    full_text = await _try_get_full_text(paper)

    if full_text:
        prompt = COMBINED_PAPER_ANALYSIS_PROMPT.format(
            research_question=research_question,
            title=paper.title,
            authors=", ".join(paper.authors[:3]) if paper.authors else "Unknown",
            year=paper.year or "N/A",
            abstract=paper.abstract or "(no abstract)",
            full_text=full_text,
        )
    else:
        prompt = ABSTRACT_ONLY_ANALYSIS_PROMPT.format(
            research_question=research_question,
            title=paper.title,
            authors=", ".join(paper.authors[:3]) if paper.authors else "Unknown",
            year=paper.year or "N/A",
            venue=getattr(paper, "venue", "") or "",
            abstract=paper.abstract or "(no abstract)",
            metadata_section=f"Citations: {paper.citations or 0}, Sources: {list(paper.sources.keys())}",
        )

    try:
        result = await router.call_json(_msg(prompt), task_type="paper_analysis")
        if isinstance(result, dict):
            return PaperAnalysis(
                paper_id=paper.paper_id,
                key_findings=result.get("key_findings", []),
                methodology=result.get("methodology", ""),
                strengths=result.get("strengths", []),
                limitations=result.get("limitations", []),
                relevance_score=float(result.get("relevance_to_question", 0.5)),
                themes=[],
            )
    except Exception as e:
        logger.warning(f"Analysis failed for {paper.paper_id}: {e}")

    return PaperAnalysis(
        paper_id=paper.paper_id,
        key_findings=[f"Abstract: {(paper.abstract or '')[:200]}"],
        methodology="Analysis unavailable",
        strengths=[],
        limitations=["Automated analysis failed"],
        relevance_score=0.5,
        themes=[],
    )


async def analyze_papers(
    papers: list[Paper],
    research_question: str,
    router,
) -> list[PaperAnalysis]:
    import asyncio

    analyses: list[PaperAnalysis] = []

    for i in range(0, len(papers), BATCH_SIZE):
        batch = papers[i : i + BATCH_SIZE]
        logger.info(f"Analyzing batch {i // BATCH_SIZE + 1} ({len(batch)} papers)")

        batch_results = await asyncio.gather(
            *[analyze_single_paper(router, p, research_question) for p in batch],
            return_exceptions=True,
        )

        for r in batch_results:
            if isinstance(r, PaperAnalysis):
                analyses.append(r)
            elif isinstance(r, Exception):
                logger.warning(f"Batch analysis error: {r}")

    logger.info(f"Analyzed {len(analyses)}/{len(papers)} papers")
    return analyses
