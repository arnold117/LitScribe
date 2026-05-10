from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langchain_openai import ChatOpenAI

from litscribe.models.review import ReviewOutput
from litscribe.models.analysis import PaperAnalysis

logger = logging.getLogger(__name__)

CRITIQUE_PROMPT = """You are a strict academic reviewer. Critique this literature review.

Research Question: {research_question}

Review text (truncated):
{review_text}

Papers available: {num_papers}

Identify specific, actionable problems:
1. UNSUPPORTED CLAIMS — claims without citations or citations that don't match
2. MISSING PERSPECTIVES — important angles not covered
3. LOGICAL GAPS — arguments that don't flow
4. WEAK SYNTHESIS — sections that just summarize papers instead of synthesizing

Output JSON:
{{
  "issues": [
    {{"type": "unsupported_claim|missing_perspective|logical_gap|weak_synthesis", "detail": "specific issue", "section": "which section", "suggestion": "how to fix"}}
  ],
  "overall_quality": "poor|fair|good|excellent"
}}"""

REVISE_PROMPT = """You are an expert academic writer. Revise this literature review based on the reviewer's critique.

Research Question: {research_question}

Current review:
{review_text}

Reviewer's critique:
{critique}

Available papers for citation:
{papers_context}

Revise the review to address ALL issues. Keep the same structure and citations format ([@key]).
Output the COMPLETE revised review text."""


async def debate_round(
    model: ChatOpenAI,
    review: ReviewOutput,
    research_question: str,
    papers_context: str,
    num_papers: int,
) -> tuple[ReviewOutput, dict]:
    # Reviewer critiques
    critique_prompt = CRITIQUE_PROMPT.format(
        research_question=research_question,
        review_text=review.text[:2000],
        num_papers=num_papers,
    )
    critique_result = await model.ainvoke(critique_prompt)
    raw = critique_result.content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        critique = json.loads(raw)
    except json.JSONDecodeError:
        critique = {"issues": [], "overall_quality": "fair"}

    issues = critique.get("issues", [])
    quality = critique.get("overall_quality", "fair")

    if not issues or quality in ("good", "excellent"):
        return review, critique

    # Synthesizer revises
    critique_text = "\n".join(
        f"- [{i.get('type', '?')}] {i.get('detail', '')} → {i.get('suggestion', '')}"
        for i in issues
    )

    revise_prompt = REVISE_PROMPT.format(
        research_question=research_question,
        review_text=review.text,
        critique=critique_text,
        papers_context=papers_context[:5000],
    )
    revised_result = await model.ainvoke(revise_prompt)
    revised_text = revised_result.content.strip()

    cjk = len(re.findall(r'[一-鿿㐀-䶿]', revised_text))
    latin = len(re.findall(r'[a-zA-Z]+', revised_text))

    revised_review = ReviewOutput(
        text=revised_text,
        citations=review.citations,
        themes=review.themes,
        word_count=cjk + latin,
        language=review.language,
    )

    return revised_review, critique


async def multi_round_debate(
    model: ChatOpenAI,
    review: ReviewOutput,
    research_question: str,
    analyses: list[PaperAnalysis],
    papers: list | None = None,
    max_rounds: int = 2,
) -> tuple[ReviewOutput, list[dict]]:
    from litscribe.tools.synthesis import _enrich_analyses_with_papers
    from litscribe.prompts.utils import format_summaries_for_prompt

    enriched = _enrich_analyses_with_papers(analyses, papers)
    papers_ctx = format_summaries_for_prompt(enriched, max_chars=5000)

    critiques = []
    current = review

    for round_num in range(max_rounds):
        t = time.time()
        current, critique = await debate_round(
            model, current, research_question, papers_ctx, len(analyses),
        )
        critiques.append(critique)

        quality = critique.get("overall_quality", "fair")
        n_issues = len(critique.get("issues", []))
        logger.info(
            f"  Debate round {round_num+1}: {n_issues} issues, quality={quality} ({time.time()-t:.1f}s)"
        )

        if quality in ("good", "excellent") or n_issues == 0:
            break

    return current, critiques
