from __future__ import annotations

import json
import logging
import re
from typing import Any

from litscribe.models.analysis import PaperAnalysis
from litscribe.models.review import ReviewOutput, Citation, Theme
from litscribe.prompts.synthesis import (
    GAP_ANALYSIS_PROMPT,
    GRAPHRAG_LITERATURE_REVIEW_PROMPT,
    LITERATURE_REVIEW_PROMPT,
    THEME_IDENTIFICATION_PROMPT,
)
from litscribe.prompts.utils import (
    build_citation_checklist,
    format_summaries_for_prompt,
    get_language_instruction,
)

logger = logging.getLogger(__name__)


def _msg(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def _count_words(text: str) -> int:
    cjk = len(re.findall(r'[一-鿿㐀-䶿]', text))
    latin = len(re.findall(r'[a-zA-Z]+', text))
    return cjk + latin


async def identify_themes(
    router,
    analyses: list[PaperAnalysis],
    research_question: str,
) -> list[dict]:
    summaries_text = format_summaries_for_prompt(
        [a.model_dump() for a in analyses], max_chars=15000
    )
    prompt = THEME_IDENTIFICATION_PROMPT.format(
        research_question=research_question,
        num_papers=len(analyses),
        paper_summaries=summaries_text,
    )
    try:
        themes = await router.call_json(_msg(prompt), task_type="synthesis")
        if isinstance(themes, list):
            return themes
    except Exception as e:
        logger.warning(f"Theme identification failed: {e}")
    return [{"theme": "General Analysis", "description": research_question,
             "paper_ids": [a.paper_id for a in analyses], "key_points": []}]


async def identify_gaps(
    router,
    analyses: list[PaperAnalysis],
    themes: list[dict],
    research_question: str,
) -> dict:
    summaries_text = format_summaries_for_prompt(
        [a.model_dump() for a in analyses], max_chars=10000
    )
    themes_text = json.dumps(themes, indent=2, ensure_ascii=False)[:3000]
    prompt = GAP_ANALYSIS_PROMPT.format(
        research_question=research_question,
        paper_summaries=summaries_text,
        themes=themes_text,
    )
    try:
        gaps = await router.call_json(_msg(prompt), task_type="synthesis")
        if isinstance(gaps, dict):
            return gaps
    except Exception as e:
        logger.warning(f"Gap analysis failed: {e}")
    return {"gaps": [], "future_directions": []}


async def write_review(
    router,
    analyses: list[PaperAnalysis],
    themes: list[dict],
    gaps: dict,
    research_question: str,
    review_type: str = "narrative",
    language: str = "en",
    graph_context: dict | None = None,
    user_instructions: str = "",
) -> ReviewOutput:
    summaries_text = format_summaries_for_prompt(
        [a.model_dump() for a in analyses], max_chars=20000
    )
    themes_text = json.dumps(themes, indent=2, ensure_ascii=False)[:5000]
    gaps_text = json.dumps(gaps, indent=2, ensure_ascii=False)[:2000]
    checklist = build_citation_checklist([a.model_dump() for a in analyses])

    target_words = max(1000, len(analyses) * 130)
    if language == "zh":
        target_words = int(target_words * 1.5)

    lang_instruction = get_language_instruction(language)

    if graph_context:
        kg_context = json.dumps(graph_context.get("communities", []), indent=2, ensure_ascii=False)[:5000]
        global_summary = graph_context.get("global_summary", "")
        prompt = GRAPHRAG_LITERATURE_REVIEW_PROMPT.format(
            review_type=review_type,
            research_question=research_question,
            knowledge_graph_context=kg_context,
            global_summary=global_summary,
            num_papers=len(analyses),
            paper_summaries=summaries_text,
            themes=themes_text,
            gaps=gaps_text,
            word_count=target_words,
            citation_checklist=checklist,
        )
    else:
        prompt = LITERATURE_REVIEW_PROMPT.format(
            review_type=review_type,
            research_question=research_question,
            num_papers=len(analyses),
            paper_summaries=summaries_text,
            themes=themes_text,
            gaps=gaps_text,
            word_count=target_words,
            citation_checklist=checklist,
        )

    prompt += lang_instruction

    if user_instructions:
        prompt += f"\n\nADDITIONAL USER INSTRUCTIONS: {user_instructions}"

    review_text = await router.call(_msg(prompt), task_type="synthesis", max_tokens=8000)

    parsed_themes = [
        Theme(
            name=t.get("theme", ""),
            description=t.get("description", ""),
            paper_ids=t.get("paper_ids", []),
        )
        for t in themes
    ]

    return ReviewOutput(
        text=review_text,
        citations=[],
        themes=parsed_themes,
        word_count=_count_words(review_text),
        language=language,
    )


async def synthesize(
    router,
    analyses: list[PaperAnalysis],
    research_question: str,
    review_type: str = "narrative",
    language: str = "en",
    graph_context: dict | None = None,
    user_instructions: str = "",
) -> ReviewOutput:
    themes = await identify_themes(router, analyses, research_question)
    gaps = await identify_gaps(router, analyses, themes, research_question)

    review = await write_review(
        router, analyses, themes, gaps,
        research_question, review_type, language, graph_context,
        user_instructions=user_instructions,
    )

    logger.info(f"Synthesis complete: {review.word_count} words, {len(review.themes)} themes")
    return review
