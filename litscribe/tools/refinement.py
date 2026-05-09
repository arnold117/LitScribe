from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from litscribe.models.review import ReviewOutput
from litscribe.prompts.refinement import REFINEMENT_CLASSIFY_PROMPT, REFINEMENT_EXECUTE_PROMPT

logger = logging.getLogger(__name__)


async def classify_instruction(
    model: ChatOpenAI,
    instruction: str,
    review_text: str,
) -> dict:
    prompt = REFINEMENT_CLASSIFY_PROMPT.format(
        instruction=instruction,
        review_excerpt=review_text[:500],
    )
    result = await model.ainvoke(prompt)
    raw = result.content.strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "action_type": "modify_content",
            "target_section": None,
            "details": instruction,
        }


async def execute_refinement(
    model: ChatOpenAI,
    instruction: dict,
    current_review: str,
    research_question: str,
    papers_context: str = "",
    language: str = "en",
) -> str:
    from litscribe.prompts.utils import get_language_instruction
    lang = get_language_instruction(language)

    prompt = REFINEMENT_EXECUTE_PROMPT.format(
        research_question=research_question,
        current_review=current_review,
        papers_context=papers_context[:5000],
        action_type=instruction.get("action_type", "modify_content"),
        target_section=instruction.get("target_section", "entire review"),
        details=instruction.get("details", ""),
    ) + lang

    result = await model.ainvoke(prompt)
    return result.content.strip()


def _count_words(text: str) -> int:
    cjk = len(re.findall(r'[一-鿿㐀-䶿]', text))
    latin = len(re.findall(r'[a-zA-Z]+', text))
    return cjk + latin


async def refine_review(
    model: ChatOpenAI,
    current_review: ReviewOutput,
    instruction_text: str,
    research_question: str,
    papers_context: str = "",
    language: str = "en",
) -> ReviewOutput:
    logger.info(f"Refining review: {instruction_text[:50]}")

    classified = await classify_instruction(
        model, instruction_text, current_review.text
    )
    logger.info(f"  Action: {classified.get('action_type')}, target: {classified.get('target_section')}")

    new_text = await execute_refinement(
        model, classified, current_review.text,
        research_question, papers_context, language,
    )

    return ReviewOutput(
        text=new_text,
        citations=current_review.citations,
        themes=current_review.themes,
        word_count=_count_words(new_text),
        language=language,
    )
