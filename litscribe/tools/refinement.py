from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from litscribe.models.paper import Paper
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


async def _search_for_new_content(
    model: ChatOpenAI,
    topic: str,
    config,
    max_papers: int = 5,
) -> tuple[list[Paper], str]:
    from litscribe.tools.search import search_all_sources
    from litscribe.tools.cite_keys import assign_cite_keys
    from litscribe.models.analysis import PaperAnalysis
    from litscribe.prompts.reading import ABSTRACT_ONLY_ANALYSIS_PROMPT

    logger.info(f"  Searching for new papers on: {topic}")
    papers = await search_all_sources([topic], config, max_per_source=max_papers, domain="")
    papers = papers[:max_papers]

    if not papers:
        return [], ""

    key_map = assign_cite_keys(papers)

    # Quick analysis of each paper
    analyses = []
    for p in papers:
        prompt = ABSTRACT_ONLY_ANALYSIS_PROMPT.format(
            research_question=topic,
            title=p.title,
            authors=", ".join(p.authors[:3]) if p.authors else "Unknown",
            year=p.year or "N/A",
            venue=p.venue or "",
            abstract=p.abstract or "(no abstract)",
            metadata_section=f"Citations: {p.citations or 0}",
        )
        try:
            raw = await model.ainvoke(prompt)
            content = raw.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```\w*\n?", "", content)
                content = re.sub(r"\n?```$", "", content)
            data = json.loads(content)
            findings = data.get("key_findings", [])
        except Exception:
            findings = [f"Abstract: {(p.abstract or '')[:200]}"]

        key = key_map.get(p.paper_id, "unknown")
        authors = ", ".join(p.authors[:2]) if p.authors else "Unknown"
        if len(p.authors) > 2:
            authors += " et al."
        analyses.append(
            f"[@{key}] {authors} ({p.year}). {p.title}\n"
            f"  Findings: {'; '.join(findings[:3])}"
        )

    context = "\n\n".join(analyses)

    # Build reference entries for new papers
    ref_entries = []
    for p in papers:
        key = key_map.get(p.paper_id, "unknown")
        authors = ", ".join(p.authors[:3]) if p.authors else "Unknown"
        if len(p.authors) > 3:
            authors += " et al."
        venue = f" *{p.venue}*." if p.venue else ""
        ref_entries.append(f"[{key}]: {authors} ({p.year}). {p.title}.{venue}")

    refs_text = "\n".join(ref_entries)
    return papers, context, refs_text, key_map


async def execute_refinement(
    model: ChatOpenAI,
    instruction: dict,
    current_review: str,
    research_question: str,
    papers_context: str = "",
    language: str = "en",
    new_papers_context: str = "",
) -> str:
    from litscribe.prompts.utils import get_language_instruction
    lang = get_language_instruction(language)

    extra = ""
    if new_papers_context:
        extra = (
            f"\n\n## NEW PAPERS for the added section (use [@key] citations):\n"
            f"{new_papers_context}\n\n"
            f"IMPORTANT:\n"
            f"- Start the new section with a ## heading (e.g., ## Delivery Methods)\n"
            f"- The new section MUST cite these new papers using their [@key]\n"
            f"- Do NOT make claims without citing a paper from this list\n"
            f"- Place the new section in a logical position in the review"
        )

    prompt = REFINEMENT_EXECUTE_PROMPT.format(
        research_question=research_question,
        current_review=current_review,
        papers_context=papers_context[:5000] + extra,
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
    config=None,
) -> ReviewOutput:
    logger.info(f"Refining review: {instruction_text[:50]}")

    classified = await classify_instruction(
        model, instruction_text, current_review.text
    )
    action = classified.get("action_type", "modify_content")
    logger.info(f"  Action: {action}, target: {classified.get('target_section')}")

    new_papers_ctx = ""
    new_refs = ""

    # For add_content: search for new papers first
    if action == "add_content" and config:
        topic = classified.get("details", instruction_text)
        papers, new_papers_ctx, new_refs, _ = await _search_for_new_content(
            model, topic, config, max_papers=5,
        )
        logger.info(f"  Found {len(papers)} new papers for '{topic[:30]}'")

    new_text = await execute_refinement(
        model, classified, current_review.text,
        research_question, papers_context, language,
        new_papers_context=new_papers_ctx,
    )

    # Append new references if any
    if new_refs:
        if "## References" in new_text:
            new_text = new_text.rstrip() + "\n" + new_refs
        else:
            new_text += f"\n\n## References\n{new_refs}"

    return ReviewOutput(
        text=new_text,
        citations=current_review.citations,
        themes=current_review.themes,
        word_count=_count_words(new_text),
        language=language,
    )
