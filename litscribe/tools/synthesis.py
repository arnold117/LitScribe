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
    REVIEW_CONCLUSION_PROMPT,
    REVIEW_INTRO_PROMPT,
    REVIEW_THEME_SECTION_PROMPT,
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
            # Cap themes based on paper count: at least 3 papers per theme
            max_themes = max(2, len(analyses) // 3)
            if len(themes) > max_themes:
                logger.info(f"Capping themes from {len(themes)} to {max_themes} (for {len(analyses)} papers)")
                themes = themes[:max_themes]
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
            return {
                "gaps": gaps.get("gaps", [])[:5],
                "future_directions": gaps.get("future_directions", [])[:5],
            }
        if isinstance(gaps, list):
            return {"gaps": gaps[:5], "future_directions": []}
    except Exception as e:
        logger.warning(f"Gap analysis JSON failed: {e}, trying plain text")
        try:
            raw = await router.call(_msg(prompt), task_type="synthesis", max_tokens=800)
            lines = [l.strip("- ").strip() for l in raw.strip().split("\n") if l.strip()]
            return {"gaps": lines[:5], "future_directions": []}
        except Exception:
            pass
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
    papers: list | None = None,
) -> ReviewOutput:
    enriched = _enrich_analyses_with_papers(analyses, papers)
    summaries_text = format_summaries_for_prompt(enriched, max_chars=20000)
    themes_text = json.dumps(themes, indent=2, ensure_ascii=False)[:5000]
    gaps_text = json.dumps(gaps, indent=2, ensure_ascii=False)[:2000]
    checklist = build_citation_checklist(enriched)

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


def _enrich_analyses_with_papers(
    analyses: list[PaperAnalysis],
    papers: list | None,
) -> list[dict]:
    paper_map = {}
    if papers:
        for p in papers:
            obj = p.model_dump() if hasattr(p, "model_dump") else p
            paper_map[obj.get("paper_id", "")] = obj

    enriched = []
    for a in analyses:
        d = a.model_dump()
        paper = paper_map.get(a.paper_id, {})
        d.setdefault("title", paper.get("title", a.paper_id))
        d.setdefault("authors", paper.get("authors", []))
        d.setdefault("year", paper.get("year", "N/A"))
        d.setdefault("venue", paper.get("venue", ""))
        enriched.append(d)
    return enriched


async def write_review_sectioned(
    router,
    analyses: list[PaperAnalysis],
    themes: list[dict],
    gaps: dict,
    research_question: str,
    review_type: str = "narrative",
    language: str = "en",
    user_instructions: str = "",
    papers: list | None = None,
) -> ReviewOutput:
    enriched = _enrich_analyses_with_papers(analyses, papers)
    lang_instruction = get_language_instruction(language)
    num_themes = max(len(themes), 1)

    target_words = max(1000, len(analyses) * 150)
    if language == "zh":
        target_words = int(target_words * 1.5)

    intro_words = int(target_words * 0.10)
    body_per_theme = int(target_words * 0.70) // num_themes
    conclusion_words = int(target_words * 0.20)

    theme_names_text = "\n".join(
        f"{i+1}. {t.get('theme', '')}: {t.get('description', '')[:100]}"
        for i, t in enumerate(themes)
    )

    sections = []

    logger.info(f"Sectioned review: {num_themes} themes, ~{target_words} words target")

    intro_summaries = format_summaries_for_prompt(enriched, max_chars=8000)
    intro_prompt = REVIEW_INTRO_PROMPT.format(
        review_type=review_type,
        research_question=research_question,
        num_papers=len(analyses),
        knowledge_graph_section="",
        theme_names=theme_names_text,
        paper_summaries=intro_summaries,
        word_count=intro_words,
    ) + lang_instruction
    if user_instructions:
        intro_prompt += f"\n\nADDITIONAL USER INSTRUCTIONS: {user_instructions}"

    try:
        intro = await router.call(_msg(intro_prompt), task_type="synthesis", max_tokens=4000)
        sections.append(intro.strip())
    except Exception as e:
        logger.warning(f"Sectioned intro failed: {e}")
        sections.append(f"# Literature Review: {research_question}\n\nThis review synthesizes {len(analyses)} papers.")

    previous_ending = sections[-1][-200:]

    for i, theme in enumerate(themes):
        theme_paper_ids = set(theme.get("paper_ids", []))
        theme_enriched = [e for e in enriched if e.get("paper_id") in theme_paper_ids]
        if not theme_enriched:
            start = (i * len(enriched)) // num_themes
            end = ((i + 1) * len(enriched)) // num_themes
            theme_enriched = enriched[start:end] if start < end else enriched[:3]

        theme_summaries = format_summaries_for_prompt(theme_enriched)
        theme_key_list = "\n".join(
            f"- `@{e.get('cite_key', '?')}` → {e.get('title', '')[:60]}"
            for e in theme_enriched if e.get("cite_key")
        )
        theme_checklist = theme_key_list or build_citation_checklist(theme_enriched)
        key_points = "\n".join(f"- {p}" for p in theme.get("key_points", [])[:5]) or "N/A"

        theme_prompt = REVIEW_THEME_SECTION_PROMPT.format(
            research_question=research_question,
            knowledge_graph_section="",
            theme_number=i + 1,
            total_themes=num_themes,
            theme_name=theme.get("theme", f"Theme {i+1}"),
            theme_description=theme.get("description", ""),
            key_points=key_points,
            num_theme_papers=len(theme_enriched),
            theme_papers=theme_summaries,
            theme_citation_checklist=theme_checklist,
            previous_ending=previous_ending,
            word_count=body_per_theme,
        ) + lang_instruction

        try:
            theme_text = await router.call(_msg(theme_prompt), task_type="synthesis", max_tokens=6000)
            sections.append(theme_text.strip())
            previous_ending = theme_text.strip()[-200:]
        except Exception as e:
            logger.warning(f"Sectioned theme '{theme.get('theme')}' failed: {e}")
            sections.append(f"## {i+1}. {theme.get('theme', '')}\n\n{theme.get('description', '')}")
            previous_ending = sections[-1][-200:]

    gaps_text = "\n".join(f"- {g}" for g in gaps.get("gaps", []))
    conclusion_prompt = REVIEW_CONCLUSION_PROMPT.format(
        research_question=research_question,
        num_papers=len(analyses),
        knowledge_graph_section="",
        theme_names=theme_names_text,
        gaps=gaps_text,
        previous_ending=previous_ending,
        uncited_section="",
        word_count=conclusion_words,
    ) + lang_instruction

    try:
        conclusion = await router.call(_msg(conclusion_prompt), task_type="synthesis", max_tokens=4000)
        sections.append(conclusion.strip())
    except Exception as e:
        logger.warning(f"Sectioned conclusion failed: {e}")
        sections.append("## Conclusion\n\nFurther research is needed.")

    full_text = "\n\n".join(sections)

    parsed_themes = [
        Theme(name=t.get("theme", ""), description=t.get("description", ""), paper_ids=t.get("paper_ids", []))
        for t in themes
    ]

    return ReviewOutput(
        text=full_text,
        citations=[],
        themes=parsed_themes,
        word_count=_count_words(full_text),
        language=language,
    )


async def _write_section(router, prompt: str) -> str:
    try:
        return await router.call(_msg(prompt), task_type="synthesis", max_tokens=4000)
    except Exception as e:
        logger.warning(f"Section generation failed: {e}")
        return ""


async def synthesize_parallel(
    router,
    analyses: list[PaperAnalysis],
    research_question: str,
    language: str = "en",
    user_instructions: str = "",
    papers: list | None = None,
) -> ReviewOutput:
    import asyncio

    enriched = _enrich_analyses_with_papers(analyses, papers)
    lang_instruction = get_language_instruction(language)
    num_papers = len(analyses)

    # Assign citation keys
    from litscribe.tools.cite_keys import assign_cite_keys, build_cite_key_table
    paper_list = papers if papers else []
    key_map = assign_cite_keys(paper_list) if paper_list else {}

    # Enrich with cite keys
    cite_counts = {p.paper_id: p.citations or 0 for p in paper_list} if paper_list else {}
    for e in enriched:
        pid = e.get("paper_id", "")
        if pid in key_map:
            e["cite_key"] = key_map[pid]
        cites = cite_counts.get(pid, 0)
        if cites > 0:
            e["impact_note"] = f"(cited {cites} times)"

    # Sort analyses by citation count (high-impact papers first)
    if papers:
        cite_map = {p.paper_id: p.citations or 0 for p in papers if hasattr(p, 'paper_id')}
        analyses = sorted(analyses, key=lambda a: cite_map.get(a.paper_id, 0), reverse=True)

    # Step 1: themes + gaps in parallel
    themes_coro = identify_themes(router, analyses, research_question)
    gaps_coro = identify_gaps(router, analyses, [], research_question)
    themes, gaps = await asyncio.gather(themes_coro, gaps_coro)

    max_themes = max(2, num_papers // 3)
    themes = themes[:max_themes]

    # Step 2: build section prompts
    summaries_text = format_summaries_for_prompt(enriched, max_chars=15000)
    cite_table = build_cite_key_table(paper_list, key_map) if key_map else ""
    checklist = cite_table or build_citation_checklist(enriched)
    theme_names = "\n".join(f"{i+1}. {t.get('theme','')}" for i, t in enumerate(themes))
    gaps_text = "\n".join(f"- {g}" for g in gaps.get("gaps", []))

    target_words_per_theme = max(200, 800 // max(len(themes), 1))
    intro_words = 150
    conclusion_words = 200

    user_suffix = f"\n\nADDITIONAL INSTRUCTIONS: {user_instructions}" if user_instructions else ""

    citation_rule = (
        "\n\nCITATION RULES (STRICT):\n"
        "1. Use Pandoc-style: [@key] for single, [@key1; @key2] for multiple\n"
        "2. Use EXACTLY the citation keys below — do NOT invent keys\n"
        "3. EVERY factual claim MUST cite at least one paper\n"
        "4. ONLY make claims that are supported by the paper's findings listed below\n"
        "5. Do NOT attribute findings to a paper unless that finding appears in the paper's summary\n"
        "6. If you're unsure whether a paper supports a claim, do NOT cite it\n"
        f"\nAvailable papers and their findings:\n{cite_table}"
    ) if cite_table else (
        "\n\nCITATION FORMAT: Use [LastName et al., Year] for every factual claim."
    )

    # Intro prompt
    intro_prompt = (
        f"Write a concise introduction (2-3 paragraphs, ~{intro_words} words) for a literature review.\n\n"
        f"Research Question: {research_question}\n"
        f"Number of papers: {num_papers}\n"
        f"Themes: {theme_names}\n\n"
        f"Papers:\n{summaries_text[:5000]}\n\n"
        f"Start with a # heading."
        f"{citation_rule}{lang_instruction}{user_suffix}"
    )

    # Theme section prompts (one per theme)
    theme_prompts = []
    for i, theme in enumerate(themes):
        theme_paper_ids = set(theme.get("paper_ids", []))
        theme_enriched = [e for e in enriched if e.get("paper_id") in theme_paper_ids]
        if not theme_enriched:
            start = (i * len(enriched)) // max(len(themes), 1)
            end = ((i + 1) * len(enriched)) // max(len(themes), 1)
            theme_enriched = enriched[start:end] if start < end else enriched[:3]

        theme_summaries = format_summaries_for_prompt(theme_enriched)
        theme_key_list = "\n".join(
            f"- `@{e.get('cite_key', '?')}` → {e.get('title', '')[:60]}"
            for e in theme_enriched if e.get("cite_key")
        )
        theme_checklist = theme_key_list or build_citation_checklist(theme_enriched)

        other_themes = [t.get("theme", "") for j, t in enumerate(themes) if j != i]
        context = f"Other themes in this review: {', '.join(other_themes)}. " if other_themes else ""

        p = (
            f"Write a theme section (~{target_words_per_theme} words) for a literature review.\n\n"
            f"Research Question: {research_question}\n"
            f"Theme: {theme.get('theme', '')}\n"
            f"Description: {theme.get('description', '')}\n"
            f"{context}Avoid overlapping with other themes.\n\n"
            f"Papers for this theme:\n{theme_summaries}\n\n"
            f"Full paper list for cross-referencing:\n{checklist}\n\n"
            f"Start with ## {theme.get('theme', '')}. Synthesize across papers — compare and contrast approaches, "
            f"highlight agreements and disagreements."
            f"{citation_rule}{lang_instruction}{user_suffix}"
        )
        theme_prompts.append(p)

    # Conclusion prompt
    conclusion_prompt = (
        f"Write conclusion sections (~{conclusion_words} words) for a literature review.\n\n"
        f"Research Question: {research_question}\n"
        f"Themes covered: {theme_names}\n"
        f"Research gaps: {gaps_text}\n\n"
        f"Write: ## Research Gaps and Future Directions (1 paragraph), then ## Conclusion (1 paragraph)."
        f"{citation_rule}{lang_instruction}"
    )

    # Step 3: generate ALL sections in parallel
    logger.info(f"Parallel synthesis: intro + {len(theme_prompts)} themes + conclusion")
    all_prompts = [intro_prompt] + theme_prompts + [conclusion_prompt]
    all_sections = await asyncio.gather(*[_write_section(router, p) for p in all_prompts])

    # Step 4: assemble
    sections = [s.strip() for s in all_sections if s.strip()]
    full_text = "\n\n".join(sections)

    parsed_themes = [
        Theme(name=t.get("theme", ""), description=t.get("description", ""), paper_ids=t.get("paper_ids", []))
        for t in themes
    ]

    return ReviewOutput(
        text=full_text,
        citations=[],
        themes=parsed_themes,
        word_count=_count_words(full_text),
        language=language,
    )


async def synthesize(
    router=None,
    analyses: list[PaperAnalysis] = None,
    research_question: str = "",
    review_type: str = "narrative",
    language: str = "en",
    graph_context: dict | None = None,
    user_instructions: str = "",
    papers: list | None = None,
    model=None,
) -> ReviewOutput:
    if router is None and model is not None:
        from litscribe.llm.adapter import ModelAdapter
        router = ModelAdapter(model)

    review = await synthesize_parallel(
        router, analyses, research_question, language,
        user_instructions=user_instructions, papers=papers,
    )

    logger.info(f"Synthesis complete: {review.word_count} words, {len(review.themes)} themes")
    return review
