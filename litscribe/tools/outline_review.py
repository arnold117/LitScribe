from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

from langchain_openai import ChatOpenAI

from litscribe.config import Config
from litscribe.models.paper import Paper
from litscribe.models.review import ReviewOutput
from litscribe.tools.outline_parser import OutlineNode, outline_to_sections, parse_outline
from litscribe.tools.pipeline import (
    _call_llm,
    step_plan,
    step_read,
    step_search,
    step_synthesize,
)
from litscribe.tools.status import PipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search query builder
# ---------------------------------------------------------------------------

def _build_search_question(section: dict, constraints: str = "") -> str:
    title = section["title"]
    path = section["path"]

    generic_keywords = (
        "存在问题", "发展趋势", "研究局限", "关键瓶颈", "产业化前景",
        "规范化发展", "挑战", "展望", "小结", "总结",
        "research gaps", "limitations", "challenges", "future",
    )
    is_generic = any(kw in title.lower() for kw in generic_keywords)

    if is_generic and len(path) >= 2:
        query = f"{path[-2]}: {title}"
    elif len(path) >= 3:
        query = f"{path[0]} {title}"
    else:
        query = title

    if constraints:
        query = f"{constraints} {query}"

    return query


# ---------------------------------------------------------------------------
# Section role classification (structural vs topical)
# ---------------------------------------------------------------------------

# Unambiguous section names only — avoid generic words like 背景/概述 that may
# legitimately appear in body section titles.
_INTRO_KEYS = ("引言", "前言", "绪论", "introduction", "preface")
_CONCL_KEYS = (
    "结论", "结语", "总结", "小结", "展望",
    "conclusion", "concluding", "summary", "future direction",
    "future work", "outlook",
)


def _section_role(title: str) -> str:
    """Classify a section as 'intro', 'conclusion', or 'body'.

    Intro/conclusion are *structural* sections written from the body, not by
    searching papers for their title (which yields meta-commentary about what
    an introduction is, rather than an introduction to the actual topic).
    """
    t = title.strip().lower()
    is_short = len(title.strip()) <= 14
    if any(k in t for k in _CONCL_KEYS) and (is_short or "conclusion" in t or "future" in t):
        return "conclusion"
    if any(k in t for k in _INTRO_KEYS) and (is_short or "introduction" in t):
        return "intro"
    return "body"


def _build_body_digest(results: list[dict], per_section_chars: int = 700) -> str:
    """Condensed view of generated body sections, to ground intro/conclusion."""
    parts = []
    for r in results:
        txt = (r.get("text") or "").strip()
        if not txt or txt.startswith("["):
            continue
        snippet = txt[:per_section_chars]
        parts.append(f"## {r['title']}\n{snippet}")
    return "\n\n".join(parts)


def _derive_topic(body_sections: list[dict], constraints: str = "") -> str:
    titles = "、".join(s["title"] for s in body_sections) if body_sections else ""
    return f"{constraints} {titles}".strip() if constraints else titles


async def _generate_structural_section(
    model, role: str, topic: str, body_digest: str,
    prior_text: str, language: str, constraints: str,
) -> str:
    """Write an intro or conclusion from the assembled body (no paper search)."""
    if language == "zh":
        cons = f"全局约束：所有内容须紧扣「{constraints}」。\n" if constraints else ""
        if role == "intro":
            prompt = (
                f"你正在为一篇文献综述撰写【引言】。\n"
                f"综述主题涵盖：{topic}\n{cons}"
                f"以下是正文各章节的内容摘要：\n{body_digest}\n\n"
                f"请基于上述正文撰写引言：\n"
                f"- 交代该研究主题的背景与重要性，引出本综述要梳理的问题\n"
                f"- 简要概述综述的结构与各部分内容\n"
                f"- 直接进入该具体主题，严禁写成“什么是引言/引言的作用/修辞空间”这类元论述\n"
                f"- 只输出正文段落，不要任何标题行，不要参考文献列表\n"
                f"- 全文使用中文，与正文风格一致"
            )
        else:
            prior = f"\n引言部分内容如下，请与之呼应、避免重复：\n{prior_text}\n" if prior_text else ""
            prompt = (
                f"你正在为一篇文献综述撰写【结论与展望】。\n"
                f"综述主题涵盖：{topic}\n{cons}"
                f"以下是正文各章节的内容摘要：\n{body_digest}\n{prior}\n"
                f"请基于上述内容撰写结论：\n"
                f"- 总结贯穿各章节的主要发现与共识\n"
                f"- 指出当前研究的空白、争议与局限\n"
                f"- 提出未来研究方向，并与引言形成首尾呼应\n"
                f"- 只输出正文段落，不要任何标题行，不要参考文献列表\n"
                f"- 全文使用中文，与正文风格一致"
            )
    else:
        cons = f"Global constraint: keep everything focused on '{constraints}'.\n" if constraints else ""
        if role == "intro":
            prompt = (
                f"You are writing the INTRODUCTION of a literature review.\n"
                f"The review covers: {topic}\n{cons}"
                f"Here are summaries of the body sections:\n{body_digest}\n\n"
                f"Write the introduction grounded in the body above:\n"
                f"- Establish the background and importance of this specific topic and the questions the review addresses\n"
                f"- Briefly outline the structure of the review\n"
                f"- Go straight into the actual subject; do NOT write meta-commentary about what an introduction is\n"
                f"- Output prose paragraphs only — no heading lines, no reference list\n"
                f"- Write in English, consistent with the body"
            )
        else:
            prior = f"\nThe introduction reads as follows; echo it and avoid repetition:\n{prior_text}\n" if prior_text else ""
            prompt = (
                f"You are writing the CONCLUSION of a literature review.\n"
                f"The review covers: {topic}\n{cons}"
                f"Here are summaries of the body sections:\n{body_digest}\n{prior}\n"
                f"Write the conclusion grounded in the body above:\n"
                f"- Synthesize the main findings and consensus across sections\n"
                f"- Identify gaps, controversies, and limitations\n"
                f"- Propose future directions, mirroring the introduction\n"
                f"- Output prose paragraphs only — no heading lines, no reference list\n"
                f"- Write in English, consistent with the body"
            )
    result = await model.ainvoke(prompt)
    return _postprocess_section(result.content.strip(), role)


# ---------------------------------------------------------------------------
# Section filtering
# ---------------------------------------------------------------------------

def _filter_sections(
    sections: list[dict], section_filter: str | None,
) -> list[dict]:
    if not section_filter:
        return sections

    filters = [f.strip() for f in section_filter.split(",")]
    selected = []
    for s in sections:
        num = s.get("number", "")
        title = s["title"]
        for f in filters:
            if num and (num == f or num.startswith(f + ".")):
                selected.append(s)
                break
            elif f.lower() in title.lower():
                selected.append(s)
                break
    return selected


# ---------------------------------------------------------------------------
# Coverage tracking
# ---------------------------------------------------------------------------

def track_coverage(
    text: str, entities: list[str],
) -> dict:
    results = {}
    text_lower = text.lower()
    for entity in entities:
        count = text_lower.count(entity.lower())
        results[entity] = count

    covered = [e for e, c in results.items() if c > 0]
    missing = [e for e, c in results.items() if c == 0]

    return {
        "total": len(entities),
        "covered": len(covered),
        "missing_count": len(missing),
        "coverage_pct": round(len(covered) / len(entities) * 100) if entities else 0,
        "covered_entities": covered,
        "missing_entities": missing,
        "counts": results,
    }


def track_coverage_by_section(
    section_results: list[dict], entities: list[str],
) -> list[dict]:
    per_section = []
    for s in section_results:
        cov = track_coverage(s.get("text", ""), entities)
        per_section.append({
            "title": s["title"],
            "number": s.get("number", ""),
            "covered": cov["covered"],
            "missing": cov["missing_count"],
            "missing_entities": cov["missing_entities"],
        })
    return per_section


# ---------------------------------------------------------------------------
# Cross-section consistency check
# ---------------------------------------------------------------------------

async def check_cross_section_consistency(
    model: ChatOpenAI,
    section_results: list[dict],
    entities: list[str],
) -> list[dict]:
    entity_mentions: dict[str, list[dict]] = {}
    for entity in entities:
        mentions = []
        for s in section_results:
            text = s.get("text", "")
            if entity.lower() in text.lower():
                # Extract sentences containing the entity
                sentences = re.split(r'[。.!？]', text)
                relevant = [
                    sent.strip()
                    for sent in sentences
                    if entity.lower() in sent.lower() and len(sent.strip()) > 20
                ]
                if relevant:
                    mentions.append({
                        "section": s["title"],
                        "number": s.get("number", ""),
                        "excerpts": relevant[:3],
                    })
        if len(mentions) >= 2:
            entity_mentions[entity] = mentions

    if not entity_mentions:
        return []

    issues = []
    for entity, mentions in entity_mentions.items():
        excerpts_text = ""
        for m in mentions:
            excerpts_text += f"\n[{m['number']} {m['section']}]:\n"
            excerpts_text += "\n".join(f"  - {e[:200]}" for e in m["excerpts"])

        prompt = (
            f"Check these excerpts about '{entity}' from different sections of a literature review. "
            f"Are there any contradictions, inconsistencies, or conflicting claims?\n"
            f"{excerpts_text}\n\n"
            f"Reply in JSON: {{\"consistent\": true/false, \"issue\": \"description if inconsistent\", "
            f"\"suggestion\": \"how to fix if inconsistent\"}}"
        )

        try:
            from litscribe.tools.pipeline import _call_llm_json
            result = await _call_llm_json(model, prompt)
            if isinstance(result, dict) and not result.get("consistent", True):
                issues.append({
                    "entity": entity,
                    "sections": [m["number"] or m["section"] for m in mentions],
                    "issue": result.get("issue", ""),
                    "suggestion": result.get("suggestion", ""),
                })
        except Exception as e:
            logger.debug(f"Consistency check failed for {entity}: {e}")

    return issues


# ---------------------------------------------------------------------------
# Main outline review pipeline
# ---------------------------------------------------------------------------

async def run_outline_review(
    model: ChatOpenAI,
    config: Config,
    outline_path: str,
    max_papers_per_section: int = 10,
    language: str = "en",
    constraints: str = "",
    section_filter: str | None = None,
    on_progress: Callable[[str, dict], None] | None = None,
) -> dict:
    t0 = time.time()

    def emit(event: str, data: dict):
        if on_progress:
            on_progress(event, data)

    roots = parse_outline(outline_path)
    all_sections = outline_to_sections(roots)

    if not all_sections:
        return {"error": "No sections found in outline"}

    sections = _filter_sections(all_sections, section_filter)
    if not sections:
        return {"error": f"No sections matched filter: {section_filter}"}

    emit("outline_parsed", {
        "total_sections": len(all_sections),
        "selected_sections": len(sections),
        "titles": [s["title"] for s in sections],
        "constraints": constraints or None,
    })

    all_papers: dict[str, Paper] = {}
    section_results: list[dict] = []

    # Generate topical body sections first (search-anchored), then write the
    # structural intro/conclusion from the assembled body — intro/conclusion
    # depend on the body, and searching for "引言"/"Introduction" yields
    # meta-commentary instead of a real section.
    body_sections = [s for s in sections if _section_role(s["title"]) == "body"]
    intro_sections = [s for s in sections if _section_role(s["title"]) == "intro"]
    conclusion_sections = [s for s in sections if _section_role(s["title"]) == "conclusion"]
    ordered = body_sections + intro_sections + conclusion_sections
    total = len(ordered)

    body_digest: str | None = None
    topic = _derive_topic(body_sections, constraints)
    intro_text = ""

    for i, section in enumerate(ordered):
        section_title = section["title"]
        context_path = " > ".join(section["path"])
        role = _section_role(section_title)

        emit("section_start", {
            "index": i,
            "total": total,
            "title": section_title,
            "path": context_path,
        })

        try:
            if role == "body":
                search_question = _build_search_question(section, constraints)
                state = PipelineState(
                    research_question=search_question,
                    language=language,
                )
                await step_plan(model, state)

                emit("section_search", {"index": i, "title": section_title})
                await step_search(model, state, config, max_papers=max_papers_per_section)

                if len(state.papers) < 2:
                    broader_q = f"{constraints} {context_path}" if constraints else context_path
                    broader_state = PipelineState(
                        research_question=broader_q,
                        language=language,
                    )
                    await step_plan(model, broader_state)
                    await step_search(model, broader_state, config, max_papers=max_papers_per_section)
                    state.papers.extend(broader_state.papers)
                    seen_ids = set()
                    deduped = []
                    for p in state.papers:
                        if p.paper_id not in seen_ids:
                            seen_ids.add(p.paper_id)
                            deduped.append(p)
                    state.papers = deduped

                emit("section_read", {"index": i, "papers": len(state.papers)})
                await step_read(model, state)

                constraint_instruction = ""
                if constraints:
                    constraint_instruction = (
                        f"- IMPORTANT CONSTRAINT: Focus specifically on {constraints}. "
                        f"All discussion must be directly relevant to these subjects.\n"
                    )

                instructions = (
                    f"You are writing one section of a larger literature review document.\n"
                    f"Document context: {context_path}\n"
                    f"Current section: {section_title}\n\n"
                    f"STRICT RULES:\n"
                    f"{constraint_instruction}"
                    f"- Write ONLY the review content for '{section_title}'. "
                    f"Do NOT include any title/heading lines (no # or ## lines).\n"
                    f"- Do NOT include 'References', '参考文献', '研究空白', '结论' or 'Conclusion' sections.\n"
                    f"- Do NOT add introductory or concluding paragraphs — just the substantive review.\n"
                    f"- All content must be directly relevant to the topic '{section_title}' "
                    f"within the context of '{context_path}'. Do NOT discuss unrelated fields.\n"
                    f"- Use consistent citation format: (Author et al., Year) in running text.\n"
                    f"- Write in {'Chinese' if language == 'zh' else 'English'}."
                )

                emit("section_synthesize", {"index": i, "title": section_title})
                await step_synthesize(model, state, instructions)

                text = state.synthesis.text if state.synthesis else ""
                text = _postprocess_section(text, section_title)
                papers_count = len(state.papers)

                for p in state.papers:
                    all_papers[p.paper_id] = p
            else:
                # Structural section: synthesize from the body, no paper search.
                if body_digest is None:
                    body_digest = _build_body_digest(section_results)

                emit("section_synthesize", {"index": i, "title": section_title})
                text = await _generate_structural_section(
                    model, role, topic, body_digest, intro_text, language, constraints,
                )
                if role == "intro":
                    intro_text = text
                papers_count = 0

            section_results.append({
                "title": section_title,
                "number": section.get("number", ""),
                "level": section["level"],
                "path": section["path"],
                "text": text,
                "papers_count": papers_count,
                "word_count": len(text.split()),
            })

            emit("section_done", {
                "index": i,
                "title": section_title,
                "papers": papers_count,
                "words": len(text.split()),
            })

        except Exception as e:
            logger.warning(f"Section '{section_title}' failed: {e}")
            section_results.append({
                "title": section_title,
                "number": section.get("number", ""),
                "level": section["level"],
                "path": section["path"],
                "text": f"[Section generation failed: {e}]",
                "papers_count": 0,
                "word_count": 0,
            })

    emit("assembling", {"sections": len(section_results)})
    full_text = _assemble_document(roots, section_results, list(all_papers.values()))
    full_text = _postprocess_document(full_text)
    total_words = len(full_text.split())
    elapsed = time.time() - t0

    # Coverage tracking
    coverage = None
    constraint_entities = _parse_constraint_entities(constraints)
    if constraint_entities:
        coverage = track_coverage(full_text, constraint_entities)
        section_coverage = track_coverage_by_section(section_results, constraint_entities)
        coverage["by_section"] = section_coverage
        emit("coverage", coverage)

    emit("complete", {
        "total_words": total_words,
        "total_papers": len(all_papers),
        "total_sections": len(section_results),
        "time": round(elapsed, 1),
        "coverage": coverage,
    })

    return {
        "text": full_text,
        "sections": section_results,
        "total_words": total_words,
        "total_papers": len(all_papers),
        "time": round(elapsed, 1),
        "coverage": coverage,
    }


def _parse_constraint_entities(constraints: str) -> list[str]:
    if not constraints:
        return []
    entities = []
    for part in re.split(r'[,;，；]', constraints):
        part = part.strip()
        latin = re.findall(r'[A-Z][a-z]+ [a-z]+', part)
        entities.extend(latin)
        genus_sp = re.findall(r'F\.\s*[a-z]+', part)
        for gs in genus_sp:
            sp = gs.replace("F. ", "").replace("F.", "").strip()
            if sp:
                entities.append(f"F. {sp}")
                entities.append(f"Ferula {sp}")
    return list(set(entities)) if entities else []


# ---------------------------------------------------------------------------
# Targeted section patching
# ---------------------------------------------------------------------------

async def patch_section(
    model: ChatOpenAI,
    config: Config,
    existing_text: str,
    section_number: str,
    patch_instruction: str,
    constraints: str = "",
    language: str = "en",
    max_papers: int = 10,
) -> dict:
    lines = existing_text.split("\n")
    section_start = -1
    section_end = len(lines)
    section_level = 0
    section_title = ""

    for i, line in enumerate(lines):
        if line.strip().startswith("#") and section_number in line:
            section_start = i
            section_level = len(re.match(r'^(#+)', line.strip()).group(1))
            section_title = re.sub(r'^#+\s*[\d.]*\s*', '', line).strip()
            continue
        if section_start >= 0 and i > section_start:
            m = re.match(r'^(#+)\s', line.strip())
            if m and len(m.group(1)) <= section_level:
                section_end = i
                break

    if section_start < 0:
        return {"error": f"Section {section_number} not found"}

    search_q = f"{constraints} {section_title}" if constraints else section_title
    if patch_instruction:
        search_q += f" {patch_instruction}"

    state = PipelineState(research_question=search_q, language=language)
    await step_plan(model, state)
    await step_search(model, state, config, max_papers=max_papers)
    await step_read(model, state)

    constraint_instr = ""
    if constraints:
        constraint_instr = f"- Focus specifically on: {constraints}\n"

    instructions = (
        f"You are patching a section of a literature review.\n"
        f"Section: {section_title}\n"
        f"User instruction: {patch_instruction}\n\n"
        f"RULES:\n"
        f"{constraint_instr}"
        f"- Write ONLY the content paragraphs. No headings, no References.\n"
        f"- Use citation format: (Author et al., Year)\n"
        f"- Write in {'Chinese' if language == 'zh' else 'English'}."
    )

    await step_synthesize(model, state, instructions)
    new_text = _postprocess_section(
        state.synthesis.text if state.synthesis else "", section_title,
    )

    # Replace section content
    new_lines = lines[:section_start + 1] + ["", new_text, ""] + lines[section_end:]
    patched = "\n".join(new_lines)

    return {
        "text": patched,
        "section": section_number,
        "section_title": section_title,
        "papers_count": len(state.papers),
        "word_count": len(new_text.split()),
    }


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _postprocess_section(text: str, section_title: str) -> str:
    lines = text.split("\n")
    cleaned: list[str] = []
    skip_rest = False

    for line in lines:
        stripped = line.strip()

        if re.match(
            r"^#{1,4}\s*(研究空白|结论|结语|小结|总结|参考文献|References|Conclusion|Summary|"
            r"Research [Gg]aps?|Future [Dd]irections?|"
            r"Methodology Comparison|Research Timeline|Statistical Summary|"
            r"Suggested Figures|方法学比较|研究时间线|统计摘要|建议图表)",
            stripped,
        ):
            skip_rest = True
            continue
        if skip_rest:
            if re.match(r"^#{1,4}\s+", stripped) and not re.match(
                r"^#{1,4}\s*(研究空白|结论|结语|小结|总结|参考文献|References|Conclusion|Summary|"
                r"Methodology Comparison|Research Timeline|Statistical Summary|"
                r"Suggested Figures|方法学比较|研究时间线|统计摘要|建议图表|"
                r"Key Advances|Significance|Phase \d|How Later|Early|Development|Recent|Frontier|"
                r"Foundational)",
                stripped,
            ):
                skip_rest = False
            else:
                continue

        if stripped.startswith("#"):
            heading_text = re.sub(r"^#{1,6}\s*", "", stripped).strip()
            if heading_text == section_title or heading_text in section_title:
                continue
            if heading_text in ("引言", "Introduction", "引论", "概述"):
                continue

        cleaned.append(line)

    result = "\n".join(cleaned).strip()

    for marker in ("## References", "## 参考文献", "# References", "# 参考文献"):
        if marker in result:
            result = result[: result.index(marker)].rstrip()

    return result


def _postprocess_document(text: str) -> str:
    text = re.sub(r"\s*\[@[\w;,\s]+\]", "", text)

    lines = text.split("\n")
    cleaned: list[str] = []
    for i, line in enumerate(lines):
        if (
            line.strip() == ""
            and i > 0
            and i < len(lines) - 1
            and lines[i - 1].strip().startswith("#")
            and lines[i + 1].strip().startswith("#")
        ):
            continue
        cleaned.append(line)

    result: list[str] = []
    for line in cleaned:
        if line.strip() == "" and result and result[-1].strip() == "":
            continue
        result.append(line)

    return "\n".join(result)


def _assemble_document(
    roots: list[OutlineNode],
    section_results: list[dict],
    papers: list[Paper],
) -> str:
    result_map = {}
    for r in section_results:
        result_map[r["title"]] = r

    parts: list[str] = []

    def _render(node: OutlineNode, depth: int = 1):
        prefix = "#" * min(depth, 4)
        heading = f"{prefix} {node.number} {node.title}".strip() if node.number else f"{prefix} {node.title}"
        parts.append(heading)

        if node.is_leaf:
            result = result_map.get(node.title)
            if result and result["text"]:
                parts.append("")
                parts.append(result["text"])
            else:
                parts.append("")
                parts.append(f"*[Content pending for: {node.title}]*")
        else:
            for child in node.children:
                _render(child, depth + 1)

        parts.append("")

    for root in roots:
        _render(root)

    from litscribe.tools.cite_keys import assign_cite_keys
    key_map = assign_cite_keys(papers)

    parts.append("## References\n")
    for p in sorted(papers, key=lambda p: (p.authors[0] if p.authors else "ZZZ", p.year or 0)):
        key = key_map.get(p.paper_id, "unknown")
        authors = ", ".join(p.authors[:3]) if p.authors else "Unknown"
        if len(p.authors) > 3:
            authors += " et al."
        venue = f" *{p.venue}*." if p.venue else ""
        doi = f" doi:{p.doi}" if p.doi else ""
        parts.append(f"- [{key}] {authors} ({p.year}). {p.title}.{venue}{doi}")

    return "\n".join(parts)
