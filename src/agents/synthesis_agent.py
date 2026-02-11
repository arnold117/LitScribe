"""Synthesis Agent for LitScribe.

This agent is responsible for:
1. Theme identification - finding common themes across papers
2. Gap analysis - identifying research gaps and future directions
3. Literature review generation - writing the final review narrative
4. Citation formatting - generating proper citations

Phase 7.5 Enhancement:
- GraphRAG deep integration: uses knowledge graph communities as themes
- Skips redundant LLM theme identification when GraphRAG data available
- Leverages entity relationships for richer synthesis
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from agents.errors import LLMError
from agents.prompts import (
    CITATION_FORMAT_PROMPT,
    GAP_ANALYSIS_PROMPT,
    GRAPHRAG_LITERATURE_REVIEW_PROMPT,
    LITERATURE_REVIEW_PROMPT,
    REVIEW_CONCLUSION_PROMPT,
    REVIEW_INTRO_PROMPT,
    REVIEW_THEME_SECTION_PROMPT,
    THEME_IDENTIFICATION_PROMPT,
    build_citation_checklist,
    format_summaries_for_prompt,
    get_language_instruction,
)
from agents.state import (
    Community,
    ExtractedEntity,
    KnowledgeGraphData,
    LitScribeState,
    PaperSummary,
    SynthesisOutput,
    ThemeCluster,
)
from agents.tools import call_llm, call_llm_for_json, extract_json

logger = logging.getLogger(__name__)


def count_words(text: str) -> int:
    """Count words in text, handling CJK (Chinese/Japanese/Korean) characters.

    For CJK text, each character is counted as one word-equivalent.
    For Latin text, standard whitespace splitting is used.
    Mixed text sums both counts.
    """
    cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uf900-\ufaff]', text))
    latin_words = len(re.findall(r'[a-zA-Zà-ÿ]+', text))
    return cjk_chars + latin_words


def communities_to_themes(
    communities: List[Community],
    entities: Dict[str, ExtractedEntity],
    analyzed_papers: List[PaperSummary],
) -> List[ThemeCluster]:
    """Convert GraphRAG communities to ThemeCluster format.

    This enables deep integration: GraphRAG communities become themes directly,
    skipping redundant LLM theme identification.

    Args:
        communities: List of detected communities from GraphRAG
        entities: Entity dictionary for looking up entity names
        analyzed_papers: List of analyzed papers for extracting key points

    Returns:
        List of ThemeCluster objects derived from communities
    """
    themes = []
    paper_findings_map = {
        p["paper_id"]: p.get("key_findings", [])
        for p in analyzed_papers
    }

    # Sort communities by level (lower = higher hierarchy) and size
    sorted_communities = sorted(
        communities,
        key=lambda c: (c.get("level", 0), -len(c.get("papers", []))),
    )

    # Use top-level communities as themes (level 0 or 1)
    top_communities = [c for c in sorted_communities if c.get("level", 0) <= 1]

    # Limit to 6 themes max
    for community in top_communities[:6]:
        community_id = community.get("community_id") or "unknown"
        entity_ids = community.get("entities") or []
        paper_ids = community.get("papers") or []
        summary = community.get("summary") or ""

        # Build theme name from top entities (defensive access)
        entity_names = []
        for eid in entity_ids[:3]:
            if eid and eid in entities:
                entity = entities[eid]
                entity_names.append(entity.get("name") or eid)
            elif eid:
                entity_names.append(str(eid))

        if entity_names:
            theme_name = " / ".join(entity_names)
        else:
            theme_name = f"Research Cluster {community_id[:8]}"

        # Use community summary as description
        description = summary if summary else f"Papers exploring {theme_name}"

        # Extract key points from papers in this community
        key_points = []
        for pid in paper_ids[:5]:
            findings = paper_findings_map.get(pid, [])
            key_points.extend(findings[:2])
        key_points = key_points[:8]  # Limit key points

        themes.append(ThemeCluster(
            theme=theme_name,
            description=description,
            paper_ids=paper_ids,
            key_points=key_points,
        ))

    # Deduplicate themes by normalized name (communities at different hierarchy
    # levels can produce identical or near-identical theme names from the same
    # entity cluster — e.g. "A / B / C" vs "B / A / C" or "A / B" vs "A / B / D")
    def _normalize_theme_name(name: str) -> str:
        parts = sorted(p.strip().lower() for p in name.split("/"))
        return " / ".join(parts)

    def _themes_overlap(name_a: str, name_b: str) -> bool:
        """Check if two theme names share >=50% of their entity components."""
        parts_a = {p.strip().lower() for p in name_a.split("/")}
        parts_b = {p.strip().lower() for p in name_b.split("/")}
        if not parts_a or not parts_b:
            return False
        overlap = len(parts_a & parts_b)
        return overlap >= min(len(parts_a), len(parts_b)) * 0.5

    unique_themes = []
    for theme in themes:
        is_dup = False
        for existing in unique_themes:
            if (_normalize_theme_name(theme["theme"]) == _normalize_theme_name(existing["theme"])
                    or _themes_overlap(theme["theme"], existing["theme"])):
                # Merge paper_ids into existing theme
                existing_ids = set(existing["paper_ids"])
                for pid in theme["paper_ids"]:
                    if pid not in existing_ids:
                        existing["paper_ids"].append(pid)
                        existing_ids.add(pid)
                is_dup = True
                break
        if not is_dup:
            unique_themes.append(theme)
    themes = unique_themes

    # If no communities, return empty list (will trigger fallback)
    if not themes:
        logger.warning("No communities found for theme conversion")

    logger.info(f"Converted {len(themes)} communities to themes")
    return themes


def format_knowledge_graph_context(
    knowledge_graph: KnowledgeGraphData,
    max_entities: int = 20,
) -> str:
    """Format knowledge graph data for inclusion in prompts.

    Args:
        knowledge_graph: The knowledge graph data
        max_entities: Maximum entities to include

    Returns:
        Formatted string with key entities and relationships
    """
    lines = []

    # Global summary
    global_summary = knowledge_graph.get("global_summary", "")
    if global_summary:
        lines.append("## Knowledge Graph Summary")
        lines.append(global_summary)
        lines.append("")

    # Key entities by type
    entities = knowledge_graph.get("entities", {})
    if entities:
        lines.append("## Key Entities")

        # Group by type
        by_type: Dict[str, List[ExtractedEntity]] = {}
        for entity in entities.values():
            etype = entity.get("entity_type", "other")
            if etype not in by_type:
                by_type[etype] = []
            by_type[etype].append(entity)

        # Sort each group by frequency and take top entities
        entity_count = 0
        for etype in ["method", "dataset", "metric", "concept", "other"]:
            if etype not in by_type:
                continue
            sorted_entities = sorted(
                by_type[etype],
                key=lambda e: e.get("frequency", 0),
                reverse=True,
            )
            lines.append(f"\n### {etype.title()}s")
            for entity in sorted_entities[:5]:
                name = entity.get("name") or "Unknown"
                desc = (entity.get("description") or "")[:100]
                freq = entity.get("frequency") or 0
                lines.append(f"- **{name}** (freq: {freq}): {desc}")
                entity_count += 1
                if entity_count >= max_entities:
                    break
            if entity_count >= max_entities:
                break

    # Community summaries (context only — not citable)
    communities = knowledge_graph.get("communities") or []
    if communities:
        lines.append("\n## Research Clusters (background context only — do NOT cite these)")
        for comm in communities[:5]:
            summary = comm.get("summary") or "Unnamed cluster"
            paper_count = len(comm.get("papers") or [])
            lines.append(f"- {summary} ({paper_count} papers)")

    return "\n".join(lines)


async def identify_themes(
    analyzed_papers: List[PaperSummary],
    research_question: str,
    model: Optional[str] = None,
    tracker=None,
) -> List[ThemeCluster]:
    """Identify major themes across analyzed papers.

    Uses LLM to find patterns and group papers into thematic clusters.

    Args:
        analyzed_papers: List of analyzed paper summaries
        research_question: The original research question
        model: LLM model to use

    Returns:
        List of identified theme clusters
    """
    if not analyzed_papers:
        return []

    paper_summaries = format_summaries_for_prompt(analyzed_papers)

    prompt = THEME_IDENTIFICATION_PROMPT.format(
        research_question=research_question,
        num_papers=len(analyzed_papers),
        paper_summaries=paper_summaries,
    )

    try:
        themes_data = await call_llm_for_json(prompt, model=model, temperature=0.4, max_tokens=2000, tracker=tracker, agent_name="synthesis")

        if not isinstance(themes_data, list):
            raise ValueError("Expected JSON array of themes")

        themes = []
        for theme_dict in themes_data[:6]:  # Max 6 themes
            themes.append(ThemeCluster(
                theme=theme_dict.get("theme", "Unknown Theme"),
                description=theme_dict.get("description", ""),
                paper_ids=theme_dict.get("paper_ids", []),
                key_points=theme_dict.get("key_points", []),
            ))

        logger.info(f"Identified {len(themes)} themes across papers")
        return themes

    except (json.JSONDecodeError, LLMError) as e:
        logger.warning(f"Theme identification failed: {e}")
        # Fallback: create a single "General Findings" theme
        return [ThemeCluster(
            theme="General Findings",
            description="Synthesized findings from analyzed papers",
            paper_ids=[p["paper_id"] for p in analyzed_papers],
            key_points=[
                finding
                for p in analyzed_papers
                for finding in p.get("key_findings", [])[:2]
            ][:10],
        )]


async def analyze_gaps(
    analyzed_papers: List[PaperSummary],
    themes: List[ThemeCluster],
    research_question: str,
    model: Optional[str] = None,
    tracker=None,
) -> Dict[str, List[str]]:
    """Identify research gaps and future directions.

    Args:
        analyzed_papers: List of analyzed paper summaries
        themes: Identified themes
        research_question: The original research question
        model: LLM model to use

    Returns:
        Dict with "gaps" and "future_directions" lists
    """
    if not analyzed_papers:
        return {"gaps": [], "future_directions": []}

    paper_summaries = format_summaries_for_prompt(analyzed_papers)
    themes_text = "\n".join(
        f"- {t['theme']}: {t['description']}"
        for t in themes
    )

    prompt = GAP_ANALYSIS_PROMPT.format(
        research_question=research_question,
        paper_summaries=paper_summaries,
        themes=themes_text,
    )

    try:
        result = await call_llm_for_json(prompt, model=model, temperature=0.4, max_tokens=1000, tracker=tracker, agent_name="synthesis")

        # Handle both dict and list responses (LLM may return a flat list)
        if isinstance(result, list):
            return {"gaps": result[:5], "future_directions": []}
        return {
            "gaps": result.get("gaps", [])[:5],
            "future_directions": result.get("future_directions", [])[:5],
        }

    except (json.JSONDecodeError, LLMError) as e:
        logger.warning(f"Gap analysis failed: {e}")
        # Fallback based on paper limitations
        limitations = []
        for paper in analyzed_papers:
            limitations.extend(paper.get("limitations", [])[:2])
        return {
            "gaps": limitations[:5] if limitations else ["Gap analysis not available"],
            "future_directions": ["Further research needed"],
        }


async def generate_review(
    analyzed_papers: List[PaperSummary],
    themes: List[ThemeCluster],
    gaps: List[str],
    research_question: str,
    review_type: str = "narrative",
    target_words: int = 2000,
    model: Optional[str] = None,
    language: str = "en",
    tracker=None,
) -> str:
    """Generate the literature review narrative.

    Args:
        analyzed_papers: List of analyzed paper summaries
        themes: Identified themes
        gaps: Research gaps
        research_question: The original research question
        review_type: Type of review (narrative, systematic, scoping)
        target_words: Target word count
        model: LLM model to use
        language: Output language code ("en", "zh", etc.)

    Returns:
        Generated review text
    """
    if not analyzed_papers:
        return "No papers available for literature review."

    paper_summaries = format_summaries_for_prompt(analyzed_papers)
    citation_checklist = build_citation_checklist(analyzed_papers)
    themes_text = "\n\n".join(
        f"**{t['theme']}**\n{t['description']}\nKey points:\n" +
        "\n".join(f"  - {p}" for p in t["key_points"][:5])
        for t in themes
    )
    gaps_text = "\n".join(f"- {g}" for g in gaps)

    prompt = LITERATURE_REVIEW_PROMPT.format(
        review_type=review_type,
        research_question=research_question,
        num_papers=len(analyzed_papers),
        paper_summaries=paper_summaries,
        citation_checklist=citation_checklist,
        themes=themes_text,
        gaps=gaps_text,
        word_count=target_words,
    ) + get_language_instruction(language)

    try:
        response = await call_llm(
            prompt,
            model=model,
            temperature=0.5,
            max_tokens=min(65536, target_words * 2),
            tracker=tracker,
            agent_name="synthesis",
            task_type="synthesis",
        )
        # Post-processing: normalize misspelled citations
        name_map = _build_citation_name_map(analyzed_papers)
        response = _normalize_citations(response, name_map)
        logger.info(f"Generated review with {count_words(response)} words")
        return response.strip()

    except LLMError as e:
        logger.error(f"Review generation failed: {e}")
        # Fallback: create basic summary
        summary_parts = [
            f"# Literature Review: {research_question}\n\n",
            "## Summary\n\n",
            f"This review synthesizes findings from {len(analyzed_papers)} papers.\n\n",
            "## Key Findings\n\n",
        ]
        for paper in analyzed_papers[:5]:
            summary_parts.append(f"**{paper['title']}** ({paper['year']})\n")
            for finding in paper.get("key_findings", [])[:2]:
                summary_parts.append(f"- {finding}\n")
            summary_parts.append("\n")

        return "".join(summary_parts)


async def generate_graphrag_review(
    analyzed_papers: List[PaperSummary],
    themes: List[ThemeCluster],
    gaps: List[str],
    research_question: str,
    knowledge_graph_context: str,
    global_summary: str,
    review_type: str = "narrative",
    target_words: int = 2500,
    model: Optional[str] = None,
    language: str = "en",
    tracker=None,
) -> str:
    """Generate literature review enhanced with GraphRAG knowledge.

    This function uses the knowledge graph context (entities, relationships,
    community summaries) to produce a richer, more connected synthesis.

    Args:
        analyzed_papers: List of analyzed paper summaries
        themes: Theme clusters (derived from GraphRAG communities)
        gaps: Research gaps
        research_question: The original research question
        knowledge_graph_context: Formatted knowledge graph data
        global_summary: GraphRAG global summary
        review_type: Type of review (narrative, systematic, scoping)
        target_words: Target word count
        model: LLM model to use
        language: Output language code ("en", "zh", etc.)

    Returns:
        Generated review text with GraphRAG enhancement
    """
    if not analyzed_papers:
        return "No papers available for literature review."

    paper_summaries = format_summaries_for_prompt(analyzed_papers)
    citation_checklist = build_citation_checklist(analyzed_papers)
    themes_text = "\n\n".join(
        f"**{t['theme']}**\n{t['description']}\nKey points:\n"
        + "\n".join(f"  - {p}" for p in t["key_points"][:5])
        for t in themes
    )
    gaps_text = "\n".join(f"- {g}" for g in gaps)

    prompt = GRAPHRAG_LITERATURE_REVIEW_PROMPT.format(
        review_type=review_type,
        research_question=research_question,
        num_papers=len(analyzed_papers),
        paper_summaries=paper_summaries,
        citation_checklist=citation_checklist,
        themes=themes_text,
        gaps=gaps_text,
        knowledge_graph_context=knowledge_graph_context,
        global_summary=global_summary,
        word_count=target_words,
    ) + get_language_instruction(language)

    try:
        response = await call_llm(
            prompt,
            model=model,
            temperature=0.5,
            max_tokens=min(65536, target_words * 2),
            tracker=tracker,
            agent_name="synthesis",
            task_type="synthesis",
        )
        # Post-processing: normalize misspelled citations
        name_map = _build_citation_name_map(analyzed_papers)
        response = _normalize_citations(response, name_map)
        logger.info(f"Generated GraphRAG-enhanced review with {count_words(response)} words")
        return response.strip()

    except LLMError as e:
        logger.error(f"GraphRAG review generation failed: {e}, falling back to standard")
        # Fallback to standard review
        return await generate_review(
            analyzed_papers=analyzed_papers,
            themes=themes,
            gaps=gaps,
            research_question=research_question,
            review_type=review_type,
            target_words=target_words,
            model=model,
            language=language,
            tracker=tracker,
        )


def _build_citation_name_map(analyzed_papers: List[PaperSummary]) -> Dict[str, str]:
    """Build a mapping of known author last names for citation normalization.

    Returns dict mapping lowercase last name -> canonical form (original casing).
    """
    from analysis.citation_grounding import _extract_last_names
    name_map = {}
    for p in analyzed_papers:
        authors = p.get("authors", [])
        if isinstance(authors, str):
            authors = [authors]
        for ln in _extract_last_names(authors):
            name_map[ln.lower()] = ln
    return name_map


def _normalize_citations(text: str, name_map: Dict[str, str]) -> str:
    """Fix misspelled author names in [Author, Year] citations.

    Uses edit distance to match misspelled names to known authors.
    Only corrects names that are within edit distance 2 of a known author.
    """
    import re as _re

    def _edit_distance(a: str, b: str) -> int:
        """Simple Levenshtein distance."""
        if len(a) < len(b):
            return _edit_distance(b, a)
        if len(b) == 0:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                cost = 0 if ca == cb else 1
                curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
            prev = curr
        return prev[len(b)]

    known_names_lower = set(name_map.keys())

    def _fix_citation(match):
        full = match.group(0)
        name_part = match.group(1).strip()
        rest = match.group(2)  # ", Year]" or " et al., Year]"

        # Check if name is already known
        if name_part.lower() in known_names_lower:
            return full  # Already correct

        # Try fuzzy match (edit distance <= 2)
        best_match = None
        best_dist = 3  # threshold
        for known_lower, canonical in name_map.items():
            dist = _edit_distance(name_part.lower(), known_lower)
            if dist < best_dist:
                best_dist = dist
                best_match = canonical
        if best_match:
            logger.info(f"Citation normalization: '{name_part}' → '{best_match}' (dist={best_dist})")
            return f"[{best_match}{rest}"
        return full

    # Match [Name, Year] and [Name et al., Year] patterns
    normalized = _re.sub(
        r'\[([A-Za-zÀ-ÿ\u4e00-\u9fff]+)((?:\s+et\s+al\.)?,\s*\d{4}\])',
        _fix_citation,
        text,
    )
    return normalized


async def generate_review_sectioned(
    analyzed_papers: List[PaperSummary],
    themes: List[ThemeCluster],
    gaps: List[str],
    research_question: str,
    review_type: str = "narrative",
    target_words: int = 5000,
    model: Optional[str] = None,
    language: str = "en",
    tracker=None,
    knowledge_graph_context: Optional[str] = None,
    global_summary: Optional[str] = None,
) -> str:
    """Generate a literature review in sections to overcome LLM output token limits.

    Splits the review into introduction, per-theme body sections, and conclusion,
    each generated as a separate LLM call (< 8K tokens each).

    Triggered when target_words > 4096 (roughly > 8K output tokens).

    Args:
        analyzed_papers: List of analyzed paper summaries
        themes: Identified themes
        gaps: Research gaps
        research_question: The original research question
        review_type: Type of review (narrative, systematic, scoping)
        target_words: Target word count for the full review
        model: LLM model to use
        language: Output language code ("en", "zh", etc.)
        tracker: Token tracker
        knowledge_graph_context: Optional GraphRAG context string
        global_summary: Optional GraphRAG global summary

    Returns:
        Full review text assembled from sections
    """
    from analysis.citation_grounding import extract_all_cited_authors

    lang_instruction = get_language_instruction(language)
    num_themes = max(len(themes), 1)

    # Word budget allocation
    intro_words = int(target_words * 0.10)
    body_total = int(target_words * 0.70)
    body_per_theme = body_total // num_themes
    conclusion_words = int(target_words * 0.20)

    # Knowledge graph section (shared across prompts)
    kg_section = ""
    if knowledge_graph_context:
        kg_section = f"## Knowledge Graph Context (background only — NOT citable):\n{knowledge_graph_context}\n"
        if global_summary:
            kg_section += f"\n## Global Research Landscape:\n{global_summary}\n"

    # Theme names for intro/conclusion
    theme_names_text = "\n".join(
        f"{i+1}. {t['theme']}: {t['description'][:100]}"
        for i, t in enumerate(themes)
    )

    sections = []

    # --- Stage 1: Introduction ---
    logger.info(f"Sectioned generation: intro (~{intro_words} words)")
    intro_prompt = REVIEW_INTRO_PROMPT.format(
        review_type=review_type,
        research_question=research_question,
        num_papers=len(analyzed_papers),
        knowledge_graph_section=kg_section,
        theme_names=theme_names_text,
        word_count=intro_words,
    ) + lang_instruction

    try:
        intro_text = await call_llm(
            intro_prompt,
            model=model,
            temperature=0.5,
            max_tokens=min(8192, intro_words * 3),
            tracker=tracker,
            agent_name="synthesis",
            task_type="synthesis",
        )
        sections.append(intro_text.strip())
    except LLMError as e:
        logger.warning(f"Sectioned intro failed: {e}")
        sections.append(f"# Literature Review: {research_question}\n\n"
                        f"This review synthesizes findings from {len(analyzed_papers)} papers.")

    # Track cited authors across all sections
    all_cited_authors = set()
    previous_ending = sections[-1][-200:] if sections else ""

    # --- Stage 2: Theme sections ---
    for i, theme in enumerate(themes):
        # Get papers for this theme — only from analyzed papers
        theme_paper_ids = set(theme.get("paper_ids", []))
        theme_papers = [p for p in analyzed_papers if p["paper_id"] in theme_paper_ids]

        # Fallback: if theme has no matched papers, assign proportionally
        if not theme_papers:
            start = (i * len(analyzed_papers)) // num_themes
            end = ((i + 1) * len(analyzed_papers)) // num_themes
            theme_papers = analyzed_papers[start:end] if start < end else analyzed_papers[:3]

        theme_summaries = format_summaries_for_prompt(theme_papers)
        theme_checklist = build_citation_checklist(theme_papers)
        key_points_text = "\n".join(f"- {p}" for p in theme.get("key_points", [])[:5])

        logger.info(
            f"Sectioned generation: theme {i+1}/{num_themes} "
            f"'{theme['theme']}' ({len(theme_papers)} papers, ~{body_per_theme} words)"
        )

        theme_prompt = REVIEW_THEME_SECTION_PROMPT.format(
            research_question=research_question,
            knowledge_graph_section=kg_section,
            theme_number=i + 1,
            total_themes=num_themes,
            theme_name=theme["theme"],
            theme_description=theme.get("description", ""),
            key_points=key_points_text or "N/A",
            num_theme_papers=len(theme_papers),
            theme_papers=theme_summaries,
            theme_citation_checklist=theme_checklist,
            previous_ending=previous_ending,
            word_count=body_per_theme,
        ) + lang_instruction

        try:
            theme_text = await call_llm(
                theme_prompt,
                model=model,
                temperature=0.5,
                max_tokens=min(8192, body_per_theme * 3),
                tracker=tracker,
                agent_name="synthesis",
                task_type="synthesis",
            )
            sections.append(theme_text.strip())
            previous_ending = theme_text.strip()[-200:]

            # Track cited authors
            section_cited = extract_all_cited_authors(theme_text)
            all_cited_authors.update(section_cited)
        except LLMError as e:
            logger.warning(f"Sectioned theme '{theme['theme']}' failed: {e}")
            sections.append(f"## {i+1}. {theme['theme']}\n\n{theme.get('description', '')}")
            previous_ending = sections[-1][-200:]

    # --- Stage 3: Conclusion ---
    # Find uncited papers
    uncited_section = ""
    if all_cited_authors:
        from analysis.citation_grounding import _extract_last_names
        uncited = []
        for p in analyzed_papers:
            authors = p.get("authors", [])
            if isinstance(authors, str):
                authors = [authors]
            last_names = _extract_last_names(authors)
            if not any(ln.lower() in all_cited_authors for ln in last_names):
                year = p.get("year", "N/A")
                cite_name = last_names[0] if last_names else "Unknown"
                uncited.append(f"- [{cite_name} et al., {year}] — {p.get('title', '')[:60]}")
        if uncited:
            uncited_section = (
                "## Uncited Papers — please incorporate these in the conclusion:\n"
                + "\n".join(uncited)
            )
            logger.info(f"Sectioned generation: {len(uncited)} uncited papers flagged for conclusion")

    gaps_text = "\n".join(f"- {g}" for g in gaps)

    logger.info(f"Sectioned generation: conclusion (~{conclusion_words} words)")
    conclusion_prompt = REVIEW_CONCLUSION_PROMPT.format(
        research_question=research_question,
        num_papers=len(analyzed_papers),
        knowledge_graph_section=kg_section,
        theme_names=theme_names_text,
        gaps=gaps_text,
        previous_ending=previous_ending,
        uncited_section=uncited_section,
        word_count=conclusion_words,
    ) + lang_instruction

    try:
        conclusion_text = await call_llm(
            conclusion_prompt,
            model=model,
            temperature=0.5,
            max_tokens=min(8192, conclusion_words * 3),
            tracker=tracker,
            agent_name="synthesis",
            task_type="synthesis",
        )
        sections.append(conclusion_text.strip())
    except LLMError as e:
        logger.warning(f"Sectioned conclusion failed: {e}")
        sections.append("## Conclusion\n\nFurther research is needed in this area.")

    # Assemble full review
    full_review = "\n\n".join(sections)

    # Post-processing: normalize misspelled citations
    name_map = _build_citation_name_map(analyzed_papers)
    full_review = _normalize_citations(full_review, name_map)

    logger.info(
        f"Sectioned generation complete: {count_words(full_review)} words "
        f"across {len(sections)} sections"
    )
    return full_review


async def format_citations(
    analyzed_papers: List[PaperSummary],
    style: str = "APA",
    model: Optional[str] = None,
    tracker=None,
) -> List[str]:
    """Format paper citations in the specified style.

    Args:
        analyzed_papers: List of analyzed paper summaries
        style: Citation style (APA, MLA, Chicago, IEEE)
        model: LLM model to use

    Returns:
        List of formatted citations
    """
    if not analyzed_papers:
        return []

    # Prepare paper info for formatting
    papers_info = []
    for paper in analyzed_papers:
        authors = paper.get("authors", [])
        if isinstance(authors, list):
            authors_str = ", ".join(authors[:3])
            if len(paper.get("authors", [])) > 3:
                authors_str += " et al."
        else:
            authors_str = str(authors)

        info = {
            "title": paper.get("title", "Unknown"),
            "authors": authors_str,
            "year": paper.get("year", "n.d."),
            "venue": paper.get("venue", ""),
            "paper_id": paper.get("paper_id", ""),
        }
        papers_info.append(info)

    papers_text = "\n".join(
        f"- Title: {p['title']}\n  Authors: {p['authors']}\n  Year: {p['year']}\n  Venue: {p['venue']}\n  ID: {p['paper_id']}"
        for p in papers_info
    )

    prompt = CITATION_FORMAT_PROMPT.format(
        citation_style=style,
        papers=papers_text,
    )

    try:
        citation_tokens = max(2000, len(papers_info) * 120)
        response = await call_llm(prompt, model=model, temperature=0.1, max_tokens=citation_tokens, tracker=tracker, agent_name="synthesis")
        citations = [line.strip() for line in response.strip().split("\n") if line.strip()]
        return citations

    except LLMError as e:
        logger.warning(f"Citation formatting failed: {e}")
        # Fallback: simple format
        return [
            f"{p['authors']} ({p['year']}). {p['title']}. {p['venue']}."
            for p in papers_info
        ]


async def synthesis_agent(state: LitScribeState) -> Dict[str, Any]:
    """Main entry point for the Synthesis Agent.

    This function is called by the LangGraph workflow to generate
    the final literature review synthesis.

    Phase 7.5 Enhancement:
    - When knowledge_graph is available from GraphRAG, uses communities
      directly as themes (skipping redundant LLM theme identification)
    - Enriches review with entity relationships and community summaries

    Args:
        state: Current workflow state

    Returns:
        State updates with synthesis output
    """
    analyzed_papers_raw = state.get("analyzed_papers", [])
    research_question = state["research_question"]
    review_type = state.get("review_type", "narrative")
    language = state.get("language", "en")
    errors = list(state.get("errors", []))
    llm_config = state.get("llm_config", {})
    model = llm_config.get("model")
    from utils.token_tracker import get_tracker
    tracker = get_tracker()

    # Deduplicate analyzed papers by paper_id
    seen_ids = set()
    analyzed_papers = []
    for p in analyzed_papers_raw:
        pid = p.get("paper_id", "")
        if pid and pid in seen_ids:
            logger.debug(f"Skipping duplicate paper: {pid}")
            continue
        if pid:
            seen_ids.add(pid)
        analyzed_papers.append(p)
    if len(analyzed_papers) < len(analyzed_papers_raw):
        logger.info(
            f"Deduplicated papers: {len(analyzed_papers_raw)} → {len(analyzed_papers)}"
        )

    # Phase 7.5: Check for GraphRAG knowledge graph
    knowledge_graph = state.get("knowledge_graph")
    use_graphrag = (
        knowledge_graph is not None
        and knowledge_graph.get("communities")
        and len(knowledge_graph.get("communities", [])) > 0
    )

    if not analyzed_papers:
        error_msg = "No analyzed papers available for synthesis"
        logger.warning(error_msg)
        errors.append(error_msg)
        return {
            "synthesis": SynthesisOutput(
                themes=[],
                gaps=["No papers analyzed"],
                future_directions=[],
                review_text="Unable to generate review - no papers analyzed.",
                citations_formatted=[],
                word_count=0,
                papers_cited=0,
            ),
            "errors": errors,
            "current_agent": "complete",
        }

    logger.info(f"Synthesis Agent starting: {len(analyzed_papers)} papers to synthesize")
    if use_graphrag:
        logger.info(
            f"GraphRAG enabled: {len(knowledge_graph.get('communities', []))} communities, "
            f"{len(knowledge_graph.get('entities', {}))} entities"
        )

    try:
        # Step 1: Identify themes (or use GraphRAG communities)
        kg_context = None
        if use_graphrag:
            # Deep integration: convert communities to themes directly
            themes = communities_to_themes(
                communities=knowledge_graph.get("communities", []),
                entities=knowledge_graph.get("entities", {}),
                analyzed_papers=analyzed_papers,
            )

            if themes:
                logger.info(f"Using {len(themes)} themes from GraphRAG communities")
                # Prepare knowledge graph context for review generation
                kg_context = format_knowledge_graph_context(knowledge_graph)
            else:
                # GraphRAG communities existed but produced no themes (e.g., all level > 1)
                # Fall back to LLM theme identification
                logger.warning("GraphRAG produced no usable themes, falling back to LLM")
                use_graphrag = False
                themes = await identify_themes(analyzed_papers, research_question, model, tracker=tracker)
        else:
            # Fallback: LLM-based theme identification
            themes = await identify_themes(analyzed_papers, research_question, model, tracker=tracker)
            kg_context = None

        # Step 2: Analyze gaps (enhanced with GraphRAG context if available)
        gap_analysis = await analyze_gaps(analyzed_papers, themes, research_question, model, tracker=tracker)
        gaps = gap_analysis["gaps"]
        future_directions = gap_analysis["future_directions"]

        # Step 3: Generate review (with GraphRAG enhancement if available)
        # Use target_words from state (tier system), with sensible fallback
        target_words = state.get("target_words", 1000 + len(analyzed_papers) * 130)
        if use_graphrag and kg_context:
            review_text = await generate_graphrag_review(
                analyzed_papers=analyzed_papers,
                themes=themes,
                gaps=gaps,
                research_question=research_question,
                knowledge_graph_context=kg_context,
                global_summary=knowledge_graph.get("global_summary", ""),
                review_type=review_type,
                target_words=target_words,
                model=model,
                language=language,
                tracker=tracker,
            )
        else:
            review_text = await generate_review(
                analyzed_papers=analyzed_papers,
                themes=themes,
                gaps=gaps,
                research_question=research_question,
                review_type=review_type,
                target_words=target_words,
                model=model,
                language=language,
                tracker=tracker,
            )

        # Step 4: Format citations for ALL analyzed papers
        # Always include all analyzed papers in references — the review should
        # cite them all, and even if the LLM missed some or misspelled names,
        # the reference list must be complete.
        from analysis.citation_grounding import (
            extract_inline_citations, extract_all_cited_authors,
            _parse_citation, _extract_last_names,
        )
        cited_authors = extract_all_cited_authors(review_text)
        if cited_authors:
            cited_count = 0
            for p in analyzed_papers:
                authors = p.get("authors", [])
                if isinstance(authors, str):
                    authors = [authors]
                last_names = _extract_last_names(authors)
                if any(ln.lower() in cited_authors for ln in last_names):
                    cited_count += 1
            logger.info(f"Citation check: {cited_count}/{len(analyzed_papers)} papers cited in review text")

        # Always format ALL analyzed papers as references
        citations = await format_citations(analyzed_papers, style="APA", model=model, tracker=tracker)

        # Build synthesis output
        synthesis = SynthesisOutput(
            themes=themes,
            gaps=gaps,
            future_directions=future_directions,
            review_text=review_text,
            citations_formatted=citations,
            word_count=count_words(review_text),
            papers_cited=len(analyzed_papers),
        )

        # Step 5: Citation grounding check (Phase 9.5)
        grounding_report = None
        try:
            from analysis.citation_grounding import check_citation_grounding
            grounding_report = check_citation_grounding(review_text, analyzed_papers)
            if grounding_report["ungrounded_count"] > 0:
                logger.warning(
                    f"Citation grounding: {grounding_report['ungrounded_count']} ungrounded citations "
                    f"(rate={grounding_report['grounding_rate']:.2%})"
                )
            else:
                logger.info(
                    f"Citation grounding: {grounding_report['grounded_count']}/{grounding_report['total_citations']} "
                    f"citations grounded (100%)"
                )
        except Exception as e:
            logger.warning(f"Citation grounding check failed: {e}")

        mode_str = "GraphRAG-enhanced" if use_graphrag else "standard"
        logger.info(
            f"Synthesis complete ({mode_str}): {synthesis['word_count']} words, "
            f"{len(themes)} themes, {len(gaps)} gaps"
        )

        result = {
            "synthesis": synthesis,
            "errors": errors,
            "current_agent": "complete",
        }
        if grounding_report:
            result["citation_grounding"] = grounding_report
        return result

    except Exception as e:
        error_msg = f"Synthesis Agent failed: {e}"
        logger.error(error_msg)
        errors.append(error_msg)
        return {
            "synthesis": SynthesisOutput(
                themes=[],
                gaps=[error_msg],
                future_directions=[],
                review_text=f"Synthesis failed: {e}",
                citations_formatted=[],
                word_count=0,
                papers_cited=0,
            ),
            "errors": errors,
            "current_agent": "complete",
        }


# Export for use in graph.py
__all__ = [
    "synthesis_agent",
    "identify_themes",
    "analyze_gaps",
    "generate_review",
    "generate_review_sectioned",
    "generate_graphrag_review",
    "format_citations",
    # GraphRAG integration helpers
    "communities_to_themes",
    "format_knowledge_graph_context",
]
