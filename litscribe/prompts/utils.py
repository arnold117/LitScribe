"""Utility functions and miscellaneous prompts — formatting, citation helpers, language instructions."""

WORKFLOW_STATUS_PROMPT = """Analyze the current workflow state and determine the next action.

Current State:
- Research Question: {research_question}
- Search Completed: {search_completed}
- Papers Found: {papers_found}
- Papers Analyzed: {papers_analyzed}
- Synthesis Generated: {synthesis_generated}
- Errors: {errors}
- Iteration Count: {iteration_count}

Determine:
1. What has been accomplished?
2. What needs to be done next?
3. Are there any issues that need attention?

Output as JSON:
{{
  "next_agent": "discovery|critical_reading|synthesis|complete",
  "reasoning": "Explanation of decision...",
  "issues": ["any issues or concerns"]
}}"""


def format_papers_for_prompt(papers: list, max_chars: int = 20000) -> str:
    """Format list of papers for inclusion in prompts.

    Args:
        papers: List of paper dictionaries
        max_chars: Maximum characters to include

    Returns:
        Formatted string representation of papers
    """
    lines = []
    total_chars = 0

    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Unknown Title")
        authors = paper.get("authors", [])
        if isinstance(authors, list):
            authors = ", ".join(authors[:3])
            if len(paper.get("authors", [])) > 3:
                authors += " et al."
        year = paper.get("year", "N/A")
        citations = paper.get("citations", 0)
        paper_id = paper.get("paper_id") or paper.get("arxiv_id") or paper.get("doi", "")
        source = paper.get("source", "unknown")
        abstract = paper.get("abstract", "")[:120]
        if abstract:
            abstract = f"\n   Abstract: {abstract}..."

        line = f"{i}. [{paper_id}] {title}\n   Authors: {authors} | Year: {year} | Citations: {citations} | Source: {source}{abstract}"

        if total_chars + len(line) > max_chars:
            lines.append(f"... and {len(papers) - i + 1} more papers")
            break

        lines.append(line)
        total_chars += len(line)

    return "\n".join(lines)


def _extract_cite_name(author: str) -> str:
    """Extract last name from an author string for citation.

    Handles: "John Smith" -> "Smith", "Smith, J." -> "Smith",
    "Ma X" -> "Ma", "Zhang ZJ" -> "Zhang" (PubMed Chinese format).
    """
    author = author.strip()
    if not author:
        return "Unknown"
    if "," in author:
        return author.split(",")[0].strip()
    parts = author.split()
    if not parts:
        return "Unknown"
    last_part = parts[-1].rstrip(".")
    # PubMed "LastName Initials" format: last word is all uppercase and short
    if len(parts) > 1 and last_part.isupper() and len(last_part) <= 3:
        return parts[0].strip()
    return parts[-1].strip()


def format_summaries_for_prompt(summaries: list, max_chars: int = 20000) -> str:
    """Format paper summaries for inclusion in prompts.

    Args:
        summaries: List of PaperSummary dictionaries
        max_chars: Maximum characters to include

    Returns:
        Formatted string representation of summaries
    """
    lines = []
    total_chars = 0

    for summary in summaries:
        paper_id = summary.get("paper_id", "Unknown")
        title = summary.get("title", "Unknown Title")
        year = summary.get("year", "N/A")
        authors = summary.get("authors", [])
        if isinstance(authors, str):
            authors = [authors]
        # Format: "LastName et al." or "LastName & LastName" for citation guidance
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."
        findings = summary.get("key_findings", [])
        cite_name = _extract_cite_name(authors[0]) if authors else "Unknown"

        section = f"""
### {title} ({year}) [ID: {paper_id}]
Authors: {author_str}
Cite as: [{cite_name} et al., {year}]
Key Findings:
{chr(10).join(f"- {f}" for f in findings[:5])}

Methodology: {summary.get("methodology", "Not analyzed")[:300]}

Strengths: {", ".join(summary.get("strengths", [])[:3])}
Limitations: {", ".join(summary.get("limitations", [])[:3])}
"""
        if total_chars + len(section) > max_chars:
            lines.append(f"\n... and {len(summaries) - len(lines)} more papers")
            break

        lines.append(section)
        total_chars += len(section)

    return "\n".join(lines)


def build_citation_checklist(summaries: list) -> str:
    """Build a numbered checklist of papers that MUST be cited in the review.

    Args:
        summaries: List of PaperSummary dictionaries

    Returns:
        Numbered checklist like "1. [Ma et al., 2006] — A survey of potential..."
    """
    lines = []
    seen = set()
    for summary in summaries:
        authors = summary.get("authors", [])
        if isinstance(authors, str):
            authors = [authors]
        year = summary.get("year", "N/A")
        title = summary.get("title", "Unknown")
        cite_name = _extract_cite_name(authors[0]) if authors else "Unknown"
        cite_key = f"[{cite_name} et al., {year}]"
        # Deduplicate by cite_key
        if cite_key in seen:
            continue
        seen.add(cite_key)
        lines.append(f"{len(lines) + 1}. {cite_key} — {title[:80]}")
    return "\n".join(lines)


# =============================================================================
# Language Instructions for Multi-Language Review Generation (Phase 8.6)
# =============================================================================

LANGUAGE_INSTRUCTIONS = {
    "en": "",  # No additional instruction needed for English
    "zh": (
        "\n\nIMPORTANT LANGUAGE REQUIREMENT: Write the entire review in Chinese (中文). "
        "Use Chinese academic writing conventions and formal scholarly tone. "
        "Section headings should be in Chinese (e.g., 引言, 主题分析, 批判性讨论, 研究空白与未来方向, 结论). "
        "Keep in-text citations in [LastName, Year] format — every citation MUST include the year (e.g., [Zhang, 2020] or [Smith et al., 2020]). NEVER omit the year. "
        "NEVER cite research clusters or themes — only cite individual papers by author name and year. "
        "You MUST cite ALL papers from the Citation Checklist — do not skip any. "
        "Do not include an English translation."
    ),
}


def get_language_instruction(language: str) -> str:
    """Get the language instruction suffix for review generation prompts.

    Args:
        language: Target language code ("en", "zh", etc.)

    Returns:
        Instruction string to append to review prompts.
        Empty string for English (default behavior).
    """
    if language in LANGUAGE_INSTRUCTIONS:
        return LANGUAGE_INSTRUCTIONS[language]

    # Generic fallback for unsupported but requested languages
    return (
        f"\n\nIMPORTANT LANGUAGE REQUIREMENT: Write the entire review in {language}. "
        f"Use appropriate academic writing conventions for this language. "
        "Section headings should also be in the target language. "
        "Keep in-text citations in [LastName, Year] format using real author surnames."
    )
