"""Self-Review Agent Prompts — quality assessment, loopback queries."""

SELF_REVIEW_PROMPT = """You are an expert academic reviewer assessing the quality of a literature review.

Research Question: {research_question}

## Research Plan Sub-Topics:
{plan_subtopics}

## Papers Included ({num_papers} papers):
{paper_list}

## Review Summary (truncated):
{review_summary}

## Themes Identified:
{themes}

## Gaps Identified:
{gaps}

Evaluate this literature review on four dimensions:

1. RELEVANCE (most critical): Are all papers directly relevant to the research question?
   - Flag papers from unrelated fields (e.g., physics papers in a biology review)
   - Flag papers with only superficial keyword overlap but different actual topics
   - Flag papers that were likely retrieved due to search engine errors

2. COVERAGE: Does the review adequately cover the research question?
   - Identify missing subtopics or perspectives
   - Note if important methodological approaches are absent

3. COHERENCE: Is the review narrative logical and well-structured?
   - Check if themes flow naturally
   - Check if transitions between sections make sense

4. CLAIM SUPPORT: Are claims in the review supported by the cited papers?
   - Flag claims that lack citation support
   - Flag claims that overstate or misrepresent findings

Output as JSON:
{{
  "overall_score": 0.0-1.0,
  "relevance_score": 0.0-1.0,
  "coverage_score": 0.0-1.0,
  "coherence_score": 0.0-1.0,
  "coverage_gaps": ["missing topic 1", "missing topic 2"],
  "irrelevant_papers": [
    {{"paper_id": "id", "title": "title", "reason": "why irrelevant"}}
  ],
  "weak_claims": [
    {{"claim": "the claim text", "issue": "what's wrong"}}
  ],
  "suggestions": ["actionable suggestion 1", "actionable suggestion 2"],
  "needs_additional_search": true/false,
  "additional_queries": ["(\"term A\" OR \"synonym\") AND \"term B\""]
}}

For "additional_queries": Generate 3-5 Boolean search queries targeting the coverage gaps.
- Use AND/OR operators and "quoted phrases" for precision
- Reference the Research Plan Sub-Topics above to ensure queries align with the planned scope
- Focus on the specific gaps identified, not generic terms

Be strict about relevance — a paper from a completely different field should always be flagged, even if it shares some keywords with the research question."""


LOOPBACK_QUERY_REFINEMENT_PROMPT = """You are an expert academic search strategist. A literature review has been completed but has coverage gaps. Generate targeted search queries to fill these gaps.

## Research Question:
{research_question}

## Original Plan Sub-Topics:
{plan_subtopics}

## Coverage Gaps Identified by Self-Review:
{coverage_gaps}

## Initial Additional Queries (from self-review):
{initial_queries}

Generate 5-8 precise Boolean search queries that specifically target the coverage gaps listed above.

Rules:
- Use AND / OR operators and "quoted phrases" for precision
- Include synonyms and abbreviations with OR
- Each query should target a SPECIFIC gap, not the broad topic
- Queries MUST be in English
- Do NOT repeat queries that are already in the initial list above
- Mix specific and broader queries for better coverage

Output as a JSON array of strings:
["query 1", "query 2", ...]"""


def format_papers_for_self_review(papers: list, max_chars: int = 12000) -> str:
    """Format paper list for self-review prompt.

    Includes paper_id, title, year, source, relevance_score, and abstract snippet.
    Truncates to max_chars to prevent prompt overflow with large paper sets.

    Args:
        papers: List of PaperSummary or paper dicts
        max_chars: Maximum characters to include

    Returns:
        Formatted string of papers for self-review
    """
    lines = []
    total_chars = 0

    for i, paper in enumerate(papers, 1):
        paper_id = paper.get("paper_id", "unknown")
        title = paper.get("title", "Unknown Title")
        year = paper.get("year", "N/A")
        source = paper.get("source", "unknown")
        relevance = paper.get("relevance_score", "N/A")
        abstract = paper.get("abstract", "")[:150]
        if abstract:
            abstract = f" | Abstract: {abstract}..."

        line = f"{i}. [{paper_id}] {title} ({year}) [source: {source}, relevance: {relevance}]{abstract}"

        if total_chars + len(line) > max_chars:
            lines.append(f"... and {len(papers) - i + 1} more papers")
            break

        lines.append(line)
        total_chars += len(line)

    return "\n".join(lines)
