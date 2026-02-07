"""LLM Prompt Templates for LitScribe Agents.

This module contains all prompt templates used by the multi-agent system.
Templates use Python format strings for variable substitution.
"""

# =============================================================================
# Discovery Agent Prompts
# =============================================================================

QUERY_EXPANSION_PROMPT = """You are an expert academic researcher. Given a research question, generate diverse search queries to find relevant academic literature.

Research Question: {research_question}

Generate exactly 5 search queries that:
1. Rephrase the original question with precise academic terminology
2. Focus on specific methodologies or techniques mentioned or implied
3. Target application domains or use cases
4. Use synonyms and related concepts from the field
5. Include combinations of key concepts

Requirements:
- Each query should be 3-8 words, optimized for academic search engines
- Avoid overly generic terms
- Include field-specific terminology
- IMPORTANT: All queries MUST be in English, even if the research question is in another language. Academic databases (arXiv, PubMed, Semantic Scholar) are English-based.

Output as a JSON array of strings:
["query1", "query2", "query3", "query4", "query5"]"""


PAPER_SELECTION_PROMPT = """You are an expert at evaluating academic papers for literature reviews.

Research Question: {research_question}

Papers Found ({total_papers} total):
{papers_list}

Select the top {max_papers} most relevant papers for this literature review.

Selection Criteria:
1. Direct relevance to the research question
2. Citation count and impact
3. Recency (prefer recent work unless seminal papers)
4. Methodological rigor
5. Diversity of perspectives

Output as a JSON array of paper IDs:
["paper_id_1", "paper_id_2", ...]"""


# =============================================================================
# Critical Reading Agent Prompts
# =============================================================================

KEY_FINDINGS_PROMPT = """You are an expert at extracting key findings from academic papers.

Paper Title: {title}
Authors: {authors}
Year: {year}

Abstract:
{abstract}

Full Text (or available sections):
{full_text}

Extract 3-5 key findings from this paper. Each finding should be:
- Specific and measurable where possible
- Supported by evidence from the paper
- Written as a complete, self-contained sentence
- Relevant to the broader research field

Output as a JSON array:
[
  "Finding 1: ...",
  "Finding 2: ...",
  ...
]"""


METHODOLOGY_ANALYSIS_PROMPT = """You are an expert at analyzing research methodologies.

Paper Title: {title}

Methods Section (or relevant content):
{methods_text}

Analyze and summarize the research methodology in 2-3 paragraphs covering:
1. Study design and approach
2. Data collection methods and sources
3. Analysis techniques used
4. Key parameters or configurations

Be specific about techniques, datasets, and metrics mentioned.
Write in academic prose suitable for a literature review."""


QUALITY_ASSESSMENT_PROMPT = """You are an expert at critically evaluating academic papers.

Paper Title: {title}
Authors: {authors}
Year: {year}

Content Summary:
{content_summary}

Key Findings:
{key_findings}

Assess the paper's quality by identifying:

STRENGTHS (2-4 points):
- What does this paper do well?
- What are its contributions to the field?
- What methodological strengths does it have?

LIMITATIONS (2-4 points):
- What are the paper's weaknesses?
- What gaps or biases exist?
- What threats to validity are present?

Output as JSON:
{{
  "strengths": ["strength1", "strength2", ...],
  "limitations": ["limitation1", "limitation2", ...]
}}"""


# =============================================================================
# Combined Critical Reading Prompt (Optimized for Speed)
# =============================================================================

COMBINED_PAPER_ANALYSIS_PROMPT = """You are an expert academic researcher performing critical reading of a research paper.

Paper Title: {title}
Authors: {authors}
Year: {year}

Abstract:
{abstract}

Full Text (or available sections):
{full_text}

Perform a complete critical analysis of this paper. Provide:

1. KEY FINDINGS (3-5 findings):
   - Specific and measurable where possible
   - Supported by evidence from the paper
   - Written as complete, self-contained sentences

2. METHODOLOGY SUMMARY (2-3 paragraphs):
   - Study design and approach
   - Data collection methods and sources
   - Analysis techniques and key parameters

3. QUALITY ASSESSMENT:
   - STRENGTHS (2-4 points): What the paper does well, contributions, methodological strengths
   - LIMITATIONS (2-4 points): Weaknesses, gaps, biases, threats to validity

Output as JSON:
{{
  "key_findings": [
    "Finding 1: ...",
    "Finding 2: ...",
    "Finding 3: ..."
  ],
  "methodology": "Study design paragraph...\\n\\nData collection paragraph...\\n\\nAnalysis techniques paragraph...",
  "strengths": ["strength1", "strength2", ...],
  "limitations": ["limitation1", "limitation2", ...]
}}"""


# =============================================================================
# Synthesis Agent Prompts
# =============================================================================

THEME_IDENTIFICATION_PROMPT = """You are an expert at synthesizing academic literature and identifying themes.

Research Question: {research_question}

Papers Analyzed ({num_papers} papers):
{paper_summaries}

Identify 3-6 major themes that emerge across these papers. For each theme:
1. Give it a clear, descriptive name
2. Explain what aspects of the research it covers
3. List which papers contribute to this theme
4. Summarize the key points within this theme

Output as JSON:
[
  {{
    "theme": "Theme Name",
    "description": "What this theme covers...",
    "paper_ids": ["id1", "id2", ...],
    "key_points": ["point1", "point2", ...]
  }},
  ...
]"""


GAP_ANALYSIS_PROMPT = """You are an expert at identifying research gaps in academic literature.

Research Question: {research_question}

Papers Analyzed:
{paper_summaries}

Identified Themes:
{themes}

Identify research gaps and future directions:

GAPS (3-5 gaps):
- What questions remain unanswered?
- What methodological limitations exist across studies?
- What populations, contexts, or scenarios are understudied?

FUTURE DIRECTIONS (3-5 directions):
- What research would address these gaps?
- What new approaches could be explored?
- What interdisciplinary connections could be made?

Output as JSON:
{{
  "gaps": ["gap1", "gap2", ...],
  "future_directions": ["direction1", "direction2", ...]
}}"""


LITERATURE_REVIEW_PROMPT = """You are an expert academic writer creating a literature review.

Review Type: {review_type}
Research Question: {research_question}

Papers ({num_papers} papers):
{paper_summaries}

Themes:
{themes}

Research Gaps:
{gaps}

Write a comprehensive {review_type} literature review that:

1. INTRODUCTION (1 paragraph)
   - Introduce the research question and its significance
   - Preview the scope and structure of the review

2. THEMATIC ANALYSIS (main body)
   - Organize by themes identified
   - For each theme:
     * Synthesize findings across papers (don't just summarize each paper)
     * Compare and contrast different approaches
     * Highlight areas of agreement and disagreement

3. CRITICAL DISCUSSION (1-2 paragraphs)
   - Discuss patterns and trends across the literature
   - Address methodological considerations
   - Note limitations in the current body of research

4. GAPS AND FUTURE DIRECTIONS (1 paragraph)
   - Summarize key research gaps
   - Suggest future research directions

5. CONCLUSION (1 paragraph)
   - Summarize main insights
   - Restate significance of findings

Requirements:
- Use in-text citations in the format [Author, Year]
- Write in formal academic prose
- Target approximately {word_count} words
- Maintain objectivity while being analytical

Write the review now:"""


CITATION_FORMAT_PROMPT = """Format the following paper information as {citation_style} citations.

Papers:
{papers}

Output each citation on a new line, formatted according to {citation_style} style.
Include all available information (authors, year, title, journal/venue, DOI if available)."""


# =============================================================================
# GraphRAG-Enhanced Synthesis Prompts (Phase 7.5)
# =============================================================================

GRAPHRAG_LITERATURE_REVIEW_PROMPT = """You are an expert academic writer creating a literature review enhanced with knowledge graph insights.

Review Type: {review_type}
Research Question: {research_question}

## Knowledge Graph Context
The following entities, relationships, and research clusters were automatically extracted from the literature. Use these to identify cross-paper connections and ensure comprehensive coverage.

{knowledge_graph_context}

## Global Research Landscape
{global_summary}

## Papers ({num_papers} papers):
{paper_summaries}

## Research Clusters (Themes):
{themes}

## Research Gaps:
{gaps}

Write a comprehensive {review_type} literature review that:

1. INTRODUCTION (1-2 paragraphs)
   - Introduce the research question and its significance
   - Preview the key research clusters/themes you will discuss
   - Briefly mention the knowledge landscape (key methods, datasets, concepts)

2. THEMATIC ANALYSIS (main body, organized by research clusters)
   For each research cluster:
   * Synthesize findings across papers (don't just summarize each paper)
   * Highlight connections between methods, datasets, and concepts identified in the knowledge graph
   * Compare and contrast different approaches
   * Reference specific papers with [Author, Year] citations
   * Note how entities (methods, datasets, metrics) flow between papers

3. CROSS-CUTTING ANALYSIS (1-2 paragraphs)
   - Discuss patterns and connections across research clusters
   - Identify key methodological trends
   - Note which entities (methods, datasets) appear across multiple clusters
   - Address areas of agreement and disagreement

4. CRITICAL DISCUSSION (1-2 paragraphs)
   - Discuss methodological considerations
   - Note limitations in the current body of research
   - Identify understudied entities or relationships

5. GAPS AND FUTURE DIRECTIONS (1 paragraph)
   - Summarize key research gaps
   - Suggest future research directions based on the knowledge graph analysis
   - Identify promising entity combinations not yet explored

6. CONCLUSION (1 paragraph)
   - Summarize main insights from the knowledge graph perspective
   - Restate significance of findings

Requirements:
- Use in-text citations in the format [Author, Year]
- Write in formal academic prose
- Target approximately {word_count} words
- Leverage the knowledge graph context to make explicit connections between papers
- Maintain objectivity while being analytical

Write the review now:"""


# =============================================================================
# Supervisor Agent Prompts
# =============================================================================

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


# =============================================================================
# Utility Functions
# =============================================================================

def format_papers_for_prompt(papers: list, max_chars: int = 8000) -> str:
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

        line = f"{i}. [{paper_id}] {title}\n   Authors: {authors} | Year: {year} | Citations: {citations}"

        if total_chars + len(line) > max_chars:
            lines.append(f"... and {len(papers) - i + 1} more papers")
            break

        lines.append(line)
        total_chars += len(line)

    return "\n".join(lines)


def format_summaries_for_prompt(summaries: list, max_chars: int = 12000) -> str:
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
        findings = summary.get("key_findings", [])

        section = f"""
### {title} ({year}) [ID: {paper_id}]
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


# =============================================================================
# Language Instructions for Multi-Language Review Generation (Phase 8.6)
# =============================================================================

LANGUAGE_INSTRUCTIONS = {
    "en": "",  # No additional instruction needed for English
    "zh": (
        "\n\nIMPORTANT LANGUAGE REQUIREMENT: Write the entire review in Chinese (中文). "
        "Use Chinese academic writing conventions and formal scholarly tone. "
        "Section headings should be in Chinese (e.g., 引言, 主题分析, 批判性讨论, 研究空白与未来方向, 结论). "
        "Keep in-text citations in their original format [Author, Year]. "
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
        "Keep in-text citations in their original format [Author, Year]."
    )
