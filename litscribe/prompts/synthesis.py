"""Synthesis Agent Prompts — themes, gaps, literature review generation."""

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
   - Cite key foundational or review papers when making background claims (e.g., [Smith et al., 2020])
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

## Citation Checklist — EVERY paper below MUST be cited at least once:
{citation_checklist}

Requirements:
- CITATION FORMAT: Use [LastName, Year] or [LastName et al., Year] with the real author surnames from the Papers section above. Copy the author names precisely — do not alter spelling. Every citation MUST include the year. NEVER omit the year. NEVER use generic placeholders.
- CITATION COVERAGE: You MUST cite ALL {num_papers} papers listed above. Every paper in the checklist MUST appear as a [LastName, Year] citation at least once. Do NOT skip any paper. If a paper seems less central, still cite it in a supporting role (e.g., "consistent with [Author, Year]" or "see also [Author, Year]").
- CITATION DENSITY: Every factual claim, finding, or method description MUST be backed by at least one citation from the provided papers. Do NOT write any factual statement based on your own knowledge without citing a source paper.
- STRUCTURE: Do NOT number the sections (no "1.", "2.", etc.). Use markdown headings (## and ###) without numbering.
- Write in formal academic prose
- Target approximately {word_count} words
- Maintain objectivity while being analytical

Write the review now:"""


CITATION_FORMAT_PROMPT = """Format the following paper information as {citation_style} citations.

Papers:
{papers}

Output each citation on a new line, formatted according to {citation_style} style.
Include all available information (authors, year, title, journal/venue, DOI if available)."""


GRAPHRAG_LITERATURE_REVIEW_PROMPT = """You are an expert academic writer creating a literature review enhanced with knowledge graph insights.

Review Type: {review_type}
Research Question: {research_question}

## Knowledge Graph Context
The following entities, relationships, and research clusters were automatically extracted from the literature. These are provided as BACKGROUND CONTEXT ONLY to help you understand the research landscape — they are NOT citable sources.

{knowledge_graph_context}

## Global Research Landscape
{global_summary}

## Papers ({num_papers} papers):
{paper_summaries}

## Thematic Organization (for structuring your review):
{themes}

## Research Gaps:
{gaps}

Write a comprehensive {review_type} literature review that:

1. INTRODUCTION (1-2 paragraphs)
   - Introduce the research question and its significance
   - Cite key foundational or review papers when making background claims (e.g., [Smith et al., 2020])
   - Preview the key themes you will discuss
   - Briefly mention the knowledge landscape (key methods, datasets, concepts)

2. THEMATIC ANALYSIS (main body, organized by themes)
   For each theme:
   * Synthesize findings across papers (don't just summarize each paper)
   * Highlight connections between methods, datasets, and concepts identified in the knowledge graph
   * Compare and contrast different approaches
   * Cite individual papers using [LastName, Year] format (e.g. [Smith, 2020], [Zhang et al., 2019])
   * Note how entities (methods, datasets, metrics) flow between papers

3. CROSS-CUTTING ANALYSIS (1-2 paragraphs)
   - Discuss patterns and connections across themes
   - Identify key methodological trends
   - Note which entities (methods, datasets) appear across multiple themes
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

## Citation Checklist — EVERY paper below MUST be cited at least once:
{citation_checklist}

Requirements:
- CITATION FORMAT: ONLY cite individual papers using [LastName, Year] or [LastName et al., Year] with the real author surnames from the Papers section above (e.g., [Smith, 2020], [Zhang et al., 2019]). Copy the author names precisely — do not alter spelling. Every citation MUST include the year. NEVER omit the year. NEVER use generic placeholders like [Author, Year].
- DO NOT cite research clusters, themes, knowledge graph summaries, or any non-paper source. "[研究集群]", "[Cluster]", "[知识图谱]" etc. are FORBIDDEN — only [AuthorName, Year] citations are allowed.
- CITATION COVERAGE: You MUST cite ALL {num_papers} papers listed above. Every paper in the checklist MUST appear as a [LastName, Year] citation at least once. Do NOT skip any paper. If a paper seems less central, still cite it in a supporting role (e.g., "consistent with [Author, Year]" or "see also [Author, Year]").
- CITATION DENSITY: Every factual claim, finding, or method description MUST be backed by at least one citation from the provided papers. Do NOT write any factual statement based on your own knowledge without citing a source paper.
- STRUCTURE: Do NOT number the sections (no "1.", "2.", etc.). Use markdown headings (## and ###) without numbering.
- Write in formal academic prose
- Target approximately {word_count} words
- Leverage the knowledge graph context to make explicit connections between papers
- Maintain objectivity while being analytical

Write the review now:"""


REVIEW_INTRO_PROMPT = """You are an expert academic writer creating the INTRODUCTION section of a literature review.

Review Type: {review_type}
Research Question: {research_question}
Number of Papers: {num_papers}
{knowledge_graph_section}
The review will cover the following themes:
{theme_names}

## Paper Summaries (use these to support factual claims):
{paper_summaries}

Write a concise introduction (2-3 paragraphs) that:
1. Introduces the research question and its significance
2. Briefly previews the scope — {num_papers} papers analyzed
3. Lists the themes that will be discussed (one sentence each)

STRICT FORMATTING RULES:
- Start with a level-1 heading: # [Your Review Title]
- Then write 2-3 paragraphs of introduction
- Cite key papers when making factual claims using [Author, Year] format (use the paper summaries above)
- Do NOT include any sub-headings, section numbers, or bullet lists
- Do NOT write a conclusion or summary within this section
- End with a transitional sentence leading into the first theme
- Target approximately {word_count} words — do NOT exceed this limit
- Write in formal academic prose

Write the introduction now:"""


REVIEW_THEME_SECTION_PROMPT = """You are an expert academic writer continuing a literature review. Write the section for ONE specific theme.

This is section {theme_number} of {total_themes} in the review.

Research Question: {research_question}
{knowledge_graph_section}
Theme: {theme_name}
Theme Description: {theme_description}

Key Points:
{key_points}

## Papers for this theme ({num_theme_papers} papers):
{theme_papers}

## Citation Checklist — EVERY paper below MUST be cited at least once in this section:
{theme_citation_checklist}

## Context from previous section (for smooth transition):
...{previous_ending}

Write the analysis for this theme:
1. Begin with a natural transition from the previous section
2. Synthesize findings across papers — do NOT just summarize each paper individually
3. Compare and contrast different approaches, methods, and findings
4. Highlight areas of agreement and disagreement

STRICT FORMATTING RULES:
- Start with EXACTLY this heading: ## {theme_number}. {theme_name}
- Use ### for sub-sections within this theme if needed
- Do NOT write a conclusion, summary, or "小结" for this section — the overall conclusion comes later
- Do NOT re-number this section — it is section {theme_number}
- CITATION FORMAT: Use [LastName, Year] or [LastName et al., Year] with the EXACT author surnames from the papers listed above. Copy the author names precisely — do not alter spelling.
- CITATION COVERAGE: You MUST cite ALL {num_theme_papers} papers listed in the checklist. Do NOT skip any paper.
- CITATION DENSITY: Every factual claim MUST be backed by at least one citation.
- Target approximately {word_count} words — do NOT exceed this limit
- Write in formal academic prose

Write this theme section now:"""


REVIEW_CONCLUSION_PROMPT = """You are an expert academic writer writing the final sections of a literature review.

Research Question: {research_question}
Number of Papers Reviewed: {num_papers}
{knowledge_graph_section}
Themes Covered:
{theme_names}

Research Gaps:
{gaps}

## Context from previous section (for smooth transition):
...{previous_ending}

{uncited_section}

Write three sections:

## Critical Discussion
(1-2 paragraphs) Discuss patterns and trends across the themes. Address methodological considerations and limitations. Note areas of agreement and disagreement.

## Research Gaps and Future Directions
(1 paragraph) Summarize key research gaps. Suggest concrete future research directions.

## Conclusion
(1 paragraph) Summarize the main insights. Restate the significance for the field.

STRICT FORMATTING RULES:
- Use EXACTLY the three ## headings shown above (Critical Discussion, Research Gaps and Future Directions, Conclusion)
- Begin with a natural transition from the previous theme section
- If uncited papers are listed above, incorporate them (e.g., "see also [Author, Year]")
- CITATION FORMAT: Use [LastName, Year] with EXACT author surnames. Copy names precisely from the papers.
- Target approximately {word_count} words — do NOT exceed this limit
- Write in formal academic prose
- This is the ONLY conclusion in the entire review — do not repeat themes in detail

Write the conclusion sections now:"""
