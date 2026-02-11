"""LLM Prompt Templates for LitScribe Agents.

This module contains all prompt templates used by the multi-agent system.
Templates use Python format strings for variable substitution.
"""

# =============================================================================
# Discovery Agent Prompts
# =============================================================================

QUERY_EXPANSION_PROMPT = """You are an expert academic researcher. Given a research question, generate diverse search queries to find relevant academic literature.

Research Question: {research_question}
Research Domain: {domain_hint}

Generate exactly 8 search queries that:
1. Rephrase the original question with precise academic terminology
2. Focus on specific methodologies or techniques mentioned or implied
3. Target application domains or use cases
4. Use synonyms and related concepts from the field
5. Include combinations of key concepts
6. Explore cross-disciplinary angles within {domain_hint}
7. Use alternative nomenclature or classification terms
8. Target review/survey papers on the topic
9. STAY WITHIN the research domain "{domain_hint}". Do NOT generate queries that would match papers from unrelated fields.

Requirements:
- Each query should be 3-8 words, optimized for academic search engines
- Avoid overly generic terms that could match unrelated fields
- Include field-specific terminology from {domain_hint}
- Maximize diversity: each query should find DIFFERENT papers, not repeat the same results
- IMPORTANT: All queries MUST be in English, even if the research question is in another language. Academic databases (arXiv, PubMed, Semantic Scholar) are English-based.

Output as a JSON array of strings:
["query1", "query2", "query3", "query4", "query5", "query6", "query7", "query8"]"""


PAPER_SELECTION_PROMPT = """You are an expert at evaluating academic papers for literature reviews.

Research Question: {research_question}
Research Domain: {domain_hint}
{sub_topics_section}
Papers Found ({total_papers} total):
{papers_list}

Select the top {max_papers} most relevant papers for this literature review.

SELECTION criteria (paper MUST meet these):
1. Directly addresses the research question or one of the sub-topics listed above
2. From the correct domain ({domain_hint}). REJECT papers from unrelated fields
3. The paper's primary focus is on the target topic, not just a passing mention
4. Citation count and impact (prefer well-cited papers)
5. Recency (prefer recent work unless seminal papers)
6. Diversity of perspectives and methods

EXCLUSION criteria (MUST reject papers matching ANY of these):
- Papers that only tangentially mention the topic (e.g., uses a keyword as a tool/example but studies something else entirely)
- Papers from clearly unrelated domains, even if they share keywords
- Broad review papers where the target topic is just one of many subjects covered
- Papers where keyword match is coincidental (e.g., same compound name in a completely different biological context)
- Clinical trials, drug delivery, or pharmacology papers when the research question is about basic science (biosynthesis, mechanisms, etc.), and vice versa
- Papers primarily about a different organism/system that happen to reference the target topic

When in doubt, prefer papers that would be cited in a focused review specifically about "{research_question}".

Output as a JSON array of paper IDs:
["paper_id_1", "paper_id_2", ...]"""


ABSTRACT_SCREENING_PROMPT = """You are screening papers for a literature review. Your job is to quickly classify each paper as "relevant" or "irrelevant" based on its abstract.

Research Question: {research_question}
Research Domain: {domain_hint}

For each paper below, decide if it is RELEVANT (should be analyzed in depth) or IRRELEVANT (should be excluded).

A paper is IRRELEVANT if:
- It is from a completely different field/domain
- It only mentions the topic in passing (not its main focus)
- The keyword match is coincidental (different context)
- It is about a different application/organism/system

A paper is RELEVANT if:
- It directly studies the research question or a closely related aspect
- Its findings would be cited in a focused review on this topic

Papers:
{papers_batch}

Output as a JSON array (one entry per paper):
[
  {{"paper_id": "...", "relevant": true, "reason": "1-sentence explanation"}},
  {{"paper_id": "...", "relevant": false, "reason": "1-sentence explanation"}}
]"""


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

Research Question Context: {research_question}

Paper Title: {title}
Authors: {authors}
Year: {year}

Abstract:
{abstract}

Full Text (or available sections):
{full_text}

Perform a complete critical analysis of this paper in the context of the research question above. Provide:

1. KEY FINDINGS (5-8 findings):
   - Specific and measurable where possible
   - Supported by evidence from the paper
   - Written as complete, self-contained sentences
   - Focus on findings relevant to the research question

2. METHODOLOGY SUMMARY (2-3 paragraphs):
   - Study design and approach
   - Data collection methods and sources
   - Analysis techniques and key parameters

3. QUALITY ASSESSMENT:
   - STRENGTHS (2-4 points): What the paper does well, contributions, methodological strengths
   - LIMITATIONS (2-4 points): Weaknesses, gaps, biases, threats to validity

4. RELEVANCE ASSESSMENT:
   - How relevant is this paper to the research question? Score 0.0-1.0
   - 1.0 = directly addresses the research question
   - 0.5 = tangentially related
   - 0.0 = completely unrelated (different field/topic)

Output as JSON:
{{
  "key_findings": [
    "Finding 1: ...",
    "Finding 2: ...",
    "Finding 3: ...",
    "Finding 4: ...",
    "Finding 5: ..."
  ],
  "methodology": "Study design paragraph...\\n\\nData collection paragraph...\\n\\nAnalysis techniques paragraph...",
  "strengths": ["strength1", "strength2", ...],
  "limitations": ["limitation1", "limitation2", ...],
  "relevance_to_question": 0.0-1.0
}}"""


ABSTRACT_ONLY_ANALYSIS_PROMPT = """You are an expert academic researcher analyzing a paper based on its ABSTRACT and metadata ONLY (full text is unavailable).

Research Question Context: {research_question}

Paper Title: {title}
Authors: {authors}
Year: {year}
Venue: {venue}

Abstract:
{abstract}

Additional Metadata:
{metadata_section}

Since you only have the abstract and metadata, focus your analysis on:
- What specific research question does this paper address?
- What methodology is described or implied in the abstract?
- What are the main claims/results explicitly stated?
- How does this relate to the research question: "{research_question}"?
- What can you infer about the methodology from the venue and field?

IMPORTANT:
- Do NOT fabricate findings not present in the abstract
- Mark uncertain inferences with [inferred from abstract]
- Be honest about the depth limitations of abstract-only analysis

Output as JSON:
{{
  "key_findings": [
    "Finding 1: ...",
    "Finding 2: ...",
    "Finding 3: ..."
  ],
  "methodology": "Brief methodology summary based on what the abstract reveals...",
  "strengths": ["strength1", "strength2"],
  "limitations": ["limitation1", "Abstract-only analysis — findings may be incomplete"],
  "relevance_to_question": 0.0-1.0
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


# =============================================================================
# GraphRAG-Enhanced Synthesis Prompts (Phase 7.5)
# =============================================================================

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


# =============================================================================
# Sectioned Review Generation Prompts (for long reviews exceeding 8K token limit)
# =============================================================================

REVIEW_INTRO_PROMPT = """You are an expert academic writer creating the INTRODUCTION section of a literature review.

Review Type: {review_type}
Research Question: {research_question}
Number of Papers: {num_papers}
{knowledge_graph_section}
The review will cover the following themes:
{theme_names}

Write a concise introduction (2-3 paragraphs) that:
1. Introduces the research question and its significance
2. Briefly previews the scope — {num_papers} papers analyzed
3. Lists the themes that will be discussed (one sentence each)

STRICT FORMATTING RULES:
- Start with a level-1 heading: # [Your Review Title]
- Then write 2-3 paragraphs of introduction
- Do NOT include any citations — citations appear in theme sections only
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
# Planning Agent Prompts (Phase 9.2)
# =============================================================================

COMPLEXITY_ASSESSMENT_PROMPT = """You are an expert academic researcher. Assess the complexity of this research question for a literature review.

Research Question: {research_question}

Rate the complexity on a scale of 1-5:
1 = Very simple, single focused topic (e.g., "BERT performance on GLUE benchmark")
2 = Simple, one main topic with clear scope (e.g., "attention mechanisms in transformers")
3 = Moderate, crosses 2-3 sub-areas (e.g., "LLM reasoning capabilities")
4 = Complex, multiple interconnected sub-topics (e.g., "AI applications in drug discovery")
5 = Very complex, broad interdisciplinary topic (e.g., "ethical implications of AI in healthcare")

Also decompose the question into sub-topics if complexity >= 3.

Additionally, identify the PRIMARY academic domain and provide search filters for each database:

Output as JSON:
{{
  "complexity_score": 1-5,
  "reasoning": "Why this complexity level...",
  "review_title": "A formal academic title for the review (in English, concise, descriptive, suitable as a paper title)",
  "needs_clarification": false,
  "clarification_questions": [],
  "domain": "Primary academic field (e.g., Biology, Computer Science, Medicine, Chemistry, Physics)",
  "arxiv_categories": ["arXiv category codes, e.g. q-bio.BM, cs.AI, cs.CL"],
  "s2_fields": ["Semantic Scholar fields, e.g. Biology, Computer Science, Medicine"],
  "pubmed_mesh": ["PubMed MeSH terms for the topic, e.g. Alkaloids, Biosynthetic Pathways"],
  "sub_topics": [
    {{
      "name": "Sub-topic name",
      "description": "What this covers",
      "estimated_papers": 5-20,
      "priority": 0.0-1.0,
      "custom_queries": ["(\"term A\" OR \"synonym\") AND \"term B\"", "query 2", "query 3"]
    }}
  ],
  "scope_estimate": "Estimated X-Y papers across N sub-topics"
}}

For "review_title": Generate a formal, publication-ready academic title (in English) based on the research question. The title should be concise (under 20 words), descriptive, and suitable as a literature review paper title. Example: if the user asks "我想看CHO CRISPR knockout相关研究", the title could be "CRISPR-Mediated Gene Knockouts in CHO Cells: Targets and Productivity Impact".

For "needs_clarification": Set to true ONLY if the research question is too vague or ambiguous to produce a meaningful search plan. Examples that need clarification:
- "I want to read about biology" (too broad, which area?)
- "Compare methods for X" (which methods? what criteria?)
- "Recent advances in Y" (how recent? which aspects?)
Do NOT set needs_clarification for questions that are simply informal or in non-English — you can infer intent from context. Most questions do NOT need clarification.

For "clarification_questions": If needs_clarification is true, provide 1-3 specific questions to help narrow the scope. Each question should be concise and actionable.

For complexity 1-2, provide 1-2 sub-topics (the question itself is essentially the topic).
For complexity 3-5, provide 3-6 sub-topics that together cover the research question comprehensively.

Ensure sub-topics are:
- Mutually exclusive (minimal overlap)
- Collectively exhaustive (cover the full question)
- Ordered by priority (most important first)
- Each with 3-5 specific search queries optimized for academic databases
- IMPORTANT: All search queries MUST be in English
- IMPORTANT: Queries MUST use Boolean syntax for precision:
  * Use AND / OR operators to combine terms
  * Use "double quotes" for exact multi-word phrases (e.g., "flux balance analysis")
  * Include synonyms and abbreviations with OR (e.g., "CHO" OR "Chinese hamster ovary")
  * Mix broad and narrow queries: some highly specific, some broader to catch related work
  * Example good query: ("sesquiterpene coumarin" OR "prenylated coumarin") AND (biosynthesis OR "biosynthetic pathway") AND (Ferula OR Apiaceae)
  * Example bad query: sesquiterpene coumarin biosynthesis plants (no operators, too vague)

For domain detection:
- arxiv_categories: Use official arXiv taxonomy (cs.*, q-bio.*, physics.*, math.*, stat.*, etc.)
- s2_fields: Use Semantic Scholar fields (Biology, Chemistry, Computer Science, Medicine, Physics, etc.)
- pubmed_mesh: Use 2-4 top-level MeSH terms that define the research scope"""


PLAN_REVISION_PROMPT = """You are an expert academic researcher. A user has rejected a proposed research plan and provided feedback. Revise the plan accordingly.

Research Question: {research_question}

## Current Plan:
{current_plan_json}

## User Feedback:
{user_feedback}

Revise the plan to address the user's feedback. You may:
- Add, remove, or modify sub-topics
- Adjust estimated paper counts
- Change priority ordering
- Modify search queries
- Adjust complexity assessment
- Update domain filters (arxiv_categories, s2_fields, pubmed_mesh)

IMPORTANT:
- Preserve any aspects of the original plan that the user did NOT criticize
- Keep all search queries in English
- Maintain the same JSON output format as the original plan

Output the revised plan as JSON:
{{
  "complexity_score": 1-5,
  "reasoning": "Why this revised complexity level...",
  "review_title": "Updated formal academic title for the review (in English)",
  "domain": "Primary academic field",
  "arxiv_categories": ["arXiv category codes"],
  "s2_fields": ["Semantic Scholar fields"],
  "pubmed_mesh": ["PubMed MeSH terms"],
  "sub_topics": [
    {{
      "name": "Sub-topic name",
      "description": "What this covers",
      "estimated_papers": 5-20,
      "priority": 0.0-1.0,
      "custom_queries": ["(\"term A\" OR \"synonym\") AND \"term B\"", "query 2", "query 3"]
    }}
  ],
  "scope_estimate": "Estimated X-Y papers across N sub-topics"
}}"""


# =============================================================================
# Self-Review Agent Prompts (Phase 9.1)
# =============================================================================

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


# =============================================================================
# Refinement Agent Prompts (Phase 9.3)
# =============================================================================

REFINEMENT_CLASSIFY_PROMPT = """You are a text editing assistant. Classify the following user instruction into an action type for modifying an academic literature review.

User Instruction: {instruction}

Current review excerpt (first 500 words):
{review_excerpt}

Classify into one of these action types:
- "add_content": Add new content, discussion, or section to the review
- "remove_content": Remove or delete a section, paragraph, or topic
- "modify_content": Change existing content (rephrase, expand, condense, restructure)
- "rewrite_section": Completely rewrite a specific section
- "add_papers": Add new papers that require additional search (NOT SUPPORTED YET)

Output as JSON:
{{
  "action_type": "add_content|remove_content|modify_content|rewrite_section|add_papers",
  "target_section": "section name or null if applies to whole review",
  "details": "Specific description of what to change"
}}"""


REFINEMENT_EXECUTE_PROMPT = """You are an expert academic writer. Modify an existing literature review based on the following instruction.

Research Question: {research_question}

## Current Review:
{current_review}

## Available Papers for Citation:
{papers_context}

## Modification Instruction:
- Action: {action_type}
- Target Section: {target_section}
- Details: {details}

## Requirements:
- Apply the requested modification precisely
- Preserve the overall structure and academic tone
- Keep all existing in-text citations in [LastName, Year] format using real author surnames
- Only use citations from the available papers list
- If adding content, integrate it naturally into the existing flow
- If removing content, ensure remaining text still flows logically
- If rewriting a section, maintain connections to surrounding sections
- Maintain the same language as the original review

Output the COMPLETE modified review text (not just the changed parts). Do not include any commentary or explanation outside the review text itself."""


# =============================================================================
# Utility Functions
# =============================================================================

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
