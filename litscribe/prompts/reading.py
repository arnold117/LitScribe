"""Critical Reading Agent Prompts — paper analysis, methodology, quality assessment."""

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
