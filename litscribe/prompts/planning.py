"""Planning Agent Prompts — complexity assessment, plan revision."""

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
      "custom_queries": ["(\"term A\" OR \"synonym\") AND \"term B\"", "query 2", "query 3", "query 4", "query 5"],
      "arxiv_categories": ["topic-specific arXiv categories, or empty [] to inherit plan-level"],
      "s2_fields": ["topic-specific S2 fields, or empty [] to inherit plan-level"],
      "pubmed_mesh": ["topic-specific MeSH terms, or empty [] to inherit plan-level"]
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
- Each with 3-5 specific search queries optimized for academic databases (up to 5 for comprehensive coverage)
- IMPORTANT: All search queries MUST be in English
- IMPORTANT: Queries MUST use Boolean syntax for precision:
  * Use AND / OR operators to combine terms
  * Use "double quotes" for exact multi-word phrases (e.g., "flux balance analysis")
  * Include synonyms and abbreviations with OR (e.g., "CHO" OR "Chinese hamster ovary")
  * Mix broad and narrow queries: some highly specific, some broader to catch related work
  * Example good query: ("sesquiterpene coumarin" OR "prenylated coumarin") AND (biosynthesis OR "biosynthetic pathway") AND (Ferula OR Apiaceae)
  * Example bad query: sesquiterpene coumarin biosynthesis plants (no operators, too vague)

For domain detection (plan-level):
- arxiv_categories: Use official arXiv taxonomy (cs.*, q-bio.*, physics.*, math.*, stat.*, etc.)
- s2_fields: Use Semantic Scholar fields (Biology, Chemistry, Computer Science, Medicine, Physics, etc.)
- pubmed_mesh: Use 2-4 top-level MeSH terms that define the research scope

For per-subtopic filters:
- If a sub-topic spans a different field from the overall domain, provide topic-specific arxiv_categories/s2_fields/pubmed_mesh
- Otherwise leave them as empty arrays [] to inherit the plan-level defaults
- Example: a bioinformatics sub-topic under a Biology plan might use ["cs.CE", "q-bio.QM"] for arxiv_categories"""


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
      "custom_queries": ["(\"term A\" OR \"synonym\") AND \"term B\"", "query 2", "query 3", "query 4", "query 5"],
      "arxiv_categories": ["topic-specific arXiv categories, or empty [] to inherit plan-level"],
      "s2_fields": ["topic-specific S2 fields, or empty [] to inherit plan-level"],
      "pubmed_mesh": ["topic-specific MeSH terms, or empty [] to inherit plan-level"]
    }}
  ],
  "scope_estimate": "Estimated X-Y papers across N sub-topics"
}}"""
