"""Discovery Agent Prompts — query expansion, paper selection, abstract screening."""

QUERY_EXPANSION_PROMPT = """You are an expert academic researcher. Given a research question, generate diverse search queries across DISTINCT dimensions to maximize literature coverage.

Research Question: {research_question}
Research Domain: {domain_hint}

Generate EXACTLY 12 search queries distributed across these 6 dimensions:

1. Core Methodology (2 queries): Target specific methods, techniques, mechanisms, or algorithms central to the topic. Use precise technical terms.
2. Application Domain (2 queries): Target real-world applications, use cases, or practical implementations. Include domain-specific terminology.
3. Review & Meta-Analysis (2 queries): Target existing reviews, surveys, systematic reviews, or meta-analyses on this topic. Include both broad and focused review queries.
4. Recent Advances (2 queries): Target cutting-edge, emerging, or state-of-the-art research from the last 2-3 years. Use terms like "novel", "recent", or specific new technique names.
5. Cross-disciplinary (2 queries): Target related sub-fields or interdisciplinary angles within or adjacent to {domain_hint}.
6. Synonyms & Alternative Nomenclature (2 queries): Target alternative names, abbreviations, historical terms, or nomenclature variants for key concepts in the research question.

Requirements:
- Each query: 3-8 words, optimized for academic search engines
- Avoid overly generic terms that could match unrelated fields
- PREFER queries within the research domain "{domain_hint}", but allow 1-2 queries reaching into adjacent fields if they could yield relevant foundational or interdisciplinary literature
- Maximize diversity: each query should find DIFFERENT papers
- All queries MUST be in English, even if the research question is in another language

Output as a JSON object:
{{
  "core_methodology": ["query1", "query2"],
  "application_domain": ["query3", "query4"],
  "review_meta": ["query5", "query6"],
  "recent_advances": ["query7", "query8"],
  "cross_disciplinary": ["query9", "query10"],
  "synonyms_nomenclature": ["query11", "query12"]
}}"""


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

A paper is IRRELEVANT if ANY of these apply:
- It is from a completely different field/domain (e.g., clinical trials when the question is about basic science, immunology when the question is about plant biochemistry)
- It only mentions the topic in passing (not its main focus)
- The keyword match is coincidental (same term used in a completely different context)
- It is about a fundamentally different application/organism/system with no transferable insights
- It is a clinical trial, epidemiology study, or medical intervention study when the research question is about biochemistry, molecular biology, or basic science (and vice versa)

A paper is RELEVANT if:
- It directly studies the research question or a closely related aspect
- It provides background knowledge, methodology, or foundational theory useful for the review
- Its findings would be cited in a focused review on this topic

Papers:
{papers_batch}

Output as a JSON array (one entry per paper):
[
  {{"paper_id": "...", "relevant": true, "reason": "1-sentence explanation"}},
  {{"paper_id": "...", "relevant": false, "reason": "1-sentence explanation"}}
]"""
