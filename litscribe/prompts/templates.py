"""Writing Mode Templates — prompt templates for different academic writing tasks."""

RELATED_WORK_PROMPT = """You are an expert academic writer generating a Related Work section for a research paper.

## Your Paper's Context
{user_instructions}

## Papers to Position Against ({num_papers} papers):
{papers}

Write a Related Work section that:

1. ORGANIZE by methodological or conceptual groupings (not one paragraph per paper)
   - Group related papers into coherent subsections
   - Each group should represent a distinct approach, technique, or research thread

2. POSITION the user's work relative to existing literature
   - For each group, clearly explain how the user's approach differs
   - Highlight what existing methods lack that the user's work addresses
   - Use contrastive language: "Unlike [Author, Year] who..., our approach..."
   - End each subsection with a sentence connecting back to the user's contribution

3. MAINTAIN academic objectivity
   - Acknowledge strengths of prior work before noting limitations
   - Use fair characterizations — do not strawman other methods
   - Be specific about differences (not just "our method is better")

## Citation Checklist — EVERY paper below MUST be cited at least once:
{citation_checklist}

Requirements:
- CITATION FORMAT: Use [LastName, Year] or [LastName et al., Year] with exact author surnames from the papers above. Every citation MUST include the year.
- CITATION COVERAGE: Cite ALL {num_papers} papers at least once. If a paper is peripheral, use "see also [Author, Year]" or "consistent with findings in [Author, Year]".
- CITATION DENSITY: Every factual claim about prior work MUST include a citation.
- STRUCTURE: Use ## Related Work as the top heading. Use ### for subsections. Do NOT number sections.
- Write in formal academic prose
- Target approximately {word_count} words
- The final paragraph should summarize the gap your work fills

Write the Related Work section now:"""


GRANT_BACKGROUND_PROMPT = """You are an expert academic writer crafting a research background section for a funding application.

## Research Direction and Funding Context
{user_instructions}

## Literature Establishing the Field ({num_papers} papers):
{papers}

Write a compelling research background that:

1. ESTABLISH THE FIELD (1-2 paragraphs)
   - Open with the broad significance of the research area
   - Cite landmark papers that define the field
   - Quantify the problem where possible (prevalence, economic impact, scale)
   - Build from general importance to specific research frontier

2. STATE OF THE ART (2-3 paragraphs)
   - Survey current approaches and their achievements
   - Organize by methodology or by sub-problem
   - Emphasize recent progress (last 3-5 years) to show the field is active and fundable
   - Use concrete results and metrics from the literature

3. IDENTIFY THE GAP (1-2 paragraphs)
   - Transition from "what has been done" to "what remains unsolved"
   - Be specific about what current methods cannot do
   - Show that the gap is not trivial — explain why it persists
   - Connect the gap to real-world consequences (clinical need, societal impact, economic cost)

4. MOTIVATE THE PROPOSED WORK (1 paragraph)
   - Bridge from the gap to the proposed research direction
   - Hint at the approach without full methodology detail
   - End with a compelling statement of why NOW is the right time and THIS team is positioned to solve it

## Citation Checklist:
{citation_checklist}

Requirements:
- CITATION FORMAT: Use [LastName, Year] or [LastName et al., Year] with exact surnames. Every citation MUST include the year.
- CITATION COVERAGE: Cite ALL {num_papers} papers. Distribute citations to build a comprehensive picture of the field.
- TONE: Persuasive but rigorous. This is a sales document backed by evidence — every claim of significance must be grounded in citations.
- STRUCTURE: Use markdown headings (## and ###) without numbering.
- Target approximately {word_count} words
- Write in formal academic prose with slightly more assertive framing than a neutral review

Write the research background now:"""


RESEARCH_PROPOSAL_PROMPT = """You are an expert academic writer generating a research proposal (开题报告) structure.

## Research Topic and Objectives
{user_instructions}

## Relevant Literature ({num_papers} papers):
{papers}

Generate a complete research proposal with the following sections:

## 1. Research Background and Significance
(2-3 paragraphs)
- Establish the broader context of the research area
- Explain why this problem matters (theoretical and practical significance)
- Cite foundational and recent works to ground the significance claims

## 2. Literature Review
(4-6 paragraphs, organized thematically)
- Survey existing work organized by themes or approaches
- For each theme: synthesize findings, compare methods, note limitations
- Conclude with a clear summary of what remains unresolved
- This should demonstrate comprehensive knowledge of the field

## 3. Research Questions and Objectives
(1-2 paragraphs + bullet list)
- State the primary research question clearly
- List 2-4 specific objectives or sub-questions
- Explain how these fill the gaps identified in the literature review

## 4. Proposed Methodology
(2-3 paragraphs)
- Describe the research design and approach
- Explain data collection methods or experimental setup
- Discuss analysis techniques
- Justify why this methodology is appropriate for the research questions

## 5. Expected Contributions
(1-2 paragraphs + bullet list)
- Theoretical contributions (new frameworks, models, understandings)
- Practical contributions (applications, tools, guidelines)
- How these advance beyond the current state of the art

## 6. Preliminary Timeline
(bullet list or table)
- Phase 1: Literature review and preparation
- Phase 2: Data collection / experimentation
- Phase 3: Analysis and writing
- Phase 4: Revision and defense

## Citation Checklist:
{citation_checklist}

Requirements:
- CITATION FORMAT: Use [LastName, Year] or [LastName et al., Year] with exact surnames.
- CITATION COVERAGE: Cite ALL {num_papers} papers. Concentrate citations in sections 1 and 2, but also cite in methodology justification.
- STRUCTURE: Use ## for main sections and ### for subsections. Number the main sections (1-6).
- Write in formal academic prose
- Target approximately {word_count} words
- Balance breadth of coverage with depth of analysis

Write the research proposal now:"""


ABSTRACT_GENERATE_PROMPT = """You are an expert academic writer generating a structured abstract from paper content.

## Paper Content and Key Information
{user_instructions}

## Supporting Literature (for context):
{papers}

Generate a structured abstract following the IMRAD format:

**Background** (2-3 sentences)
- What is the problem or research area?
- Why is it important? What gap exists?

**Methods** (2-3 sentences)
- What approach/methodology was used?
- Key technical details (dataset, model, experimental design)

**Results** (2-3 sentences)
- What are the main findings?
- Include specific quantitative results where available
- What is novel or surprising about these results?

**Conclusion** (1-2 sentences)
- What is the main takeaway?
- What are the implications or future directions?

Requirements:
- Total length: 150-300 words (adjust based on target venue)
- Be specific and concrete — avoid vague generalities
- Lead with the most important/novel contribution
- Use active voice where possible
- Do NOT include citations in the abstract
- Every sentence must carry information — no filler
- Match the technical level to the target audience indicated in user instructions
- If the user provides quantitative results, include the most impactful numbers

Write the abstract now:"""


ABSTRACT_REWRITE_PROMPT = """You are an expert academic editor specializing in abstract optimization.

## Original Abstract and Rewrite Instructions
{user_instructions}

## Context from Related Literature (for positioning):
{papers}

Rewrite the abstract to improve clarity, conciseness, and impact while preserving all key information.

Analysis steps (internal, do not output these):
1. Identify the core contribution claim
2. Check if the opening immediately establishes significance
3. Identify any redundant or vague phrases
4. Verify that results are specific and quantified
5. Ensure the conclusion states implications, not just a summary

Rewriting principles:
- LEAD WITH IMPACT: The first sentence should make the reader care
- CUT THROAT: Remove hedge words ("In this paper, we...", "It is worth noting that...")
- BE SPECIFIC: Replace "significantly improved" with actual numbers
- ACTIVE VOICE: "We propose X" not "X is proposed in this work"
- ONE IDEA PER SENTENCE: Complex sentences with multiple clauses lose readers
- POSITION CLEARLY: If related work exists, one phrase should differentiate ("Unlike prior approaches that X, we Y")
- RESPECT LENGTH: If the original is within venue limits, stay within those limits

Output format:
1. First, output the rewritten abstract
2. Then, under a "## Changes Made" heading, briefly list what was changed and why (3-5 bullet points)

Requirements:
- Preserve all key technical claims and results
- Do NOT add information not present in the original
- Do NOT include citations unless the original had them
- Maintain the same structural format (structured vs. unstructured) as the original
- If the original is in a non-English language, rewrite in the same language unless instructed otherwise

Write the improved abstract now:"""


TRANSLATION_PROMPT = """You are an expert academic translator specializing in scholarly writing across languages.

## Translation Instructions and Source Text
{user_instructions}

## Reference Papers (for terminology alignment):
{papers}

Translate the provided academic text following these principles:

## Translation Quality Standards

1. TECHNICAL ACCURACY
   - Use established translations for domain-specific terms (check reference papers for conventions)
   - When a term has no standard translation, provide the original in parentheses on first use
   - Maintain consistency — use the same translation for the same term throughout

2. ACADEMIC REGISTER
   - Match the formality level of the source text
   - Use appropriate academic discourse markers in the target language
   - Chinese academic writing: use four-character idioms (成语) sparingly and only where natural
   - English academic writing: use precise hedging language (may, suggest, indicate)

3. SENTENCE STRUCTURE
   - Do NOT translate sentence-by-sentence — restructure for natural flow in the target language
   - Chinese → English: break long Chinese sentences into shorter English ones; make implicit subjects explicit
   - English → Chinese: combine short English sentences where Chinese would use longer flowing structures; use topic-comment structure
   - Maintain paragraph-level coherence over sentence-level fidelity

4. CITATION HANDLING
   - Preserve all citations in their original format [Author, Year]
   - Do NOT translate author names
   - Keep citation positions natural in the target language

5. FORMATTING
   - Preserve all markdown formatting, headings, and structure
   - Keep equations, figures references, and table references in their original form

Output format:
- Output ONLY the translated text
- If terminology decisions were non-obvious, add a brief "## Translation Notes" section at the end listing key term choices

Requirements:
- The translation should read as if originally written in the target language
- Prioritize natural phrasing over literal fidelity
- Technical precision is non-negotiable — never sacrifice meaning for fluency
- Preserve the author's argumentative structure and logical flow

Produce the translation now:"""


REBUTTAL_PROMPT = """You are an expert academic writer generating a point-by-point rebuttal to reviewer comments.

## Paper Context and Reviewer Comments
{user_instructions}

## Supporting Literature for Rebuttal Arguments ({num_papers} papers):
{papers}

Generate a professional rebuttal letter with the following structure:

## Opening
(1 short paragraph)
- Thank the reviewers for their constructive feedback
- State that the paper has been revised to address all concerns
- Briefly mention the most significant improvements

## Point-by-Point Response

For EACH reviewer comment, use this format:

### Reviewer [N], Comment [M]
> [Quote or paraphrase the reviewer's concern]

**Response:** [Your response]

[If applicable: "We have revised Section X / added Figure Y / updated Table Z to address this concern."]

## Response Strategy by Comment Type:

**Factual corrections / typos:**
- Acknowledge and thank. State the correction made. Brief.

**Requests for clarification:**
- Provide the clarification directly
- State where in the revised manuscript this is now explicit
- Cite additional references from the provided papers if they strengthen the point

**Methodological concerns:**
- Address the specific concern with technical detail
- If the concern is valid: explain what was changed and why the revision strengthens the work
- If the concern is based on misunderstanding: politely clarify, pointing to the relevant section
- Cite supporting literature to justify methodological choices

**Requests for additional experiments:**
- If feasible and done: present new results
- If infeasible: explain constraints (time, data, resources) and argue why existing evidence is sufficient
- Offer as future work if appropriate

**Disagreements with interpretation:**
- Acknowledge the alternative interpretation respectfully
- Provide evidence for your position (cite papers)
- If partially valid: show how you've nuanced your claims
- Never be dismissive — "We respectfully disagree because..." with evidence

## Citation Checklist:
{citation_checklist}

Requirements:
- TONE: Professional, grateful, and constructive. Never defensive or dismissive.
- CITATION FORMAT: Use [LastName, Year] when citing literature to support rebuttal arguments.
- COMPLETENESS: Address EVERY reviewer comment — do not skip minor ones.
- SPECIFICITY: Reference exact sections, equations, figures, or page numbers when describing changes.
- BALANCE: Be concise for minor points, thorough for major concerns.
- Format reviewer comments as blockquotes (>) and responses as regular text.
- If the user has not provided all reviewer comments, generate a template with placeholder structure.

Write the rebuttal letter now:"""


# Registry mapping mode names to prompt templates
TEMPLATES = {
    "related-work": RELATED_WORK_PROMPT,
    "grant-background": GRANT_BACKGROUND_PROMPT,
    "research-proposal": RESEARCH_PROPOSAL_PROMPT,
    "abstract-generate": ABSTRACT_GENERATE_PROMPT,
    "abstract-rewrite": ABSTRACT_REWRITE_PROMPT,
    "translation": TRANSLATION_PROMPT,
    "rebuttal": REBUTTAL_PROMPT,
}
