"""GraphRAG-specific prompts for entity extraction and summarization."""

# =============================================================================
# Entity Extraction Prompts
# =============================================================================

ENTITY_EXTRACTION_SYSTEM = """You are an expert at extracting structured entities from academic papers.
Your task is to identify key entities that represent important concepts, methods, datasets, and metrics.
Be precise and consistent in naming - use the most canonical/common name for each entity.

IMPORTANT: You MUST respond with ONLY valid JSON. No markdown, no explanations, no code blocks. Just pure JSON."""

ENTITY_EXTRACTION_PROMPT = """Extract all significant entities from this academic paper.

**Paper Information:**
- Title: {title}
- Authors: {authors}
- Year: {year}
- Abstract: {abstract}

{content_section}

**Entity Types to Extract:**
1. **METHODS**: Algorithms, techniques, architectures, models, frameworks
   - Examples: "Transformer", "BERT", "Gradient Descent", "Random Forest", "ResNet"

2. **DATASETS**: Evaluation datasets, benchmarks, corpora
   - Examples: "ImageNet", "GLUE", "SQuAD", "COCO", "WikiText-103"

3. **METRICS**: Evaluation metrics and measures
   - Examples: "F1 Score", "BLEU", "Perplexity", "ROUGE", "mAP", "AUC"

4. **CONCEPTS**: Key theoretical concepts, paradigms, phenomena
   - Examples: "Attention Mechanism", "Transfer Learning", "Overfitting", "Emergent Behavior"

**Instructions:**
- Extract 5-15 entities per paper
- Use canonical names (e.g., "BERT" not "bert model")
- Include common aliases/abbreviations
- Write brief descriptions (1-2 sentences)
- Only extract entities actually mentioned in the paper

**Output Format (JSON):**
{{
  "entities": [
    {{
      "name": "Transformer",
      "type": "method",
      "aliases": ["Transformer model", "Transformer architecture"],
      "description": "A neural network architecture based on self-attention mechanisms, enabling parallel processing of sequences."
    }},
    {{
      "name": "BLEU",
      "type": "metric",
      "aliases": ["BLEU score", "BiLingual Evaluation Understudy"],
      "description": "A metric for evaluating machine translation quality by comparing n-gram overlap with reference translations."
    }}
  ]
}}"""

ENTITY_EXTRACTION_CONTENT_TEMPLATE = """**Full Text Content (first 3000 chars):**
{content}"""

# =============================================================================
# Community Summarization Prompts
# =============================================================================

COMMUNITY_SUMMARY_SYSTEM = """You are an expert at synthesizing academic research themes.
Your task is to summarize the research focus of a community of related papers and concepts."""

COMMUNITY_SUMMARY_PROMPT = """Summarize the research theme represented by this community of papers and entities.

**Research Question Context:**
{research_question}

**Entities in this Community:**
{entities_list}

**Papers in this Community:**
{papers_list}

**Key Relationships:**
{relationships}

**Instructions:**
Write a 2-3 paragraph summary covering:
1. What is the main research theme/topic of this community?
2. What methods/approaches are most prominent?
3. What are the key findings, contributions, and trends?

Be specific and cite the methods/datasets by name. Focus on synthesis, not just listing.

**Summary:**"""

# =============================================================================
# Global Summary Prompts
# =============================================================================

GLOBAL_SUMMARY_SYSTEM = """You are an expert at synthesizing academic literature.
Your task is to create a high-level overview of the research landscape from community summaries."""

GLOBAL_SUMMARY_PROMPT = """Create a global summary of the research landscape based on these community summaries.

**Research Question:**
{research_question}

**Community Summaries:**
{community_summaries}

**Graph Statistics:**
- Total Papers: {num_papers}
- Total Entities: {num_entities}
- Total Communities: {num_communities}
- Key Methods: {top_methods}
- Key Datasets: {top_datasets}

**Instructions:**
Write a comprehensive 3-5 paragraph synthesis that:
1. Identifies the major research themes and how they relate
2. Highlights the most influential methods and their adoption
3. Notes the primary evaluation approaches (datasets, metrics)
4. Describes the evolution and current state of the field
5. Points to emerging trends and open questions

This summary should serve as a high-level overview that helps readers understand the research landscape before diving into details.

**Global Summary:**"""

# =============================================================================
# Technology Comparison Prompts
# =============================================================================

TECHNOLOGY_COMPARISON_SYSTEM = """You are an expert at comparing research methods and approaches.
Your task is to create a structured comparison of methods found in the literature."""

TECHNOLOGY_COMPARISON_PROMPT = """Create a technology comparison table based on the extracted entities and papers.

**Research Question:**
{research_question}

**Methods Found:**
{methods_list}

**Datasets Used:**
{datasets_list}

**Papers Analyzed:**
{papers_summary}

**Instructions:**
Create a structured comparison with:
1. List the top 5-10 most important methods
2. For each method, identify:
   - Key characteristics/approach
   - Datasets it was evaluated on
   - Reported performance (if available)
   - Main strengths
   - Main limitations
3. Identify patterns across methods

**Output Format (JSON):**
{{
  "methods": [
    {{
      "name": "Method Name",
      "description": "Brief description of approach",
      "datasets_evaluated": ["Dataset1", "Dataset2"],
      "key_strength": "Main advantage",
      "key_limitation": "Main drawback",
      "papers": ["paper_id1", "paper_id2"]
    }}
  ],
  "comparison_axes": ["accuracy", "efficiency", "scalability"],
  "key_insights": [
    "Insight about patterns across methods"
  ]
}}"""
