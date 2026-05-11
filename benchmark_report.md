# LitScribe Benchmark Report

## Setup
- **Model**: DeepSeek V4 Flash (via DeepSeek API)
- **Papers per query**: 12 (max)
- **Pipeline**: Plan → Search (6 sources) → Read → Contradictions → Synthesize → Ground → Review
- **Date**: 2026-05-11

## Results

| Label | Domain | Papers | Words | Score | Citations | Grounding | Time |
|-------|--------|--------|-------|-------|-----------|-----------|------|
| BIO-1 | Biology (CHO CRISPR) | 9 | 1649 | **0.82** | 29 | 83% | 106s |
| CS-2 | Computer Science (LLM reasoning) | 12 | 1535 | 0.72 | 6 | 63% | 121s |
| MED-1 | Medicine (scRNA-seq TME) | 12 | 1709 | 0.65 | 4 | 68% | 115s |
| CS-1 | Computer Science (transformer attention) | 12 | 1643 | 0.55 | 8 | 62% | 153s |
| CHEM-1 | Chemistry (sesquiterpene coumarin) | 3 | 1338 | 0.55 | 15 | 100% | 103s |

### Summary
- **Average score**: 0.66
- **Average grounding**: 75%
- **Average time**: 120s (~2 minutes per review)
- **Average words**: 1,575

## Analysis

### What works well
- **Biology domain** achieves highest score (0.82) — PubMed + Europe PMC provide excellent coverage
- **Citation grounding** catches hallucinated citations — CHEM-1 achieved 100% grounding accuracy
- **Speed** — full 12-paper review in ~2 minutes including search, analysis, synthesis, verification
- **Relevance filtering** — adaptive threshold (0.2-0.5 based on result count) removes off-topic papers

### Current limitations
- **CS domain lower scores** — arXiv rate-limiting (HTTP 429) reduces paper pool quality
- **Niche topics** (CHEM-1) find fewer relevant papers — only 3 after filtering
- **Score variance** — ±0.10 between runs due to search API variability and LLM non-determinism
- **Self-review as metric** — scores reflect LLM self-assessment, not human evaluation

### Unique capabilities (not in competing tools)
1. **Contradiction-Aware Synthesis** — contradictions injected into review narrative
2. **Citation Grounding** — every [@key] verified against source paper
3. **Multi-agent Debate** — reviewer ↔ synthesizer iterative improvement
4. **Metacognitive Quality Loop** — system decides which steps to re-run
5. **Cross-lingual** — Chinese questions → English search → Chinese/English review
6. **6-source search** — OpenAlex, Europe PMC, PubMed, Semantic Scholar, arXiv, CrossRef

## Methodology

Each benchmark query runs through the lightweight pipeline (no comparison table, timeline, statistics, or figure suggestions to reduce API calls):

1. **Plan**: LLM decomposes question into sub-topics
2. **Search**: 6 sources queried with sub-topic keywords, domain-aware filtering
3. **Read**: Each paper analyzed for findings, methodology, strengths, limitations
4. **Filter**: Relevance < threshold removed (adaptive: 0.5 for >8 papers, 0.3 for 4-8, 0.2 for <4)
5. **Contradictions**: Pairwise comparison of findings
6. **Synthesize**: Parallel section generation with [@key] citations
7. **Ground**: Each citation verified against source paper
8. **Review**: Self-evaluation on relevance, coverage, coherence, claim support

Score is the self-review `overall_score` (0-1). Grounding is `verified / total_citations`.
