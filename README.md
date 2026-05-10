# LitScribe

AI-powered multi-agent literature review engine. Input a research question, get a comprehensive review with citations — in under 2 minutes.

**Architecture**: DeepAgents supervisor + deterministic pipeline + parallel synthesis

```
User → Supervisor (DeepAgents)
         └→ run_review tool (deterministic pipeline)
              ├─ Plan    (LLM: decompose into sub-topics)
              ├─ Search  (API: 5 academic sources, parallel)
              ├─ Read    (LLM: analyze each paper)
              ├─ GraphRAG (LLM: knowledge graph)
              ├─ Synthesize (LLM: parallel section generation)
              └─ Review  (LLM: quality evaluation + loop-back)
```

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/arnold117/LitScribe.git
cd LitScribe
conda create -n litscribe python=3.12
conda activate litscribe
pip install -e ".[dev]"

# 2. Configure (any OpenAI-compatible endpoint)
cp .env.example .env
# Edit .env: set llm-key, llm-location, llm-model

# 3. Run
litscribe chat              # Interactive chat
litscribe review "CRISPR CHO knockout productivity"  # Direct review
litscribe serve             # Web UI at http://localhost:8000
```

## Features

### Literature Review Pipeline
- **Plan** — LLM decomposes research question into sub-topics with domain-specific search queries
- **Search** — Parallel multi-source search across arXiv, OpenAlex, Europe PMC, PubMed, Semantic Scholar
- **Read** — Batch paper analysis (PDF full-text via Unpaywall OA lookup, falls back to abstract)
- **GraphRAG** — Knowledge graph with entity extraction, community detection (Leiden), summarization
- **Synthesize** — Parallel section generation: intro + themes + conclusion generated concurrently
- **Contradictions** — Pairwise comparison of findings across papers, auto-detects opposing conclusions
- **Review** — Self-evaluation with loop-back (if score < 0.65, refines queries and re-searches)

### Domain-Aware Search
- 5 sources: arXiv (CS/physics/math), OpenAlex (250M+ all domains), Europe PMC (40M+ life science), PubMed (35M+ biomedical), Semantic Scholar (200M+)
- Auto-skips arXiv for biology/medicine/chemistry queries
- Keyword relevance filter removes off-topic papers
- In-memory search cache (same query → instant)

### Natural Language Control
```
# The supervisor extracts preferences from your message:
"Review transformer attention, 10 papers, focus on efficiency"
"帮我写一个关于CRISPR递送方法的综述，中文，20篇论文"
"Quick review on single-cell RNA-seq in tumor microenvironment"
```

### Review Refinement
After a review is generated, modify it with natural language:
```
"Add a section about delivery methods"
"Expand the methodology discussion"
"Rewrite the conclusion in Chinese"
"Remove the part about glycosylation"
```

### Web UI
`litscribe serve` starts a web interface with:
- Real-time progress bar (SSE streaming)
- Review display with refine + export
- Export to Markdown or BibTeX

### Multi-Language
- English and Chinese review generation
- Chinese queries auto-expanded with jieba segmentation
- Search queries always in English for maximum coverage

## Performance

| Metric | Value |
|--------|-------|
| Pipeline time (8 papers) | ~60s |
| Review quality score | 0.65-0.85 |
| Citation accuracy | Pandoc-style [@key], auto-generated BibTeX |
| Search coverage | 325M+ papers across 5 sources |
| LLM provider | Any OpenAI-compatible (DeepSeek, DashScope, OpenAI, Ollama) |

### Benchmark by Domain

| Domain | Score | Papers | Words | Time |
|--------|-------|--------|-------|------|
| Biology (CHO CRISPR) | 0.85 | 8 | 1796 | 77s |
| Computer Science (transformer) | 0.65 | 8 | 1873 | 91s |
| Chemistry (sesquiterpene coumarin) | 0.55 | 8 | 1548 | 90s |

## Configuration

```bash
# .env — only 3 required vars
llm-key=sk-your-key          # API key
llm-location=https://api.deepseek.com/  # Endpoint
llm-model=deepseek-v4-flash  # Model name

# Optional search APIs (improve coverage)
NCBI_EMAIL=your@email.com
NCBI_API_KEY=your-key
S2_API_KEY=your-key
```

Works with any OpenAI-compatible endpoint: DeepSeek, DashScope (百炼), OpenAI, Ollama, vLLM, etc.

## Project Structure

```
litscribe/
├── agents.py          # DeepAgents supervisor + tool factory
├── config.py          # 3 env vars: llm-key/location/model
├── cli/main.py        # Typer CLI (chat, review, serve, skills)
├── api/               # FastAPI + SSE + Web UI
├── tools/
│   ├── pipeline.py    # Deterministic 9-step pipeline
│   ├── search.py      # Multi-source academic search
│   ├── contradictions.py  # Pairwise contradiction detection
│   ├── grounding.py   # Citation verification against source papers
│   ├── synthesis.py   # Parallel section generation
│   ├── cite_keys.py   # Pandoc [@key] + BibTeX generation
│   ├── review.py      # Self-evaluation
│   ├── refinement.py  # Search-augmented review modification
│   └── status.py      # PipelineState + routing
├── services/          # arXiv, PubMed, S2, OpenAlex, Europe PMC, Zotero, PDF
├── models/            # Pydantic v2: Paper, ResearchPlan, PaperAnalysis, ReviewOutput
├── evolution/         # Memory system (episodic/semantic/procedural) — experimental
├── plugins/graphrag/  # Entity extraction, Leiden communities, summarization
├── exporters/         # BibTeX, citation formatter (APA/MLA/IEEE/Chicago/GB_T), Pandoc
├── middleware/        # Evolution skill injection, token tracking, DeepSeek compat
├── prompts/           # All LLM prompts (planning, reading, synthesis, review, etc.)
└── llm/adapter.py     # ChatOpenAI → router interface adapter
```

## Development

```bash
conda activate litscribe
pytest tests/                    # 37 tests
litscribe --help                 # CLI commands
```

## License

MIT
