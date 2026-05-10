# LitScribe

AI-powered multi-agent literature review engine. Input a research question, get a comprehensive, citation-grounded review — in under 2 minutes.

**Architecture**: DeepAgents supervisor + deterministic 11-step pipeline + parallel synthesis

```
User → Supervisor (DeepAgents)
         └→ run_review (deterministic pipeline)
              ├─ Plan           (LLM: decompose into sub-topics)
              ├─ Search         (API: 5 academic sources + Unpaywall OA)
              ├─ Read           (LLM: full-text or abstract analysis)
              ├─ Contradictions (LLM: pairwise finding comparison)
              ├─ GraphRAG       (LLM: knowledge graph extraction)
              ├─ Synthesize     (LLM: parallel section generation)
              ├─ Debate         (LLM: reviewer ↔ synthesizer, 2 rounds)
              ├─ Ground         (LLM: verify citations against sources)
              ├─ Review         (LLM: quality evaluation + loop-back)
              └─ Save           (SQLite: session + knowledge base)
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
litscribe chat                          # Interactive chat
litscribe review "CRISPR CHO knockout"  # Direct review
litscribe serve                         # Web UI at localhost:8000
litscribe evaluate                      # Benchmark across 5 domains
litscribe sessions                      # View past reviews
```

## Features

### 11-Step Literature Review Pipeline
- **Plan** — LLM decomposes research question into sub-topics with domain-specific search queries
- **Search** — Parallel multi-source search with connection retry and in-memory cache
- **Unpaywall** — OA PDF lookup by DOI (26% → 68% full-text coverage)
- **Read** — PDF full-text analysis when available, abstract fallback
- **Contradictions** — Pairwise comparison of findings across papers, auto-detects opposing conclusions
- **GraphRAG** — Knowledge graph with entity extraction, Leiden community detection, summarization
- **Synthesize** — Parallel section generation (intro + themes + conclusion concurrent)
- **Debate** — Reviewer critiques → Synthesizer revises → up to 2 rounds
- **Ground** — Citation verification: each [@key] claim checked against source paper, auto-fix hallucinations
- **Review** — Self-evaluation with loop-back (score < 0.65 → refine queries and re-search)
- **Save** — Session + knowledge base persisted to SQLite

### Academic Search (5 sources, 325M+ papers)
- OpenAlex (250M+), Europe PMC (40M+), PubMed (35M+), Semantic Scholar (200M+), arXiv (2M+)
- Domain-aware: auto-skips arXiv for biology/medicine/chemistry
- DOI deduplication across sources
- Keyword relevance filter
- Connection retry with backoff (2 consecutive failures → skip source)
- Cross-lingual: CJK queries preserved alongside English expanded queries

### Citation System
- Pandoc-style `[@key]` citations (deterministic: author+year)
- Auto-generated BibTeX with matching keys
- `## References` section at review end
- Citation grounding: verified ✓ / unsupported ✗ / unverified ?
- Claim chain API: each claim linked to source evidence

### Review Refinement
Search-augmented modification — when adding new content, searches for papers first:
```
"Add a section about delivery methods"
→ searches 5 papers on delivery methods
→ writes new ## section citing [@newkey1; @newkey2]
→ appends new references
```

Also supports: modify, remove, rewrite existing sections.

### Contradiction Detection
Pairwise comparison of paper findings:
```
[@cho2025] reports: "FUT8 knockout had no negative effect on cell growth"
[@lin2020] finds: "FUT8 knockout reduced cell viability by 40%"
→ Classified as "opposing_conclusions" (major severity)
→ Injected into review as critical analysis section
```

### Multi-Agent Debate
After synthesis, reviewer and synthesizer debate:
1. Reviewer identifies unsupported claims, missing perspectives, weak synthesis
2. Synthesizer revises addressing all issues
3. Up to 2 rounds, stops when quality is "good/excellent"

### Web UI
`litscribe serve` starts a web interface with:
- Real-time SSE progress (plan → search → read → contradictions → synthesize → debate → ground → review)
- Grounding report: "12 citations verified, 2 auto-fixed (85% accuracy)"
- Contradiction count in progress bar
- Past reviews list with click-to-load
- Refine input + Export (Markdown / BibTeX)
- Shareable review link: `/api/share/{session_id}`
- Comments API for collaborative review

### Knowledge Base
Cross-session knowledge accumulation:
- Key findings from each review saved to SQLite with HMAC integrity signatures
- New reviews automatically query past findings in the same domain
- Injected as "Prior Knowledge" context into synthesis
- Tampered entries automatically detected and rejected

### Source Credibility Weighting
- Papers sorted by citation count (high-impact first in LLM context)
- Impact note "(cited N times)" in paper summaries

### Benchmark Framework
```bash
litscribe evaluate --max-papers 8 --output report.md
```
Runs 5 queries across Biology, CS, Medicine, Chemistry. Generates markdown report with score/papers/words/time per domain.

### Security
- HMAC-SHA256 integrity on knowledge base entries (detect data poisoning)
- Prompt injection filter (regex patterns for common injection attempts)
- HTML escaping on all user-generated content (XSS prevention)
- Rate limiting (10 req/min per endpoint)
- PDF URL domain whitelist (SSRF prevention)
- Parameterized SQL queries throughout

### Multi-Language
- English and Chinese review generation
- Chinese queries auto-expanded with jieba segmentation
- Cross-lingual search: both CJK original + English expanded queries

## Performance

| Metric | Value |
|--------|-------|
| Pipeline time (8 papers) | ~60-90s |
| Review quality score | 0.65-0.85 |
| Citation format | Pandoc [@key], auto BibTeX |
| Full-text coverage | 68% (via Unpaywall) |
| Search coverage | 325M+ papers across 5 sources |
| LLM provider | Any OpenAI-compatible |

## Configuration

```bash
# .env — 3 required vars
llm-key=sk-your-key
llm-location=https://api.deepseek.com/
llm-model=deepseek-v4-flash

# Optional
NCBI_EMAIL=your@email.com
NCBI_API_KEY=your-key
SEMANTIC_SCHOLAR_API_KEY=your-key
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/review | Run review (SSE streaming) |
| POST | /api/refine | Modify existing review |
| POST | /api/chat | Conversational endpoint |
| GET | /api/state | Current pipeline state |
| GET | /api/sessions | List past reviews |
| GET | /api/sessions/{id} | Get session details |
| POST | /api/sessions/{id}/comments | Add comment |
| GET | /api/sessions/{id}/comments | List comments |
| GET | /api/share/{id} | Shareable HTML page |
| GET | /api/claims | Citation verification data |
| GET | /api/export/{format} | Export (markdown/bibtex) |

## Project Structure

```
litscribe/                    # 86 files, 10,250 lines
├── agents.py                 # DeepAgents supervisor + tool factory
├── config.py                 # 3 env vars: llm-key/location/model
├── errors.py                 # Error types + retry decorators
├── cli/main.py               # Typer CLI (8 commands)
├── api/                      # FastAPI + SSE + Web UI
│   ├── app.py                # 11 API endpoints + rate limiting
│   └── static/index.html     # Single-page frontend
├── tools/
│   ├── pipeline.py           # 11-step deterministic pipeline
│   ├── search.py             # Multi-source search (parallel sources, sequential queries)
│   ├── contradictions.py     # Pairwise contradiction detection
│   ├── debate.py             # Multi-round reviewer ↔ synthesizer
│   ├── grounding.py          # Citation verification against sources
│   ├── claim_chain.py        # Claim → evidence tracing
│   ├── synthesis.py          # Parallel section generation
│   ├── cite_keys.py          # Pandoc [@key] + BibTeX generation
│   ├── refinement.py         # Search-augmented review modification
│   ├── benchmark.py          # Multi-domain evaluation framework
│   ├── sanitize.py           # Prompt injection filter
│   ├── integrity.py          # HMAC data integrity
│   ├── export.py             # Markdown/BibTeX export
│   └── status.py             # PipelineState + routing
├── services/                 # arXiv, PubMed, S2, OpenAlex, Europe PMC, Zotero, PDF, Unpaywall
├── models/                   # Pydantic v2: Paper, ResearchPlan, PaperAnalysis, ReviewOutput
├── store/                    # SQLite sessions, knowledge base, vectors
├── evolution/                # 3-tier memory + SkillEvolver (experimental)
├── plugins/graphrag/         # Entity extraction, Leiden communities, summarization
├── exporters/                # BibTeX, citation formatter, Pandoc
├── prompts/                  # All LLM prompts
└── llm/adapter.py            # ChatOpenAI adapter
```

## Development

```bash
conda activate litscribe
pytest tests/                 # 49 tests
litscribe evaluate            # Benchmark (requires API key)
litscribe --help
```

## License

MIT
