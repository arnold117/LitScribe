# LitScribe

**LitScribe** is an autonomous academic synthesis engine designed to transform how researchers conduct literature reviews. Built on a **LangGraph Multi-Agent** architecture with **GraphRAG** knowledge synthesis, LitScribe goes beyond simple summarization to provide deep, cross-paper synthesis and gap analysis.

> **Latest run**: 24-paper CHO metabolism review, 7,679 words, 100% citation grounding, $0.098 total cost, 15 minutes end-to-end.

---

## Vision

The goal of LitScribe is to act as a rigorous "Digital Scribe" for scholars—faithfully organizing human knowledge while eliminating the hallucinations common in vanilla LLM outputs.

## Key Features

### Multi-Agent Literature Review
- **Supervisor Agent**: LangGraph-based workflow orchestration with state routing, self-review loop-back
- **Planning Agent**: Complexity assessment (1-5), sub-topic decomposition, **domain detection** (arXiv categories, S2 fields, PubMed MeSH), **interactive plan revision** (user feedback loop, up to 3 rounds)
- **Discovery Agent**: **Per-sub-topic search**, domain-aware query expansion, **two-stage abstract screening** (keyword + LLM), **multi-round snowball with co-citation analysis**, **review tier system** (standard/large/massive)
- **Critical Reading Agent**: PDF parsing, 5-8 key findings extraction, methodology analysis, **LLM-assessed relevance scoring**, **pre-synthesis filtering** (removes papers with relevance < 0.4)
- **GraphRAG Agent**: Knowledge graph construction, entity extraction with research context, **relevance-gated** (filters low-relevance papers)
- **Synthesis Agent**: GraphRAG-enhanced theme identification, gap analysis, **citation name normalization** (edit-distance fuzzy matching), **auto-generated review titles**
- **Self-Review Agent**: Quality assessment, **actionable irrelevant paper removal**, **incremental loop-back** to Discovery (targeted re-search, not full restart)
- **Refinement Agent**: Iterative review modification via natural language instructions, in-process refinement from post-review menu

### Multi-Source Search
- Unified search across **arXiv**, **PubMed**, **Semantic Scholar**
- **Domain filtering**: arXiv category, Semantic Scholar `fields_of_study`, PubMed MeSH terms
- **Word-boundary keyword matching** — prevents false positives (e.g., "bio" won't match "biography")
- **Zotero** integration for personal library management (auto-search when configured)
- **Unpaywall** open-access PDF acquisition (legal OA lookup by DOI)
- Intelligent deduplication and merging across sources
- Citation tracking, paper recommendations, and **co-citation analysis**

### Export & Citations
- **BibTeX** export with auto-detected entry types
- **5 citation styles**: APA, MLA, IEEE, Chicago, GB/T 7714
- **Multi-format export**: Word (.docx), PDF, HTML, LaTeX, Markdown

### Caching & Persistence
- **SQLite cache**: Local-first search, PDF caching, parse results
- **GraphRAG cache**: Entity extraction results, graph edges, community data
- **Checkpointing**: Resume interrupted reviews via `thread_id`
- **Incremental updates**: Only fetch what's missing, reuse cached entities

### PDF Processing
- High-fidelity PDF-to-Markdown conversion
- LaTeX equation preservation
- Dual backend: `pymupdf4llm` (fast) / `marker-pdf` (OCR)

### Local-First Search & Zotero Integration
- **Local-first**: SQLite cache → Zotero library → external APIs (only fetch what's missing)
- **Zotero auto-search**: Automatically searches your Zotero library before external APIs (requires `ZOTERO_API_KEY` + `ZOTERO_LIBRARY_ID`)
- **Collection filtering**: `--zotero-collection` to search/save within a specific collection (supports key or name)
- **Auto-save**: When a collection is specified, discovered papers are automatically saved to it
- **Local PDF injection**: Include your own PDFs alongside searched papers
- **User config**: Persistent preferences via `~/.litscribe/config.yaml`

### Multi-Language Review Generation
- **Direct generation**: Write reviews in the target language (not translate after)
- `--lang zh`: Chinese academic writing with formal scholarly tone
- **CJK-aware word counting**: Correctly counts Chinese/Japanese/Korean characters (regex-based, not `split()`)
- **Language mismatch detection**: Warns on English query + `--lang zh`, suggests correction
- Generic fallback for other languages
- Search queries always in English for optimal database coverage

### GraphRAG Knowledge Synthesis
- **Entity extraction**: Automatic identification of methods, datasets, metrics, concepts
- **Entity linking**: Cross-paper entity deduplication using embeddings
- **Knowledge graph**: NetworkX-based graph with papers, entities, and relationships
- **Community detection**: Leiden algorithm for hierarchical clustering
- **Global synthesis**: Multi-level summarization from entity → community → global
- **Deep integration**: Communities used directly as themes in synthesis

### Quality Assurance & Iterative Refinement
- **Self-Review**: Automated scoring (relevance, coverage, coherence, argumentation), **actionable** irrelevant paper removal, **incremental** loop-back to Discovery
- **Planning**: Complexity-aware sub-topic decomposition, **domain detection** for search filtering, **interactive plan revision** (feedback loop)
- **Citation Normalization**: Post-generation fuzzy matching corrects misspelled author names (Levenshtein distance <= 2)
- **Citation Grounding**: Verifies every `[Author, Year]` citation maps to an actual paper (100% grounding rate achieved)
- **Token Tracking**: Per-agent and per-model cost breakdown with Qwen/DeepSeek/Anthropic pricing
- **Refinement**: Natural language instructions to modify reviews (add/remove/modify/rewrite sections)
- **Post-Review Menu**: Interactive choices after review — refine in-process, show full text, or save & exit
- **Session Management**: Git-like version tracking with unified diffs, rollback support
- **Non-blocking**: All quality agents use graceful fallbacks — never block the main workflow

## Tech Stack

- **Language:** Python 3.12+
- **Orchestration:** LangGraph (multi-agent framework with state management)
- **Async Processing:** asyncio with concurrent batching and semaphore control
- **Interface:** CLI + optional MCP server for external clients
- **Storage:** SQLite (cache, checkpointing, GraphRAG data)
- **Knowledge Graph:** NetworkX + graspologic (Leiden community detection)
- **Embeddings:** sentence-transformers (entity linking)
- **Cloud LLM:** Qwen3-Max via DashScope (default) / DeepSeek Chat / Claude Opus 4.5 (via LiteLLM)
- **PDF Processing:** pymupdf4llm (default) / marker-pdf (OCR)

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          LitScribe Architecture                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Research Question                                                       │
│        │                                                                 │
│        ▼                                                                 │
│  ┌──────────────┐                                                        │
│  │  Supervisor  │◄──────────────────────────────────────┐               │
│  └──────┬───────┘                                       │               │
│         │                                                │               │
│         ▼                                                │               │
│  ┌──────────────┐                                        │               │
│  │  Planning    │  Complexity assessment (1-5)           │               │
│  │    Agent     │  Domain detection + search filters     │               │
│  └──────┬───────┘                                        │               │
│         │                                                │               │
│         ▼                                                │               │
│  ┌──────────────┐     ┌─────────────────┐               │               │
│  │  Discovery   │────▶│ Search APIs     │               │               │
│  │    Agent     │◄┐   │ • arXiv/PubMed  │               │               │
│  │ • Domain-    │ │   │ • Semantic S.   │               │               │
│  │   aware      │ │   │ • Zotero        │               │               │
│  └──────┬───────┘ │   └─────────────────┘               │               │
│         │         │                                      │               │
│         ▼         │                                      │               │
│  ┌──────────────┐ │   ┌─────────────────┐               │               │
│  │  Critical    │ │──▶│ PDF Parser      │               │               │
│  │  Reading     │ │   │ • pymupdf4llm   │               │               │
│  └──────┬───────┘ │   └─────────────────┘               │               │
│         │         │                                      │               │
│         ▼         │                                      │               │
│  ┌──────────────┐ │   ┌─────────────────┐               │               │
│  │  GraphRAG    │ │──▶│ Knowledge Graph │               │               │
│  │    Agent     │ │   │ • Entities      │               │               │
│  │ • Relevance  │ │   │ • Communities   │               │               │
│  │   gated      │ │   └─────────────────┘               │               │
│  └──────┬───────┘ │                                      │               │
│         │         │                                      │               │
│         ▼         │                                      │               │
│  ┌──────────────┐ │                                      │               │
│  │  Synthesis   │─┼──────────────────────────────────────┘               │
│  │    Agent     │ │                                                      │
│  └──────┬───────┘ │                                                      │
│         │         │                                                      │
│         ▼         │  loop back                                           │
│  ┌──────────────┐ │  (score < 0.6)    ┌─────────────────┐               │
│  │ Self-Review  │─┘                   │ SQLite Cache    │               │
│  │ • Scoring    │                     │ • Papers/PDFs   │               │
│  │ • Remove bad │                     │ • Sessions      │               │
│  │ • Loop-back  │                     │ • Versions      │               │
│  └──────┬───────┘                     └─────────────────┘               │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────┐     ┌─────────────────┐                               │
│  │  Session     │     │ Export          │                               │
│  │  Created     │     │ • BibTeX        │                               │
│  │  (auto v1)   │     │ • Word/PDF/HTML │                               │
│  └──────┬───────┘     └─────────────────┘                               │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────┐                                                        │
│  │ Interactive  │  [1] Save & exit                                       │
│  │  Menu / Re-  │  [2] Refine → Refinement Agent → save version         │
│  │  finement    │  [3] Show full text                                    │
│  └──────────────┘                                                        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Workflow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Supervisor
    participant Planning
    participant Discovery
    participant CriticalReading
    participant GraphRAG
    participant Synthesis
    participant SelfReview
    participant Refinement

    User->>CLI: litscribe review "question"

    CLI->>Planning: Assess complexity + detect domain
    Planning-->>CLI: Research plan (domain, arXiv cats, S2 fields, MeSH)
    CLI-->>User: Show plan (if complex)
    User-->>CLI: Confirm plan

    CLI->>Supervisor: Start workflow (inject plan)

    Supervisor->>Discovery: Search papers
    Discovery->>Discovery: Domain-aware query expansion + filtered search
    Discovery-->>Supervisor: Papers found (keyword relevance scored)

    Supervisor->>CriticalReading: Analyze papers
    CriticalReading->>CriticalReading: PDF parse + 5-8 findings + relevance assessment
    CriticalReading-->>Supervisor: Paper summaries

    Supervisor->>GraphRAG: Build knowledge graph
    GraphRAG->>GraphRAG: Relevance-gated entity extraction + community detection
    GraphRAG-->>Supervisor: Communities & themes

    Supervisor->>Synthesis: Generate review
    Synthesis->>Synthesis: Theme synthesis + gap analysis + citations
    Synthesis-->>Supervisor: Review text

    Supervisor->>SelfReview: Quality assessment
    SelfReview->>SelfReview: Score + remove irrelevant papers

    alt Score < 0.6 (online mode)
        SelfReview-->>Supervisor: Loop back to Discovery
        Supervisor->>Discovery: Re-search (clear synthesis/graphrag)
    else Score >= 0.6
        SelfReview-->>Supervisor: Complete
    end

    Supervisor-->>CLI: Final state + auto-create session (v1)
    CLI-->>User: Save .md + .json + interactive menu

    Note over User,Refinement: Post-review interactive menu
    User->>CLI: [2] Refine review + instruction
    CLI->>Refinement: classify + execute
    Refinement-->>CLI: Updated review (v2) + references
```

## Quick Start

### Prerequisites

- Python 3.12+ (via mamba/conda)
- API keys: DashScope (Qwen3, default LLM) or DeepSeek / Anthropic (Claude), optional: NCBI, Semantic Scholar, Zotero

### Installation

```bash
# Clone the repository
git clone https://github.com/arnold117/LitScribe.git
cd LitScribe

# Create environment
mamba env create -f environment.yml
mamba activate litscribe

# Install as editable package
pip install -e .

# Configure API keys
cp .env.example .env
# Edit .env with your keys

# Verify
litscribe --help
```

### Usage

```bash
# === Literature Review ===
litscribe review "What are the latest advances in LLM reasoning?"
litscribe review "CRISPR applications" -s pubmed,arxiv -p 15
litscribe review "石杉碱甲生物合成" --lang zh           # Chinese review
litscribe review "topic" --local-files a.pdf b.pdf      # Include local PDFs

# === Zotero Integration ===
litscribe review "topic" --zotero-collection "My Review"     # Search + auto-save to collection (by name)
litscribe review "topic" --zotero-collection ABC123XY        # Search + auto-save to collection (by key)

# === Planning & GraphRAG ===
litscribe review "LLM fine-tuning methods" -p 10 --enable-graphrag
litscribe review "transformer architectures" -p 20 --disable-graphrag
litscribe review "complex multi-domain topic" --plan-only    # Show plan only

# === Session & Refinement ===
litscribe session list                                       # List all sessions
litscribe session show <session_id>                          # Session details + versions
litscribe session refine <id> -i "Add discussion about LoRA" # Iterative refinement
litscribe session diff <id> 1 2                              # Compare two versions
litscribe session rollback <id> 1                            # Rollback to version 1

# === Export ===
litscribe export review.json -f docx -s apa      # Word (APA style)
litscribe export review.json -f pdf -s ieee      # PDF (IEEE style)
litscribe export review.json -f bibtex           # BibTeX only
litscribe export review.json -f md -l zh         # Chinese Markdown

# === Search ===
litscribe search "transformer attention" --sources arxiv,semantic_scholar
litscribe search "CRISPR" --sources pubmed --limit 20 --sort citations

# === Paper / PDF ===
litscribe paper arXiv:1706.03762 --verbose
litscribe parse paper.pdf --output paper.md

# === Config & Cache ===
litscribe config show                 # Show current config
litscribe config set max_papers 20    # Set default
litscribe cache stats                 # Cache statistics
litscribe cache clear --expired       # Clear expired entries
```

### Output Files

When running `litscribe review`, outputs are saved to `output/`:
- `review_*.md` - Literature review in Markdown
- `review_*.json` - Full data (papers, analysis, synthesis)

Export generates additional formats:
- `review_*.bib` - BibTeX citations
- `review_*.docx` - Word document
- `review_*.pdf` - PDF (requires Pandoc + LaTeX)

## Project Status

### Completed

| Phase | Description | Status |
|-------|-------------|--------|
| MVP | Search services, unified search, CLI | ✅ Done |
| Iteration 2 | Multi-agent system, LangGraph | ✅ Done |
| Phase 6.5 | SQLite cache, checkpointing | ✅ Done |
| Phase 7 | BibTeX, export, citation styles | ✅ Done |
| Phase 7.5 | GraphRAG, scale-up (50-500 papers) | ✅ Done |
| Phase 8 | Zotero sync, local files, multi-lang generation, user config | ✅ Done |
| Phase 9.1 | Self-Review Agent (auto quality assessment, scoring, gap detection) | ✅ Done |
| Phase 9.2 | Planning Agent (complexity assessment, sub-topic decomposition, `--plan-only`) | ✅ Done |
| **Phase 9.3** | **Refinement Agent, session management, version tracking, diff & rollback** | **✅ Done** |
| **Phase 9.5** | **Evaluation & Instrumentation: token tracking, citation grounding, evaluation framework, ablation flags, failure analysis** | **✅ Done** |
| **Phase 10** | **MCP cleanup → services/, unified MCP server, GraphRAG optimization (threshold clustering, retry, concurrency)** | **✅ Done** |
| **Post-10 Patches** | **Unpaywall PDF, two-stage screening, co-citation snowball, tier system, plan iteration, CJK word count, citation normalization** | **✅ Done** |
| **Qwen3 Migration** | **Switch to Qwen3-Max (65K output), `enable_thinking: False`, `<think>` stripping, Qwen pricing** | **✅ Done** |

### Planned

| Phase | Description | Priority |
|-------|-------------|----------|
| Phase 11 | Local LLM support (Ollama/MLX/vLLM) | Medium |
| Phase 12 | Subscription system, daily digest | Medium |
| Phase 13 | Web UI (React + FastAPI) | Medium |

## Testing

```bash
# Run all tests (272 tests)
pytest

# Run a single test file
pytest tests/test_token_tracker.py -v
```

| File | Module | Tests | Key Coverage |
|------|--------|-------|-------------|
| `test_search_quality.py` | Search filtering & relevance | 34 | Word-boundary matching, MeSH logic, pre-synthesis filter |
| `test_reasoning_model.py` | Model routing | 26 | Reasoning model detection, Qwen3 thinking mode |
| `test_citation_grounding.py` | Citation grounding | 25 | Citation extraction, author matching, fuzzy grounding |
| `test_tier_system.py` | Review tiers | 22 | Standard/large/massive tiers, per-sub-topic search |
| `test_graphrag.py` | GraphRAG pipeline | 22 | Entity normalization, threshold clustering, retry logic |
| `test_exporters.py` | BibTeX / Citation / Pandoc | 20 | BibTeX export, citation formatting |
| `test_zotero_integration.py` | Zotero integration | 18 | Collection resolution, auto-save, state wiring |
| `test_evaluator.py` | Review evaluation | 18 | Search quality, theme coverage, domain purity |
| `test_plan_override.py` | Plan iteration | 14 | Revision prompt, feedback loop, MAX_PLAN_REVISIONS |
| `test_token_tracker.py` | Token tracking | 13 | Multi-model cost, Qwen pricing, CLI format |
| `test_unpaywall.py` | Unpaywall PDF | 12 | OA lookup, timeout handling, integration points |
| `test_screening_snowball.py` | Abstract screening | 11 | Two-stage screening, co-citation snowball |
| `test_review_title.py` | Title generation | 11 | Auto title, plan-based titles |
| `test_loopback.py` | Self-review loop-back | 10 | Incremental re-search, state clearing |
| `test_abstract_analysis.py` | Abstract-only analysis | 10 | No-PDF fallback, findings extraction |
| `test_cache_manual.py` | Cache DB | 9 | Cache CRUD, async ops |
| `test_arxiv_ratelimit.py` | arXiv rate limit | 8 | 429 retry, exponential backoff |
| `test_checkpointing.py` | LangGraph checkpointing | 6 | SQLite saver, graph compilation, ablation flags |
| `test_critical_reading_cache.py` | Critical Reading + Cache | 5 | PDF/parse caching |
| `test_discovery_cache.py` | Discovery + Cache | 4 | State creation, cache-enabled search |
| **Total** | **20 test files** | **272** | |

## Development Notes

### PDF Parsing Backend

| Backend | Speed | OCR | Stability | Use Case |
|---------|-------|-----|-----------|----------|
| `pymupdf4llm` | Fast | No | Stable | Native text PDFs |
| `marker-pdf` | Slow | Yes | Unstable (MPS) | Scanned PDFs |

### Rate Limits

| API | Limit | Mitigation |
|-----|-------|------------|
| Semantic Scholar | 1 req/s | AsyncRateLimiter |
| PubMed | 3 req/s (no key) | Cache + batch |
| arXiv | No limit | Polite delay |

### LLM Strategy

Default LLM is **Qwen3-Max** via DashScope (65K max output tokens, non-thinking mode). Configurable via `LITELLM_MODEL` env var or `litscribe config set model`. All models routed through LiteLLM.

| Task | Default | Alternative | Cost (per 1M tokens) |
|------|---------|-------------|---------------------|
| All agents | Qwen3-Max | DeepSeek Chat | $0.35 in / $1.39 out |
| Review synthesis (reasoning) | DeepSeek-Reasoner | Opus 4.5 | $0.55 in / $2.19 out |

**Qwen3 specifics**: `enable_thinking: False` is automatically set for `qwen3*` models to avoid `<think>` blocks consuming output tokens. Fallback `<think>` stripping is applied in both `call_llm()` and `extract_json()`.

**Supported providers**: `dashscope/qwen3-max-*`, `deepseek/deepseek-chat`, `deepseek/deepseek-reasoner`, `anthropic/claude-*`, `openai/gpt-*`, or any LiteLLM-compatible model string.

### Architecture Notes

**Service Layer**: Internal agents directly import service functions (`src/services/`) for lower latency and tighter LangGraph integration. A unified MCP server (`src/mcp_server.py`) is also available for external client access via stdio/streamable-http.

**Agent Architecture**: 8 agents orchestrated by LangGraph StateGraph:
- Main pipeline: Supervisor → Planning → Discovery → Critical Reading → GraphRAG → Synthesis → Self-Review
- **Discovery pipeline**: Per-sub-topic search → keyword extraction → second-round search → two-stage abstract screening (keyword + LLM) → co-citation snowball → relevance filtering → paper selection
- **Loop-back**: Self-Review can route back to Discovery (score < 0.6), **incremental** re-search using `additional_queries` (not full restart)
- **Pre-workflow planning**: CLI runs Planning Agent before the graph, injects the approved plan; users can **iterate** on the plan with feedback (up to 3 rounds)
- **PDF acquisition chain**: Cache → arXiv → Zotero → Unpaywall (OA by DOI) → direct URL
- Standalone: Refinement Agent (bypasses main graph, invoked via `session refine` or post-review interactive menu)
- All quality agents (Self-Review, Refinement) use graceful fallbacks and never block the main workflow

---

## License

MIT License - feel free to use and contribute.

## Contact

*Created by Arnold - Exploring the future of AI4Sci.*
