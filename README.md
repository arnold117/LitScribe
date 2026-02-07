# LitScribe

**LitScribe** is an autonomous academic synthesis engine designed to transform how researchers conduct literature reviews. By leveraging the **Model Context Protocol (MCP)** and a **Multi-Agent** architecture, LitScribe goes beyond simple summarization to provide deep, cross-paper synthesis and gap analysis.

---

## Vision

The goal of LitScribe is to act as a rigorous "Digital Scribe" for scholars—faithfully organizing human knowledge while eliminating the hallucinations common in vanilla LLM outputs.

## Key Features

### Multi-Agent Literature Review
- **Supervisor Agent**: LangGraph-based workflow orchestration with state routing
- **Planning Agent**: Complexity assessment (1-5), sub-topic decomposition, `--plan-only` mode
- **Discovery Agent**: Query expansion, multi-source search (parallel), snowball sampling
- **Critical Reading Agent**: PDF parsing, key findings extraction, methodology analysis (batched)
- **GraphRAG Agent**: Knowledge graph construction, entity extraction, community detection
- **Synthesis Agent**: GraphRAG-enhanced theme identification, gap analysis, review generation
- **Self-Review Agent**: Automated quality assessment (relevance, coverage, coherence, argumentation)
- **Refinement Agent**: Iterative review modification via natural language instructions

### Multi-Source Search
- Unified search across **arXiv**, **PubMed**, **Semantic Scholar**
- **Zotero** integration for personal library management
- Intelligent deduplication and merging across sources
- Citation tracking and paper recommendations

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
- **Zotero bidirectional sync**: Import from collections, auto-save discoveries, write analysis notes back
- **Local PDF injection**: Include your own PDFs alongside searched papers
- **User config**: Persistent preferences via `~/.litscribe/config.yaml`

### Multi-Language Review Generation
- **Direct generation**: Write reviews in the target language (not translate after)
- `--lang zh`: Chinese academic writing with formal scholarly tone
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
- **Self-Review**: Automated scoring (relevance, coverage, coherence, argumentation), irrelevant paper detection
- **Planning**: Complexity-aware sub-topic decomposition with interactive confirmation
- **Refinement**: Natural language instructions to modify reviews (add/remove/modify/rewrite sections)
- **Session Management**: Git-like version tracking with unified diffs, rollback support
- **Non-blocking**: All quality agents use graceful fallbacks — never block the main workflow

## Tech Stack

- **Language:** Python 3.12+
- **Orchestration:** LangGraph (multi-agent framework with state management)
- **Async Processing:** asyncio with concurrent batching and semaphore control
- **Interface:** FastMCP 2.0 (direct import, not protocol-based)
- **Storage:** SQLite (cache, checkpointing, GraphRAG data)
- **Knowledge Graph:** NetworkX + graspologic (Leiden community detection)
- **Embeddings:** sentence-transformers (entity linking)
- **Cloud LLM:** Claude Opus 4.5 / Sonnet 4.5 / DeepSeek-R1
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
│  │    Agent     │  Sub-topic decomposition               │               │
│  └──────┬───────┘                                        │               │
│         │                                                │               │
│         ▼                                                │               │
│  ┌──────────────┐     ┌─────────────────┐               │               │
│  │  Discovery   │────▶│ MCP Servers     │               │               │
│  │    Agent     │     │ • arXiv/PubMed  │               │               │
│  │              │     │ • Semantic S.   │               │               │
│  │ • Query exp  │     │ • Zotero        │               │               │
│  └──────┬───────┘     └─────────────────┘               │               │
│         │                                                │               │
│         ▼                                                │               │
│  ┌──────────────┐     ┌─────────────────┐               │               │
│  │  Critical    │────▶│ PDF Parser      │               │               │
│  │  Reading     │     │ • pymupdf4llm   │               │               │
│  └──────┬───────┘     └─────────────────┘               │               │
│         │                                                │               │
│         ▼                                                │               │
│  ┌──────────────┐     ┌─────────────────┐               │               │
│  │  GraphRAG    │────▶│ Knowledge Graph │               │               │
│  │    Agent     │     │ • Entities      │               │               │
│  │              │     │ • Communities   │               │               │
│  └──────┬───────┘     └─────────────────┘               │               │
│         │                                                │               │
│         ▼                                                │               │
│  ┌──────────────┐                                        │               │
│  │  Synthesis   │────────────────────────────────────────┘               │
│  │    Agent     │                                                        │
│  └──────┬───────┘                                                        │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────┐     ┌─────────────────┐                               │
│  │ Self-Review  │     │ SQLite Cache    │                               │
│  │    Agent     │     │ • Papers/PDFs   │                               │
│  │ • Scoring    │     │ • Sessions      │                               │
│  │ • Gap detect │     │ • Versions      │                               │
│  └──────┬───────┘     └─────────────────┘                               │
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
│  │ Refinement   │  User instruction → classify → execute → save version │
│  │    Agent     │  Standalone entry (not in main graph)                  │
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
    CLI->>Supervisor: Start workflow

    Supervisor->>Planning: Assess complexity
    Planning->>Planning: Sub-topic decomposition
    Planning-->>Supervisor: Research plan

    Supervisor->>Discovery: Search papers
    Discovery->>Discovery: Query expansion + multi-source search
    Discovery-->>Supervisor: Papers found

    Supervisor->>CriticalReading: Analyze papers
    CriticalReading->>CriticalReading: PDF parse & extract
    CriticalReading-->>Supervisor: Paper summaries

    Supervisor->>GraphRAG: Build knowledge graph
    GraphRAG->>GraphRAG: Entity extraction + community detection
    GraphRAG-->>Supervisor: Communities & themes

    Supervisor->>Synthesis: Generate review
    Synthesis->>Synthesis: Theme synthesis + gap analysis
    Synthesis-->>Supervisor: Review text

    Supervisor->>SelfReview: Quality assessment
    SelfReview-->>Supervisor: Scores + suggestions

    Supervisor-->>CLI: Final state + auto-create session (v1)
    CLI-->>User: Save .md + .json + show session_id

    Note over User,Refinement: Iterative refinement (optional)
    User->>CLI: litscribe session refine <id> -i "add LoRA discussion"
    CLI->>Refinement: classify + execute
    Refinement-->>CLI: Updated review (v2) + diff
```

## Quick Start

### Prerequisites

- Python 3.12+ (via mamba/conda)
- API keys: Anthropic (Claude), optional: NCBI, Semantic Scholar, Zotero

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LitScribe.git
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
| MVP | MCP servers, unified search, CLI | ✅ Done |
| Iteration 2 | Multi-agent system, LangGraph | ✅ Done |
| Phase 6.5 | SQLite cache, checkpointing | ✅ Done |
| Phase 7 | BibTeX, export, citation styles | ✅ Done |
| Phase 7.5 | GraphRAG, scale-up (50-500 papers) | ✅ Done |
| Phase 8 | Zotero sync, local files, multi-lang generation, user config | ✅ Done |
| Phase 9.1 | Self-Review Agent (auto quality assessment, scoring, gap detection) | ✅ Done |
| Phase 9.2 | Planning Agent (complexity assessment, sub-topic decomposition, `--plan-only`) | ✅ Done |
| **Phase 9.3** | **Refinement Agent, session management, version tracking, diff & rollback** | **✅ Done** |

### Planned

| Phase | Description | Priority |
|-------|-------------|----------|
| Phase 10 | MCP architecture cleanup, GraphRAG optimization | Medium-High |
| Phase 11 | Local LLM support (Ollama/MLX/vLLM) | Medium |
| Phase 12 | Subscription system, daily digest | Medium |
| Phase 13 | Web UI (React + FastAPI) | Medium |

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

| Task | Model | Reason |
|------|-------|--------|
| Query expansion | Haiku | Simple, low cost |
| Paper analysis | Sonnet 4.5 | Balance quality/cost |
| Entity extraction | Sonnet 4.5 | Structured output |
| Community summary | Sonnet 4.5 | Synthesis quality |
| Review synthesis | Opus 4.5 | Complex reasoning |
| Batch processing | DeepSeek-R3 | Cost-effective |

### Architecture Notes

**MCP Integration**: The project uses FastMCP decorators but **directly imports** functions rather than using MCP protocol (stdio/HTTP). This design prioritizes lower latency, simpler error handling, and tighter integration with LangGraph. Phase 10 will add a true MCP server mode for external client access.

**Agent Architecture**: 8 agents orchestrated by LangGraph StateGraph:
- Main pipeline: Supervisor → Planning → Discovery → Critical Reading → GraphRAG → Synthesis → Self-Review
- Standalone: Refinement Agent (bypasses main graph, invoked via `session refine`)
- All quality agents (Self-Review, Refinement) use graceful fallbacks and never block the main workflow

---

## License

MIT License - feel free to use and contribute.

## Contact

*Created by Arnold - Exploring the future of AI4Sci.*
