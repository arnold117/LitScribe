# LitScribe

**LitScribe** is an autonomous academic synthesis engine designed to transform how researchers conduct literature reviews. By leveraging the **Model Context Protocol (MCP)** and a **Multi-Agent** architecture, LitScribe goes beyond simple summarization to provide deep, cross-paper synthesis and gap analysis.

---

## Vision

The goal of LitScribe is to act as a rigorous "Digital Scribe" for scholars—faithfully organizing human knowledge while eliminating the hallucinations common in vanilla LLM outputs.

## Key Features

### Multi-Agent Literature Review
- **Discovery Agent**: Query expansion, multi-source search, snowball sampling
- **Critical Reading Agent**: PDF parsing, key findings extraction, methodology analysis
- **Synthesis Agent**: Theme identification, gap analysis, review generation
- **Supervisor Agent**: LangGraph-based workflow orchestration

### Multi-Source Search
- Unified search across **arXiv**, **PubMed**, **Semantic Scholar**
- **Zotero** integration for personal library management
- Intelligent deduplication and merging across sources
- Citation tracking and paper recommendations

### Export & Citations
- **BibTeX** export with auto-detected entry types
- **5 citation styles**: APA, MLA, IEEE, Chicago, GB/T 7714
- **Multi-format export**: Word (.docx), PDF, HTML, LaTeX, Markdown
- **Multi-language**: English and Chinese output

### Caching & Persistence
- **SQLite cache**: Local-first search, PDF caching, parse results
- **Checkpointing**: Resume interrupted reviews via `thread_id`
- **Incremental updates**: Only fetch what's missing

### PDF Processing
- High-fidelity PDF-to-Markdown conversion
- LaTeX equation preservation
- Dual backend: `pymupdf4llm` (fast) / `marker-pdf` (OCR)

## Tech Stack

- **Language:** Python 3.12+
- **Orchestration:** LangGraph (multi-agent framework)
- **Interface:** Model Context Protocol (MCP) via FastMCP 2.0
- **Storage:** SQLite (cache, checkpointing)
- **Cloud LLM:** Claude Opus 4.5 / Sonnet 4.5 / DeepSeek-R1
- **PDF Processing:** pymupdf4llm (default) / marker-pdf (OCR)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LitScribe Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Research Question                                                  │
│        │                                                            │
│        ▼                                                            │
│  ┌──────────────┐                                                   │
│  │  Supervisor  │◄────────────────────────────────┐                │
│  │    Agent     │                                 │                │
│  └──────┬───────┘                                 │                │
│         │                                         │                │
│         ▼                                         │                │
│  ┌──────────────┐     ┌─────────────────┐        │                │
│  │  Discovery   │────▶│ MCP Servers     │        │                │
│  │    Agent     │     │ • arXiv         │        │                │
│  │              │     │ • PubMed        │        │                │
│  │ • Query exp  │     │ • Semantic S.   │        │                │
│  │ • Search     │     │ • Zotero        │        │                │
│  │ • Snowball   │     └─────────────────┘        │                │
│  └──────┬───────┘                                │                │
│         │                                         │                │
│         ▼                                         │                │
│  ┌──────────────┐     ┌─────────────────┐        │                │
│  │  Critical    │────▶│ PDF Parser      │        │                │
│  │  Reading     │     │ • pymupdf4llm   │        │                │
│  │    Agent     │     │ • marker-pdf    │        │                │
│  │              │     └─────────────────┘        │                │
│  │ • Parse PDF  │                                │                │
│  │ • Findings   │     ┌─────────────────┐        │                │
│  │ • Quality    │────▶│ SQLite Cache    │        │                │
│  └──────┬───────┘     │ • Papers        │        │                │
│         │             │ • PDFs          │        │                │
│         ▼             │ • Checkpoints   │        │                │
│  ┌──────────────┐     └─────────────────┘        │                │
│  │  Synthesis   │                                │                │
│  │    Agent     │────────────────────────────────┘                │
│  │              │                                                  │
│  │ • Themes     │     ┌─────────────────┐                         │
│  │ • Gaps       │────▶│ Export          │                         │
│  │ • Review     │     │ • BibTeX        │                         │
│  └──────────────┘     │ • Word/PDF      │                         │
│                       │ • Markdown      │                         │
│                       └─────────────────┘                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Workflow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Supervisor
    participant Discovery
    participant CriticalReading
    participant Synthesis
    participant Cache

    User->>CLI: litscribe review "question"
    CLI->>Supervisor: Start workflow

    loop Until complete
        Supervisor->>Discovery: Search papers
        Discovery->>Cache: Check local cache
        Cache-->>Discovery: Cached results
        Discovery->>Discovery: Query expansion
        Discovery->>Discovery: Multi-source search
        Discovery-->>Supervisor: Papers found

        Supervisor->>CriticalReading: Analyze papers
        CriticalReading->>Cache: Get cached PDFs
        CriticalReading->>CriticalReading: Parse & extract
        CriticalReading-->>Supervisor: Paper summaries

        Supervisor->>Synthesis: Generate review
        Synthesis->>Synthesis: Themes & gaps
        Synthesis-->>Supervisor: Review complete
    end

    Supervisor-->>CLI: Final state
    CLI-->>User: Save .md + .json
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

# === Export ===
litscribe export review.json -f docx -s apa      # Word (APA style)
litscribe export review.json -f pdf -s ieee      # PDF (IEEE style)
litscribe export review.json -f bibtex           # BibTeX only
litscribe export review.json -f md -l zh         # Chinese Markdown

# === Search ===
litscribe search "transformer attention" --sources arxiv,semantic_scholar
litscribe search "CRISPR" --sources pubmed --limit 20 --sort citations

# === Paper Info ===
litscribe paper arXiv:1706.03762 --verbose
litscribe citations arXiv:1706.03762 --limit 10

# === PDF Processing ===
litscribe parse paper.pdf --output paper.md

# === Cache Management ===
litscribe cache stats              # Show statistics
litscribe cache clear --expired    # Clear expired entries
litscribe cache vacuum             # Optimize database
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
| MVP | MCP servers, unified search, CLI | Done |
| Iteration 2 | Multi-agent system, LangGraph | Done |
| Phase 6.5 | SQLite cache, checkpointing | Done |
| Phase 7 | BibTeX, export, multi-language | Done |

### Planned

| Phase | Description | Priority |
|-------|-------------|----------|
| Phase 7.5 | GraphRAG, scale-up (50-500 papers) | Next |
| Phase 8 | Claude Code plugin (MCP Server) | Medium |
| Phase 9 | Visualization (plots, networks) | Medium |
| Phase 10 | Peer Review Agent | Low |

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
| Review synthesis | Opus 4.5 | Complex reasoning |
| Batch processing | DeepSeek-R3 | Cost-effective |

---

## License

MIT License - feel free to use and contribute.

## Contact

*Created by Arnold - Exploring the future of AI4Sci.*
