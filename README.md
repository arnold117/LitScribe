# 🖋️ LitScribe

**LitScribe** is an autonomous academic synthesis engine designed to transform how researchers conduct literature reviews. By leveraging the **Model Context Protocol (MCP)** and a **Multi-Agent** architecture, LitScribe goes beyond simple summarization to provide deep, cross-paper synthesis and gap analysis.

---

## 🌟 Vision
The goal of LitScribe is to act as a rigorous "Digital Scribe" for scholars—faithfully organizing human knowledge while eliminating the hallucinations common in vanilla LLM outputs.

## 🚀 Key Features

### Iteration 2 (Current) ✅
- **Multi-Agent Literature Review**: Automated end-to-end literature review generation
  - **Discovery Agent**: Query expansion, multi-source search, snowball sampling
  - **Critical Reading Agent**: PDF parsing, key findings extraction, methodology analysis
  - **Synthesis Agent**: Theme identification, gap analysis, review generation
- **LangGraph Orchestration**: State-based workflow with supervisor routing
- **Auto-Save Output**: Reviews saved to `output/` with full JSON data

### MVP ✅
- **Multi-Source Literature Search**: Unified search across arXiv, PubMed, Semantic Scholar
- **MCP Integration**: Standardized connectors for academic databases and Zotero
- **Intelligent Deduplication**: Smart merging of papers from multiple sources
- **PDF-to-Markdown Parsing**: High-fidelity conversion of academic PDFs with LaTeX support
- **CLI Tool**: Full-featured command line interface for all operations
- **Semantic Scholar API**: Citation tracking, paper recommendations, TL;DR summaries

### Planned Features (Iteration 3+)
- **BibTeX Export**: Automatic citation generation in multiple formats
- **Claude Code Plugin**: MCP Server mode for IDE integration
- **Visualization**: Forest plots, citation networks, timelines
- **Peer Review Agent**: Self-critique and verification
- **Multi-Agent Debate**: Resolve conflicting claims across papers

## 🛠️ Tech Stack
- **Language:** Python 3.12+ (managed by `mamba`)
- **Orchestration:** LangGraph (production-ready multi-agent framework)
- **Interface:** Model Context Protocol (MCP) via FastMCP 2.0
- **Local LLM:** Qwen 3 (32B/14B) via MLX/Ollama - optimized for M4
- **Cloud LLM:** Claude Opus 4.5 / Sonnet 4.5 for complex synthesis
- **PDF Processing:** pymupdf4llm (default) / marker-pdf (optional, OCR support)

## ⚡ Quick Start

### Prerequisites
- macOS with Apple Silicon (M4 recommended)
- [Mamba](https://mamba.readthedocs.io/) or Conda
- API keys: Anthropic (Claude), optional: NCBI (PubMed), Semantic Scholar, Zotero

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LitScribe.git
cd LitScribe

# Create mamba environment
mamba env create -f environment.yml
mamba activate litscribe

# Install as editable package (for CLI)
pip install -e .

# Copy and configure environment variables
cp .env.example .env
# Edit .env and add your API keys

# Verify installation
litscribe --help
```

### Usage

```bash
# === Literature Review (Multi-Agent System) ===
# Generate a complete literature review
litscribe review "What are the latest advances in LLM reasoning?"

# Specify sources and paper count
litscribe review "CRISPR applications in cancer" -s pubmed,arxiv -p 15

# Custom output path
litscribe review "transformer architecture" -o my_review

# === Search & Discovery ===
# Search papers across multiple sources
litscribe search "transformer attention" --sources arxiv,semantic_scholar

# Search PubMed for biomedical literature
litscribe search "CRISPR" --sources pubmed --limit 20 --sort citations

# Get detailed paper info (supports DOI, arXiv ID, PMID)
litscribe paper arXiv:1706.03762 --verbose

# Get papers citing a specific paper
litscribe citations arXiv:1706.03762 --limit 10

# === PDF Processing ===
# Parse a PDF to markdown
litscribe parse paper.pdf --output paper.md

# === Demo ===
litscribe demo
```

### Output Files

When running `litscribe review`, outputs are saved to `output/`:
- `review_*.md` - The literature review in Markdown format
- `review_*.json` - Full data (search results, paper analyses, synthesis)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LitScribe Workflow                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Research Question                                          │
│        │                                                    │
│        ▼                                                    │
│  ┌──────────┐     ┌───────────────┐     ┌─────────────┐   │
│  │Supervisor│────▶│Discovery Agent│────▶│  Critical   │   │
│  │  Agent   │     │               │     │Reading Agent│   │
│  └──────────┘     │ • Query expand│     │             │   │
│        ▲          │ • Multi-search│     │ • PDF parse │   │
│        │          │ • Snowball    │     │ • Findings  │   │
│        │          └───────────────┘     │ • Quality   │   │
│        │                                └──────┬──────┘   │
│        │                                       │          │
│        │          ┌───────────────┐            │          │
│        └──────────│Synthesis Agent│◀───────────┘          │
│                   │               │                        │
│                   │ • Themes      │                        │
│                   │ • Gaps        │                        │
│                   │ • Review text │                        │
│                   └───────────────┘                        │
│                          │                                 │
│                          ▼                                 │
│                   Literature Review                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📂 Project Status

### ✅ Iteration 1: MVP Complete
- [x] Project structure and configuration
- [x] Environment setup (mamba)
- [x] **arXiv MCP Server** - search, download PDFs
- [x] **PubMed MCP Server** - search, citations, MeSH terms
- [x] **Zotero MCP Server** - library search, collections
- [x] **PDF Parser MCP Server** - dual backend (pymupdf4llm/marker-pdf)
- [x] **Semantic Scholar MCP Server** - citations, references, TL;DR
- [x] **Unified Search Aggregator** - parallel search, deduplication
- [x] **CLI Tool** - search, paper, citations, parse, demo

### ✅ Iteration 2: Multi-Agent System Complete
- [x] LangGraph state management (`state.py`)
- [x] Error handling and retry logic (`errors.py`)
- [x] MCP tool wrappers (`tools.py`)
- [x] LLM prompt templates (`prompts.py`)
- [x] **Discovery Agent** - query expansion, multi-source search, paper selection
- [x] **Critical Reading Agent** - PDF acquisition, key findings, methodology
- [x] **Synthesis Agent** - themes, gaps, review generation
- [x] **Supervisor Agent** - workflow routing
- [x] **LangGraph Graph** - state machine orchestration
- [x] **CLI Integration** - `litscribe review` command

### 📋 Iteration 3: Planned
- [ ] BibTeX export functionality
- [ ] Claude Code plugin mode (MCP Server)
- [ ] Visualization generation (forest plots, citation networks)
- [ ] Peer Review Agent

## 📝 Development Notes

### PDF Parsing Backend Selection
We provide two PDF parsing backends:

| Backend | Speed | OCR | Stability | Use Case |
|---------|-------|-----|-----------|----------|
| `pymupdf4llm` | Fast | No | Stable | Native text PDFs (most papers) |
| `marker-pdf` | Slow | Yes | Unstable on macOS MPS | Scanned PDFs, complex layouts |

**Default:** `pymupdf4llm` for stability and speed.

### Rate Limits
- **Semantic Scholar API**: 1 request/second (with API key)
- **PubMed**: 3 requests/second (without key), 10/second (with NCBI key)

---

## 📄 License
MIT License - feel free to use and contribute.

## 🤝 Contact
*Created by Arnold - Exploring the future of AI4Sci.*
