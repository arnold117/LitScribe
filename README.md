# 🖋️ LitScribe

**LitScribe** is an autonomous academic synthesis engine designed to transform how researchers conduct literature reviews. By leveraging the **Model Context Protocol (MCP)** and a **Multi-Agent** architecture, LitScribe goes beyond simple summarization to provide deep, cross-paper synthesis and gap analysis.

---

## 🌟 Vision
The goal of LitScribe is to act as a rigorous "Digital Scribe" for scholars—faithfully organizing human knowledge while eliminating the hallucinations common in vanilla LLM outputs.

## 🚀 Key Features

### MVP (Current Phase) ✅
- **Multi-Source Literature Search**: Unified search across arXiv, PubMed, Semantic Scholar
- **MCP Integration**: Standardized connectors for academic databases and Zotero
- **Intelligent Deduplication**: Smart merging of papers from multiple sources
- **PDF-to-Markdown Parsing**: High-fidelity conversion of academic PDFs with LaTeX support
- **CLI Tool**: Full-featured command line interface for all operations
- **Semantic Scholar API**: Citation tracking, paper recommendations, TL;DR summaries

### Planned Features
- **Multi-Agent Debate**: Resolve conflicting claims across papers
- **Agentic Workflow**: Specialized agents for *Discovery*, *Critical Reading*, and *Synthesis*
- **Traceable Citations**: Every claim backed by direct evidence from source PDFs
- **M4 Optimized**: Native support for Apple Silicon via MLX

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

# Copy and configure environment variables
cp .env.example .env
# Edit .env and add your API keys

# Verify installation
python -c "import fastmcp; import langgraph; print('✅ LitScribe ready!')"
```

### Usage

```bash
# Search papers across multiple sources
python src/cli/litscribe_cli.py search "transformer attention" --sources arxiv,semantic_scholar

# Search PubMed for biomedical literature
python src/cli/litscribe_cli.py search "CRISPR" --sources pubmed --limit 20 --sort citations

# Get detailed paper info (supports DOI, arXiv ID, PMID)
python src/cli/litscribe_cli.py paper arXiv:1706.03762 --verbose

# Get papers citing a specific paper
python src/cli/litscribe_cli.py citations arXiv:1706.03762 --limit 10

# Parse a PDF to markdown
python src/cli/litscribe_cli.py parse paper.pdf --output paper.md

# Run interactive demo
python src/cli/litscribe_cli.py demo

# Run full demo workflow
python scripts/demo_workflow.py --demo all
```

## 📂 Project Status

### ✅ MVP Complete
- [x] Project structure and configuration
- [x] Environment setup (mamba)
- [x] Dependency management
- [x] **arXiv MCP Server** - search, download PDFs, batch operations
- [x] **PubMed MCP Server** - search, citations, MeSH terms
- [x] **Zotero MCP Server** - library search, collections, PDF paths
- [x] **PDF Parser MCP Server** - dual backend (pymupdf4llm/marker-pdf)
- [x] **Semantic Scholar MCP Server** - citations, references, recommendations, TL;DR
- [x] **Unified Search Aggregator** - parallel search, deduplication, ranking
- [x] **CLI Tool** - search, paper, citations, parse, demo commands
- [x] **Demo Workflow** - end-to-end demonstration scripts

### 📋 Next Phase
- [ ] Semantic Scholar API key integration (pending approval)
- [ ] Multi-agent synthesis system
- [ ] Debate mechanism for conflicting papers
- [ ] Web interface (FastAPI + React)
- [ ] Advanced hallucination control

## 📝 Development Notes

### PDF Parsing Backend Selection
We provide two PDF parsing backends:

| Backend | Speed | OCR | Stability | Use Case |
|---------|-------|-----|-----------|----------|
| `pymupdf4llm` | Fast | No | Stable | Native text PDFs (most papers) |
| `marker-pdf` | Slow | Yes | Unstable on macOS MPS | Scanned PDFs, complex layouts |

**Default:** `pymupdf4llm` for stability and speed.

**Known Issue:** `marker-pdf` has dependency conflicts and segfaults on macOS with MPS (Apple Silicon). If you need OCR support, consider running on Linux/CUDA or using CPU-only mode.

---

## 📄 License
MIT License - feel free to use and contribute.

## 🤝 Contact
*Created by Arnold - Exploring the future of AI4Sci.*