# 🖋️ LitScribe

**LitScribe** is an autonomous academic synthesis engine designed to transform how researchers conduct literature reviews. By leveraging the **Model Context Protocol (MCP)** and a **Multi-Agent** architecture, LitScribe goes beyond simple summarization to provide deep, cross-paper synthesis and gap analysis.

---

## 🌟 Vision
The goal of LitScribe is to act as a rigorous "Digital Scribe" for scholars—faithfully organizing human knowledge while eliminating the hallucinations common in vanilla LLM outputs.

## 🚀 Key Features

### MVP (Current Phase)
- **Multi-Source Literature Search**: Unified search across arXiv, PubMed, and Google Scholar
- **MCP Integration**: Standardized connectors for academic databases and Zotero
- **Intelligent Deduplication**: Smart merging of papers from multiple sources
- **PDF-to-Markdown Parsing**: High-fidelity conversion of academic PDFs with LaTeX support
- **Semantic Search**: Vector-based search in your Zotero library

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
- **PDF Processing:** marker-pdf (4x faster than nougat)

## ⚡ Quick Start

### Prerequisites
- macOS with Apple Silicon (M4 recommended)
- [Mamba](https://mamba.readthedocs.io/) or Conda
- API keys for Claude, SerpApi (optional: NCBI)

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
python -m litscribe.cli search "transformer architecture" --sources arxiv,pubmed,scholar

# Parse a PDF
python -m litscribe.cli parse paper.pdf --output markdown

# Run demo workflow
python scripts/demo_workflow.py
```

## 📂 Project Status

### ✅ Completed
- [x] Project structure and configuration
- [x] Environment setup (mamba)
- [x] Dependency management

### 🚧 In Progress (MVP)
- [ ] arXiv MCP Server
- [ ] PubMed MCP Server
- [ ] Google Scholar MCP Server
- [ ] Unified search aggregator
- [ ] PDF parser MCP Server
- [ ] Zotero integration with semantic search
- [ ] CLI tool and demo scripts

### 📋 Future Iterations
- [ ] Multi-agent synthesis system
- [ ] Debate mechanism for conflicting papers
- [ ] Web interface (FastAPI + React)
- [ ] Advanced hallucination control

---

## 📄 License
MIT License - feel free to use and contribute.

## 🤝 Contact
*Created by Arnold - Exploring the future of AI4Sci.*