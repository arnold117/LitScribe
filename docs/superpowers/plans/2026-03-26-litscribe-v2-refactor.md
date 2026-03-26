# LitScribe v2 Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite LitScribe from LangGraph to Agno with Hermes-style self-evolution, 百炼 API, unified storage, and API-first architecture.

**Architecture:** Agno Workflow for deterministic pipeline (Plan→Discovery→Reading→Synthesis→Self-Review), Agno Team for parallel sub-tasks, three-tier memory (Episodic FTS5 + Semantic ChromaDB + Procedural Skills), FastAPI core API with Typer CLI thin client.

**Tech Stack:** Python 3.11+, agno, chromadb, litellm, aiosqlite, FastAPI, Typer, httpx, sentence-transformers, networkx, graspologic, pymupdf4llm

**Design Spec:** `docs/superpowers/specs/2026-03-26-litscribe-v2-refactor-design.md`

---

## File Structure

```
litscribe/                      # New top-level package (replaces src/)
├── __init__.py
├── config.py                   # Central config (env + yaml)
│
├── store/                      # Unified storage layer
│   ├── __init__.py
│   ├── unified.py              # UnifiedStore facade
│   ├── sqlite.py               # SQLite ops + schema v5 + FTS5
│   └── vectors.py              # ChromaDB wrapper
│
├── llm/                        # LLM abstraction
│   ├── __init__.py
│   ├── router.py               # LLMRouter per-task routing
│   └── tracker.py              # TokenTracker
│
├── models/                     # Pydantic data contracts (shared)
│   ├── __init__.py
│   ├── paper.py                # Paper, SearchMeta
│   ├── plan.py                 # ResearchPlan, SubTopic, ReviewTier
│   ├── analysis.py             # PaperAnalysis, ParsedDoc
│   ├── review.py               # ReviewOutput, Citation, Theme
│   └── assessment.py           # ReviewAssessment
│
├── services/                   # Search API clients
│   ├── __init__.py
│   ├── base.py                 # SearchService Protocol
│   ├── arxiv.py                # Adapted from src/services/arxiv.py
│   ├── pubmed.py               # Adapted from src/services/pubmed.py
│   ├── semantic_scholar.py     # Adapted from src/services/semantic_scholar.py
│   ├── openalex.py             # Adapted from src/services/openalex.py
│   ├── europe_pmc.py           # Adapted from src/services/europe_pmc.py
│   ├── zotero.py               # Adapted from src/services/zotero.py
│   └── pdf.py                  # PDF fetch chain + parse
│
├── agents/                     # Agno agents + pipeline
│   ├── __init__.py
│   ├── pipeline.py             # Agno Workflow: LitScribePipeline
│   ├── planner.py              # Planning agent
│   ├── discovery.py            # Discovery Team (multi-source parallel)
│   ├── reader.py               # Critical reading agent
│   ├── synthesizer.py          # Synthesis agent
│   └── reviewer.py             # Self-review agent
│
├── evolution/                  # Self-evolution layer
│   ├── __init__.py
│   ├── memory_manager.py       # Unified MemoryManager interface
│   ├── episodic.py             # EpisodicMemory (SQLite + FTS5)
│   ├── semantic.py             # SemanticMemory (ChromaDB)
│   ├── procedural.py           # ProceduralMemory (filesystem skills)
│   └── skill_evolver.py        # SkillEvolver: extract/patch/inject
│
├── plugins/                    # Optional plugins
│   ├── __init__.py
│   └── graphrag/
│       ├── __init__.py
│       ├── plugin.py           # GraphRAGPlugin entry point
│       ├── entity_extractor.py # Adapted from src/graphrag/
│       ├── linker.py           # Adapted from src/graphrag/
│       ├── graph_builder.py    # Adapted from src/graphrag/
│       ├── community_detector.py
│       └── summarizer.py
│
├── exporters/                  # Copy from src/exporters/ with minor adaptations
│   ├── __init__.py
│   ├── bibtex.py
│   ├── pandoc.py
│   └── citation_formatter.py
│
├── api/                        # FastAPI layer
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, lifespan, CORS
│   ├── deps.py                 # Dependency injection
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── reviews.py          # POST/GET/DELETE /api/reviews
│   │   ├── sessions.py         # GET /api/sessions
│   │   ├── memory.py           # GET/PUT /api/memory/*
│   │   └── config_routes.py    # GET/PUT /api/config
│   └── websocket.py            # WS /ws/reviews/{id}
│
└── cli/                        # Typer thin client
    ├── __init__.py
    └── main.py                 # ~200 lines, HTTP calls to API

tests/
├── conftest.py                 # Shared fixtures
├── test_config.py
├── test_store/
│   ├── test_sqlite.py
│   └── test_vectors.py
├── test_llm/
│   ├── test_router.py
│   └── test_tracker.py
├── test_models/
│   └── test_contracts.py
├── test_services/
│   └── test_base.py
├── test_agents/
│   ├── test_pipeline.py
│   ├── test_planner.py
│   ├── test_discovery.py
│   ├── test_reader.py
│   ├── test_synthesizer.py
│   └── test_reviewer.py
├── test_evolution/
│   ├── test_episodic.py
│   ├── test_semantic.py
│   ├── test_procedural.py
│   └── test_skill_evolver.py
├── test_plugins/
│   └── test_graphrag.py
├── test_api/
│   ├── test_reviews.py
│   └── test_memory.py
└── test_cli/
    └── test_main.py
```

---

## Phase 1: Project Skeleton + Config + Models

### Task 1: Project skeleton and dependencies

**Files:**
- Create: `litscribe/__init__.py`
- Create: `pyproject.toml` (new, replaces existing)
- Create: `tests/conftest.py`

- [ ] **Step 1: Create package structure**

```bash
mkdir -p litscribe/{store,llm,models,services,agents,evolution,plugins/graphrag,exporters,api/routes,cli}
touch litscribe/__init__.py litscribe/store/__init__.py litscribe/llm/__init__.py litscribe/models/__init__.py litscribe/services/__init__.py litscribe/agents/__init__.py litscribe/evolution/__init__.py litscribe/plugins/__init__.py litscribe/plugins/graphrag/__init__.py litscribe/exporters/__init__.py litscribe/api/__init__.py litscribe/api/routes/__init__.py litscribe/cli/__init__.py
mkdir -p tests/{test_store,test_llm,test_models,test_services,test_agents,test_evolution,test_plugins,test_api,test_cli}
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[project]
name = "litscribe"
version = "2.0.0"
description = "Self-evolving multi-agent literature review engine"
requires-python = ">=3.11"
dependencies = [
    "agno>=2.5.0",
    "litellm>=1.40.0",
    "aiosqlite>=0.20.0",
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "networkx>=3.0",
    "graspologic>=3.0",
    "pymupdf4llm>=0.0.10",
    "tenacity>=8.0",
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    "typer>=0.12.0",
    "httpx>=0.27.0",
    "pydantic>=2.7.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23.0",
    "pytest-httpx>=0.30.0",
    "httpx>=0.27.0",
]

[project.scripts]
litscribe = "litscribe.cli.main:app"
litscribe-server = "litscribe.api.main:run"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 3: Write tests/conftest.py**

```python
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary directory for test data (SQLite, ChromaDB, skills)."""
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    return db_dir


@pytest.fixture
def tmp_skills_dir(tmp_path):
    """Temporary directory for procedural skills."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    return skills_dir


@pytest.fixture(autouse=True)
def env_override(monkeypatch, tmp_data_dir):
    """Override environment for all tests."""
    monkeypatch.setenv("LITSCRIBE_DATA_DIR", str(tmp_data_dir))
    monkeypatch.setenv("LITSCRIBE_TESTING", "1")
```

- [ ] **Step 4: Install in dev mode and verify**

Run: `pip install -e ".[dev]"`
Expected: Clean install, no errors

- [ ] **Step 5: Commit**

```bash
git add litscribe/ tests/conftest.py pyproject.toml
git commit -m "feat(v2): project skeleton with Agno + ChromaDB dependencies"
```

---

### Task 2: Central configuration

**Files:**
- Create: `litscribe/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import os
import pytest
from pathlib import Path


def test_config_loads_defaults():
    from litscribe.config import Config

    cfg = Config()
    assert cfg.llm.default_model == "openai/qwen-plus"
    assert cfg.llm.api_base == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert cfg.data_dir.is_absolute()


def test_config_loads_env(monkeypatch):
    from litscribe.config import Config

    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test123")
    monkeypatch.setenv("LITSCRIBE_DEFAULT_MODEL", "openai/qwen-max")
    cfg = Config()
    assert cfg.llm.api_key == "sk-test123"
    assert cfg.llm.default_model == "openai/qwen-max"


def test_config_task_models_default():
    from litscribe.config import Config

    cfg = Config()
    assert "synthesis" in cfg.llm.task_models
    assert "query_expansion" in cfg.llm.task_models


def test_config_yaml_override(tmp_path):
    from litscribe.config import Config

    yaml_content = """
llm:
  default_model: "openai/qwen-turbo"
  task_models:
    synthesis: "openai/deepseek-r1"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    cfg = Config(config_path=config_file)
    assert cfg.llm.default_model == "openai/qwen-turbo"
    assert cfg.llm.task_models["synthesis"] == "openai/deepseek-r1"
    # Non-overridden task models still have defaults
    assert "query_expansion" in cfg.llm.task_models


def test_config_data_directories(tmp_path, monkeypatch):
    from litscribe.config import Config

    monkeypatch.setenv("LITSCRIBE_DATA_DIR", str(tmp_path))
    cfg = Config()
    assert cfg.data_dir == tmp_path
    assert cfg.db_path == tmp_path / "litscribe.db"
    assert cfg.chroma_path == tmp_path / "vectors"
    assert cfg.skills_dir == Path.home() / ".litscribe" / "skills"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'litscribe.config'`

- [ ] **Step 3: Write implementation**

```python
# litscribe/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_TASK_MODELS = {
    "query_expansion": "openai/qwen-turbo",
    "planning": "openai/qwen-plus",
    "paper_analysis": "openai/qwen-plus",
    "entity_extraction": "openai/qwen-turbo",
    "synthesis": "openai/qwen-max",
    "self_review": "openai/qwen-plus",
    "refinement": "openai/deepseek-r1",
    "skill_extraction": "openai/qwen-turbo",
}

_DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class LLMConfig:
    api_base: str = _DEFAULT_API_BASE
    api_key: str = ""
    default_model: str = "openai/qwen-plus"
    reasoning_model: str = "openai/deepseek-r1"
    task_models: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_TASK_MODELS))


@dataclass
class ServiceConfig:
    ncbi_email: str = ""
    ncbi_api_key: str = ""
    semantic_scholar_api_key: str = ""
    zotero_api_key: str = ""
    zotero_library_id: str = ""
    unpaywall_email: str = ""
    max_concurrent_requests: int = 5


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    services: ServiceConfig = field(default_factory=ServiceConfig)
    data_dir: Path = field(default_factory=lambda: Path.home() / ".litscribe" / "data")
    skills_dir: Path = field(default_factory=lambda: Path.home() / ".litscribe" / "skills")
    graphrag_enabled: bool = True
    debug: bool = False

    def __init__(self, config_path: Path | None = None):
        self.llm = LLMConfig()
        self.services = ServiceConfig()
        self.data_dir = Path(os.getenv("LITSCRIBE_DATA_DIR", Path.home() / ".litscribe" / "data"))
        self.skills_dir = Path(os.getenv("LITSCRIBE_SKILLS_DIR", Path.home() / ".litscribe" / "skills"))
        self.graphrag_enabled = os.getenv("LITSCRIBE_GRAPHRAG_ENABLED", "1") == "1"
        self.debug = os.getenv("LITSCRIBE_DEBUG", "0") == "1"

        self._load_env()
        if config_path and config_path.exists():
            self._load_yaml(config_path)
        else:
            default_yaml = Path.home() / ".litscribe" / "config.yaml"
            if default_yaml.exists():
                self._load_yaml(default_yaml)

    def _load_env(self):
        self.llm.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.llm.api_base = os.getenv("LITSCRIBE_API_BASE", _DEFAULT_API_BASE)
        self.llm.default_model = os.getenv("LITSCRIBE_DEFAULT_MODEL", self.llm.default_model)

        self.services.ncbi_email = os.getenv("NCBI_EMAIL", "")
        self.services.ncbi_api_key = os.getenv("NCBI_API_KEY", "")
        self.services.semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        self.services.zotero_api_key = os.getenv("ZOTERO_API_KEY", "")
        self.services.zotero_library_id = os.getenv("ZOTERO_LIBRARY_ID", "")
        self.services.unpaywall_email = os.getenv("UNPAYWALL_EMAIL", "")

    def _load_yaml(self, path: Path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        llm_data = data.get("llm", {})
        if "default_model" in llm_data:
            self.llm.default_model = llm_data["default_model"]
        if "api_base" in llm_data:
            self.llm.api_base = llm_data["api_base"]
        if "api_key" in llm_data:
            self.llm.api_key = llm_data["api_key"]
        if "reasoning_model" in llm_data:
            self.llm.reasoning_model = llm_data["reasoning_model"]
        if "task_models" in llm_data:
            self.llm.task_models.update(llm_data["task_models"])

        services_data = data.get("services", {})
        for key, val in services_data.items():
            if hasattr(self.services, key):
                setattr(self.services, key, val)

        if "graphrag_enabled" in data:
            self.graphrag_enabled = data["graphrag_enabled"]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "litscribe.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "vectors"

    def ensure_directories(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/config.py tests/test_config.py
git commit -m "feat(v2): central config with env + yaml + 百炼 defaults"
```

---

### Task 3: Pydantic data contracts

**Files:**
- Create: `litscribe/models/paper.py`
- Create: `litscribe/models/plan.py`
- Create: `litscribe/models/analysis.py`
- Create: `litscribe/models/review.py`
- Create: `litscribe/models/assessment.py`
- Create: `tests/test_models/test_contracts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models/test_contracts.py
import pytest
from pydantic import ValidationError


def test_paper_model():
    from litscribe.models.paper import Paper

    p = Paper(
        paper_id="arxiv:2412.15115",
        title="Test Paper",
        authors=["Alice", "Bob"],
        abstract="An abstract.",
        year=2024,
        sources={"arxiv": "2412.15115"},
    )
    assert p.paper_id == "arxiv:2412.15115"
    assert p.relevance_score == 0.0  # default
    assert p.pdf_urls == []  # default


def test_paper_requires_title():
    from litscribe.models.paper import Paper

    with pytest.raises(ValidationError):
        Paper(paper_id="x", authors=[], abstract="", year=2024, sources={})


def test_research_plan_model():
    from litscribe.models.plan import ResearchPlan, SubTopic, ReviewTier

    plan = ResearchPlan(
        question="LLM reasoning",
        sub_topics=[SubTopic(name="chain of thought", keywords=["CoT", "reasoning"])],
        domain="NLP/AI",
        tier=ReviewTier.STANDARD,
    )
    assert len(plan.sub_topics) == 1
    assert plan.tier == ReviewTier.STANDARD


def test_review_output_model():
    from litscribe.models.review import ReviewOutput, Citation, Theme

    review = ReviewOutput(
        text="A review.",
        citations=[Citation(paper_id="x", claim="claim", section="intro")],
        themes=[Theme(name="theme1", description="desc", paper_ids=["x"])],
        word_count=100,
    )
    assert review.word_count == 100


def test_review_assessment_model():
    from litscribe.models.assessment import ReviewAssessment

    assessment = ReviewAssessment(
        passed=False,
        score=0.45,
        feedback="Needs more coverage",
        refined_queries=["LLM reasoning chains"],
    )
    assert not assessment.passed
    assert assessment.refined_queries is not None


def test_paper_analysis_model():
    from litscribe.models.analysis import PaperAnalysis

    analysis = PaperAnalysis(
        paper_id="arxiv:2412.15115",
        key_findings=["Finding 1"],
        methodology="Experimental",
        strengths=["Strong design"],
        limitations=["Small sample"],
        relevance_score=0.85,
    )
    assert analysis.relevance_score == 0.85


def test_models_are_serializable():
    from litscribe.models.paper import Paper

    p = Paper(
        paper_id="x",
        title="Test",
        authors=["A"],
        abstract="abs",
        year=2024,
        sources={"arxiv": "123"},
    )
    data = p.model_dump()
    p2 = Paper.model_validate(data)
    assert p == p2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models/test_contracts.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write models**

```python
# litscribe/models/paper.py
from pydantic import BaseModel


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int
    sources: dict[str, str]  # {"arxiv": "2412.15115", "scholar": "12345"}
    venue: str = ""
    citations: int = 0
    doi: str = ""
    pdf_urls: list[str] = []
    relevance_score: float = 0.0
    completeness_score: float = 0.0


class SearchMeta(BaseModel):
    total_found: int = 0
    sources_queried: list[str] = []
    queries_used: list[str] = []
    snowball_rounds: int = 0
```

```python
# litscribe/models/plan.py
from enum import Enum

from pydantic import BaseModel


class ReviewTier(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class SubTopic(BaseModel):
    name: str
    keywords: list[str] = []
    estimated_papers: int = 10


class ResearchPlan(BaseModel):
    question: str
    sub_topics: list[SubTopic]
    domain: str
    tier: ReviewTier
    max_papers: int = 40
    language: str = "en"
    target_words: int = 0
```

```python
# litscribe/models/analysis.py
from pydantic import BaseModel


class ParsedDoc(BaseModel):
    paper_id: str
    markdown: str = ""
    sections: list[dict] = []
    word_count: int = 0


class PaperAnalysis(BaseModel):
    paper_id: str
    key_findings: list[str]
    methodology: str = ""
    strengths: list[str] = []
    limitations: list[str] = []
    relevance_score: float = 0.0
    themes: list[str] = []
```

```python
# litscribe/models/review.py
from pydantic import BaseModel


class Citation(BaseModel):
    paper_id: str
    claim: str
    section: str = ""


class Theme(BaseModel):
    name: str
    description: str
    paper_ids: list[str] = []


class ReviewOutput(BaseModel):
    text: str
    citations: list[Citation] = []
    themes: list[Theme] = []
    word_count: int = 0
    language: str = "en"
```

```python
# litscribe/models/assessment.py
from pydantic import BaseModel


class ReviewAssessment(BaseModel):
    passed: bool
    score: float
    feedback: str
    refined_queries: list[str] | None = None
    coverage_score: float = 0.0
    weak_claims: list[str] = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models/test_contracts.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/models/ tests/test_models/
git commit -m "feat(v2): Pydantic data contracts for inter-agent communication"
```

---

## Phase 2: Infrastructure Layer

### Task 4: SQLite store with FTS5

**Files:**
- Create: `litscribe/store/sqlite.py`
- Create: `tests/test_store/test_sqlite.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store/test_sqlite.py
import pytest
import pytest_asyncio
import asyncio


@pytest_asyncio.fixture
async def sqlite_store(tmp_data_dir):
    from litscribe.store.sqlite import SQLiteStore

    store = SQLiteStore(tmp_data_dir / "test.db")
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_initialize_creates_tables(sqlite_store):
    tables = await sqlite_store.list_tables()
    assert "papers" in tables
    assert "parsed_docs" in tables
    assert "episodes" in tables
    assert "sessions" in tables
    assert "skills_meta" in tables


@pytest.mark.asyncio
async def test_save_and_get_paper(sqlite_store):
    from litscribe.models.paper import Paper

    paper = Paper(
        paper_id="arxiv:2412.15115",
        title="Test Paper",
        authors=["Alice"],
        abstract="Abstract text",
        year=2024,
        sources={"arxiv": "2412.15115"},
    )
    await sqlite_store.save_papers([paper])
    result = await sqlite_store.get_paper("arxiv:2412.15115")
    assert result is not None
    assert result.title == "Test Paper"


@pytest.mark.asyncio
async def test_get_paper_not_found(sqlite_store):
    result = await sqlite_store.get_paper("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_save_and_get_parsed_doc(sqlite_store):
    from litscribe.models.analysis import ParsedDoc

    doc = ParsedDoc(paper_id="x", markdown="# Title\nContent", word_count=2)
    await sqlite_store.save_parsed("x", doc)
    result = await sqlite_store.get_parsed("x")
    assert result is not None
    assert result.markdown == "# Title\nContent"


@pytest.mark.asyncio
async def test_episode_save_and_fts5_recall(sqlite_store):
    await sqlite_store.save_episode(
        session_id="sess1",
        question="LLM reasoning capabilities",
        outcome_score=0.85,
        summary="Searched arxiv and S2 for reasoning papers. Found 35 papers.",
    )
    await sqlite_store.save_episode(
        session_id="sess2",
        question="Protein folding methods",
        outcome_score=0.7,
        summary="Searched PubMed for protein structure prediction. Found 20 papers.",
    )
    results = await sqlite_store.recall("reasoning LLM", limit=5)
    assert len(results) >= 1
    assert results[0]["session_id"] == "sess1"


@pytest.mark.asyncio
async def test_episode_recall_empty(sqlite_store):
    results = await sqlite_store.recall("nonexistent topic")
    assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store/test_sqlite.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# litscribe/store/sqlite.py
from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from litscribe.models.paper import Paper
from litscribe.models.analysis import ParsedDoc

SCHEMA_VERSION = 5

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,  -- JSON array
    abstract TEXT NOT NULL,
    year INTEGER NOT NULL,
    sources TEXT NOT NULL,  -- JSON dict
    venue TEXT DEFAULT '',
    citations INTEGER DEFAULT 0,
    doi TEXT DEFAULT '',
    pdf_urls TEXT DEFAULT '[]',  -- JSON array
    relevance_score REAL DEFAULT 0.0,
    completeness_score REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pdfs (
    paper_id TEXT PRIMARY KEY REFERENCES papers(paper_id),
    pdf_path TEXT NOT NULL,
    file_hash TEXT,
    file_size INTEGER,
    downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS parsed_docs (
    paper_id TEXT PRIMARY KEY REFERENCES papers(paper_id),
    markdown TEXT NOT NULL,
    sections TEXT DEFAULT '[]',  -- JSON array
    word_count INTEGER DEFAULT 0,
    parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    outcome_score REAL,
    summary TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    question, summary, content=episodes, content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, question, summary)
    VALUES (new.id, new.question, new.summary);
END;

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    research_question TEXT NOT NULL,
    review_type TEXT DEFAULT 'standard',
    language TEXT DEFAULT 'en',
    state_snapshot TEXT DEFAULT '{}',  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS review_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    version_number INTEGER NOT NULL,
    review_text TEXT NOT NULL,
    word_count INTEGER DEFAULT 0,
    instruction TEXT DEFAULT '',
    diff_text TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, version_number)
);

CREATE TABLE IF NOT EXISTS skills_meta (
    skill_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    domain TEXT DEFAULT '',
    version INTEGER DEFAULT 1,
    success_rate REAL DEFAULT 0.0,
    use_count INTEGER DEFAULT 0,
    last_used TIMESTAMP,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class SQLiteStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA_SQL)
        await self._db.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None

    async def list_tables(self) -> list[str]:
        cursor = await self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    # -- Papers --

    async def save_papers(self, papers: list[Paper]):
        for p in papers:
            await self._db.execute(
                """INSERT OR REPLACE INTO papers
                   (paper_id, title, authors, abstract, year, sources, venue,
                    citations, doi, pdf_urls, relevance_score, completeness_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    p.paper_id, p.title, json.dumps(p.authors), p.abstract,
                    p.year, json.dumps(p.sources), p.venue, p.citations,
                    p.doi, json.dumps(p.pdf_urls), p.relevance_score,
                    p.completeness_score,
                ),
            )
        await self._db.commit()

    async def get_paper(self, paper_id: str) -> Paper | None:
        cursor = await self._db.execute(
            "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return Paper(
            paper_id=row["paper_id"],
            title=row["title"],
            authors=json.loads(row["authors"]),
            abstract=row["abstract"],
            year=row["year"],
            sources=json.loads(row["sources"]),
            venue=row["venue"],
            citations=row["citations"],
            doi=row["doi"],
            pdf_urls=json.loads(row["pdf_urls"]),
            relevance_score=row["relevance_score"],
            completeness_score=row["completeness_score"],
        )

    # -- Parsed Docs --

    async def save_parsed(self, paper_id: str, doc: ParsedDoc):
        await self._db.execute(
            """INSERT OR REPLACE INTO parsed_docs
               (paper_id, markdown, sections, word_count)
               VALUES (?, ?, ?, ?)""",
            (paper_id, doc.markdown, json.dumps(doc.sections), doc.word_count),
        )
        await self._db.commit()

    async def get_parsed(self, paper_id: str) -> ParsedDoc | None:
        cursor = await self._db.execute(
            "SELECT * FROM parsed_docs WHERE paper_id = ?", (paper_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return ParsedDoc(
            paper_id=row["paper_id"],
            markdown=row["markdown"],
            sections=json.loads(row["sections"]),
            word_count=row["word_count"],
        )

    # -- Episodes (Episodic Memory) --

    async def save_episode(
        self, session_id: str, question: str, outcome_score: float, summary: str
    ):
        await self._db.execute(
            """INSERT INTO episodes (session_id, question, outcome_score, summary)
               VALUES (?, ?, ?, ?)""",
            (session_id, question, outcome_score, summary),
        )
        await self._db.commit()

    async def recall(self, query: str, limit: int = 5) -> list[dict]:
        cursor = await self._db.execute(
            """SELECT e.session_id, e.question, e.outcome_score, e.summary, e.created_at
               FROM episodes_fts fts
               JOIN episodes e ON e.id = fts.rowid
               WHERE episodes_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "session_id": row["session_id"],
                "question": row["question"],
                "outcome_score": row["outcome_score"],
                "summary": row["summary"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_store/test_sqlite.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/store/sqlite.py tests/test_store/test_sqlite.py
git commit -m "feat(v2): SQLite store with FTS5 episodic memory"
```

---

### Task 5: ChromaDB vector store

**Files:**
- Create: `litscribe/store/vectors.py`
- Create: `tests/test_store/test_vectors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store/test_vectors.py
import pytest


@pytest.fixture
def vector_store(tmp_data_dir):
    from litscribe.store.vectors import VectorStore

    store = VectorStore(tmp_data_dir / "vectors")
    return store


def test_add_and_search(vector_store):
    vector_store.add_texts(
        collection="test",
        texts=["LLM reasoning with chain of thought", "Protein folding prediction"],
        metadatas=[{"domain": "NLP"}, {"domain": "biology"}],
        ids=["doc1", "doc2"],
    )
    results = vector_store.search("reasoning language models", collection="test", n=1)
    assert len(results) == 1
    assert results[0]["id"] == "doc1"


def test_search_empty_collection(vector_store):
    results = vector_store.search("anything", collection="empty", n=5)
    assert results == []


def test_add_to_named_collections(vector_store):
    vector_store.add_texts(
        collection="semantic_memory",
        texts=["Knowledge chunk 1"],
        metadatas=[{"source": "paper1"}],
        ids=["k1"],
    )
    vector_store.add_texts(
        collection="skill_embeddings",
        texts=["NLP search strategy"],
        metadatas=[{"skill": "nlp_search"}],
        ids=["s1"],
    )
    sem_results = vector_store.search("knowledge", collection="semantic_memory", n=1)
    skill_results = vector_store.search("search", collection="skill_embeddings", n=1)
    assert len(sem_results) == 1
    assert len(skill_results) == 1
    assert sem_results[0]["id"] == "k1"
    assert skill_results[0]["id"] == "s1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store/test_vectors.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# litscribe/store/vectors.py
from __future__ import annotations

from pathlib import Path

import chromadb


class VectorStore:
    def __init__(self, path: Path):
        self._client = chromadb.PersistentClient(path=str(path))

    def _get_or_create(self, name: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(name=name)

    def add_texts(
        self,
        collection: str,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ):
        coll = self._get_or_create(collection)
        coll.add(documents=texts, metadatas=metadatas, ids=ids)

    def search(
        self, query: str, collection: str, n: int = 10
    ) -> list[dict]:
        try:
            coll = self._client.get_collection(collection)
        except ValueError:
            return []
        count = coll.count()
        if count == 0:
            return []
        results = coll.query(query_texts=[query], n_results=min(n, count))
        output = []
        for i in range(len(results["ids"][0])):
            output.append(
                {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                }
            )
        return output

    def delete(self, collection: str, ids: list[str]):
        try:
            coll = self._client.get_collection(collection)
            coll.delete(ids=ids)
        except ValueError:
            pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_store/test_vectors.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/store/vectors.py tests/test_store/test_vectors.py
git commit -m "feat(v2): ChromaDB vector store for semantic memory"
```

---

### Task 6: UnifiedStore facade

**Files:**
- Create: `litscribe/store/unified.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store/test_unified.py
import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def store(tmp_data_dir):
    from litscribe.store.unified import UnifiedStore

    s = UnifiedStore(db_path=tmp_data_dir / "test.db", chroma_path=tmp_data_dir / "vectors")
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_unified_store_paper_roundtrip(store):
    from litscribe.models.paper import Paper

    paper = Paper(
        paper_id="test1", title="T", authors=["A"], abstract="abs",
        year=2024, sources={"arxiv": "123"},
    )
    await store.save_papers([paper])
    result = await store.get_paper("test1")
    assert result is not None
    assert result.title == "T"


@pytest.mark.asyncio
async def test_unified_store_semantic_search(store):
    store.add_embeddings(
        texts=["deep learning for NLP"], metadatas=[{"source": "p1"}],
        ids=["e1"], collection="semantic_memory",
    )
    results = store.semantic_search("NLP neural networks", collection="semantic_memory", n=1)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_unified_store_episode_roundtrip(store):
    await store.save_episode(
        session_id="s1", question="test query",
        outcome_score=0.8, summary="found papers",
    )
    results = await store.recall("test query")
    assert len(results) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store/test_unified.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# litscribe/store/unified.py
from __future__ import annotations

from pathlib import Path

from litscribe.models.paper import Paper
from litscribe.models.analysis import ParsedDoc
from litscribe.store.sqlite import SQLiteStore
from litscribe.store.vectors import VectorStore


class UnifiedStore:
    def __init__(self, db_path: Path, chroma_path: Path):
        self.sqlite = SQLiteStore(db_path)
        self.vectors = VectorStore(chroma_path)

    async def initialize(self):
        await self.sqlite.initialize()

    async def close(self):
        await self.sqlite.close()

    # -- Paper operations (delegate to SQLite) --

    async def save_papers(self, papers: list[Paper]):
        await self.sqlite.save_papers(papers)

    async def get_paper(self, paper_id: str) -> Paper | None:
        return await self.sqlite.get_paper(paper_id)

    # -- Parse operations --

    async def save_parsed(self, paper_id: str, doc: ParsedDoc):
        await self.sqlite.save_parsed(paper_id, doc)

    async def get_parsed(self, paper_id: str) -> ParsedDoc | None:
        return await self.sqlite.get_parsed(paper_id)

    # -- Episodic memory --

    async def save_episode(
        self, session_id: str, question: str, outcome_score: float, summary: str
    ):
        await self.sqlite.save_episode(session_id, question, outcome_score, summary)

    async def recall(self, query: str, limit: int = 5) -> list[dict]:
        return await self.sqlite.recall(query, limit)

    # -- Vector operations (delegate to ChromaDB) --

    def add_embeddings(
        self, texts: list[str], metadatas: list[dict], ids: list[str], collection: str
    ):
        self.vectors.add_texts(collection, texts, metadatas, ids)

    def semantic_search(self, query: str, collection: str, n: int = 10) -> list[dict]:
        return self.vectors.search(query, collection, n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_store/test_unified.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/store/unified.py tests/test_store/test_unified.py
git commit -m "feat(v2): UnifiedStore facade over SQLite + ChromaDB"
```

---

### Task 7: LLM Router

**Files:**
- Create: `litscribe/llm/router.py`
- Create: `litscribe/llm/tracker.py`
- Create: `tests/test_llm/test_router.py`
- Create: `tests/test_llm/test_tracker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm/test_tracker.py
def test_tracker_record_and_summary():
    from litscribe.llm.tracker import TokenTracker

    tracker = TokenTracker()
    tracker.record("discovery", "openai/qwen-turbo", {"prompt_tokens": 100, "completion_tokens": 50})
    tracker.record("synthesis", "openai/qwen-max", {"prompt_tokens": 500, "completion_tokens": 300})

    summary = tracker.summary()
    assert summary["total_prompt_tokens"] == 600
    assert summary["total_completion_tokens"] == 350
    assert "discovery" in summary["by_agent"]
    assert "synthesis" in summary["by_agent"]


def test_tracker_empty_summary():
    from litscribe.llm.tracker import TokenTracker

    tracker = TokenTracker()
    summary = tracker.summary()
    assert summary["total_prompt_tokens"] == 0
```

```python
# tests/test_llm/test_router.py
import pytest


def test_resolve_model_default():
    from litscribe.llm.router import LLMRouter
    from litscribe.config import Config

    config = Config()
    router = LLMRouter(config)
    assert router.resolve_model("query_expansion") == "openai/qwen-turbo"
    assert router.resolve_model("synthesis") == "openai/qwen-max"
    assert router.resolve_model("unknown_task") == "openai/qwen-plus"


def test_resolve_model_custom_config(tmp_path):
    from litscribe.llm.router import LLMRouter
    from litscribe.config import Config

    yaml_content = """
llm:
  default_model: "openai/qwen-turbo"
  task_models:
    synthesis: "openai/deepseek-r1"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)
    config = Config(config_path=config_file)
    router = LLMRouter(config)
    assert router.resolve_model("synthesis") == "openai/deepseek-r1"
    assert router.resolve_model("unknown") == "openai/qwen-turbo"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm/ -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementations**

```python
# litscribe/llm/tracker.py
from __future__ import annotations

from collections import defaultdict


class TokenTracker:
    def __init__(self):
        self._records: list[dict] = []

    def record(self, agent_name: str, model: str, usage: dict):
        self._records.append({
            "agent": agent_name,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        })

    def summary(self) -> dict:
        total_prompt = sum(r["prompt_tokens"] for r in self._records)
        total_completion = sum(r["completion_tokens"] for r in self._records)
        by_agent: dict[str, dict] = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
        for r in self._records:
            by_agent[r["agent"]]["prompt_tokens"] += r["prompt_tokens"]
            by_agent[r["agent"]]["completion_tokens"] += r["completion_tokens"]
            by_agent[r["agent"]]["calls"] += 1
        return {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_calls": len(self._records),
            "by_agent": dict(by_agent),
        }
```

```python
# litscribe/llm/router.py
from __future__ import annotations

import re
import logging
from typing import Any

import litellm

from litscribe.config import Config
from litscribe.llm.tracker import TokenTracker

logger = logging.getLogger(__name__)

REASONING_PATTERNS = re.compile(r"reasoner|deepseek-r1|o1-|o3-|o4-", re.IGNORECASE)


class LLMRouter:
    def __init__(self, config: Config):
        self.config = config
        self.tracker = TokenTracker()

    def resolve_model(self, task_type: str | None = None) -> str:
        if task_type and task_type in self.config.llm.task_models:
            return self.config.llm.task_models[task_type]
        return self.config.llm.default_model

    def _is_reasoning_model(self, model: str) -> bool:
        return bool(REASONING_PATTERNS.search(model))

    async def call(
        self,
        messages: list[dict],
        task_type: str | None = None,
        model_override: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        agent_name: str = "unknown",
    ) -> str:
        model = model_override or self.resolve_model(task_type)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if self.config.llm.api_key:
            kwargs["api_key"] = self.config.llm.api_key
        if self.config.llm.api_base:
            kwargs["api_base"] = self.config.llm.api_base

        if not self._is_reasoning_model(model):
            kwargs["temperature"] = temperature

        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content or ""

        # Strip <think> blocks from reasoning models
        if self._is_reasoning_model(model):
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        if response.usage:
            self.tracker.record(agent_name, model, {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            })

        return content

    async def call_json(
        self,
        messages: list[dict],
        task_type: str | None = None,
        agent_name: str = "unknown",
        max_retries: int = 2,
    ) -> dict:
        import json

        for attempt in range(max_retries + 1):
            temp = 0.7 if attempt == 0 else max(0.3, 0.7 - attempt * 0.2)
            raw = await self.call(
                messages, task_type=task_type, temperature=temp, agent_name=agent_name
            )
            try:
                # Try to extract JSON from response
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```\w*\n?", "", raw)
                    raw = re.sub(r"\n?```$", "", raw)
                return json.loads(raw)
            except json.JSONDecodeError:
                if attempt == max_retries:
                    raise
                logger.warning(f"JSON parse failed (attempt {attempt + 1}), retrying with lower temperature")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm/ -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/llm/ tests/test_llm/
git commit -m "feat(v2): LLM router with per-task model routing + token tracker"
```

---

### Task 8: Search service protocol + base adapter

**Files:**
- Create: `litscribe/services/base.py`
- Create: `tests/test_services/test_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_services/test_base.py
import pytest
import pytest_asyncio

from litscribe.models.paper import Paper


class MockSearchService:
    """Test that any class following the protocol works."""

    source_name = "mock"

    async def search(self, query: str, max_results: int = 10, **filters) -> list[Paper]:
        return [
            Paper(
                paper_id="mock:1",
                title=f"Result for: {query}",
                authors=["Author"],
                abstract="Abstract",
                year=2024,
                sources={"mock": "1"},
            )
        ]


def test_mock_service_implements_protocol():
    from litscribe.services.base import SearchService

    service = MockSearchService()
    assert isinstance(service, SearchService)


@pytest.mark.asyncio
async def test_mock_service_returns_papers():
    service = MockSearchService()
    results = await service.search("test query")
    assert len(results) == 1
    assert results[0].paper_id == "mock:1"


def test_dedup_papers():
    from litscribe.services.base import dedup_papers

    papers = [
        Paper(paper_id="a", title="Paper A", authors=["X"], abstract="abs", year=2024, sources={"arxiv": "1"}),
        Paper(paper_id="b", title="Paper B", authors=["Y"], abstract="abs", year=2024, sources={"s2": "2"}),
        Paper(paper_id="a", title="Paper A dup", authors=["X"], abstract="abs", year=2024, sources={"pubmed": "3"}),
    ]
    deduped = dedup_papers(papers)
    assert len(deduped) == 2
    # Merged sources for paper "a"
    paper_a = next(p for p in deduped if p.paper_id == "a")
    assert "arxiv" in paper_a.sources
    assert "pubmed" in paper_a.sources
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_services/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# litscribe/services/base.py
from __future__ import annotations

from typing import Protocol, runtime_checkable

from litscribe.models.paper import Paper


@runtime_checkable
class SearchService(Protocol):
    source_name: str

    async def search(self, query: str, max_results: int = 10, **filters) -> list[Paper]:
        ...


def dedup_papers(papers: list[Paper]) -> list[Paper]:
    """Deduplicate papers by paper_id, merging sources."""
    seen: dict[str, Paper] = {}
    for p in papers:
        if p.paper_id in seen:
            existing = seen[p.paper_id]
            merged_sources = {**existing.sources, **p.sources}
            merged_urls = list(set(existing.pdf_urls + p.pdf_urls))
            seen[p.paper_id] = existing.model_copy(
                update={
                    "sources": merged_sources,
                    "pdf_urls": merged_urls,
                    "relevance_score": max(existing.relevance_score, p.relevance_score),
                }
            )
        else:
            seen[p.paper_id] = p
    return list(seen.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_services/test_base.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/services/base.py tests/test_services/test_base.py
git commit -m "feat(v2): SearchService protocol + paper dedup utility"
```

---

## Phase 3: Self-Evolution Layer

### Task 9: Episodic memory

**Files:**
- Create: `litscribe/evolution/episodic.py`
- Create: `tests/test_evolution/test_episodic.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evolution/test_episodic.py
import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def episodic(tmp_data_dir):
    from litscribe.evolution.episodic import EpisodicMemory
    from litscribe.store.sqlite import SQLiteStore

    store = SQLiteStore(tmp_data_dir / "test.db")
    await store.initialize()
    mem = EpisodicMemory(store)
    yield mem
    await store.close()


@pytest.mark.asyncio
async def test_record_and_recall(episodic):
    await episodic.record(
        session_id="s1",
        question="LLM reasoning capabilities",
        outcome_score=0.85,
        key_events=["Searched arXiv", "Found 35 papers", "Synthesis score 0.85"],
    )
    results = await episodic.recall("LLM reasoning")
    assert len(results) >= 1
    assert results[0]["session_id"] == "s1"
    assert "arXiv" in results[0]["summary"]


@pytest.mark.asyncio
async def test_recall_returns_most_relevant(episodic):
    await episodic.record(
        session_id="s1", question="LLM reasoning",
        outcome_score=0.8, key_events=["searched arxiv for reasoning"],
    )
    await episodic.record(
        session_id="s2", question="protein folding",
        outcome_score=0.9, key_events=["searched pubmed for protein"],
    )
    results = await episodic.recall("protein biology")
    assert results[0]["session_id"] == "s2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolution/test_episodic.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# litscribe/evolution/episodic.py
from __future__ import annotations

from litscribe.store.sqlite import SQLiteStore


class EpisodicMemory:
    """Cross-session recall of task experiences via FTS5."""

    def __init__(self, store: SQLiteStore):
        self._store = store

    async def record(
        self,
        session_id: str,
        question: str,
        outcome_score: float,
        key_events: list[str],
    ):
        summary = "; ".join(key_events)
        await self._store.save_episode(session_id, question, outcome_score, summary)

    async def recall(self, query: str, limit: int = 5) -> list[dict]:
        return await self._store.recall(query, limit)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evolution/test_episodic.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/evolution/episodic.py tests/test_evolution/test_episodic.py
git commit -m "feat(v2): episodic memory — FTS5-based cross-session recall"
```

---

### Task 10: Semantic memory

**Files:**
- Create: `litscribe/evolution/semantic.py`
- Create: `tests/test_evolution/test_semantic.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evolution/test_semantic.py
import pytest


@pytest.fixture
def semantic(tmp_data_dir):
    from litscribe.evolution.semantic import SemanticMemory
    from litscribe.store.vectors import VectorStore

    vectors = VectorStore(tmp_data_dir / "vectors")
    return SemanticMemory(vectors)


def test_absorb_and_search(semantic):
    from litscribe.models.analysis import PaperAnalysis

    analyses = [
        PaperAnalysis(
            paper_id="p1",
            key_findings=["Transformers enable parallel processing", "Attention is all you need"],
            methodology="Experimental",
            relevance_score=0.9,
        ),
        PaperAnalysis(
            paper_id="p2",
            key_findings=["BERT achieves SOTA on GLUE", "Pre-training helps downstream tasks"],
            methodology="Benchmark evaluation",
            relevance_score=0.8,
        ),
    ]
    semantic.absorb(analyses)
    results = semantic.search("transformer architecture")
    assert len(results) >= 1


def test_update_user_profile(semantic):
    semantic.update_user_profile(
        user_id="default",
        domain="NLP/AI",
        preferences={"tier": "standard", "language": "en"},
    )
    profile = semantic.get_user_profile("default")
    assert profile is not None
    assert "NLP" in profile["document"]


def test_search_empty(semantic):
    results = semantic.search("nonexistent topic")
    assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolution/test_semantic.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# litscribe/evolution/semantic.py
from __future__ import annotations

from litscribe.models.analysis import PaperAnalysis
from litscribe.store.vectors import VectorStore

SEMANTIC_COLLECTION = "semantic_memory"
USER_COLLECTION = "user_profiles"


class SemanticMemory:
    """Domain knowledge accumulation + user modeling via vector search."""

    def __init__(self, vectors: VectorStore):
        self._vectors = vectors

    def absorb(self, analyses: list[PaperAnalysis]):
        texts = []
        metadatas = []
        ids = []
        for a in analyses:
            for i, finding in enumerate(a.key_findings):
                chunk_id = f"{a.paper_id}:finding:{i}"
                texts.append(finding)
                metadatas.append({"paper_id": a.paper_id, "type": "finding"})
                ids.append(chunk_id)
        if texts:
            self._vectors.add_texts(SEMANTIC_COLLECTION, texts, metadatas, ids)

    def search(self, query: str, n: int = 10) -> list[dict]:
        return self._vectors.search(query, SEMANTIC_COLLECTION, n)

    def update_user_profile(self, user_id: str, domain: str, preferences: dict):
        text = f"User domain: {domain}. Preferences: {preferences}"
        self._vectors.add_texts(
            USER_COLLECTION,
            texts=[text],
            metadatas=[{"user_id": user_id, "domain": domain, **preferences}],
            ids=[user_id],
        )

    def get_user_profile(self, user_id: str) -> dict | None:
        results = self._vectors.search(user_id, USER_COLLECTION, n=1)
        if results and results[0]["id"] == user_id:
            return results[0]
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evolution/test_semantic.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/evolution/semantic.py tests/test_evolution/test_semantic.py
git commit -m "feat(v2): semantic memory — ChromaDB knowledge + user modeling"
```

---

### Task 11: Procedural memory (skills)

**Files:**
- Create: `litscribe/evolution/procedural.py`
- Create: `tests/test_evolution/test_procedural.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evolution/test_procedural.py
import pytest
from pathlib import Path


@pytest.fixture
def procedural(tmp_skills_dir, tmp_data_dir):
    from litscribe.evolution.procedural import ProceduralMemory
    from litscribe.store.vectors import VectorStore

    vectors = VectorStore(tmp_data_dir / "vectors")
    return ProceduralMemory(tmp_skills_dir, vectors)


def test_save_and_load_skill(procedural):
    procedural.save_skill(
        name="NLP Search Strategy",
        domain="NLP/AI",
        trigger="When research question involves NLP",
        strategy="1. Search arXiv cs.CL\n2. Snowball from top cited",
        learned_adjustments=[],
    )
    skill = procedural.get_skill("nlp-search-strategy")
    assert skill is not None
    assert skill["name"] == "NLP Search Strategy"
    assert skill["domain"] == "NLP/AI"
    assert skill["version"] == 1


def test_list_skills(procedural):
    procedural.save_skill(name="Skill A", domain="NLP", trigger="t", strategy="s", learned_adjustments=[])
    procedural.save_skill(name="Skill B", domain="Bio", trigger="t", strategy="s", learned_adjustments=[])
    skills = procedural.list_skills()
    assert len(skills) == 2


def test_patch_skill_increments_version(procedural):
    procedural.save_skill(name="My Skill", domain="NLP", trigger="t", strategy="old strategy", learned_adjustments=[])
    procedural.patch_skill("my-skill", strategy="new strategy", adjustment="Added OpenAlex source")
    skill = procedural.get_skill("my-skill")
    assert skill["version"] == 2
    assert "new strategy" in skill["strategy"]
    assert "Added OpenAlex" in skill["raw_content"]


def test_find_relevant_skills(procedural):
    procedural.save_skill(name="NLP Search", domain="NLP", trigger="NLP queries", strategy="arxiv cs.CL", learned_adjustments=[])
    procedural.save_skill(name="Bio Search", domain="Biology", trigger="biomedical queries", strategy="pubmed mesh", learned_adjustments=[])
    results = procedural.find_relevant("NLP language models")
    assert len(results) >= 1
    assert results[0]["name"] == "NLP Search"


def test_get_nonexistent_skill(procedural):
    assert procedural.get_skill("nonexistent") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolution/test_procedural.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# litscribe/evolution/procedural.py
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import yaml

from litscribe.store.vectors import VectorStore

SKILL_COLLECTION = "skill_embeddings"


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class ProceduralMemory:
    """Filesystem-based skill documents with vector search for retrieval."""

    def __init__(self, skills_dir: Path, vectors: VectorStore):
        self._dir = skills_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._vectors = vectors

    def save_skill(
        self,
        name: str,
        domain: str,
        trigger: str,
        strategy: str,
        learned_adjustments: list[str],
    ) -> str:
        slug = _slugify(name)
        path = self._dir / f"{slug}.md"

        frontmatter = {
            "name": name,
            "domain": domain,
            "version": 1,
            "success_rate": 0.0,
            "last_used": datetime.now().strftime("%Y-%m-%d"),
        }
        adjustments_text = ""
        if learned_adjustments:
            adjustments_text = "\n".join(f"- {a}" for a in learned_adjustments)

        content = f"""---
{yaml.dump(frontmatter, default_flow_style=False).strip()}
---

## Trigger
{trigger}

## Strategy
{strategy}

## Learned Adjustments
{adjustments_text}
"""
        path.write_text(content)

        # Index in vector store for similarity search
        self._vectors.add_texts(
            SKILL_COLLECTION,
            texts=[f"{name}: {trigger}. {strategy}"],
            metadatas=[{"slug": slug, "domain": domain, "name": name}],
            ids=[slug],
        )
        return slug

    def get_skill(self, slug: str) -> dict | None:
        path = self._dir / f"{slug}.md"
        if not path.exists():
            return None
        return self._parse_skill(path)

    def list_skills(self) -> list[dict]:
        skills = []
        for path in sorted(self._dir.glob("*.md")):
            skill = self._parse_skill(path)
            if skill:
                skills.append(skill)
        return skills

    def patch_skill(self, slug: str, strategy: str | None = None, adjustment: str | None = None):
        path = self._dir / f"{slug}.md"
        if not path.exists():
            return
        raw = path.read_text()
        skill = self._parse_skill(path)
        if not skill:
            return

        # Increment version in frontmatter
        new_version = skill["version"] + 1
        raw = re.sub(r"version:\s*\d+", f"version: {new_version}", raw)
        raw = re.sub(
            r"last_used:\s*\S+",
            f"last_used: {datetime.now().strftime('%Y-%m-%d')}",
            raw,
        )

        if strategy:
            raw = re.sub(
                r"(## Strategy\n).*?(\n## )",
                f"\\g<1>{strategy}\n\\g<2>",
                raw,
                flags=re.DOTALL,
            )

        if adjustment:
            raw = raw.rstrip() + f"\n- v{new_version}: {adjustment}\n"

        path.write_text(raw)

        # Re-index
        self._vectors.delete(SKILL_COLLECTION, [slug])
        self._vectors.add_texts(
            SKILL_COLLECTION,
            texts=[f"{skill['name']}: {strategy or skill['trigger']}"],
            metadatas=[{"slug": slug, "domain": skill["domain"], "name": skill["name"]}],
            ids=[slug],
        )

    def find_relevant(self, query: str, n: int = 5) -> list[dict]:
        results = self._vectors.search(query, SKILL_COLLECTION, n)
        skills = []
        for r in results:
            slug = r["metadata"]["slug"]
            skill = self.get_skill(slug)
            if skill:
                skills.append(skill)
        return skills

    def _parse_skill(self, path: Path) -> dict | None:
        raw = path.read_text()
        match = re.match(r"^---\n(.+?)\n---\n(.+)", raw, re.DOTALL)
        if not match:
            return None
        frontmatter = yaml.safe_load(match.group(1))
        body = match.group(2)

        trigger_match = re.search(r"## Trigger\n(.+?)(?=\n## )", body, re.DOTALL)
        strategy_match = re.search(r"## Strategy\n(.+?)(?=\n## )", body, re.DOTALL)

        return {
            "slug": path.stem,
            "name": frontmatter.get("name", ""),
            "domain": frontmatter.get("domain", ""),
            "version": frontmatter.get("version", 1),
            "success_rate": frontmatter.get("success_rate", 0.0),
            "last_used": frontmatter.get("last_used", ""),
            "trigger": trigger_match.group(1).strip() if trigger_match else "",
            "strategy": strategy_match.group(1).strip() if strategy_match else "",
            "raw_content": raw,
            "file_path": str(path),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evolution/test_procedural.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/evolution/procedural.py tests/test_evolution/test_procedural.py
git commit -m "feat(v2): procedural memory — filesystem skills with vector search"
```

---

### Task 12: Skill evolver

**Files:**
- Create: `litscribe/evolution/skill_evolver.py`
- Create: `tests/test_evolution/test_skill_evolver.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evolution/test_skill_evolver.py
import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def evolver(tmp_data_dir, tmp_skills_dir):
    from litscribe.evolution.skill_evolver import SkillEvolver
    from litscribe.evolution.episodic import EpisodicMemory
    from litscribe.evolution.procedural import ProceduralMemory
    from litscribe.store.sqlite import SQLiteStore
    from litscribe.store.vectors import VectorStore

    sqlite = SQLiteStore(tmp_data_dir / "test.db")
    await sqlite.initialize()
    vectors = VectorStore(tmp_data_dir / "vectors")
    episodic = EpisodicMemory(sqlite)
    procedural = ProceduralMemory(tmp_skills_dir, vectors)
    evolver = SkillEvolver(episodic=episodic, procedural=procedural)
    yield evolver
    await sqlite.close()


@pytest.mark.asyncio
async def test_should_extract_skill_high_score_complex(evolver):
    assert evolver.should_extract_skill(score=0.85, complexity=7) is True


@pytest.mark.asyncio
async def test_should_not_extract_skill_low_score(evolver):
    assert evolver.should_extract_skill(score=0.4, complexity=7) is False


@pytest.mark.asyncio
async def test_should_not_extract_skill_low_complexity(evolver):
    assert evolver.should_extract_skill(score=0.9, complexity=2) is False


@pytest.mark.asyncio
async def test_record_failure(evolver):
    await evolver.record_failure(
        session_id="s1",
        question="test question",
        score=0.3,
        feedback="Coverage too low",
    )
    results = await evolver.episodic.recall("test question")
    assert len(results) == 1
    assert "FAILURE" in results[0]["summary"]


@pytest.mark.asyncio
async def test_inject_skills_adds_to_instructions(evolver):
    evolver.procedural.save_skill(
        name="NLP Search",
        domain="NLP",
        trigger="NLP queries",
        strategy="Search arXiv cs.CL first",
        learned_adjustments=[],
    )
    instructions = evolver.inject_skills(domain="NLP", task_type="discovery")
    assert "NLP Search" in instructions
    assert "arXiv" in instructions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolution/test_skill_evolver.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# litscribe/evolution/skill_evolver.py
from __future__ import annotations

from litscribe.evolution.episodic import EpisodicMemory
from litscribe.evolution.procedural import ProceduralMemory

SCORE_THRESHOLD = 0.7
COMPLEXITY_THRESHOLD = 5
FAILURE_THRESHOLD = 0.5


class SkillEvolver:
    """Evaluates task outcomes and manages skill lifecycle."""

    def __init__(self, episodic: EpisodicMemory, procedural: ProceduralMemory):
        self.episodic = episodic
        self.procedural = procedural

    def should_extract_skill(self, score: float, complexity: int) -> bool:
        return score >= SCORE_THRESHOLD and complexity >= COMPLEXITY_THRESHOLD

    async def post_task_evaluate(
        self,
        session_id: str,
        question: str,
        score: float,
        complexity: int,
        domain: str,
        trace_summary: str,
        used_skills: list[str] | None = None,
    ):
        if self.should_extract_skill(score, complexity):
            existing = self.procedural.find_relevant(question, n=1)
            if existing and existing[0].get("domain") == domain:
                self.procedural.patch_skill(
                    existing[0]["slug"],
                    adjustment=f"Refined from session {session_id} (score={score})",
                )
            else:
                self.procedural.save_skill(
                    name=f"{domain} strategy from {session_id[:8]}",
                    domain=domain,
                    trigger=f"Research questions about {domain}",
                    strategy=trace_summary,
                    learned_adjustments=[],
                )

        if score < FAILURE_THRESHOLD:
            await self.record_failure(session_id, question, score, trace_summary)
            if used_skills:
                for slug in used_skills:
                    skill = self.procedural.get_skill(slug)
                    if skill:
                        self.procedural.patch_skill(
                            slug, adjustment=f"NEEDS REVIEW: failed in session {session_id}"
                        )

    async def record_failure(
        self, session_id: str, question: str, score: float, feedback: str
    ):
        await self.episodic.record(
            session_id=session_id,
            question=question,
            outcome_score=score,
            key_events=[f"FAILURE (score={score}): {feedback}"],
        )

    def inject_skills(self, domain: str, task_type: str) -> str:
        relevant = self.procedural.find_relevant(f"{domain} {task_type}", n=3)
        if not relevant:
            return ""
        parts = ["\n--- Relevant Skills ---"]
        for skill in relevant:
            parts.append(f"\n### {skill['name']} (v{skill['version']}, success={skill['success_rate']})")
            parts.append(f"Trigger: {skill['trigger']}")
            parts.append(f"Strategy:\n{skill['strategy']}")
        parts.append("\n--- End Skills ---\n")
        return "\n".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evolution/test_skill_evolver.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/evolution/skill_evolver.py tests/test_evolution/test_skill_evolver.py
git commit -m "feat(v2): skill evolver — post-task evaluation + skill lifecycle"
```

---

### Task 13: Memory manager facade

**Files:**
- Create: `litscribe/evolution/memory_manager.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evolution/test_memory_manager.py
import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def memory(tmp_data_dir, tmp_skills_dir):
    from litscribe.evolution.memory_manager import MemoryManager

    mgr = MemoryManager(
        db_path=tmp_data_dir / "test.db",
        chroma_path=tmp_data_dir / "vectors",
        skills_dir=tmp_skills_dir,
    )
    await mgr.initialize()
    yield mgr
    await mgr.close()


@pytest.mark.asyncio
async def test_memory_manager_has_all_layers(memory):
    assert memory.episodic is not None
    assert memory.semantic is not None
    assert memory.procedural is not None
    assert memory.evolver is not None


@pytest.mark.asyncio
async def test_memory_manager_episode_roundtrip(memory):
    await memory.episodic.record(
        session_id="s1", question="test", outcome_score=0.8, key_events=["event1"]
    )
    results = await memory.episodic.recall("test")
    assert len(results) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolution/test_memory_manager.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# litscribe/evolution/memory_manager.py
from __future__ import annotations

from pathlib import Path

from litscribe.evolution.episodic import EpisodicMemory
from litscribe.evolution.semantic import SemanticMemory
from litscribe.evolution.procedural import ProceduralMemory
from litscribe.evolution.skill_evolver import SkillEvolver
from litscribe.store.sqlite import SQLiteStore
from litscribe.store.vectors import VectorStore


class MemoryManager:
    """Unified interface to all three memory tiers + skill evolver."""

    def __init__(self, db_path: Path, chroma_path: Path, skills_dir: Path):
        self._sqlite = SQLiteStore(db_path)
        self._vectors = VectorStore(chroma_path)
        self.episodic = EpisodicMemory(self._sqlite)
        self.semantic = SemanticMemory(self._vectors)
        self.procedural = ProceduralMemory(skills_dir, self._vectors)
        self.evolver = SkillEvolver(self.episodic, self.procedural)

    async def initialize(self):
        await self._sqlite.initialize()

    async def close(self):
        await self._sqlite.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evolution/test_memory_manager.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/evolution/memory_manager.py tests/test_evolution/test_memory_manager.py
git commit -m "feat(v2): MemoryManager — unified three-tier memory facade"
```

---

## Phase 4: Agents (Agno Pipeline)

### Task 14: Planner agent

**Files:**
- Create: `litscribe/agents/planner.py`
- Create: `tests/test_agents/test_planner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agents/test_planner.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_planner_creates_research_plan():
    from litscribe.agents.planner import create_plan
    from litscribe.models.plan import ReviewTier

    mock_llm = AsyncMock(return_value='{"sub_topics": [{"name": "chain of thought", "keywords": ["CoT", "reasoning"], "estimated_papers": 15}], "domain": "NLP/AI"}')

    plan = await create_plan(
        question="LLM reasoning capabilities",
        tier=ReviewTier.STANDARD,
        max_papers=40,
        language="en",
        llm_call=mock_llm,
    )
    assert plan.question == "LLM reasoning capabilities"
    assert len(plan.sub_topics) >= 1
    assert plan.domain == "NLP/AI"
    assert plan.tier == ReviewTier.STANDARD


@pytest.mark.asyncio
async def test_planner_handles_llm_returning_string_subtopics():
    from litscribe.agents.planner import create_plan
    from litscribe.models.plan import ReviewTier

    # LLM might return sub_topics as a string instead of list
    mock_llm = AsyncMock(return_value='{"sub_topics": "chain of thought, prompt engineering", "domain": "NLP"}')

    plan = await create_plan(
        question="LLM reasoning",
        tier=ReviewTier.QUICK,
        max_papers=20,
        language="en",
        llm_call=mock_llm,
    )
    assert len(plan.sub_topics) >= 1
    assert all(st.name for st in plan.sub_topics)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_planner.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# litscribe/agents/planner.py
from __future__ import annotations

import json
import logging
from typing import Callable, Awaitable

from litscribe.models.plan import ResearchPlan, SubTopic, ReviewTier

logger = logging.getLogger(__name__)

PLANNING_PROMPT = """You are a research planning agent. Given a research question, decompose it into sub-topics for a literature review.

Research Question: {question}
Review Tier: {tier} (max {max_papers} papers)
Language: {language}

Return a JSON object with:
- "sub_topics": list of objects, each with "name" (str), "keywords" (list of str), "estimated_papers" (int)
- "domain": the primary research domain (e.g., "NLP/AI", "Biomedical", "Social Science")

Return ONLY valid JSON, no markdown."""

TIER_SUBTOPIC_LIMITS = {
    ReviewTier.QUICK: 3,
    ReviewTier.STANDARD: 5,
    ReviewTier.COMPREHENSIVE: 8,
}


def _calculate_target_words(max_papers: int, language: str) -> int:
    base = 1000 + max_papers * 130
    if language in ("zh", "ja", "ko"):
        base = int(base * 1.5)
    return base


async def create_plan(
    question: str,
    tier: ReviewTier,
    max_papers: int,
    language: str,
    llm_call: Callable[..., Awaitable[str]],
    memory_context: str = "",
) -> ResearchPlan:
    prompt = PLANNING_PROMPT.format(
        question=question, tier=tier.value, max_papers=max_papers, language=language
    )
    if memory_context:
        prompt += f"\n\nRelevant past experience:\n{memory_context}"

    raw = await llm_call(prompt)

    # Parse JSON from response
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    data = json.loads(raw)

    # Handle sub_topics as string (LLM quirk)
    sub_topics_raw = data.get("sub_topics", [])
    if isinstance(sub_topics_raw, str):
        sub_topics_raw = [
            {"name": t.strip(), "keywords": [], "estimated_papers": max_papers // 3}
            for t in sub_topics_raw.split(",")
            if t.strip()
        ]

    limit = TIER_SUBTOPIC_LIMITS.get(tier, 5)
    sub_topics = [
        SubTopic(
            name=st.get("name", st) if isinstance(st, dict) else str(st),
            keywords=st.get("keywords", []) if isinstance(st, dict) else [],
            estimated_papers=st.get("estimated_papers", 10) if isinstance(st, dict) else 10,
        )
        for st in sub_topics_raw[:limit]
    ]

    domain = data.get("domain", "General")

    return ResearchPlan(
        question=question,
        sub_topics=sub_topics,
        domain=domain,
        tier=tier,
        max_papers=max_papers,
        language=language,
        target_words=_calculate_target_words(max_papers, language),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents/test_planner.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/agents/planner.py tests/test_agents/test_planner.py
git commit -m "feat(v2): planner agent with tier-based sub-topic decomposition"
```

---

### Task 15: Pipeline skeleton (Agno Workflow)

**Files:**
- Create: `litscribe/agents/pipeline.py`
- Create: `tests/test_agents/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agents/test_pipeline.py
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_pipeline_runs_all_steps():
    from litscribe.agents.pipeline import LitScribePipeline

    call_log = []

    async def mock_plan(question, **kwargs):
        call_log.append("plan")
        from litscribe.models.plan import ResearchPlan, SubTopic, ReviewTier
        return ResearchPlan(
            question=question, sub_topics=[SubTopic(name="t1", keywords=["k"])],
            domain="NLP", tier=ReviewTier.QUICK, max_papers=10, language="en",
        )

    async def mock_discover(plan, **kwargs):
        call_log.append("discover")
        from litscribe.models.paper import Paper
        return [Paper(paper_id="p1", title="P1", authors=["A"], abstract="abs", year=2024, sources={"arxiv": "1"})]

    async def mock_read(papers, **kwargs):
        call_log.append("read")
        from litscribe.models.analysis import PaperAnalysis
        return [PaperAnalysis(paper_id="p1", key_findings=["f1"], relevance_score=0.9)]

    async def mock_synthesize(analyses, **kwargs):
        call_log.append("synthesize")
        from litscribe.models.review import ReviewOutput
        return ReviewOutput(text="Review text", word_count=100)

    async def mock_review(output, **kwargs):
        call_log.append("review")
        from litscribe.models.assessment import ReviewAssessment
        return ReviewAssessment(passed=True, score=0.85, feedback="Good")

    pipeline = LitScribePipeline(
        plan_fn=mock_plan,
        discover_fn=mock_discover,
        read_fn=mock_read,
        synthesize_fn=mock_synthesize,
        review_fn=mock_review,
    )

    result = await pipeline.run("LLM reasoning", max_papers=10)
    assert result.text == "Review text"
    assert call_log == ["plan", "discover", "read", "synthesize", "review"]


@pytest.mark.asyncio
async def test_pipeline_loops_on_failed_review():
    from litscribe.agents.pipeline import LitScribePipeline
    from litscribe.models.plan import ResearchPlan, SubTopic, ReviewTier
    from litscribe.models.paper import Paper
    from litscribe.models.analysis import PaperAnalysis
    from litscribe.models.review import ReviewOutput
    from litscribe.models.assessment import ReviewAssessment

    review_call_count = 0

    async def mock_plan(question, **kwargs):
        return ResearchPlan(
            question=question, sub_topics=[SubTopic(name="t1", keywords=["k"])],
            domain="NLP", tier=ReviewTier.QUICK, max_papers=10, language="en",
        )

    async def mock_discover(plan, **kwargs):
        return [Paper(paper_id="p1", title="P1", authors=["A"], abstract="abs", year=2024, sources={})]

    async def mock_read(papers, **kwargs):
        return [PaperAnalysis(paper_id="p1", key_findings=["f1"], relevance_score=0.9)]

    async def mock_synthesize(analyses, **kwargs):
        return ReviewOutput(text="Review", word_count=100)

    async def mock_review(output, **kwargs):
        nonlocal review_call_count
        review_call_count += 1
        if review_call_count < 2:
            return ReviewAssessment(passed=False, score=0.4, feedback="Needs work", refined_queries=["more papers"])
        return ReviewAssessment(passed=True, score=0.8, feedback="OK")

    pipeline = LitScribePipeline(
        plan_fn=mock_plan, discover_fn=mock_discover,
        read_fn=mock_read, synthesize_fn=mock_synthesize,
        review_fn=mock_review, max_iterations=3,
    )

    result = await pipeline.run("test", max_papers=10)
    assert result.text == "Review"
    assert review_call_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# litscribe/agents/pipeline.py
from __future__ import annotations

import logging
from typing import Callable, Awaitable, Any

from litscribe.models.plan import ResearchPlan, ReviewTier
from litscribe.models.paper import Paper
from litscribe.models.analysis import PaperAnalysis
from litscribe.models.review import ReviewOutput
from litscribe.models.assessment import ReviewAssessment

logger = logging.getLogger(__name__)

PlanFn = Callable[..., Awaitable[ResearchPlan]]
DiscoverFn = Callable[..., Awaitable[list[Paper]]]
ReadFn = Callable[..., Awaitable[list[PaperAnalysis]]]
SynthesizeFn = Callable[..., Awaitable[ReviewOutput]]
ReviewFn = Callable[..., Awaitable[ReviewAssessment]]
ProgressCallback = Callable[[str, str], None] | None


class LitScribePipeline:
    """Deterministic pipeline: Plan → Discover → Read → Synthesize → Self-Review loop."""

    def __init__(
        self,
        plan_fn: PlanFn,
        discover_fn: DiscoverFn,
        read_fn: ReadFn,
        synthesize_fn: SynthesizeFn,
        review_fn: ReviewFn,
        graphrag_fn: Callable[..., Awaitable[Any]] | None = None,
        max_iterations: int = 3,
        on_progress: ProgressCallback = None,
    ):
        self.plan_fn = plan_fn
        self.discover_fn = discover_fn
        self.read_fn = read_fn
        self.synthesize_fn = synthesize_fn
        self.review_fn = review_fn
        self.graphrag_fn = graphrag_fn
        self.max_iterations = max_iterations
        self.on_progress = on_progress

    def _emit(self, step: str, detail: str = ""):
        if self.on_progress:
            self.on_progress(step, detail)
        logger.info(f"[pipeline] {step}: {detail}")

    async def run(
        self,
        question: str,
        max_papers: int = 40,
        tier: ReviewTier = ReviewTier.STANDARD,
        language: str = "en",
        graphrag_enabled: bool = False,
    ) -> ReviewOutput:
        self._emit("planning", question)
        plan = await self.plan_fn(question, tier=tier, max_papers=max_papers, language=language)

        knowledge_graph = None

        for iteration in range(self.max_iterations):
            self._emit("discovery", f"iteration {iteration + 1}")
            papers = await self.discover_fn(plan, iteration=iteration)

            self._emit("reading", f"{len(papers)} papers")
            analyses = await self.read_fn(papers, plan=plan)

            if graphrag_enabled and self.graphrag_fn:
                self._emit("graphrag", f"{len(analyses)} analyses")
                knowledge_graph = await self.graphrag_fn(analyses)

            self._emit("synthesis", f"{len(analyses)} analyses")
            review = await self.synthesize_fn(
                analyses, plan=plan, knowledge_graph=knowledge_graph
            )

            self._emit("self_review", f"iteration {iteration + 1}")
            assessment = await self.review_fn(review, plan=plan)

            if assessment.passed:
                self._emit("complete", f"score={assessment.score}")
                return review

            logger.info(
                f"Loop back: score={assessment.score}, feedback={assessment.feedback}"
            )
            # Refine plan for next iteration
            if assessment.refined_queries:
                plan.sub_topics[0].keywords.extend(assessment.refined_queries)

        self._emit("complete", "max iterations reached")
        return review
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents/test_pipeline.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/agents/pipeline.py tests/test_agents/test_pipeline.py
git commit -m "feat(v2): LitScribePipeline — deterministic workflow with loop-back"
```

---

## Phase 5: API + CLI

### Task 16: FastAPI core + reviews endpoint

**Files:**
- Create: `litscribe/api/main.py`
- Create: `litscribe/api/deps.py`
- Create: `litscribe/api/routes/reviews.py`
- Create: `tests/test_api/test_reviews.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api/test_reviews.py
import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    from litscribe.api.main import create_app
    return create_app()


@pytest.mark.asyncio
async def test_create_review(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/reviews", json={
            "question": "LLM reasoning",
            "max_papers": 10,
            "tier": "quick",
        })
        assert resp.status_code == 202
        data = resp.json()
        assert "review_id" in data
        assert data["status"] == "started"


@pytest.mark.asyncio
async def test_get_review_not_found(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/reviews/nonexistent")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_reviews(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/reviews")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api/test_reviews.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementations**

```python
# litscribe/api/deps.py
from __future__ import annotations

from functools import lru_cache

from litscribe.config import Config


@lru_cache
def get_config() -> Config:
    return Config()
```

```python
# litscribe/api/routes/reviews.py
from __future__ import annotations

import uuid
import asyncio
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/reviews", tags=["reviews"])

# In-memory review tracking (will be backed by store in production)
_reviews: dict[str, dict] = {}


class ReviewRequest(BaseModel):
    question: str
    max_papers: int = 40
    tier: str = "standard"
    language: str = "en"
    model: str | None = None
    graphrag_enabled: bool = False


class ReviewResponse(BaseModel):
    review_id: str
    status: str
    question: str
    created_at: str


@router.post("", status_code=202)
async def create_review(req: ReviewRequest) -> ReviewResponse:
    review_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()
    _reviews[review_id] = {
        "review_id": review_id,
        "status": "started",
        "question": req.question,
        "config": req.model_dump(),
        "created_at": now,
        "result": None,
    }
    # TODO: dispatch pipeline as background task in production
    return ReviewResponse(
        review_id=review_id, status="started",
        question=req.question, created_at=now,
    )


@router.get("")
async def list_reviews() -> list[dict]:
    return [
        {"review_id": r["review_id"], "status": r["status"], "question": r["question"]}
        for r in _reviews.values()
    ]


@router.get("/{review_id}")
async def get_review(review_id: str):
    if review_id not in _reviews:
        raise HTTPException(status_code=404, detail="Review not found")
    return _reviews[review_id]


@router.delete("/{review_id}")
async def delete_review(review_id: str):
    if review_id not in _reviews:
        raise HTTPException(status_code=404, detail="Review not found")
    del _reviews[review_id]
    return {"status": "deleted"}
```

```python
# litscribe/api/main.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from litscribe.api.routes.reviews import router as reviews_router


def create_app() -> FastAPI:
    app = FastAPI(title="LitScribe", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(reviews_router)

    return app


def run():
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api/test_reviews.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/api/ tests/test_api/
git commit -m "feat(v2): FastAPI core with reviews CRUD endpoint"
```

---

### Task 17: Memory API endpoints

**Files:**
- Create: `litscribe/api/routes/memory.py`
- Create: `tests/test_api/test_memory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api/test_memory.py
import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    from litscribe.api.main import create_app
    return create_app()


@pytest.mark.asyncio
async def test_list_skills(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/memory/skills")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_search_episodes(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/memory/episodes", params={"q": "test"})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api/test_memory.py -v`
Expected: FAIL — router not registered

- [ ] **Step 3: Write implementation**

```python
# litscribe/api/routes/memory.py
from __future__ import annotations

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("/skills")
async def list_skills() -> list:
    # Will be wired to MemoryManager in production
    return []


@router.get("/episodes")
async def search_episodes(q: str = Query("")) -> list:
    # Will be wired to EpisodicMemory in production
    return []


@router.put("/skills/{skill_id}")
async def update_skill(skill_id: str, body: dict) -> dict:
    return {"status": "updated", "skill_id": skill_id}
```

- [ ] **Step 4: Register router in main.py**

Add to `litscribe/api/main.py` after `reviews_router` import:

```python
from litscribe.api.routes.memory import router as memory_router
```

And in `create_app()`:

```python
    app.include_router(memory_router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_api/test_memory.py -v`
Expected: All 2 tests PASS

- [ ] **Step 6: Commit**

```bash
git add litscribe/api/routes/memory.py litscribe/api/main.py tests/test_api/test_memory.py
git commit -m "feat(v2): memory API — skills listing + episodic search endpoints"
```

---

### Task 18: Typer CLI thin client

**Files:**
- Create: `litscribe/cli/main.py`
- Create: `tests/test_cli/test_main.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli/test_main.py
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_help():
    from litscribe.cli.main import app

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LitScribe" in result.output


def test_cli_review_help():
    from litscribe.cli.main import app

    result = runner.invoke(app, ["review", "--help"])
    assert result.exit_code == 0
    assert "question" in result.output.lower() or "QUESTION" in result.output


def test_cli_skills_help():
    from litscribe.cli.main import app

    result = runner.invoke(app, ["skills", "--help"])
    assert result.exit_code == 0


def test_cli_config_help():
    from litscribe.cli.main import app

    result = runner.invoke(app, ["config", "--help"])
    assert result.exit_code == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli/test_main.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# litscribe/cli/main.py
from __future__ import annotations

import sys
from typing import Optional

import typer
import httpx

app = typer.Typer(name="litscribe", help="LitScribe — Self-evolving literature review engine")

API_BASE = "http://localhost:8000"


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE, timeout=300)


@app.command()
def review(
    question: str = typer.Argument(..., help="Research question"),
    tier: str = typer.Option("standard", help="Review tier: quick/standard/comprehensive"),
    max_papers: int = typer.Option(40, help="Maximum papers to analyze"),
    model: Optional[str] = typer.Option(None, help="LLM model override"),
    language: str = typer.Option("en", help="Output language"),
    graphrag: bool = typer.Option(False, help="Enable GraphRAG"),
):
    """Start a literature review."""
    with _client() as client:
        resp = client.post("/api/reviews", json={
            "question": question,
            "max_papers": max_papers,
            "tier": tier,
            "language": language,
            "model": model,
            "graphrag_enabled": graphrag,
        })
        if resp.status_code == 202:
            data = resp.json()
            typer.echo(f"Review started: {data['review_id']}")
        else:
            typer.echo(f"Error: {resp.text}", err=True)
            raise typer.Exit(1)


@app.command()
def sessions():
    """List historical review sessions."""
    with _client() as client:
        resp = client.get("/api/sessions")
        if resp.status_code == 200:
            for s in resp.json():
                typer.echo(f"  {s.get('session_id', 'N/A')}  {s.get('question', '')}")
        else:
            typer.echo(f"Error: {resp.text}", err=True)


@app.command()
def skills(
    action: str = typer.Argument("list", help="Action: list / show / delete"),
    name: Optional[str] = typer.Argument(None, help="Skill name (for show/delete)"),
):
    """Manage procedural skills."""
    with _client() as client:
        if action == "list":
            resp = client.get("/api/memory/skills")
            if resp.status_code == 200:
                for skill in resp.json():
                    typer.echo(f"  {skill.get('name', 'N/A')} (v{skill.get('version', '?')})")
            else:
                typer.echo("No skills found.")
        else:
            typer.echo(f"Action '{action}' not yet implemented.")


@app.command()
def config(
    key: Optional[str] = typer.Argument(None, help="Config key to show/set"),
    value: Optional[str] = typer.Argument(None, help="Value to set"),
):
    """View or modify configuration."""
    with _client() as client:
        if key is None:
            resp = client.get("/api/config")
            if resp.status_code == 200:
                import json
                typer.echo(json.dumps(resp.json(), indent=2))
            else:
                typer.echo("Could not fetch config.")
        elif value is not None:
            resp = client.put("/api/config", json={"key": key, "value": value})
            typer.echo(f"Set {key} = {value}")
        else:
            typer.echo(f"Config key: {key} (show not yet implemented)")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli/test_main.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/cli/main.py tests/test_cli/test_main.py
git commit -m "feat(v2): Typer CLI thin client (~150 lines)"
```

---

## Phase 6: Service Adapters + Exporters

### Task 19: Adapt existing services

**Files:**
- Create: `litscribe/services/arxiv.py` (adapted from `src/services/arxiv.py`)
- Create: `litscribe/services/pubmed.py` (adapted from `src/services/pubmed.py`)
- Create: `litscribe/services/semantic_scholar.py` (adapted)
- Create: `litscribe/services/openalex.py` (adapted)
- Create: `litscribe/services/europe_pmc.py` (adapted)
- Create: `litscribe/services/zotero.py` (adapted)
- Create: `litscribe/services/pdf.py` (adapted)

This task adapts existing service code to conform to the `SearchService` protocol and use the new `Paper` model. The internal logic (rate limiting, API parsing) is preserved.

- [ ] **Step 1: For each service, adapt the search function to return `list[Paper]`**

The key change per service is wrapping the existing result dict into a `Paper` model. Example pattern for each:

```python
# litscribe/services/arxiv.py (sketch — adapt from src/services/arxiv.py)
from litscribe.models.paper import Paper
from litscribe.services.base import SearchService

class ArxivService:
    source_name = "arxiv"

    async def search(self, query: str, max_results: int = 10, **filters) -> list[Paper]:
        # ... existing arxiv search logic ...
        # Convert results to Paper model:
        return [
            Paper(
                paper_id=f"arxiv:{result['arxiv_id']}",
                title=result["title"],
                authors=result["authors"],
                abstract=result["abstract"],
                year=result["year"],
                sources={"arxiv": result["arxiv_id"]},
                pdf_urls=[result.get("pdf_url", "")],
            )
            for result in raw_results
        ]
```

- [ ] **Step 2: Copy and adapt each service file from `src/services/` into `litscribe/services/`**

Preserve: rate limiting, API key handling, error handling, pagination logic.
Change: return type to `list[Paper]`, add `source_name` attribute.

- [ ] **Step 3: Copy exporters as-is**

```bash
cp src/exporters/bibtex_exporter.py litscribe/exporters/bibtex.py
cp src/exporters/pandoc_exporter.py litscribe/exporters/pandoc.py
cp src/exporters/citation_formatter.py litscribe/exporters/citation_formatter.py
```

Minimal changes: update any imports from `src.` to `litscribe.`.

- [ ] **Step 4: Run all tests to verify nothing is broken**

Run: `pytest tests/ -v`
Expected: All previous tests still PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/services/ litscribe/exporters/
git commit -m "feat(v2): adapt service adapters + exporters to new package structure"
```

---

## Phase 7: GraphRAG Plugin

### Task 20: Adapt GraphRAG as optional plugin

**Files:**
- Create: `litscribe/plugins/graphrag/plugin.py`
- Create: `litscribe/plugins/graphrag/entity_extractor.py` (adapted)
- Create: `litscribe/plugins/graphrag/linker.py` (adapted)
- Create: `litscribe/plugins/graphrag/graph_builder.py` (adapted)
- Create: `litscribe/plugins/graphrag/community_detector.py` (adapted)
- Create: `litscribe/plugins/graphrag/summarizer.py` (adapted)
- Create: `tests/test_plugins/test_graphrag.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plugins/test_graphrag.py
import pytest
from unittest.mock import AsyncMock

from litscribe.models.analysis import PaperAnalysis


@pytest.mark.asyncio
async def test_graphrag_plugin_process():
    from litscribe.plugins.graphrag.plugin import GraphRAGPlugin

    analyses = [
        PaperAnalysis(
            paper_id="p1",
            key_findings=["Transformers use attention mechanisms"],
            methodology="Experimental",
            relevance_score=0.9,
        ),
    ]

    mock_llm = AsyncMock(return_value='[{"name": "Transformer", "entity_type": "method", "description": "Neural architecture", "aliases": []}]')

    plugin = GraphRAGPlugin(llm_call=mock_llm)
    result = await plugin.process(analyses)
    assert result is not None
    assert "entities" in result or hasattr(result, "entities")
```

- [ ] **Step 2: Adapt GraphRAG modules from `src/graphrag/` into `litscribe/plugins/graphrag/`**

Preserve: entity ID generation (md5), entity linking algorithm, NetworkX graph construction, community detection.
Change: LLM calls go through the injected `llm_call` callable, imports updated to `litscribe.*`.

- [ ] **Step 3: Write plugin entry point**

```python
# litscribe/plugins/graphrag/plugin.py
from __future__ import annotations

from typing import Any, Callable, Awaitable

from litscribe.models.analysis import PaperAnalysis


class GraphRAGPlugin:
    """Optional GraphRAG plugin — loaded when config.graphrag_enabled is True."""

    def __init__(self, llm_call: Callable[..., Awaitable[str]]):
        self.llm_call = llm_call

    async def process(self, analyses: list[PaperAnalysis]) -> dict[str, Any]:
        from litscribe.plugins.graphrag.entity_extractor import extract_entities
        from litscribe.plugins.graphrag.linker import link_entities
        from litscribe.plugins.graphrag.graph_builder import build_graph
        from litscribe.plugins.graphrag.community_detector import detect_communities

        entities, mentions = await extract_entities(analyses, self.llm_call)
        linked = link_entities(entities)
        graph = build_graph(linked, mentions)
        communities = detect_communities(graph)

        return {
            "entities": {e["entity_id"]: e for e in linked},
            "mentions": mentions,
            "graph": graph,
            "communities": communities,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_plugins/test_graphrag.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add litscribe/plugins/graphrag/ tests/test_plugins/
git commit -m "feat(v2): GraphRAG as optional plugin with injected LLM calls"
```

---

## Phase 8: Integration + Wiring

### Task 21: Wire everything together

**Files:**
- Modify: `litscribe/api/main.py` (add lifespan, dependency injection)
- Modify: `litscribe/api/deps.py` (add store, memory, pipeline dependencies)
- Modify: `litscribe/api/routes/reviews.py` (wire to pipeline)

- [ ] **Step 1: Update deps.py with application-scoped dependencies**

```python
# litscribe/api/deps.py
from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache

from litscribe.config import Config
from litscribe.store.unified import UnifiedStore
from litscribe.evolution.memory_manager import MemoryManager
from litscribe.llm.router import LLMRouter


@lru_cache
def get_config() -> Config:
    cfg = Config()
    cfg.ensure_directories()
    return cfg


# Application-scoped singletons (initialized in lifespan)
_store: UnifiedStore | None = None
_memory: MemoryManager | None = None
_llm: LLMRouter | None = None


async def init_app():
    global _store, _memory, _llm
    cfg = get_config()
    _store = UnifiedStore(db_path=cfg.db_path, chroma_path=cfg.chroma_path)
    await _store.initialize()
    _memory = MemoryManager(db_path=cfg.db_path, chroma_path=cfg.chroma_path, skills_dir=cfg.skills_dir)
    await _memory.initialize()
    _llm = LLMRouter(cfg)


async def shutdown_app():
    if _store:
        await _store.close()
    if _memory:
        await _memory.close()


def get_store() -> UnifiedStore:
    assert _store is not None
    return _store


def get_memory() -> MemoryManager:
    assert _memory is not None
    return _memory


def get_llm() -> LLMRouter:
    assert _llm is not None
    return _llm
```

- [ ] **Step 2: Update main.py with lifespan**

```python
# litscribe/api/main.py
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from litscribe.api.deps import init_app, shutdown_app
from litscribe.api.routes.reviews import router as reviews_router
from litscribe.api.routes.memory import router as memory_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_app()
    yield
    await shutdown_app()


def create_app() -> FastAPI:
    app = FastAPI(title="LitScribe", version="2.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(reviews_router)
    app.include_router(memory_router)

    return app


def run():
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add litscribe/api/
git commit -m "feat(v2): wire dependencies — store, memory, LLM router in API lifespan"
```

---

### Task 22: Full integration test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
import pytest
from unittest.mock import AsyncMock

from litscribe.models.plan import ReviewTier


@pytest.mark.asyncio
async def test_full_pipeline_with_mock_llm(tmp_data_dir, tmp_skills_dir):
    """End-to-end: config → store → memory → pipeline → skill evolution."""
    from litscribe.config import Config
    from litscribe.store.unified import UnifiedStore
    from litscribe.evolution.memory_manager import MemoryManager
    from litscribe.agents.pipeline import LitScribePipeline
    from litscribe.agents.planner import create_plan
    from litscribe.models.paper import Paper
    from litscribe.models.analysis import PaperAnalysis
    from litscribe.models.review import ReviewOutput
    from litscribe.models.assessment import ReviewAssessment

    # Setup
    store = UnifiedStore(db_path=tmp_data_dir / "test.db", chroma_path=tmp_data_dir / "vectors")
    await store.initialize()
    memory = MemoryManager(
        db_path=tmp_data_dir / "test.db",
        chroma_path=tmp_data_dir / "vectors",
        skills_dir=tmp_skills_dir,
    )
    await memory.initialize()

    # Mock LLM for planner
    mock_llm = AsyncMock(return_value='{"sub_topics": [{"name": "CoT", "keywords": ["chain of thought"], "estimated_papers": 10}], "domain": "NLP/AI"}')

    async def plan_fn(question, **kw):
        return await create_plan(question, ReviewTier.QUICK, 10, "en", mock_llm)

    async def discover_fn(plan, **kw):
        return [Paper(paper_id="p1", title="CoT Paper", authors=["A"], abstract="About CoT", year=2024, sources={"arxiv": "1"})]

    async def read_fn(papers, **kw):
        return [PaperAnalysis(paper_id="p1", key_findings=["CoT improves reasoning"], relevance_score=0.9)]

    async def synthesize_fn(analyses, **kw):
        return ReviewOutput(text="CoT review text", word_count=500)

    async def review_fn(output, **kw):
        return ReviewAssessment(passed=True, score=0.85, feedback="Good coverage")

    pipeline = LitScribePipeline(
        plan_fn=plan_fn, discover_fn=discover_fn, read_fn=read_fn,
        synthesize_fn=synthesize_fn, review_fn=review_fn,
    )

    result = await pipeline.run("LLM reasoning", max_papers=10, tier=ReviewTier.QUICK)
    assert result.text == "CoT review text"

    # Verify skill evolution would trigger
    assert memory.evolver.should_extract_skill(score=0.85, complexity=5)

    # Verify episodic memory works
    await memory.episodic.record(
        session_id="test", question="LLM reasoning",
        outcome_score=0.85, key_events=["Found 1 paper", "Score 0.85"],
    )
    recalled = await memory.episodic.recall("LLM reasoning")
    assert len(recalled) == 1

    # Verify semantic memory works
    from litscribe.models.analysis import PaperAnalysis as PA
    memory.semantic.absorb([PA(paper_id="p1", key_findings=["CoT improves reasoning"], relevance_score=0.9)])
    sem_results = memory.semantic.search("chain of thought reasoning")
    assert len(sem_results) >= 1

    await store.close()
    await memory.close()
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test(v2): end-to-end integration test — pipeline + memory + skill evolution"
```

---

## Summary

| Phase | Tasks | What it delivers |
|-------|-------|------------------|
| 1: Skeleton + Config + Models | 1-3 | Project structure, config system, data contracts |
| 2: Infrastructure | 4-8 | SQLite+FTS5, ChromaDB, UnifiedStore, LLM Router, SearchService protocol |
| 3: Self-Evolution | 9-13 | Three-tier memory (Episodic + Semantic + Procedural) + SkillEvolver + MemoryManager |
| 4: Agents | 14-15 | Planner agent + Pipeline (deterministic workflow with loop-back) |
| 5: API + CLI | 16-18 | FastAPI endpoints + Typer CLI thin client |
| 6: Services + Exporters | 19 | Adapted search services + exporters from v1 |
| 7: GraphRAG Plugin | 20 | Optional GraphRAG as plugin |
| 8: Integration | 21-22 | Wiring + full integration test |

**Total: 22 tasks, ~40 files created, targeting ~3000-4000 lines of new code.**

**Post-plan tasks** (not in scope but noted for follow-up):
- Adapt each service file's internal logic (Task 19 is a sketch — needs per-service work)
- WebSocket progress streaming
- Production background task runner for reviews
- Session persistence and version diffing
- Migrate existing SQLite v4 data to v5 schema
