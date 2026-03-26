# LitScribe v2 Refactoring Design

## Overview

Full rewrite of LitScribe from LangGraph to Agno, adding Hermes-style self-evolving agent capabilities (three-tier memory, procedural skills, self-improvement loop), 百炼 API as primary LLM backend, and API-first architecture.

## Decisions

| Decision | Choice |
|----------|--------|
| Migration strategy | Full rewrite on new branch |
| Agent framework | Agno (Workflow + Team) replacing LangGraph |
| Self-evolution | Full Hermes-style: Episodic + Semantic + Procedural memory + Skill evolution loop |
| GraphRAG | Redesigned as optional plugin, entity/summarizer prompts enter skill evolution |
| LLM backend | 百炼 (DashScope) primary, litellm retained for flexibility |
| Storage | Unified SQLite + ChromaDB, replacing 7 scattered cache modules |
| UI strategy | FastAPI core API first, CLI as thin Typer client, Web UI future |

---

## 1. Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Interface Layer                        │
│  CLI (Typer, thin)  ←→  FastAPI  ←→  Web UI (future)    │
│                          ↕ SSE/WebSocket                 │
├──────────────────────────────────────────────────────────┤
│                   Orchestration Layer                     │
│  Agno Workflow: LitScribe Pipeline                       │
│  ┌─────┐  ┌──────────┐  ┌────────┐  ┌─────────┐        │
│  │Plan │→ │Discovery │→ │Reading │→ │Synthesis│        │
│  └─────┘  │(Team:    │  └────────┘  └─────────┘        │
│           │ parallel │       ↑            ↓              │
│           │ search)  │       └── Self-Review ──→ loop?   │
│           └──────────┘                                   │
│  Optional Plugin: GraphRAG Agent                         │
├──────────────────────────────────────────────────────────┤
│                  Self-Evolution Layer                     │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │  Episodic   │ │  Semantic    │ │  Procedural      │  │
│  │  Memory     │ │  Memory      │ │  Skills          │  │
│  │  (FTS5)     │ │  (ChromaDB)  │ │  (~/.litscribe/) │  │
│  └─────────────┘ └──────────────┘ └──────────────────┘  │
│  SkillEvolver: evaluate → extract → store → reuse → patch│
├──────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                    │
│  Unified Store    │  LLM Router (litellm)  │  Services   │
│  (SQLite+Chroma)  │  → 百炼/others         │  (APIs)     │
└──────────────────────────────────────────────────────────┘
```

Layer dependency rule: each layer depends only on layers below it, never upward or sideways.

---

## 2. Orchestration Layer

### 2.1 Pipeline

Agno **Workflow** for the deterministic pipeline steps. Agno **Team** (coordinate mode) for sub-tasks requiring LLM-driven delegation (e.g., multi-source parallel search within Discovery).

```python
class LitScribePipeline(Workflow):
    planner: Agent
    discovery: Team           # coordinate mode, parallel multi-source search
    critical_reader: Agent
    graphrag: GraphRAGPlugin  # optional, injected via config
    synthesizer: Agent
    self_reviewer: Agent

    def run(self, question, config):
        plan = self.planner.run(question)
        for iteration in range(max_iterations):  # default 3
            papers = self.discovery.run(plan, memory_context)
            analyses = self.critical_reader.run(papers)
            if config.graphrag_enabled:
                kg = self.graphrag.process(analyses)
            review = self.synthesizer.run(analyses, kg?)
            assessment = self.self_reviewer.run(review)
            if assessment.passed:
                break
            plan = refine_plan(plan, assessment)
        return review
```

### 2.2 Routing

- **Deterministic**: Workflow steps execute in fixed order (Plan → Discovery → Reading → Synthesis → Self-Review)
- **Loop-back**: Self-Review score < 0.65 triggers re-entry to Discovery with refined queries (max 3 loop-backs, matching current circuit breaker logic)
- **Refinement**: After pipeline completes, user can request iterative refinement via `/api/reviews/{id}/refine`, handled by the Synthesizer agent with refinement instructions
- **Local LLM routing**: Discovery Team leader decides search source allocation per sub-topic

### 2.3 Inter-Agent Data Contracts

Pydantic models replace the monolithic TypedDict state:

```python
class ResearchPlan(BaseModel):
    question: str
    sub_topics: list[SubTopic]
    domain: str
    tier: ReviewTier

class DiscoveryResult(BaseModel):
    papers: list[Paper]
    search_metadata: SearchMeta

class AnalysisResult(BaseModel):
    analyses: list[PaperAnalysis]
    parsed_docs: list[ParsedDoc]

class ReviewOutput(BaseModel):
    text: str
    citations: list[Citation]
    themes: list[Theme]

class ReviewAssessment(BaseModel):
    passed: bool
    score: float
    feedback: str
    refined_queries: list[str] | None
```

Each agent receives only the data it needs, not a god-state.

### 2.4 GraphRAG Plugin

Optional module, loaded when `config.graphrag_enabled = True`:

```python
class GraphRAGPlugin:
    entity_extractor: Agent   # prompt participates in skill evolution
    linker: EntityLinker      # pure algorithm, not an agent
    graph_builder: GraphBuilder
    community_detector: CommunityDetector
    summarizer: Agent         # prompt participates in skill evolution

    async def process(self, analyses: list[PaperAnalysis]) -> KnowledgeGraph:
        entities = await self.entity_extractor.run(analyses)
        linked = self.linker.link(entities)
        graph = self.graph_builder.build(linked)
        communities = self.community_detector.detect(graph)
        summaries = await self.summarizer.run(communities)
        return KnowledgeGraph(graph, communities, summaries)
```

---

## 3. Self-Evolution Layer

### 3.1 Three-Tier Memory

```
MemoryManager (unified interface)
├── EpisodicMemory  — "what happened" — SQLite + FTS5
├── SemanticMemory  — "what we know"  — ChromaDB vectors
└── ProceduralMemory — "how to do it" — ~/.litscribe/skills/ markdown
```

**Episodic Memory**: Records task experiences for cross-session recall.
- After each review task, LLM summarizes key events into a compressed episode
- FTS5 full-text indexed for fast retrieval
- Stores: session_id, question, outcome (score), summary, key_events, timestamp

**Semantic Memory**: Accumulates domain knowledge and user modeling.
- Extracts knowledge chunks from paper analyses, embeds into ChromaDB
- Maintains user profile (research domains, preferred tiers, writing style)
- Semantic vector search for contextual retrieval

**Procedural Memory (Skills)**: Reusable workflow documents the agent creates and maintains.
- Stored as Markdown/YAML in `~/.litscribe/skills/`
- Organized by type: `search_strategies/`, `analysis_patterns/`, `domain_knowledge/`
- Each skill has metadata: name, domain, version, success_rate, last_used

Skill file format:
```yaml
---
name: NLP Deep Search Strategy
domain: NLP/AI
version: 3
success_rate: 0.85
last_used: 2026-03-26
---
## Trigger
When research question involves NLP/AI domain

## Strategy
1. Query expansion: topic + method + dataset terms
2. Priority sources: arXiv (cs.CL) > S2 > ACL Anthology
3. Snowball: seed from citation count > 100

## Learned Adjustments
- v2: Added OpenAlex as supplementary source
- v3: Removed MeSH (not applicable to NLP)
```

### 3.2 Skill Evolution Loop

```
Task complete
  → SelfEvaluator assesses quality
    → score >= 0.7 AND complexity >= 5 tool calls?
      → SkillExtractor extracts skill from execution trace
        → similar skill exists?
          → YES: SkillPatcher merges diff, version++
          → NO:  save new skill file
    → score < 0.5?
      → mark related skills as needs_review
      → EpisodicMemory records failure reason
```

```python
class SkillEvolver:
    def post_task_evaluate(self, session, result, assessment):
        if assessment.score >= 0.7 and session.complexity >= 5:
            skill = self.extract_skill(session.trace)
            existing = self.find_similar_skill(skill)
            if existing:
                self.patch_skill(existing, skill)
            else:
                self.save_skill(skill)
        if assessment.score < 0.5:
            self.mark_skills_for_review(session.used_skills)
            self.episodic.save_failure(session, assessment.feedback)

    def inject_skills(self, agent, context):
        relevant = self.find_skills(context.domain, context.task_type)
        agent.instructions += format_skills(relevant)
```

### 3.3 Memory Application Points

| Pipeline Step | Memory Usage |
|---|---|
| Planning | Episodic: recall similar past plans; Semantic: known sub-topics for domain |
| Discovery | Procedural: domain search strategy skill; Semantic: known key authors/papers |
| Critical Reading | Procedural: domain analysis template; Semantic: domain knowledge aids comprehension |
| Synthesis | Procedural: review writing pattern; Episodic: user's preferred writing style |
| Self-Review | Episodic: common issues from past tasks; triggers SkillEvolver |

---

## 4. Infrastructure Layer

### 4.1 Unified Store

Two storage engines replacing 7 cache modules:

**SQLite** (litscribe.db):
- `papers` — paper metadata (merged paper_cache + search_cache)
- `pdfs` — PDF download records
- `parsed_docs` — parsed markdown + sections
- `episodes` — episodic memory with FTS5 index
- `sessions` — task sessions + version history
- `skills_meta` — skill usage statistics and scores
- `user_profiles` — user preferences

**ChromaDB** (litscribe_vectors/):
- `semantic_memory` — domain knowledge vectors
- `paper_embeddings` — paper abstract vectors (reused for semantic dedup)
- `skill_embeddings` — skill vectors (similar skill retrieval)

```python
class UnifiedStore:
    def __init__(self, db_path, chroma_path):
        self.db = aiosqlite.connect(db_path)
        self.chroma = chromadb.PersistentClient(chroma_path)

    # Paper operations
    async def get_paper(self, paper_id) -> Paper | None
    async def save_papers(self, papers: list[Paper])
    async def search_cached(self, query, source) -> list[Paper] | None

    # Parse operations
    async def get_parsed(self, paper_id) -> ParsedDoc | None
    async def save_parsed(self, paper_id, doc: ParsedDoc)

    # Semantic search
    def semantic_search(self, query, collection, n=10) -> list
    def add_embeddings(self, texts, metadata, collection)

    # Episodic memory
    async def save_episode(self, episode: Episode)
    async def recall(self, query, limit=5) -> list[Episode]
```

Schema migration: v4 → v5, auto-migrate on startup.

### 4.2 LLM Router

```python
class LLMRouter:
    def __init__(self, config):
        self.default_model = config.default_model   # openai/qwen-plus
        self.task_models = config.task_models
        self.api_base = config.api_base              # 百炼 endpoint
        self.tracker = TokenTracker()

    async def call(self, prompt, task_type, **kwargs) -> str:
        model = self.resolve_model(task_type)
        response = await litellm.acompletion(
            model=model, messages=prompt,
            api_base=self.api_base, **kwargs
        )
        self.tracker.record(task_type, model, response.usage)
        return response.choices[0].message.content

    async def call_json(self, prompt, task_type, **kwargs) -> dict:
        """JSON extraction with retry on parse failure"""

    def resolve_model(self, task_type) -> str:
        return self.task_models.get(task_type, self.default_model)
```

Default 百炼 configuration:
```yaml
llm:
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  default_model: "openai/qwen-plus"
  task_models:
    query_expansion: "openai/qwen-turbo"
    planning: "openai/qwen-plus"
    paper_analysis: "openai/qwen-plus"
    entity_extraction: "openai/qwen-turbo"
    synthesis: "openai/qwen-max"
    self_review: "openai/qwen-plus"
    refinement: "openai/deepseek-r1"
    skill_extraction: "openai/qwen-turbo"
```

### 4.3 Services

Existing services refactored to a unified interface:

```python
class SearchService(Protocol):
    source_name: str
    async def search(self, query, max_results, **filters) -> list[Paper]
```

Implementations: ArxivService, PubMedService, SemanticScholarService, OpenAlexService, EuropePMCService, ZoteroService.

PDF fetch chain unchanged: Unpaywall → PMC → S2 → arXiv → publisher.

### 4.4 API Layer

```python
app = FastAPI(title="LitScribe")

# Review lifecycle
POST   /api/reviews              # start new review
GET    /api/reviews/{id}         # query status/result
POST   /api/reviews/{id}/refine  # iterative refinement
DELETE /api/reviews/{id}         # cancel

# Session management
GET    /api/sessions             # list historical sessions
GET    /api/sessions/{id}        # session detail + version history

# Memory system
GET    /api/memory/episodes      # episodic search
GET    /api/memory/skills        # skill listing
PUT    /api/memory/skills/{id}   # manual skill editing

# Configuration
GET    /api/config
PUT    /api/config

# Real-time progress
WS     /ws/reviews/{id}          # WebSocket agent progress stream
```

### 4.5 CLI

Typer-based thin client over FastAPI, target ~200 lines (down from 1773):

```python
app = typer.Typer()

@app.command()
def review(question: str, tier: str = "standard", model: str = None):
    """Start a literature review"""

@app.command()
def sessions():
    """List historical sessions"""

@app.command()
def skills(action: str = "list"):
    """Manage skills: list / show / edit / delete"""

@app.command()
def config(key: str = None, value: str = None):
    """View/modify configuration"""
```

---

## 5. Project Structure (Target)

```
litscribe/
├── api/
│   ├── main.py                 # FastAPI app, CORS, middleware
│   ├── routes/
│   │   ├── reviews.py          # review lifecycle
│   │   ├── sessions.py         # session management
│   │   ├── memory.py           # memory/skills endpoints
│   │   └── config.py           # configuration
│   └── websocket.py            # real-time progress
│
├── agents/
│   ├── pipeline.py             # Agno Workflow: LitScribePipeline
│   ├── planner.py              # Planning agent
│   ├── discovery.py            # Discovery Team (multi-source)
│   ├── reader.py               # Critical reading agent
│   ├── synthesizer.py          # Synthesis agent
│   ├── reviewer.py             # Self-review agent
│   └── models.py               # Pydantic data contracts
│
├── evolution/
│   ├── memory_manager.py       # Unified memory interface
│   ├── episodic.py             # Episodic memory (FTS5)
│   ├── semantic.py             # Semantic memory (ChromaDB)
│   ├── procedural.py           # Procedural skills (filesystem)
│   └── skill_evolver.py        # Skill extraction, patching, injection
│
├── plugins/
│   └── graphrag/
│       ├── plugin.py           # GraphRAGPlugin entry point
│       ├── entity_extractor.py
│       ├── linker.py
│       ├── graph_builder.py
│       ├── community_detector.py
│       └── summarizer.py
│
├── services/
│   ├── arxiv.py
│   ├── pubmed.py
│   ├── semantic_scholar.py
│   ├── openalex.py
│   ├── europe_pmc.py
│   ├── zotero.py
│   └── pdf.py                  # PDF fetch + parse
│
├── store/
│   ├── unified.py              # UnifiedStore interface
│   ├── sqlite.py               # SQLite operations + schema
│   └── vectors.py              # ChromaDB operations
│
├── llm/
│   ├── router.py               # LLMRouter with per-task routing
│   └── tracker.py              # Token tracking
│
├── cli/
│   └── main.py                 # Typer thin client (~200 lines)
│
├── exporters/
│   ├── bibtex.py
│   ├── pandoc.py
│   └── citation_formatter.py
│
└── config.py                   # Central configuration
```

---

## 6. New Dependencies

| Package | Purpose | Replaces |
|---------|---------|----------|
| `agno` | Agent framework | `langgraph`, `langgraph-checkpoint-sqlite`, `langchain-core` |
| `chromadb` | Vector store for semantic memory | (new) |
| `typer` | CLI framework | `argparse` (manual) |
| `httpx` | CLI → API HTTP client | (new) |

Retained: `litellm`, `aiosqlite`, `sentence-transformers`, `networkx`, `graspologic`, `pymupdf4llm`, `tenacity`, `fastapi`, `uvicorn`.

Removed: `langgraph`, `langgraph-checkpoint-sqlite`, `langchain-core`.

---

## 7. Migration Notes

- All existing services (arxiv, pubmed, s2, openalex, europe_pmc, zotero, pdf) are reused with interface normalization
- SQLite schema v4 → v5 auto-migration preserves cached papers, parsed docs, and session history
- Existing prompts in `agents/prompts.py` are ported to individual agent files
- GraphRAG algorithms (NetworkX, graspologic) are unchanged; only the orchestration wrapper changes
- Exporters (bibtex, pandoc, citation_formatter) are copied as-is
