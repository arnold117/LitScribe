from __future__ import annotations

import json
import logging
from typing import Any

from deepagents import SubAgent, create_deep_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from litscribe.config import Config
from litscribe.evolution.memory_manager import MemoryManager
from litscribe.middleware.evolution import EvolutionMiddleware
from litscribe.middleware.token_tracking import TokenTrackingMiddleware
from litscribe.models.assessment import ReviewAssessment
from litscribe.models.plan import ResearchPlan
from litscribe.models.review import ReviewOutput
from litscribe.prompts.planner_system import (
    PLANNER_SYSTEM_PROMPT,
    READER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
)
from litscribe.prompts.supervisor import SUPERVISOR_PROMPT
from litscribe.tools.status import PipelineState, check_status as _check_status

logger = logging.getLogger(__name__)


def _build_model(config: Config) -> ChatOpenAI:
    model_name = config.llm.default_model
    if "/" in model_name:
        model_name = model_name.split("/", 1)[1]

    kwargs = dict(
        model=model_name,
        openai_api_key=config.llm.api_key,
        openai_api_base=config.llm.api_base,
        temperature=0.1,
        timeout=300,
        max_retries=2,
    )
    if "deepseek" in config.llm.api_base.lower() and any(
        k in model_name.lower() for k in ("v4", "reasoner")
    ):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    return ChatOpenAI(**kwargs)


def _build_subagents() -> list[SubAgent]:
    planner: SubAgent = {
        "name": "planner",
        "description": (
            "Decompose a research question into sub-topics with search queries and domain classification. "
            "Returns a structured research plan with sub-topics, keywords, and estimated paper counts."
        ),
        "system_prompt": PLANNER_SYSTEM_PROMPT,
        "tools": [],
    }
    reader: SubAgent = {
        "name": "reader",
        "description": (
            "Critically analyze academic papers. Given paper titles and abstracts, "
            "extract key findings, methodology, strengths, limitations, and relevance scores."
        ),
        "system_prompt": READER_SYSTEM_PROMPT,
        "tools": [],
    }
    synthesizer: SubAgent = {
        "name": "synthesizer",
        "description": (
            "Write a comprehensive literature review from paper analyses. "
            "Organize by themes, cite every paper with [Author, Year] format."
        ),
        "system_prompt": SYNTHESIZER_SYSTEM_PROMPT,
        "tools": [],
    }
    reviewer: SubAgent = {
        "name": "reviewer",
        "description": (
            "Evaluate review quality on relevance, coverage, coherence, and claim support. "
            "Return a score and suggestions for improvement."
        ),
        "system_prompt": REVIEWER_SYSTEM_PROMPT,
        "tools": [],
    }
    return [planner, reader, synthesizer, reviewer]


def create_pipeline_tools(config: Config, state: PipelineState):

    @tool
    async def search_papers(queries: str, max_papers: int = 40) -> str:
        """Search academic databases with given queries (comma-separated). Pure API search, no LLM."""
        from litscribe.tools.search import search_all_sources

        query_list = [q.strip() for q in queries.split(",") if q.strip()]
        if not query_list:
            return "No queries provided."

        papers = await search_all_sources(query_list, config, max_per_source=max_papers)
        state.papers = papers[:max_papers]
        state.iteration += 1

        return (
            f"Found {len(papers)} papers, kept top {len(state.papers)}. "
            f"Sources searched: arXiv, OpenAlex, Europe PMC, PubMed, S2. "
            f"Call check_pipeline_status for next step."
        )

    @tool
    async def build_knowledge_graph() -> str:
        """Build a knowledge graph from paper analyses. Call when ≥5 papers analyzed."""
        from litscribe.tools.graphrag import build_knowledge_graph as _build

        if len(state.analyses) < 5:
            return f"Only {len(state.analyses)} analyses, skipping graph (need ≥5)."

        kg_model = _build_model(config)

        async def llm_call(prompt: str, **kwargs) -> str:
            result = await kg_model.ainvoke(prompt)
            return result.content

        graph = await _build(state.analyses, llm_call)
        state.graph = graph
        n = len(graph.get("communities", []))
        return f"Built knowledge graph: {n} communities. Call check_pipeline_status."

    @tool
    def check_pipeline_status() -> str:
        """Check pipeline progress and get routing recommendation. Call after EVERY step."""
        result = _check_status(state)
        return json.dumps(result, indent=2, ensure_ascii=False)

    @tool
    async def export_results(format: str = "markdown", style: str = "apa") -> str:
        """Export the review. Formats: markdown, bibtex, citations."""
        from litscribe.tools.export import export_review

        if state.synthesis is None:
            return "No review to export."

        result = await export_review(state.synthesis, state.papers, format, style)
        return result.get("content", "Export failed")[:3000]

    return [search_papers, build_knowledge_graph, check_pipeline_status, export_results]


def create_litscribe_agent(
    config: Config,
    memory: MemoryManager | None = None,
):
    model = _build_model(config)
    state = PipelineState()

    tools = create_pipeline_tools(config, state)
    subagents = _build_subagents()

    middleware = []
    if memory:
        evolution_mw = EvolutionMiddleware(memory, pipeline_state=state)
        middleware.append(evolution_mw)
    token_mw = TokenTrackingMiddleware()
    middleware.append(token_mw)

    agent = create_deep_agent(
        model=model,
        tools=tools,
        subagents=subagents,
        system_prompt=SUPERVISOR_PROMPT,
        middleware=middleware,
        name="litscribe",
    )

    return agent, state, token_mw
