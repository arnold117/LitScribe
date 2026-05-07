from __future__ import annotations

import asyncio
import json
import logging
import os
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
    SYNTHESIZER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
)
from litscribe.prompts.supervisor import SUPERVISOR_PROMPT
from litscribe.tools.status import PipelineState, check_status as _check_status

logger = logging.getLogger(__name__)


def _build_model(config: Config) -> ChatOpenAI:
    model_name = config.llm.default_model
    # Strip litellm provider prefix if present (e.g. "openai/gpt-4" → "gpt-4")
    if "/" in model_name:
        model_name = model_name.split("/", 1)[1]

    return ChatOpenAI(
        base_url=config.llm.api_base,
        api_key=config.llm.api_key,
        model=model_name,
    )


def create_pipeline_tools(config: Config, state: PipelineState):
    from litscribe.llm.router import LLMRouter
    router = LLMRouter(config)

    @tool
    async def create_plan(research_question: str, domain: str = "General") -> str:
        """Create a research plan by decomposing the question into sub-topics. MUST be called first before discovery."""
        from litscribe.prompts.planning import COMPLEXITY_ASSESSMENT_PROMPT

        state.research_question = research_question
        state.domain = domain

        prompt = COMPLEXITY_ASSESSMENT_PROMPT.format(research_question=research_question)
        try:
            result = await router.call_json(_msg(prompt), task_type="planning")
            if isinstance(result, dict):
                from litscribe.models.plan import ResearchPlan, SubTopic
                sub_topics = []
                for st in result.get("sub_topics", []):
                    sub_topics.append(SubTopic(
                        name=st.get("name", ""),
                        keywords=st.get("custom_queries", st.get("keywords", [])),
                        estimated_papers=st.get("estimated_papers", 10),
                    ))
                state.plan = ResearchPlan(
                    question=research_question,
                    sub_topics=sub_topics,
                    domain=result.get("domain", domain),
                    tier="standard",
                    max_papers=40,
                    language=state.language,
                    target_words=max(1000, sum(st.estimated_papers for st in sub_topics) * 130),
                )
                state.domain = result.get("domain", domain)
                topics_str = ", ".join(st.name for st in sub_topics)
                return (
                    f"Plan created: {len(sub_topics)} sub-topics ({topics_str}), "
                    f"domain={state.domain}. Call check_pipeline_status for next step."
                )
        except Exception as e:
            logger.warning(f"Plan creation failed: {e}")

        from litscribe.models.plan import ResearchPlan, SubTopic
        state.plan = ResearchPlan(
            question=research_question,
            sub_topics=[SubTopic(name=research_question, keywords=[research_question], estimated_papers=20)],
            domain=domain, tier="standard", max_papers=40, language=state.language, target_words=3000,
        )
        return f"Fallback plan created (single topic). Call check_pipeline_status."

    @tool
    async def discover_papers(
        research_question: str,
        max_papers: int = 40,
        extra_queries: str = "",
    ) -> str:
        """Search academic databases for papers. Pass max_papers based on user preference (default 40)."""
        from litscribe.tools.discovery import discover_papers as _discover

        extra = [q.strip() for q in extra_queries.split(",") if q.strip()] if extra_queries else []

        # Use sub-topic queries from plan if available
        if state.plan and state.plan.sub_topics:
            for st in state.plan.sub_topics:
                extra.extend(st.keywords[:3])

        result = await _discover(
            research_question=research_question,
            domain=state.domain,
            config=config,
            router=router,
            max_papers=max_papers,
            extra_queries=extra,
        )

        state.papers = result["papers"]
        state.iteration += 1
        return (
            f"Found {result['total_found']} papers, selected {result['total_selected']}. "
            f"Used {result['queries_used']} queries. "
            f"Call check_status for next step."
        )

    @tool
    async def analyze_papers() -> str:
        """Critically analyze all discovered papers. Call this after discovery."""
        from litscribe.tools.reading import analyze_papers as _analyze

        if not state.papers:
            return "No papers to analyze. Run discover_papers first."

        analyses = await _analyze(state.papers, state.research_question, router)
        state.analyses = analyses
        return (
            f"Analyzed {len(analyses)} papers. "
            f"Average relevance: {sum(a.relevance_score for a in analyses) / max(len(analyses), 1):.2f}. "
            f"Call check_status for next step."
        )

    @tool
    async def build_knowledge_graph() -> str:
        """Build a knowledge graph from paper analyses. Call when ≥5 papers analyzed."""
        from litscribe.tools.graphrag import build_knowledge_graph as _build

        if len(state.analyses) < 5:
            return f"Only {len(state.analyses)} analyses, skipping graph (need ≥5)."

        async def llm_call(prompt: str, **kwargs) -> str:
            return await router.call([{"role": "user", "content": prompt}], **kwargs)

        graph = await _build(state.analyses, llm_call)
        state.graph = graph
        n_communities = len(graph.get("communities", []))
        return f"Built knowledge graph: {n_communities} communities. Call check_status."

    @tool
    async def write_review(instructions: str = "") -> str:
        """Write the literature review. Pass user preferences in instructions (e.g. 'focus on efficiency, 3 themes, compare methods')."""
        from litscribe.tools.synthesis import synthesize

        if not state.analyses:
            return "No analyses available. Run analyze_papers first."

        review = await synthesize(
            router=router,
            analyses=state.analyses,
            research_question=state.research_question,
            language=state.language,
            graph_context=state.graph,
            user_instructions=instructions,
            papers=state.papers,
        )
        state.synthesis = review
        return (
            f"Review written: {review.word_count} words, {len(review.themes)} themes. "
            f"Call check_status for next step."
        )

    @tool
    async def evaluate_review() -> str:
        """Evaluate the review quality. Call after writing the review."""
        from litscribe.tools.review import evaluate_review as _evaluate

        if state.synthesis is None:
            return "No review to evaluate. Run write_review first."

        assessment = await _evaluate(
            router=router,
            review=state.synthesis,
            analyses=state.analyses,
            plan=state.plan,
            research_question=state.research_question,
        )
        state.assessment = assessment
        passed = "PASSED" if assessment.passed else "NEEDS IMPROVEMENT"
        return (
            f"Review evaluation: {passed} (score={assessment.score:.2f}, "
            f"coverage={assessment.coverage_score:.2f}). "
            f"Call check_status for next step."
        )

    @tool
    def check_pipeline_status() -> str:
        """Check current pipeline progress and get routing recommendation. Call after EVERY step."""
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

    return [
        create_plan,
        discover_papers,
        analyze_papers,
        build_knowledge_graph,
        write_review,
        evaluate_review,
        check_pipeline_status,
        export_results,
    ]


def _build_subagents() -> list[SubAgent]:
    planner: SubAgent = {
        "name": "planner",
        "description": "Decompose a research question into sub-topics with search strategy and domain classification",
        "system_prompt": PLANNER_SYSTEM_PROMPT,
        "tools": [],
        "response_format": ResearchPlan,
    }
    reader: SubAgent = {
        "name": "reader",
        "description": "Critically analyze academic papers — extract findings, methodology, strengths, limitations",
        "system_prompt": READER_SYSTEM_PROMPT,
        "tools": [],
    }
    synthesizer: SubAgent = {
        "name": "synthesizer",
        "description": "Write a comprehensive literature review from paper analyses, organized by themes",
        "system_prompt": SYNTHESIZER_SYSTEM_PROMPT,
        "tools": [],
        "response_format": ReviewOutput,
    }
    reviewer: SubAgent = {
        "name": "reviewer",
        "description": "Evaluate review quality — relevance, coverage, coherence, claim support",
        "system_prompt": REVIEWER_SYSTEM_PROMPT,
        "tools": [],
        "response_format": ReviewAssessment,
    }
    return [planner, reader, synthesizer, reviewer]


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
