"""Factory that builds a LitScribePipeline wired with Agno agents."""
from __future__ import annotations

import json
import re
from typing import Any

from agno.agent import Agent

from litscribe.config import Config
from litscribe.agents.agno_agents import (
    create_planner_agent,
    create_reader_agent,
    create_synthesizer_agent,
    create_reviewer_agent,
)
from litscribe.agents.pipeline import LitScribePipeline
from litscribe.evolution.memory_manager import MemoryManager
from litscribe.llm.router import LLMRouter
from litscribe.models.plan import ResearchPlan, ReviewTier, SubTopic
from litscribe.models.paper import Paper
from litscribe.models.analysis import PaperAnalysis
from litscribe.models.review import ReviewOutput, Citation, Theme
from litscribe.models.assessment import ReviewAssessment


def _extract_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON from an LLM response."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    return json.loads(cleaned)


def _agent_run_text(agent: Agent, message: str) -> str:
    """Run an Agno agent synchronously and return the response text."""
    response = agent.run(message)
    if hasattr(response, "content"):
        return response.content or ""
    return str(response)


async def _agent_run_text_async(agent: Agent, message: str) -> str:
    """Run an Agno agent asynchronously and return the response text."""
    response = await agent.arun(message)
    if hasattr(response, "content"):
        return response.content or ""
    return str(response)


def create_agno_pipeline(config: Config, memory: MemoryManager | None = None) -> LitScribePipeline:
    """Factory that creates a LitScribePipeline wired with Agno agents and config."""
    router = LLMRouter(config)

    api_key = config.llm.api_key
    api_base = config.llm.api_base

    planner = create_planner_agent(
        model=router.resolve_model("planning"),
        api_key=api_key,
        api_base=api_base,
    )
    reader = create_reader_agent(
        model=router.resolve_model("paper_analysis"),
        api_key=api_key,
        api_base=api_base,
    )
    synthesizer = create_synthesizer_agent(
        model=router.resolve_model("synthesis"),
        api_key=api_key,
        api_base=api_base,
    )
    reviewer = create_reviewer_agent(
        model=router.resolve_model("self_review"),
        api_key=api_key,
        api_base=api_base,
    )

    # ------------------------------------------------------------------ #
    # Step functions                                                       #
    # ------------------------------------------------------------------ #

    async def plan_fn(question: str, **kwargs: Any) -> ResearchPlan:
        tier_raw = kwargs.get("tier", ReviewTier.STANDARD)
        tier = tier_raw if isinstance(tier_raw, ReviewTier) else ReviewTier(tier_raw)
        max_papers: int = kwargs.get("max_papers", 40)
        language: str = kwargs.get("language", "en")

        prompt = (
            f"Research question: {question}\n"
            f"Review tier: {tier.value}\n"
            f"Max papers: {max_papers}\n"
            f"Language: {language}\n\n"
            "Return ONLY valid JSON with keys: "
            '"sub_topics" (list of {name, keywords, estimated_papers}) and "domain".'
        )
        raw = await _agent_run_text_async(planner, prompt)
        try:
            data = _extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            data = {}

        raw_subtopics = data.get("sub_topics", [])
        sub_topics: list[SubTopic] = []
        if isinstance(raw_subtopics, list):
            for item in raw_subtopics:
                if isinstance(item, dict):
                    sub_topics.append(
                        SubTopic(
                            name=item.get("name", ""),
                            keywords=item.get("keywords", []),
                            estimated_papers=int(item.get("estimated_papers", 10)),
                        )
                    )
        if not sub_topics:
            sub_topics = [SubTopic(name=question, keywords=[question])]

        base_words = 1000 + max_papers * 130
        target_words = int(base_words * 1.5) if language[:2] in {"zh", "ja", "ko"} else base_words

        return ResearchPlan(
            question=question,
            sub_topics=sub_topics,
            domain=data.get("domain", "General"),
            tier=tier,
            max_papers=max_papers,
            language=language,
            target_words=target_words,
        )

    async def discover_fn(plan: ResearchPlan, **kwargs: Any) -> list[Paper]:
        """Discovery uses external services, not an LLM agent.

        This stub returns an empty list so the pipeline can be instantiated
        without live service dependencies.  Callers should replace this with
        a real service-backed implementation when building a production
        pipeline.
        """
        return kwargs.get("_papers", [])

    async def read_fn(papers: list[Paper], **kwargs: Any) -> list[PaperAnalysis]:
        if not papers:
            return []

        summaries = "\n\n".join(
            f"ID: {p.paper_id}\nTitle: {p.title}\nAbstract: {p.abstract or ''}"
            for p in papers
        )
        prompt = (
            "Analyze the following academic papers. "
            "For each paper return a JSON object with keys: "
            "paper_id, key_findings (list), methodology, strengths (list), "
            "limitations (list), relevance_score (0-1), themes (list).\n\n"
            f"{summaries}\n\n"
            "Return a JSON array of analysis objects."
        )
        raw = await _agent_run_text_async(reader, prompt)
        try:
            data = _extract_json(raw)
            if isinstance(data, dict):
                data = data.get("analyses", data.get("papers", [data]))
        except (json.JSONDecodeError, ValueError):
            data = []

        analyses: list[PaperAnalysis] = []
        for item in data if isinstance(data, list) else []:
            if isinstance(item, dict):
                analyses.append(
                    PaperAnalysis(
                        paper_id=item.get("paper_id", ""),
                        key_findings=item.get("key_findings", []),
                        methodology=item.get("methodology", ""),
                        strengths=item.get("strengths", []),
                        limitations=item.get("limitations", []),
                        relevance_score=float(item.get("relevance_score", 0.0)),
                        themes=item.get("themes", []),
                    )
                )
        # Fallback: if parsing fails, emit a minimal analysis per paper
        if not analyses:
            analyses = [
                PaperAnalysis(paper_id=p.paper_id, key_findings=[]) for p in papers
            ]
        return analyses

    async def synthesize_fn(
        analyses: list[PaperAnalysis], **kwargs: Any
    ) -> ReviewOutput:
        plan: ResearchPlan | None = kwargs.get("plan")
        question = plan.question if plan else "the research question"
        language = (plan.language if plan else None) or "en"

        summaries = "\n\n".join(
            f"Paper {a.paper_id}: findings={a.key_findings}, themes={a.themes}"
            for a in analyses
        )
        prompt = (
            f"Write a comprehensive literature review for: {question}\n"
            f"Language: {language}\n\n"
            f"Paper analyses:\n{summaries}\n\n"
            "Return JSON with keys: text (review markdown), "
            "citations (list of {paper_id, claim, section}), "
            "themes (list of {name, description, paper_ids})."
        )
        raw = await _agent_run_text_async(synthesizer, prompt)
        try:
            data = _extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            data = {}

        text = data.get("text", raw)
        citations = [
            Citation(
                paper_id=c.get("paper_id", ""),
                claim=c.get("claim", ""),
                section=c.get("section", ""),
            )
            for c in data.get("citations", [])
            if isinstance(c, dict)
        ]
        themes = [
            Theme(
                name=t.get("name", ""),
                description=t.get("description", ""),
                paper_ids=t.get("paper_ids", []),
            )
            for t in data.get("themes", [])
            if isinstance(t, dict)
        ]
        return ReviewOutput(
            text=text,
            citations=citations,
            themes=themes,
            word_count=len(text.split()),
            language=language,
        )

    async def review_fn(output: ReviewOutput, **kwargs: Any) -> ReviewAssessment:
        plan: ResearchPlan | None = kwargs.get("plan")
        question = plan.question if plan else "the research question"

        prompt = (
            f"Evaluate the following literature review for: {question}\n\n"
            f"Review text:\n{output.text}\n\n"
            "Return JSON with keys: score (float 0-1), passed (bool, true if score >= 0.65), "
            "feedback (string), refined_queries (list of strings for follow-up searches)."
        )
        raw = await _agent_run_text_async(reviewer, prompt)
        try:
            data = _extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            data = {}

        score = float(data.get("score", 0.5))
        return ReviewAssessment(
            passed=bool(data.get("passed", score >= 0.65)),
            score=score,
            feedback=data.get("feedback", raw),
            refined_queries=data.get("refined_queries") or [],
        )

    return LitScribePipeline(
        plan_fn=plan_fn,
        discover_fn=discover_fn,
        read_fn=read_fn,
        synthesize_fn=synthesize_fn,
        review_fn=review_fn,
        memory=memory,
    )
