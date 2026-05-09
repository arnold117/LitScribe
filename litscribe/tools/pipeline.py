from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_openai import ChatOpenAI

from litscribe.config import Config
from litscribe.models.paper import Paper
from litscribe.models.plan import ResearchPlan, SubTopic
from litscribe.tools.status import PipelineState

logger = logging.getLogger(__name__)


async def _call_llm(model: ChatOpenAI, prompt: str) -> str:
    result = await model.ainvoke(prompt)
    return result.content


async def _call_llm_json(model: ChatOpenAI, prompt: str, retries: int = 2) -> dict | list:
    import re
    for attempt in range(retries + 1):
        raw = await _call_llm(model, prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt == retries:
                logger.warning(f"JSON parse failed after {retries+1} attempts")
                return {}
    return {}


FAST_PLAN_PROMPT = """Decompose this research question into 2-4 sub-topics for a literature review.

Research Question: {research_question}

Output JSON:
{{
  "domain": "Primary academic field (e.g. Biology, Computer Science, Chemistry)",
  "sub_topics": [
    {{"name": "sub-topic name", "custom_queries": ["search query 1", "search query 2"]}}
  ]
}}

Keep queries in English, 3-8 words each, optimized for academic search engines."""


async def step_plan(model: ChatOpenAI, state: PipelineState) -> None:
    logger.info("Pipeline step: PLAN")
    t = time.time()

    prompt = FAST_PLAN_PROMPT.format(research_question=state.research_question)
    result = await _call_llm_json(model, prompt)

    if isinstance(result, dict) and result.get("sub_topics"):
        sub_topics = []
        for st in result["sub_topics"]:
            sub_topics.append(SubTopic(
                name=st.get("name", ""),
                keywords=st.get("custom_queries", st.get("keywords", [])),
                estimated_papers=st.get("estimated_papers", 10),
            ))
        state.plan = ResearchPlan(
            question=state.research_question,
            sub_topics=sub_topics,
            domain=result.get("domain", "General"),
            tier="standard",
            max_papers=40,
            language=state.language,
            target_words=max(1000, sum(st.estimated_papers for st in sub_topics) * 130),
        )
        state.domain = result.get("domain", "General")
    else:
        state.plan = ResearchPlan(
            question=state.research_question,
            sub_topics=[SubTopic(name=state.research_question, keywords=[state.research_question], estimated_papers=20)],
            domain="General", tier="standard", max_papers=40,
            language=state.language, target_words=3000,
        )

    logger.info(f"  PLAN done: {len(state.plan.sub_topics)} sub-topics, domain={state.domain} ({time.time()-t:.1f}s)")


async def step_search(model: ChatOpenAI, state: PipelineState, config: Config, max_papers: int = 40) -> None:
    from litscribe.tools.search import search_all_sources

    logger.info("Pipeline step: SEARCH")
    t = time.time()

    queries = [state.research_question]
    if state.plan:
        for st in state.plan.sub_topics:
            queries.extend(st.keywords[:2])
    if state.extra_queries:
        queries.extend(state.extra_queries)

    seen: set[str] = set()
    unique = []
    for q in queries:
        ql = q.strip().lower()
        if ql and ql not in seen:
            seen.add(ql)
            unique.append(q.strip())

    papers = await search_all_sources(unique[:6], config, max_per_source=max_papers, domain=state.domain)
    state.papers = papers[:max_papers]
    state.iteration += 1

    logger.info(f"  SEARCH done: {len(papers)} found → {len(state.papers)} kept ({time.time()-t:.1f}s)")


async def _try_get_full_text(paper: Paper) -> str | None:
    if not paper.pdf_urls:
        return None
    try:
        from litscribe.services.pdf import PDFService
        pdf_svc = PDFService()
        parsed = await pdf_svc.parse(paper.pdf_urls[0])
        if parsed and parsed.markdown and len(parsed.markdown) > 200:
            return parsed.markdown[:8000]
    except Exception:
        pass
    return None


async def step_read(model: ChatOpenAI, state: PipelineState) -> None:
    import asyncio
    from litscribe.prompts.reading import ABSTRACT_ONLY_ANALYSIS_PROMPT, COMBINED_PAPER_ANALYSIS_PROMPT
    from litscribe.models.analysis import PaperAnalysis

    logger.info(f"Pipeline step: READ ({len(state.papers)} papers)")
    t = time.time()

    async def analyze_one(paper: Paper) -> PaperAnalysis:
        full_text = await _try_get_full_text(paper)

        if full_text:
            prompt = COMBINED_PAPER_ANALYSIS_PROMPT.format(
                research_question=state.research_question,
                title=paper.title,
                authors=", ".join(paper.authors[:3]) if paper.authors else "Unknown",
                year=paper.year or "N/A",
                abstract=paper.abstract or "(no abstract)",
                full_text=full_text,
            )
        else:
            prompt = ABSTRACT_ONLY_ANALYSIS_PROMPT.format(
                research_question=state.research_question,
                title=paper.title,
                authors=", ".join(paper.authors[:3]) if paper.authors else "Unknown",
                year=paper.year or "N/A",
                venue=paper.venue or "",
                abstract=paper.abstract or "(no abstract)",
                metadata_section=f"Citations: {paper.citations or 0}",
            )
        try:
            result = await _call_llm_json(model, prompt, retries=1)
            if isinstance(result, dict):
                return PaperAnalysis(
                    paper_id=paper.paper_id,
                    key_findings=result.get("key_findings", []),
                    methodology=result.get("methodology", ""),
                    strengths=result.get("strengths", []),
                    limitations=result.get("limitations", []),
                    relevance_score=float(result.get("relevance_to_question", 0.5)),
                    themes=[],
                )
        except Exception as e:
            logger.warning(f"Analysis failed for {paper.paper_id}: {e}")

        return PaperAnalysis(
            paper_id=paper.paper_id,
            key_findings=[f"Abstract: {(paper.abstract or '')[:200]}"],
            methodology="Analysis unavailable", strengths=[], limitations=[],
            relevance_score=0.5, themes=[],
        )

    batch_size = 5
    analyses = []
    for i in range(0, len(state.papers), batch_size):
        batch = state.papers[i:i + batch_size]
        results = await asyncio.gather(*[analyze_one(p) for p in batch], return_exceptions=True)
        for r in results:
            if isinstance(r, PaperAnalysis):
                analyses.append(r)

    state.analyses = analyses
    logger.info(f"  READ done: {len(analyses)} analyzed ({time.time()-t:.1f}s)")


async def step_graphrag(model: ChatOpenAI, state: PipelineState) -> None:
    if len(state.analyses) < 5:
        logger.info(f"Pipeline step: GRAPHRAG skipped ({len(state.analyses)} < 5)")
        return

    from litscribe.tools.graphrag import build_knowledge_graph

    logger.info("Pipeline step: GRAPHRAG")
    t = time.time()

    async def llm_call(prompt: str, **kwargs) -> str:
        return await _call_llm(model, prompt)

    try:
        state.graph = await build_knowledge_graph(state.analyses, llm_call)
    except Exception as e:
        logger.warning(f"GraphRAG failed: {e}, skipping")
        state.graph = None
    n = len(state.graph.get("communities", [])) if state.graph else 0
    logger.info(f"  GRAPHRAG done: {n} communities ({time.time()-t:.1f}s)")


async def step_synthesize(model: ChatOpenAI, state: PipelineState, user_instructions: str = "") -> None:
    from litscribe.tools.synthesis import synthesize

    logger.info("Pipeline step: SYNTHESIZE")
    t = time.time()

    review = await synthesize(
        router=None, analyses=state.analyses,
        research_question=state.research_question,
        language=state.language, graph_context=state.graph,
        user_instructions=user_instructions, papers=state.papers,
        model=model,
    )
    state.synthesis = review
    logger.info(f"  SYNTHESIZE done: {review.word_count} words, {len(review.themes)} themes ({time.time()-t:.1f}s)")


async def step_review(model: ChatOpenAI, state: PipelineState) -> None:
    from litscribe.tools.review import evaluate_review

    logger.info("Pipeline step: REVIEW")
    t = time.time()

    assessment = await evaluate_review(
        router=None, review=state.synthesis, analyses=state.analyses,
        plan=state.plan, research_question=state.research_question,
        model=model,
    )
    state.assessment = assessment
    logger.info(f"  REVIEW done: score={assessment.score:.2f}, passed={assessment.passed} ({time.time()-t:.1f}s)")


async def run_review(
    model: ChatOpenAI,
    config: Config,
    state: PipelineState,
    max_papers: int = 40,
    user_instructions: str = "",
    memory=None,
) -> str:
    total_start = time.time()

    # 1. Plan
    await step_plan(model, state)

    for iteration in range(state.max_iterations):
        # 2. Search
        await step_search(model, state, config, max_papers)

        if len(state.papers) < 3:
            return f"Only found {len(state.papers)} papers. Try a broader research question."

        # 3. Read
        await step_read(model, state)

        # 4. GraphRAG
        await step_graphrag(model, state)

        # 5. Synthesize
        await step_synthesize(model, state, user_instructions)

        # 6. Review
        await step_review(model, state)

        # Check if good enough
        if state.assessment.passed or state.assessment.score >= 0.65:
            break

        # Loop-back
        if iteration < state.max_iterations - 1:
            logger.info(f"Loop-back: score={state.assessment.score:.2f}, refining queries")
            state.extra_queries = getattr(state.assessment, "refined_queries", []) or []
            state.synthesis = None
            state.assessment = None
            state.graph = None

    # Evolution: post-task evaluate
    if memory and state.assessment:
        try:
            from litscribe.evolution.skill_evolver import TaskMetrics
            metrics = TaskMetrics(
                sub_topic_count=len(state.plan.sub_topics) if state.plan else 0,
                papers_found=len(state.papers),
                papers_relevant=len(state.analyses),
                loop_back_count=max(0, state.iteration - 1),
                source_count=len({s for p in state.papers for s in p.sources}),
            )
            memory.evolver.post_task_evaluate(
                session_id=f"review-{int(total_start)}",
                domain=state.domain,
                score=state.assessment.score,
                metrics=metrics,
            )
            logger.info(f"Evolution: post_task_evaluate done (score={state.assessment.score:.2f})")
        except Exception as e:
            logger.warning(f"Evolution post_task_evaluate failed: {e}")

    elapsed = time.time() - total_start
    logger.info(f"Pipeline complete in {elapsed:.1f}s")

    return (
        f"Literature review complete.\n"
        f"- Papers: {len(state.papers)} found, {len(state.analyses)} analyzed\n"
        f"- Review: {state.synthesis.word_count} words, {len(state.synthesis.themes)} themes\n"
        f"- Score: {state.assessment.score:.2f}\n"
        f"- Time: {elapsed:.0f}s\n\n"
        f"{state.synthesis.text[:2000]}"
    )
