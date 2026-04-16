"""LitScribePipeline — deterministic workflow with loop-back on failed review.

When a MemoryManager is attached the pipeline automatically:
- injects relevant skills into the planner prompt (entry)
- absorbs paper analyses into semantic memory (each iteration)
- evaluates the task and evolves skills (exit)
"""
from __future__ import annotations

import uuid
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from litscribe.models.plan import ResearchPlan
from litscribe.models.paper import Paper
from litscribe.models.analysis import PaperAnalysis
from litscribe.models.review import ReviewOutput
from litscribe.models.assessment import ReviewAssessment

if TYPE_CHECKING:
    from litscribe.evolution.memory_manager import MemoryManager

# Type aliases for the injectable step functions
PlanFn = Callable[..., Awaitable[ResearchPlan]]
DiscoverFn = Callable[..., Awaitable[list[Paper]]]
ReadFn = Callable[..., Awaitable[list[PaperAnalysis]]]
SynthesizeFn = Callable[..., Awaitable[ReviewOutput]]
ReviewFn = Callable[..., Awaitable[ReviewAssessment]]
GraphRAGFn = Callable[..., Awaitable[Any]] | None
ProgressFn = Callable[[str, Any], None] | None


class LitScribePipeline:
    """Orchestrates: plan → (discover → read → [graphrag] → synthesize → review)×N."""

    def __init__(
        self,
        plan_fn: PlanFn,
        discover_fn: DiscoverFn,
        read_fn: ReadFn,
        synthesize_fn: SynthesizeFn,
        review_fn: ReviewFn,
        graphrag_fn: GraphRAGFn = None,
        max_iterations: int = 3,
        on_progress: ProgressFn = None,
        memory: MemoryManager | None = None,
    ) -> None:
        self.plan_fn = plan_fn
        self.discover_fn = discover_fn
        self.read_fn = read_fn
        self.synthesize_fn = synthesize_fn
        self.review_fn = review_fn
        self.graphrag_fn = graphrag_fn
        self.max_iterations = max_iterations
        self.on_progress = on_progress
        self.memory = memory

    def _progress(self, step: str, data: Any = None) -> None:
        if self.on_progress:
            self.on_progress(step, data)

    async def run(self, question: str, **kwargs: Any) -> ReviewOutput:
        """Execute the pipeline and return the final ReviewOutput."""
        session_id = kwargs.pop("session_id", None) or str(uuid.uuid4())

        # --- Entry: inject skills into planner context ---
        if self.memory is not None:
            domain_hint = kwargs.get("domain", "General")
            skill_context = self.memory.evolver.inject_skills(domain_hint, "planning")
            if skill_context:
                kwargs["memory_context"] = skill_context

        self._progress("planning", question)
        plan: ResearchPlan = await self.plan_fn(question, **kwargs)
        self._progress("plan_ready", plan)

        extra_queries: list[str] = []
        output: ReviewOutput | None = None
        all_analyses: list[PaperAnalysis] = []
        all_papers: list[Paper] = []
        loop_back_count = 0
        sources_seen: set[str] = set()

        for iteration in range(self.max_iterations):
            self._progress("iteration_start", iteration)

            # Discovery — pass refined queries if we have them
            discover_kwargs = dict(kwargs)
            if extra_queries:
                discover_kwargs["extra_queries"] = extra_queries

            self._progress("discovering", plan)
            papers: list[Paper] = await self.discover_fn(plan, **discover_kwargs)
            self._progress("discovered", papers)
            all_papers.extend(papers)
            for p in papers:
                sources_seen.update(p.sources.keys())

            # Critical reading
            self._progress("reading", papers)
            analyses: list[PaperAnalysis] = await self.read_fn(papers, **kwargs)
            self._progress("read", analyses)
            all_analyses.extend(analyses)

            # --- Absorb into semantic memory after each read ---
            if self.memory is not None:
                self.memory.semantic.absorb(analyses)

            # Optional GraphRAG enrichment
            if self.graphrag_fn is not None:
                self._progress("graphrag", analyses)
                analyses = await self.graphrag_fn(analyses, **kwargs) or analyses
                self._progress("graphrag_done", analyses)

            # Synthesis
            self._progress("synthesizing", analyses)
            output = await self.synthesize_fn(analyses, plan=plan, **kwargs)
            self._progress("synthesized", output)

            # Self-review
            self._progress("reviewing", output)
            assessment: ReviewAssessment = await self.review_fn(
                output, plan=plan, **kwargs
            )
            self._progress("reviewed", assessment)

            if assessment.passed:
                break

            # Prepare for next iteration using refined queries from assessment
            extra_queries = assessment.refined_queries or []
            loop_back_count += 1

        assert output is not None, "Pipeline produced no output"

        # --- Exit: evaluate task and evolve skills ---
        if self.memory is not None:
            from litscribe.evolution.skill_evolver import TaskMetrics

            relevant_count = sum(
                1 for a in all_analyses if a.relevance_score >= 0.7
            )
            metrics = TaskMetrics(
                sub_topic_count=len(plan.sub_topics),
                papers_found=len(all_papers),
                papers_relevant=relevant_count,
                loop_back_count=loop_back_count,
                source_count=len(sources_seen),
            )
            await self.memory.evolver.post_task_evaluate(
                session_id=session_id,
                question=question,
                score=assessment.score,
                metrics=metrics,
                domain=plan.domain,
                trace_summary=(
                    f"Question: {question}. "
                    f"Sub-topics: {[st.name for st in plan.sub_topics]}. "
                    f"Papers: {metrics.papers_relevant}/{metrics.papers_found} relevant. "
                    f"Loop-backs: {loop_back_count}. Score: {assessment.score}."
                ),
            )

        return output
