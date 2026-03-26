"""LitScribePipeline — deterministic workflow with loop-back on failed review."""
from __future__ import annotations

from typing import Any, Callable, Awaitable

from litscribe.models.plan import ResearchPlan
from litscribe.models.paper import Paper
from litscribe.models.analysis import PaperAnalysis
from litscribe.models.review import ReviewOutput
from litscribe.models.assessment import ReviewAssessment

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
    ) -> None:
        self.plan_fn = plan_fn
        self.discover_fn = discover_fn
        self.read_fn = read_fn
        self.synthesize_fn = synthesize_fn
        self.review_fn = review_fn
        self.graphrag_fn = graphrag_fn
        self.max_iterations = max_iterations
        self.on_progress = on_progress

    def _progress(self, step: str, data: Any = None) -> None:
        if self.on_progress:
            self.on_progress(step, data)

    async def run(self, question: str, **kwargs: Any) -> ReviewOutput:
        """Execute the pipeline and return the final ReviewOutput."""
        self._progress("planning", question)
        plan: ResearchPlan = await self.plan_fn(question, **kwargs)
        self._progress("plan_ready", plan)

        extra_queries: list[str] = []
        output: ReviewOutput | None = None

        for iteration in range(self.max_iterations):
            self._progress("iteration_start", iteration)

            # Discovery — pass refined queries if we have them
            discover_kwargs = dict(kwargs)
            if extra_queries:
                discover_kwargs["extra_queries"] = extra_queries

            self._progress("discovering", plan)
            papers: list[Paper] = await self.discover_fn(plan, **discover_kwargs)
            self._progress("discovered", papers)

            # Critical reading
            self._progress("reading", papers)
            analyses: list[PaperAnalysis] = await self.read_fn(papers, **kwargs)
            self._progress("read", analyses)

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

        assert output is not None, "Pipeline produced no output"
        return output
