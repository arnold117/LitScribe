from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from litscribe.config import Config
from litscribe.tools.status import PipelineState

logger = logging.getLogger(__name__)

BENCHMARK_QUERIES = [
    {"question": "CHO CRISPR knockout productivity", "domain": "Biology", "label": "BIO-1"},
    {"question": "transformer attention mechanisms", "domain": "Computer Science", "label": "CS-1"},
    {"question": "single-cell RNA-seq tumor microenvironment", "domain": "Medicine", "label": "MED-1"},
    {"question": "sesquiterpene coumarin biosynthesis", "domain": "Chemistry", "label": "CHEM-1"},
    {"question": "large language model reasoning", "domain": "Computer Science", "label": "CS-2"},
]


@dataclass
class BenchmarkResult:
    label: str
    question: str
    domain: str
    papers: int = 0
    words: int = 0
    score: float = 0.0
    citations: int = 0
    contradictions: int = 0
    grounding_accuracy: float = 0.0
    time_seconds: float = 0.0
    error: str = ""


async def run_single_benchmark(
    model,
    config: Config,
    query: dict,
    max_papers: int = 8,
) -> BenchmarkResult:
    import re
    from litscribe.tools.pipeline import (
        step_plan, step_search, step_read, step_contradictions,
        step_synthesize, step_ground, step_review,
    )

    result = BenchmarkResult(
        label=query["label"],
        question=query["question"],
        domain=query["domain"],
    )

    state = PipelineState(
        research_question=query["question"],
        language="en",
    )

    t = time.time()
    try:
        await step_plan(model, state)
        await step_search(model, state, config, max_papers)
        await step_read(model, state)
        await step_contradictions(model, state)

        # Lightweight synthesize for benchmark (skip comparison/timeline/stats/figures)
        from litscribe.tools.synthesis import synthesize
        review = await synthesize(
            router=None, analyses=state.analyses,
            research_question=state.research_question,
            language="en", papers=state.papers, model=model,
        )
        state.synthesis = review

        await step_ground(model, state)
        await step_review(model, state)

        result.papers = len(state.papers)
        result.words = state.synthesis.word_count if state.synthesis else 0
        result.score = state.assessment.score if state.assessment else 0.0
        result.citations = len(re.findall(r'\[@[\w]+\]', state.synthesis.text)) if state.synthesis else 0
        result.contradictions = state.contradiction_report.count if state.contradiction_report else 0
        if state.grounding_report:
            result.grounding_accuracy = state.grounding_report.accuracy
    except Exception as e:
        result.error = str(e)[:100]
        logger.error(f"Benchmark {query['label']} failed: {e}")

    result.time_seconds = time.time() - t
    return result


async def run_benchmark(
    config: Config,
    model,
    queries: list[dict] | None = None,
    max_papers: int = 8,
) -> list[BenchmarkResult]:
    if queries is None:
        queries = BENCHMARK_QUERIES

    results = []
    for q in queries:
        logger.info(f"Benchmark: {q['label']} — {q['question'][:40]}")
        r = await run_single_benchmark(model, config, q, max_papers)
        results.append(r)
        logger.info(
            f"  {r.label}: score={r.score:.2f}, papers={r.papers}, "
            f"words={r.words}, time={r.time_seconds:.0f}s"
        )

    return results


def format_benchmark_report(results: list[BenchmarkResult]) -> str:
    lines = [
        "# LitScribe Benchmark Report\n",
        "| Label | Domain | Papers | Words | Score | Citations | Contradictions | Grounding | Time |",
        "|-------|--------|--------|-------|-------|-----------|---------------|-----------|------|",
    ]

    for r in results:
        if r.error:
            lines.append(f"| {r.label} | {r.domain} | ERROR | | | | | | {r.error[:30]} |")
        else:
            lines.append(
                f"| {r.label} | {r.domain} | {r.papers} | {r.words} | "
                f"{r.score:.2f} | {r.citations} | {r.contradictions} | "
                f"{r.grounding_accuracy:.0%} | {r.time_seconds:.0f}s |"
            )

    avg_score = sum(r.score for r in results if not r.error) / max(sum(1 for r in results if not r.error), 1)
    avg_time = sum(r.time_seconds for r in results if not r.error) / max(sum(1 for r in results if not r.error), 1)
    lines.append(f"\n**Average**: score={avg_score:.2f}, time={avg_time:.0f}s")

    return "\n".join(lines)
