from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from litscribe.tools.status import PipelineState

logger = logging.getLogger(__name__)

METACOG_PROMPT = """You are a metacognitive quality controller for a literature review pipeline.

Current pipeline results:
- Papers found: {papers_found}
- Papers analyzed: {analyses_count}
- Review score: {score}
- Coverage score: {coverage}
- Contradictions found: {contradictions}
- Grounding accuracy: {grounding_accuracy}%
- Word count: {word_count}
- Themes: {themes}

Review weaknesses (from self-review):
{feedback}

Decide which pipeline steps should be RE-RUN to improve quality:

Available steps to re-run:
1. SEARCH — if too few papers or poor coverage
2. READ — if analyses are shallow
3. SYNTHESIZE — if review structure or flow is weak
4. GROUND — if citation accuracy is low

Rules:
- Only re-run steps that would actually help
- If score >= 0.8 and grounding >= 70%, no re-run needed
- Maximum 2 steps to re-run

Output JSON:
{{
  "should_rerun": true/false,
  "steps_to_rerun": ["SEARCH", "SYNTHESIZE"],
  "reasoning": "Why these steps need re-running",
  "strategy_adjustment": "What to do differently this time"
}}"""


async def metacognitive_evaluate(
    model: ChatOpenAI,
    state: PipelineState,
) -> dict:
    score = state.assessment.score if state.assessment else 0
    coverage = state.assessment.coverage_score if state.assessment else 0
    grounding_acc = int(state.grounding_report.accuracy * 100) if state.grounding_report else 0
    contradictions = state.contradiction_report.count if state.contradiction_report else 0
    themes = [t.name for t in state.synthesis.themes] if state.synthesis else []
    feedback = state.assessment.feedback if state.assessment else ""

    prompt = METACOG_PROMPT.format(
        papers_found=len(state.papers),
        analyses_count=len(state.analyses),
        score=f"{score:.2f}",
        coverage=f"{coverage:.2f}",
        contradictions=contradictions,
        grounding_accuracy=grounding_acc,
        word_count=state.synthesis.word_count if state.synthesis else 0,
        themes=", ".join(themes),
        feedback=feedback[:500],
    )

    try:
        result = await model.ainvoke(prompt)
        raw = result.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        decision = json.loads(raw)
        logger.info(
            f"Metacognition: rerun={decision.get('should_rerun')}, "
            f"steps={decision.get('steps_to_rerun', [])}, "
            f"reason={decision.get('reasoning', '')[:60]}"
        )
        return decision
    except Exception as e:
        logger.warning(f"Metacognitive evaluation failed: {e}")
        return {"should_rerun": False, "steps_to_rerun": [], "reasoning": "evaluation failed"}


async def save_strategy(config, domain: str, strategy: str):
    try:
        from litscribe.store.knowledge import KnowledgeStore
        kb = KnowledgeStore(config.db_path)
        from litscribe.models.analysis import PaperAnalysis
        strategy_entry = PaperAnalysis(
            paper_id="strategy",
            key_findings=[f"Strategy for {domain}: {strategy}"],
            methodology="", strengths=[], limitations=[],
            relevance_score=1.0, themes=[],
        )
        await kb.save_findings(domain, "pipeline_strategy", [strategy_entry])
    except Exception as e:
        logger.debug(f"Strategy save failed: {e}")
