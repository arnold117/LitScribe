from __future__ import annotations

import json
import logging
import re
from typing import Any

from litscribe.models.assessment import ReviewAssessment
from litscribe.models.analysis import PaperAnalysis
from litscribe.models.review import ReviewOutput
from litscribe.models.plan import ResearchPlan
from litscribe.prompts.review import SELF_REVIEW_PROMPT, format_papers_for_self_review

logger = logging.getLogger(__name__)


def _msg(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def _count_words(text: str) -> int:
    cjk = len(re.findall(r'[一-鿿㐀-䶿]', text))
    latin = len(re.findall(r'[a-zA-Z]+', text))
    return cjk + latin


def _truncate_review(text: str, max_words: int = 3000) -> str:
    wc = _count_words(text)
    if wc <= max_words:
        return text
    max_chars = max_words * 2
    return text[:max_chars] + f"\n\n[... truncated, {wc} words total]"


def _fallback_assessment(error_msg: str) -> ReviewAssessment:
    return ReviewAssessment(
        passed=True,
        score=0.5,
        feedback=f"Self-review failed: {error_msg}",
        refined_queries=[],
        coverage_score=0.5,
        weak_claims=[],
    )


async def evaluate_review(
    router,
    review: ReviewOutput,
    analyses: list[PaperAnalysis],
    plan: ResearchPlan | None,
    research_question: str,
) -> ReviewAssessment:
    paper_dicts = []
    for a in analyses:
        paper_dicts.append({
            "paper_id": a.paper_id,
            "title": a.paper_id,
            "year": "N/A",
            "source": "analyzed",
            "relevance_score": a.relevance_score,
            "abstract": "; ".join(a.key_findings[:2]),
        })

    paper_list_text = format_papers_for_self_review(paper_dicts)
    review_summary = _truncate_review(review.text)

    plan_subtopics = ""
    if plan and plan.sub_topics:
        plan_subtopics = "\n".join(
            f"- {st.name}: {', '.join(st.keywords)}" for st in plan.sub_topics
        )

    themes_text = "\n".join(f"- {t.name}: {t.description}" for t in review.themes) if review.themes else "None identified"
    gaps_text = "Not analyzed"

    prompt = SELF_REVIEW_PROMPT.format(
        research_question=research_question,
        plan_subtopics=plan_subtopics or "Not available",
        num_papers=len(analyses),
        paper_list=paper_list_text,
        review_summary=review_summary,
        themes=themes_text,
        gaps=gaps_text,
    )

    try:
        result = await router.call_json(_msg(prompt), task_type="self_review")
        if isinstance(result, dict):
            score = float(result.get("overall_score", 0.5))
            coverage = float(result.get("coverage_score", 0.5))
            raw_claims = result.get("weak_claims", [])
            weak_claims = []
            for c in raw_claims:
                if isinstance(c, dict):
                    weak_claims.append(f"{c.get('claim', '')}: {c.get('issue', '')}")
                elif isinstance(c, str):
                    weak_claims.append(c)

            return ReviewAssessment(
                passed=score >= 0.65,
                score=score,
                feedback=json.dumps(result.get("suggestions", []), ensure_ascii=False),
                refined_queries=result.get("additional_queries", []),
                coverage_score=coverage,
                weak_claims=weak_claims,
            )
    except Exception as e:
        logger.warning(f"Self-review LLM failed: {e}")
        return _fallback_assessment(str(e))

    return _fallback_assessment("Unexpected response format")
