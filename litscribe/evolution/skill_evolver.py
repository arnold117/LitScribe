"""Skill evolver — post-task evaluation and skill lifecycle management.

Uses multi-signal trigger conditions instead of a single complexity int
to decide whether a task outcome is worth extracting as a reusable skill.
"""
from __future__ import annotations

from dataclasses import dataclass

from litscribe.evolution.episodic import EpisodicMemory
from litscribe.evolution.procedural import ProceduralMemory

SCORE_THRESHOLD = 0.7
FAILURE_THRESHOLD = 0.5


@dataclass
class TaskMetrics:
    """Signals collected from a single pipeline run."""

    sub_topic_count: int = 1
    papers_found: int = 0
    papers_relevant: int = 0
    loop_back_count: int = 0
    source_count: int = 1

    @property
    def signal_to_noise(self) -> float:
        """Fraction of discovered papers that were actually relevant."""
        if self.papers_found == 0:
            return 1.0
        return self.papers_relevant / self.papers_found

    @property
    def is_non_trivial(self) -> bool:
        """True when any trigger condition fires — meaning the task was
        complex enough that a successful strategy is worth remembering."""
        return (
            self.sub_topic_count >= 3
            or self.signal_to_noise < 0.5
            or self.loop_back_count >= 2
            or self.source_count > 2
        )


class SkillEvolver:
    """Evaluates task outcomes and evolves the procedural skill library."""

    def __init__(self, episodic: EpisodicMemory, procedural: ProceduralMemory):
        self.episodic = episodic
        self.procedural = procedural

    def should_extract_skill(self, score: float, metrics: TaskMetrics) -> bool:
        """Return True when the outcome warrants extracting a reusable skill.

        High score + non-trivial task = worth remembering.
        """
        return score >= SCORE_THRESHOLD and metrics.is_non_trivial

    async def post_task_evaluate(
        self,
        session_id: str,
        question: str,
        score: float,
        metrics: TaskMetrics,
        domain: str,
        trace_summary: str,
        used_skills: list[str] | None = None,
    ) -> None:
        """Record the outcome and update the skill library accordingly."""
        # Record episode regardless of outcome
        await self.episodic.record(
            session_id=session_id,
            question=question,
            outcome_score=score,
            key_events=[
                f"score={score}, subs={metrics.sub_topic_count}, "
                f"papers={metrics.papers_relevant}/{metrics.papers_found}, "
                f"loops={metrics.loop_back_count}, sources={metrics.source_count}",
                trace_summary,
            ],
        )

        if self.should_extract_skill(score, metrics):
            existing = self.procedural.find_relevant(question, n=1)
            if existing and existing[0].get("domain") == domain:
                self.procedural.patch_skill(
                    existing[0]["slug"],
                    adjustment=f"Refined from session {session_id[:8]} (score={score})",
                    score=score,
                )
            else:
                self.procedural.save_skill(
                    name=f"{domain} strategy from {session_id[:8]}",
                    domain=domain,
                    trigger=f"Research questions about {domain}",
                    strategy=trace_summary,
                    learned_adjustments=[],
                    success_rate=score,
                )

        if score < FAILURE_THRESHOLD:
            await self.record_failure(session_id, question, score, trace_summary)
            if used_skills:
                for slug in used_skills:
                    skill = self.procedural.get_skill(slug)
                    if skill:
                        self.procedural.patch_skill(
                            slug,
                            adjustment=f"NEEDS REVIEW: failed in session {session_id[:8]}",
                        )

    async def record_failure(
        self,
        session_id: str,
        question: str,
        score: float,
        feedback: str,
    ) -> None:
        """Persist a failure episode for future learning."""
        await self.episodic.record(
            session_id=session_id,
            question=question,
            outcome_score=score,
            key_events=[f"FAILURE (score={score}): {feedback}"],
        )

    def inject_skills(self, domain: str, task_type: str) -> str:
        """Build a prompt snippet with relevant skills for the given context."""
        relevant = self.procedural.find_relevant(f"{domain} {task_type}", n=3)
        if not relevant:
            return ""
        parts = ["\n--- Relevant Skills ---"]
        for skill in relevant:
            parts.append(
                f"\n### {skill['name']} (v{skill['version']}, success={skill['success_rate']})"
            )
            parts.append(f"Trigger: {skill['trigger']}")
            parts.append(f"Strategy:\n{skill['strategy']}")
        parts.append("\n--- End Skills ---\n")
        return "\n".join(parts)
