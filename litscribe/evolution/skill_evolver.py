"""Skill evolver — post-task evaluation and skill lifecycle management."""
from __future__ import annotations

from litscribe.evolution.episodic import EpisodicMemory
from litscribe.evolution.procedural import ProceduralMemory

SCORE_THRESHOLD = 0.7
COMPLEXITY_THRESHOLD = 5
FAILURE_THRESHOLD = 0.5


class SkillEvolver:
    """Evaluates task outcomes and evolves the procedural skill library."""

    def __init__(self, episodic: EpisodicMemory, procedural: ProceduralMemory):
        self.episodic = episodic
        self.procedural = procedural

    def should_extract_skill(self, score: float, complexity: int) -> bool:
        """Return True when the outcome warrants extracting a reusable skill."""
        return score >= SCORE_THRESHOLD and complexity >= COMPLEXITY_THRESHOLD

    async def post_task_evaluate(
        self,
        session_id: str,
        question: str,
        score: float,
        complexity: int,
        domain: str,
        trace_summary: str,
        used_skills: list[str] | None = None,
    ) -> None:
        """Record the outcome and update the skill library accordingly."""
        if self.should_extract_skill(score, complexity):
            existing = self.procedural.find_relevant(question, n=1)
            if existing and existing[0].get("domain") == domain:
                self.procedural.patch_skill(
                    existing[0]["slug"],
                    adjustment=f"Refined from session {session_id[:8]} (score={score})",
                )
            else:
                self.procedural.save_skill(
                    name=f"{domain} strategy from {session_id[:8]}",
                    domain=domain,
                    trigger=f"Research questions about {domain}",
                    strategy=trace_summary,
                    learned_adjustments=[],
                )
        if score < FAILURE_THRESHOLD:
            await self.record_failure(session_id, question, score, trace_summary)
            if used_skills:
                for slug in used_skills:
                    skill = self.procedural.get_skill(slug)
                    if skill:
                        self.procedural.patch_skill(
                            slug,
                            adjustment=f"NEEDS REVIEW: failed in session {session_id}",
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
