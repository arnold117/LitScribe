from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI

from litscribe.models.analysis import PaperAnalysis

logger = logging.getLogger(__name__)


@dataclass
class Contradiction:
    paper_a_id: str
    paper_b_id: str
    claim_a: str
    claim_b: str
    contradiction_type: str  # "methodological", "data_inconsistency", "opposing_conclusions"
    explanation: str
    severity: str  # "minor", "moderate", "major"


@dataclass
class ContradictionReport:
    total_pairs_checked: int = 0
    contradictions: list[Contradiction] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.contradictions)


CONTRADICTION_PROMPT = """Compare these two papers' findings and determine if there are any contradictions.

Paper A ({paper_a_id}):
Key findings:
{findings_a}

Paper B ({paper_b_id}):
Key findings:
{findings_b}

Are there any contradictions between these papers? A contradiction is when:
- Paper A's findings directly conflict with Paper B's findings
- They report opposite effects of the same intervention
- Their data or conclusions are inconsistent
- They use similar methods but get different results

Output JSON:
{{
  "has_contradiction": true/false,
  "contradictions": [
    {{
      "claim_a": "what paper A claims",
      "claim_b": "what paper B claims (contradicting A)",
      "type": "methodological|data_inconsistency|opposing_conclusions",
      "explanation": "why these contradict each other",
      "severity": "minor|moderate|major"
    }}
  ]
}}

If no contradiction, return {{"has_contradiction": false, "contradictions": []}}"""


async def _check_pair(
    model: ChatOpenAI,
    a: PaperAnalysis,
    b: PaperAnalysis,
) -> list[Contradiction]:
    findings_a = "\n".join(f"- {f}" for f in a.key_findings[:5])
    findings_b = "\n".join(f"- {f}" for f in b.key_findings[:5])

    if not findings_a.strip() or not findings_b.strip():
        return []

    prompt = CONTRADICTION_PROMPT.format(
        paper_a_id=a.paper_id,
        paper_b_id=b.paper_id,
        findings_a=findings_a,
        findings_b=findings_b,
    )

    try:
        result = await model.ainvoke(prompt)
        raw = result.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        data = json.loads(raw)
        if not data.get("has_contradiction"):
            return []

        contras = []
        for c in data.get("contradictions", []):
            contras.append(Contradiction(
                paper_a_id=a.paper_id,
                paper_b_id=b.paper_id,
                claim_a=c.get("claim_a", ""),
                claim_b=c.get("claim_b", ""),
                contradiction_type=c.get("type", "opposing_conclusions"),
                explanation=c.get("explanation", ""),
                severity=c.get("severity", "moderate"),
            ))
        return contras

    except Exception as e:
        logger.debug(f"Contradiction check failed for {a.paper_id} vs {b.paper_id}: {e}")
        return []


async def detect_contradictions(
    model: ChatOpenAI,
    analyses: list[PaperAnalysis],
    max_pairs: int = 15,
) -> ContradictionReport:
    from itertools import combinations

    pairs = list(combinations(analyses, 2))
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    logger.info(f"Contradiction detection: checking {len(pairs)} pairs from {len(analyses)} papers")

    sem = asyncio.Semaphore(3)

    async def _guarded_check(a, b):
        async with sem:
            return await _check_pair(model, a, b)

    results = await asyncio.gather(
        *[_guarded_check(a, b) for a, b in pairs],
        return_exceptions=True,
    )

    all_contradictions = []
    for r in results:
        if isinstance(r, list):
            all_contradictions.extend(r)

    report = ContradictionReport(
        total_pairs_checked=len(pairs),
        contradictions=all_contradictions,
    )

    logger.info(f"Contradictions: {report.count} found across {report.total_pairs_checked} pairs")
    return report


def format_contradictions_for_synthesis(
    report: ContradictionReport,
    key_map: dict[str, str] | None = None,
) -> str:
    if not report.contradictions:
        return ""

    lines = ["## Notable Contradictions in the Literature\n"]
    for i, c in enumerate(report.contradictions, 1):
        key_a = key_map.get(c.paper_a_id, c.paper_a_id) if key_map else c.paper_a_id
        key_b = key_map.get(c.paper_b_id, c.paper_b_id) if key_map else c.paper_b_id

        lines.append(
            f"{i}. **{c.contradiction_type.replace('_', ' ').title()}** "
            f"({c.severity}): [@{key_a}] reports: \"{c.claim_a}\", "
            f"while [@{key_b}] finds: \"{c.claim_b}\". "
            f"{c.explanation}"
        )

    return "\n".join(lines)
