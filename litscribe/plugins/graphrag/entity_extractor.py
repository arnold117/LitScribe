"""Entity extractor — uses injected LLM call to extract named entities from paper analyses."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Awaitable

from litscribe.models.analysis import PaperAnalysis


def _entity_id(name: str, entity_type: str) -> str:
    normalized = f"{name.lower().strip()}:{entity_type.lower().strip()}"
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


async def extract_entities(
    analyses: list[PaperAnalysis],
    llm_call: Callable[..., Awaitable[str]],
) -> tuple[list[dict], list[dict]]:
    """Extract entities from a list of PaperAnalysis objects via LLM.

    Returns (entities, mentions) where:
    - entities: list of entity dicts with keys: entity_id, name, entity_type,
      aliases, description, paper_ids, frequency
    - mentions: list of mention dicts with keys: entity_id, paper_id, context,
      section, confidence
    """
    entities: list[dict] = []
    mentions: list[dict] = []

    for analysis in analyses:
        prompt = (
            f"Extract entities from: {'; '.join(analysis.key_findings)}\n"
            "Return JSON array of objects with: name, entity_type "
            "(method/dataset/metric/concept), description, aliases (list)."
        )
        raw = await llm_call(prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

        try:
            extracted = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if not isinstance(extracted, list):
            extracted = [extracted]

        for e in extracted:
            eid = _entity_id(e.get("name", ""), e.get("entity_type", "concept"))
            entity: dict[str, Any] = {
                "entity_id": eid,
                "name": e.get("name", ""),
                "entity_type": e.get("entity_type", "concept"),
                "aliases": e.get("aliases", []),
                "description": e.get("description", ""),
                "paper_ids": [analysis.paper_id],
                "frequency": 1,
            }
            entities.append(entity)
            mentions.append({
                "entity_id": eid,
                "paper_id": analysis.paper_id,
                "context": "; ".join(analysis.key_findings),
                "section": "findings",
                "confidence": 0.8,
            })

    return entities, mentions
