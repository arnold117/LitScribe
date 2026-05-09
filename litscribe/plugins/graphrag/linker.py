"""Entity linker — deduplicates and merges entity records by entity_id."""
from __future__ import annotations


def link_entities(entities: list[dict]) -> list[dict]:
    """Merge duplicate entities (same entity_id) by accumulating frequency,
    paper_ids and aliases.

    Returns a deduplicated list of entity dicts.
    """
    seen: dict[str, dict] = {}
    for e in entities:
        eid = e["entity_id"]
        if eid in seen:
            seen[eid]["frequency"] += e["frequency"]
            seen[eid]["paper_ids"] = list(set(seen[eid]["paper_ids"] + e["paper_ids"]))
            seen[eid]["aliases"] = list(set(seen[eid]["aliases"] + e["aliases"]))
        else:
            seen[eid] = dict(e)
    return list(seen.values())
