from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def build_claim_chain(
    review_text: str,
    grounding_report,
    key_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not grounding_report or not grounding_report.claims:
        return {"claims": [], "text": review_text}

    claims_data = []
    for claim in grounding_report.claims:
        cite_key = ""
        if key_map and claim.paper_id:
            cite_key = key_map.get(claim.paper_id, claim.paper_id)

        claims_data.append({
            "author": claim.author,
            "year": claim.year,
            "cite_key": cite_key,
            "paper_id": claim.paper_id or "",
            "claim": claim.claim[:200],
            "supported": claim.supported,
            "evidence": claim.evidence[:200] if claim.evidence else "",
        })

    return {
        "claims": claims_data,
        "total": grounding_report.total_citations,
        "verified": grounding_report.verified,
        "unsupported": grounding_report.unsupported,
        "accuracy": grounding_report.accuracy,
        "text": review_text,
    }


def claims_to_html_annotations(review_text: str, claims_data: list[dict]) -> str:
    annotated = review_text
    for c in claims_data:
        if not c.get("cite_key"):
            continue

        key = c["cite_key"]
        pattern = f"[@{key}]"

        if c.get("supported") is True:
            badge = f'<span title="Verified: {c.get("evidence", "")[:100]}" style="color:green">[@{key}]✓</span>'
        elif c.get("supported") is False:
            badge = f'<span title="Unsupported — auto-fixed" style="color:red">[@{key}]✗</span>'
        else:
            badge = f'<span title="Unverified" style="color:orange">[@{key}]?</span>'

        annotated = annotated.replace(pattern, badge, 1)

    return annotated
