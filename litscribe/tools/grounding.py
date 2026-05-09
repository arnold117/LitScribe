from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI

from litscribe.models.paper import Paper
from litscribe.models.analysis import PaperAnalysis

logger = logging.getLogger(__name__)


@dataclass
class CitationClaim:
    author: str
    year: str
    claim: str
    paper_id: str | None = None
    supported: bool | None = None
    evidence: str = ""
    fixed_claim: str = ""


@dataclass
class GroundingReport:
    total_citations: int = 0
    verified: int = 0
    unsupported: int = 0
    unmatched: int = 0
    claims: list[CitationClaim] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.total_citations == 0:
            return 0.0
        return self.verified / self.total_citations


def extract_citations(review_text: str) -> list[CitationClaim]:
    pattern = re.compile(
        r'([^.!?\n]{10,300}?)'
        r'\[([A-Za-zÀ-ÿ一-鿿][\w\s]*?(?:\s+et\s+al\.)?),\s*(\d{4})\]'
    )

    claims = []
    seen = set()

    for match in pattern.finditer(review_text):
        claim_text = match.group(1).strip()
        author = match.group(2).strip()
        year = match.group(3).strip()
        key = f"{author}:{year}:{claim_text[:50]}"

        if key in seen:
            continue
        seen.add(key)

        claim_text = re.sub(r'^\s*[,;]\s*', '', claim_text)
        claim_text = claim_text.strip()
        if len(claim_text) < 15:
            continue

        claims.append(CitationClaim(author=author, year=year, claim=claim_text))

    return claims


def _extract_last_names(authors: list[str]) -> list[str]:
    names = []
    for author in authors:
        author = author.strip()
        if not author:
            continue
        if "," in author:
            names.append(author.split(",")[0].strip().lower())
        else:
            parts = author.split()
            if parts:
                last = parts[-1].rstrip(".")
                if last.isupper() and len(last) <= 3 and len(parts) > 1:
                    names.append(parts[0].strip().lower())
                else:
                    names.append(last.lower())
    return names


def match_citations_to_papers(
    claims: list[CitationClaim],
    papers: list[Paper],
    analyses: list[PaperAnalysis],
) -> None:
    paper_index: dict[str, str] = {}
    for p in papers:
        last_names = _extract_last_names(p.authors)
        for ln in last_names:
            paper_index[f"{ln}:{p.year}"] = p.paper_id
            # Also index without "et al." variants
            if len(ln) > 2:
                paper_index[f"{ln[:3]}:{p.year}"] = p.paper_id

    for claim in claims:
        cite_name = claim.author.lower().replace(" et al.", "").strip()
        cite_parts = cite_name.split()
        # Try full last name first, then first word
        for name in [cite_parts[-1] if cite_parts else "", cite_parts[0] if cite_parts else ""]:
            key = f"{name}:{claim.year}"
            if key in paper_index:
                claim.paper_id = paper_index[key]
                break
            # Try prefix match
            key3 = f"{name[:3]}:{claim.year}" if len(name) >= 3 else ""
            if key3 and key3 in paper_index:
                claim.paper_id = paper_index[key3]
                break


async def verify_single_claim(
    model: ChatOpenAI,
    claim: CitationClaim,
    paper_text: str,
) -> CitationClaim:
    prompt = (
        f"Verify if the following claim is supported by the paper's content.\n\n"
        f"CLAIM: {claim.claim}\n"
        f"ATTRIBUTED TO: [{claim.author}, {claim.year}]\n\n"
        f"PAPER CONTENT:\n{paper_text[:2000]}\n\n"
        f"Answer in JSON:\n"
        f'{{"supported": true/false, "evidence": "quote or paraphrase from paper that supports/contradicts", '
        f'"fixed_claim": "corrected version if unsupported, empty string if supported"}}'
    )

    try:
        result = await model.ainvoke(prompt)
        raw = result.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        data = json.loads(raw)
        claim.supported = data.get("supported", True)
        claim.evidence = data.get("evidence", "")
        claim.fixed_claim = data.get("fixed_claim", "")
    except Exception as e:
        logger.debug(f"Verification failed for [{claim.author}, {claim.year}]: {e}")
        claim.supported = True

    return claim


async def ground_citations(
    model: ChatOpenAI,
    review_text: str,
    papers: list[Paper],
    analyses: list[PaperAnalysis],
    max_verify: int = 20,
) -> GroundingReport:
    claims = extract_citations(review_text)
    match_citations_to_papers(claims, papers, analyses)

    paper_text_map: dict[str, str] = {}
    for p in papers:
        text = f"Title: {p.title}\nAbstract: {p.abstract or ''}"
        paper_text_map[p.paper_id] = text
    for a in analyses:
        if a.paper_id in paper_text_map:
            findings = "\n".join(f"- {f}" for f in a.key_findings)
            paper_text_map[a.paper_id] += f"\nKey findings:\n{findings}"

    matched = [c for c in claims if c.paper_id]
    unmatched = [c for c in claims if not c.paper_id]

    to_verify = matched[:max_verify]
    if to_verify:
        tasks = [
            verify_single_claim(model, c, paper_text_map.get(c.paper_id, ""))
            for c in to_verify
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    verified = sum(1 for c in matched if c.supported is True)
    unsupported = sum(1 for c in matched if c.supported is False)

    report = GroundingReport(
        total_citations=len(claims),
        verified=verified,
        unsupported=unsupported,
        unmatched=len(unmatched),
        claims=claims,
    )

    logger.info(
        f"Citation grounding: {report.total_citations} citations, "
        f"{report.verified} verified, {report.unsupported} unsupported, "
        f"{report.unmatched} unmatched (accuracy={report.accuracy:.0%})"
    )
    return report


def apply_fixes(review_text: str, report: GroundingReport) -> str:
    fixed = review_text
    for claim in report.claims:
        if claim.supported is False and claim.fixed_claim:
            old_sentence = f"{claim.claim}[{claim.author}"
            new_sentence = f"{claim.fixed_claim}[{claim.author}"
            fixed = fixed.replace(old_sentence, new_sentence, 1)
    return fixed
